"""
BF16-vs-FP4 accuracy comparison.

Runs the same prompt set against two endpoints (partner FP4 service and
your own BF16 reference) and reports:

  - Accuracy on each
  - Accuracy gap (BF16 - FP4)
  - Per-prompt disagreement breakdown (BF16 right & FP4 wrong, etc.)
  - Output divergence: at temperature=0, how many prompts produce
    byte-identical outputs vs different outputs

This is what the validation plan calls for: "compare partner FP4 to our
own BF16 reference on the same prompts."

Usage:
  python -m benchmarks.compare_bf16_fp4 \\
      --partner-url https://partner/v1 \\
      --partner-model gpt-oss-120b \\
      --reference-url http://your-vllm:8000/v1 \\
      --reference-model openai/gpt-oss-120b \\
      --prompts prompt_sets/gsm8k_standard.jsonl \\
                prompt_sets/gsm8k_perturbed.jsonl \\
                prompt_sets/mmlu_standard.jsonl \\
                prompt_sets/mmlu_permuted.jsonl \\
      --output-dir out/compare
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from common import Endpoint, stream_chat, write_json
from accuracy_harness import (
    accuracy,
    evaluate,
    grade_mmlu_letter,
    grade_numeric,
    load_jsonl,
)


async def run_endpoint(
    name: str, endpoint: Endpoint, prompts: list[dict],
    concurrency: int, max_tokens: int,
) -> list[dict]:
    print(f"  [{name}] running {len(prompts)} prompts against "
          f"{endpoint.base_url} ...")
    results = await evaluate(
        endpoint, prompts, concurrency=concurrency, max_tokens=max_tokens
    )
    acc = accuracy(results)
    print(f"  [{name}] accuracy: {acc:.4f} "
          f"({sum(1 for r in results if r['correct'])}/{len(results)})")
    return results


def cross_tabulate(
    bf16_results: list[dict], fp4_results: list[dict]
) -> dict:
    """4-way breakdown of per-prompt correctness."""
    bf16_by_id = {r["id"]: r for r in bf16_results}
    fp4_by_id = {r["id"]: r for r in fp4_results}

    both_right = 0
    both_wrong = 0
    bf16_only_right = 0   # FP4 regressed
    fp4_only_right = 0    # FP4 happened to get one BF16 missed
    output_identical = 0
    output_different = 0

    disagreement_examples: list[dict] = []

    for pid, bf16 in bf16_by_id.items():
        fp4 = fp4_by_id.get(pid)
        if fp4 is None:
            continue
        b = bf16["correct"]
        f = fp4["correct"]
        if b and f:
            both_right += 1
        elif not b and not f:
            both_wrong += 1
        elif b and not f:
            bf16_only_right += 1
            if len(disagreement_examples) < 10:
                disagreement_examples.append({
                    "id": pid,
                    "expected": bf16["expected"],
                    "bf16_response": bf16["response"][:200],
                    "fp4_response": fp4["response"][:200],
                    "regression": "FP4 wrong, BF16 right",
                })
        else:
            fp4_only_right += 1

        # Output identity check (regardless of correctness)
        if bf16["response"].strip() == fp4["response"].strip():
            output_identical += 1
        else:
            output_different += 1

    n = both_right + both_wrong + bf16_only_right + fp4_only_right
    return {
        "n_compared": n,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "fp4_regressed": bf16_only_right,  # the concerning category
        "fp4_recovered": fp4_only_right,
        "output_identical_count": output_identical,
        "output_different_count": output_different,
        "output_identity_rate": (
            output_identical / (output_identical + output_different)
            if (output_identical + output_different) > 0 else 0.0
        ),
        "disagreement_examples": disagreement_examples,
    }


async def main_async(args: argparse.Namespace) -> None:
    partner = Endpoint(
        base_url=args.partner_url,
        api_key=args.partner_key or "dummy",
        model=args.partner_model,
    )
    reference = Endpoint(
        base_url=args.reference_url,
        api_key=args.reference_key or "dummy",
        model=args.reference_model,
    )

    print(f"[compare] partner   = {partner.base_url} ({partner.model})")
    print(f"[compare] reference = {reference.base_url} ({reference.model})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_set_summary: dict[str, dict] = {}
    verdicts: list[str] = []

    for prompt_file in args.prompts:
        name = Path(prompt_file).stem
        print(f"\n[compare] === {name} ===")
        prompts = load_jsonl(Path(prompt_file))
        if args.limit:
            prompts = prompts[: args.limit]

        bf16_results = await run_endpoint(
            "BF16 ref", reference, prompts,
            args.concurrency, args.max_tokens,
        )
        fp4_results = await run_endpoint(
            "FP4 partner", partner, prompts,
            args.concurrency, args.max_tokens,
        )

        bf16_acc = accuracy(bf16_results)
        fp4_acc = accuracy(fp4_results)
        gap = bf16_acc - fp4_acc
        tabs = cross_tabulate(bf16_results, fp4_results)

        per_set_summary[name] = {
            "n": len(prompts),
            "bf16_accuracy": bf16_acc,
            "fp4_accuracy": fp4_acc,
            "gap_bf16_minus_fp4": gap,
            "cross_tab": {k: v for k, v in tabs.items()
                          if k != "disagreement_examples"},
            "disagreement_examples": tabs["disagreement_examples"],
        }

        # Per-set verdict
        print(f"  BF16 accuracy: {bf16_acc:.4f}")
        print(f"  FP4  accuracy: {fp4_acc:.4f}")
        print(f"  Gap (BF16 - FP4): {gap:+.4f}")
        print(f"  Output identical: {tabs['output_identity_rate']:.2%} "
              f"({tabs['output_identical_count']}/{tabs['output_identical_count'] + tabs['output_different_count']})")
        print(f"  FP4 regressed on {tabs['fp4_regressed']} prompts that BF16 got right")

        if gap > 0.03:
            verdicts.append(
                f"[{name}] FP4 accuracy is {gap*100:.1f} pp below BF16 "
                f"reference ({fp4_acc:.3f} vs {bf16_acc:.3f}). "
                "Exceeds 3 pp red-flag threshold; investigate quantization."
            )
        elif gap > 0.01:
            verdicts.append(
                f"[{name}] FP4 accuracy is {gap*100:.1f} pp below BF16 "
                "(1-3 pp; within expected quantization noise but worth tracking)."
            )

        if tabs["output_identity_rate"] < 0.1 and args.max_tokens > 64:
            verdicts.append(
                f"[{name}] FP4 and BF16 produce identical output on only "
                f"{tabs['output_identity_rate']:.1%} of prompts at temperature=0. "
                "High divergence suggests aggressive quantization, different "
                "sampling, or different checkpoints. Confirm the partner is "
                "running the same model checkpoint."
            )

        # Save per-set raw results
        write_json(out_dir / f"{name}_bf16.json", {"results": bf16_results})
        write_json(out_dir / f"{name}_fp4.json", {"results": fp4_results})

    # Aggregate verdicts
    if per_set_summary:
        gaps = {n: s["gap_bf16_minus_fp4"] for n, s in per_set_summary.items()}
        # Look for differential degradation: bigger gap on perturbed than standard
        for name in per_set_summary:
            if "perturbed" in name or "permuted" in name or "filler" in name:
                base = name.replace("_perturbed", "_standard") \
                           .replace("_permuted", "_standard") \
                           .replace("_filler", "_standard")
                if base in gaps:
                    differential = gaps[name] - gaps[base]
                    if differential > 0.02:
                        verdicts.append(
                            f"[{name}] FP4 degrades MORE on perturbed set than "
                            f"on standard set (gap differential = {differential*100:+.1f} pp). "
                            "Strongly suggests benchmark-tuned quantization "
                            "or surface-pattern overfitting -- the FP4 model "
                            "performs well on standard benchmark surface but "
                            "worse on equivalent reformulations."
                        )

    print("\n=== OVERALL VERDICTS ===")
    if verdicts:
        for v in verdicts:
            print(f"  - {v}")
    else:
        print("  (no red flags above thresholds)")

    write_json(
        out_dir / "compare_summary.json",
        {
            "partner": {
                "base_url": partner.base_url, "model": partner.model
            },
            "reference": {
                "base_url": reference.base_url, "model": reference.model
            },
            "per_set": per_set_summary,
            "verdicts": verdicts,
        },
    )
    print(f"\n[compare] wrote {out_dir / 'compare_summary.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--partner-url", required=True)
    p.add_argument("--partner-key", default="dummy")
    p.add_argument("--partner-model", required=True)
    p.add_argument("--reference-url", required=True,
                   help="Your own BF16 vLLM (or similar) endpoint URL")
    p.add_argument("--reference-key", default="dummy")
    p.add_argument("--reference-model", required=True,
                   help="Model string the reference endpoint expects")
    p.add_argument("--prompts", nargs="+", required=True,
                   help="JSONL prompt sets to compare on")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap prompts per set (0 = use all)")
    p.add_argument("--output-dir", default="out/compare")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
