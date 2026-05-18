"""
Accuracy harness.

Two paths, both supported here:

1. Standard benchmarks via lm-evaluation-harness.
   Uses the harness's openai-chat-completions backend pointed at the
   partner endpoint. Run MMLU, GSM8K, MMLU-Pro, HumanEval (via OpenAI eval
   harness; HumanEval+ recommended via EvalPlus separately).

   See cmd_standard() below for the exact commands.

2. Custom perturbation-based evaluation: run a held-out set against the
   partner endpoint AND against your own BF16 reference, compare
   accuracy on (a) standard, (b) perturbed, (c) held-out. This is what
   actually catches FP4 calibration overfit.

This file implements (2). For (1) we just emit the commands.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx

from common import Endpoint, stream_chat, write_json


# ---------------------------------------------------------------------------
# (1) Standard-benchmark commands (for reproducibility -- copy/paste)
# ---------------------------------------------------------------------------

def cmd_standard(endpoint: Endpoint, output_dir: Path) -> str:
    """Emit lm-eval-harness commands to run against partner endpoint."""
    return f"""\
# Standard accuracy benchmarks via lm-evaluation-harness
# Install: pip install lm-eval[api]
# Docs: https://github.com/EleutherAI/lm-evaluation-harness

export OPENAI_API_KEY={endpoint.api_key}

# --- Knowledge / reasoning ---
lm_eval --model local-chat-completions \\
    --tasks mmlu,gsm8k_cot,mmlu_pro \\
    --model_args base_url={endpoint.base_url}/chat/completions,model={endpoint.model},num_concurrent=4,max_retries=3 \\
    --apply_chat_template \\
    --output_path {output_dir}/lm_eval_standard \\
    --log_samples

# --- Code (HumanEval+ via EvalPlus -- separate tool) ---
# pip install evalplus[vllm]
# evalplus.evaluate --model {endpoint.model} \\
#     --backend openai --base-url {endpoint.base_url} \\
#     --dataset humaneval --greedy

# --- Long context: RULER at 8K ---
# Use github.com/hsiehjackson/RULER ; configure for OpenAI-compatible API.

# --- LiveBench (contamination-resistant) ---
# git clone https://github.com/LiveBench/LiveBench
# Configure with your endpoint; LiveBench is rotating so always pull latest.
"""


# ---------------------------------------------------------------------------
# (2) Custom evaluator with perturbations
# ---------------------------------------------------------------------------

async def _ask(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    prompt: str,
    max_tokens: int = 256,
) -> str:
    r = await stream_chat(
        client, endpoint, prompt,
        max_tokens=max_tokens, temperature=0.0,
    )
    if r.error:
        return ""
    return r.full_text


# ----- GSM8K-style grading -----

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_number(text: str) -> float | None:
    """Pull the last number from the model's response."""
    # Prefer text after "####" if the model used the standard format
    if "####" in text:
        text = text.split("####")[-1]
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def grade_numeric(expected: str, response: str, tol: float = 1e-3) -> bool:
    """Check if the response's final number matches the expected value."""
    try:
        e = float(expected.replace(",", "").strip())
    except ValueError:
        return False
    got = extract_final_number(response)
    if got is None:
        return False
    return abs(got - e) <= tol * max(1.0, abs(e))


# ----- MMLU-style grading -----

_MMLU_LETTER_RE = re.compile(r"\b([A-J])\b")


def grade_mmlu_letter(expected_letter: str, response: str) -> bool:
    """Look for the answer letter (A-J) in the response."""
    expected = expected_letter.strip().upper()
    # Common patterns: "The answer is X", "Answer: X", or just "X"
    m = re.search(r"answer[^A-J]*([A-J])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper() == expected
    # Fall back to last standalone letter
    matches = _MMLU_LETTER_RE.findall(response)
    if not matches:
        return False
    return matches[-1].upper() == expected


# ----- Prompt set loader -----

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL prompt set. Each record needs:
        {"id": str, "prompt": str, "expected": str, "type": "numeric"|"letter"}
    Optional: "perturbed_from": str (id of original problem)
    """
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


async def evaluate(
    endpoint: Endpoint,
    prompts: list[dict],
    concurrency: int = 4,
    max_tokens: int = 512,
) -> list[dict]:
    """Run all prompts, grade, return per-prompt result records."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(prompts)  # type: ignore

    async def run_one(idx: int, item: dict) -> None:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                response = await _ask(
                    client, endpoint, item["prompt"], max_tokens=max_tokens
                )
            grade_type = item.get("type", "numeric")
            if grade_type == "numeric":
                correct = grade_numeric(item["expected"], response)
            elif grade_type == "letter":
                correct = grade_mmlu_letter(item["expected"], response)
            else:
                correct = response.strip().lower() == item["expected"].strip().lower()
            results[idx] = {
                "id": item["id"],
                "perturbed_from": item.get("perturbed_from"),
                "expected": item["expected"],
                "response": response,
                "correct": correct,
            }

    await asyncio.gather(*[run_one(i, p) for i, p in enumerate(prompts)])
    return results


def accuracy(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["correct"]) / len(records)


async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print standard-benchmark commands for the operator to run separately
    cmds_path = out_dir / "standard_benchmark_commands.sh"
    cmds_path.write_text(cmd_standard(endpoint, out_dir))
    print(f"[accuracy] wrote standard benchmark commands to {cmds_path}")
    print("[accuracy] run those separately for MMLU / GSM8K / MMLU-Pro / HumanEval+")

    if not args.prompts:
        print("[accuracy] no custom prompt set provided (--prompts); done")
        return

    # Custom perturbation eval
    all_results = {}
    for prompt_file in args.prompts:
        name = Path(prompt_file).stem
        print(f"\n[accuracy] === {name} ===")
        prompts = load_jsonl(Path(prompt_file))
        print(f"  loaded {len(prompts)} prompts")
        results = await evaluate(
            endpoint, prompts,
            concurrency=args.concurrency, max_tokens=args.max_tokens
        )
        acc = accuracy(results)
        print(f"  accuracy: {acc:.3f} ({sum(1 for r in results if r['correct'])}/{len(results)})")
        all_results[name] = {
            "n": len(results),
            "accuracy": acc,
            "results": results,
        }

    # Report: gap between standard and perturbed sets
    if len(all_results) >= 2:
        print(f"\n=== ACCURACY GAPS ===")
        names = list(all_results.keys())
        baseline_name = names[0]
        baseline_acc = all_results[baseline_name]["accuracy"]
        print(f"  baseline: {baseline_name} = {baseline_acc:.3f}")
        for n in names[1:]:
            a = all_results[n]["accuracy"]
            gap = baseline_acc - a
            flag = ""
            if gap > 0.03:
                flag = " <<< >3% gap; possible FP4 brittleness or benchmark overfit"
            print(f"  {n}: {a:.3f}  (gap {gap:+.3f}){flag}")

    write_json(out_dir / "custom_eval_results.json", all_results)
    print(f"\n[accuracy] wrote {out_dir / 'custom_eval_results.json'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--prompts", nargs="*", default=[],
                   help="Path(s) to JSONL prompt sets. Run multiple to "
                        "compare accuracy across (standard, perturbed, held-out).")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--output-dir", default="out/accuracy")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
