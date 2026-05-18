"""
Precision sleuthing probe.

At temperature=0, a BF16 and an FP4 serving of the same model checkpoint
should diverge -- but by how much, and how soon, is a strong signal of
how aggressive the partner's quantization actually is.

What this measures, per prompt:
  - First-divergence token index: at what output position do partner and
    reference first emit different text? Earlier divergence = more
    aggressive quantization.
  - Longest common prefix (LCP) in characters and tokens.
  - Full-output identity rate.
  - Decoded length agreement.

What this does NOT do:
  - Compare against the partner's own claimed precision. Just measures
    divergence from a known BF16 reference.
  - Tell you whether divergence is "bad" -- some divergence is expected
    even between two BF16 runs across different stacks (numerics differ).
    The signal is comparative: how does the partner compare to your own
    FP8 or smaller-quant runs?

Recommended usage:
  1. Run against partner endpoint + your own BF16 reference. Establish
     a divergence baseline.
  2. (Optional) Run against your own FP8 and INT8 servings of the same
     checkpoint as a calibration: this tells you what divergence "looks
     like" at known precisions.
  3. Compare partner's divergence numbers to your calibration ladder.
     If partner divergence > your INT8 divergence, partner is running
     more aggressive quantization than INT8.

Usage:
  python -m probes.probe_precision_sleuth \\
      --partner-url ... --partner-model ... \\
      --reference-url ... --reference-model ... \\
      --n 20 --max-tokens 256 \\
      --output out/precision_sleuth.json
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
from pathlib import Path

import httpx

from common import (
    Endpoint, random_prompt, stream_chat, summary_stats, warmup, write_json,
)


def longest_common_prefix(a: str, b: str) -> int:
    """Length of the longest common prefix (in characters)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


async def run_pair(
    partner_client: httpx.AsyncClient,
    reference_client: httpx.AsyncClient,
    partner: Endpoint,
    reference: Endpoint,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Run the same prompt at temperature=0 against both endpoints."""
    # Run serially within a pair to avoid noise from concurrent inflight load
    pr = await stream_chat(
        partner_client, partner, prompt,
        max_tokens=max_tokens, temperature=0.0, ignore_eos=True,
    )
    rr = await stream_chat(
        reference_client, reference, prompt,
        max_tokens=max_tokens, temperature=0.0, ignore_eos=True,
    )

    if pr.error or rr.error:
        return {"error": pr.error or rr.error}

    p_text = pr.full_text
    r_text = rr.full_text
    lcp_chars = longest_common_prefix(p_text, r_text)
    identical = p_text == r_text

    # Find first divergence position (character-level) and estimate
    # tokens by chunk-emission ordering. Coarser than true tokenization
    # but doesn't require the tokenizer.
    return {
        "partner_len_chars": len(p_text),
        "reference_len_chars": len(r_text),
        "partner_tokens": pr.completion_tokens,
        "reference_tokens": rr.completion_tokens,
        "lcp_chars": lcp_chars,
        "identical": identical,
        "diverged_at_char": lcp_chars if not identical else None,
        "lcp_ratio": (
            lcp_chars / max(1, min(len(p_text), len(r_text)))
        ),
        "partner_excerpt": p_text[:300],
        "reference_excerpt": r_text[:300],
    }


async def main_async(args: argparse.Namespace) -> None:
    partner = Endpoint(
        base_url=args.partner_url, api_key=args.partner_key,
        model=args.partner_model,
    )
    reference = Endpoint(
        base_url=args.reference_url, api_key=args.reference_key,
        model=args.reference_model,
    )

    print(f"[sleuth] partner   = {partner.base_url} ({partner.model})")
    print(f"[sleuth] reference = {reference.base_url} ({reference.model})")

    print("[sleuth] warmup both endpoints...")
    await warmup(partner, n=args.warmup)
    await warmup(reference, n=args.warmup)

    results = []
    async with httpx.AsyncClient() as pc, httpx.AsyncClient() as rc:
        for i in range(args.n):
            # Use random English prompts to defeat any cache and provide
            # diverse content for the divergence measure.
            prompt = random_prompt(args.input_tokens, seed=7000 + i)
            res = await run_pair(
                pc, rc, partner, reference, prompt,
                max_tokens=args.max_tokens,
            )
            results.append(res)
            if "error" in res:
                print(f"  [{i+1:02d}/{args.n}] ERROR: {res['error']}")
                continue
            print(f"  [{i+1:02d}/{args.n}] "
                  f"identical={res['identical']}  "
                  f"LCP={res['lcp_chars']}/{res['partner_len_chars']} chars "
                  f"({res['lcp_ratio']:.1%})")

    valid = [r for r in results if "error" not in r]
    if not valid:
        print("[sleuth] no successful pair runs; aborting")
        return

    # Stats
    identity_rate = sum(1 for r in valid if r["identical"]) / len(valid)
    lcp_ratios = [r["lcp_ratio"] for r in valid]
    lcp_chars = [r["lcp_chars"] for r in valid]
    diverge_positions = [r["diverged_at_char"] for r in valid
                          if r["diverged_at_char"] is not None]

    summary = {
        "n_pairs": len(valid),
        "identity_rate": identity_rate,
        "lcp_ratio_stats": summary_stats(lcp_ratios),
        "lcp_char_stats": summary_stats([float(x) for x in lcp_chars]),
        "first_diverge_char_stats": (
            summary_stats([float(x) for x in diverge_positions])
            if diverge_positions else {}
        ),
    }

    # Verdicts
    verdicts = []
    mean_lcp = summary["lcp_ratio_stats"].get("mean", 0.0)

    print(f"\n=== SUMMARY ===")
    print(f"  identical outputs: {identity_rate:.1%} of pairs")
    print(f"  mean LCP ratio:    {mean_lcp:.1%}")
    print(f"  p50 LCP ratio:     {summary['lcp_ratio_stats'].get('p50', 0.0):.1%}")
    if diverge_positions:
        print(f"  mean first-divergence position: char "
              f"{statistics.fmean(diverge_positions):.0f}")

    if identity_rate > 0.8:
        verdicts.append(
            "Partner and reference produce identical output on >80% of "
            "prompts at temperature=0. Either same precision class or "
            "very tight numerics. Consistent with reference-grade FP8 or BF16 "
            "serving; less consistent with aggressive INT4/FP4 quantization."
        )
    elif identity_rate < 0.1 and mean_lcp < 0.3:
        verdicts.append(
            f"Partner and reference diverge rapidly: only {identity_rate:.0%} "
            f"identical outputs, mean longest common prefix {mean_lcp:.0%}. "
            "Either aggressive quantization, different sampling, or different "
            "checkpoint. Confirm partner is running the SAME checkpoint hash "
            "as your reference before reading further into this."
        )
    elif mean_lcp < 0.5:
        verdicts.append(
            f"Moderate divergence: mean LCP {mean_lcp:.0%}. Consistent "
            "with FP8 or aggressive quantization, but inconclusive on its "
            "own. Cross-reference with the roofline result -- if roofline "
            "implied precision is 'default' but divergence looks like FP4, "
            "there's a contradiction."
        )

    print("\n=== VERDICTS ===")
    if verdicts:
        for v in verdicts:
            print(f"  - {v}")
    else:
        print("  (no clear signal)")

    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "partner": {
                "base_url": partner.base_url, "model": partner.model
            },
            "reference": {
                "base_url": reference.base_url, "model": reference.model
            },
            "n_pairs": args.n,
            "max_tokens": args.max_tokens,
            "summary": summary,
            "per_pair": results,
            "verdicts": verdicts,
        },
    )
    print(f"\n[sleuth] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--partner-url", required=True)
    p.add_argument("--partner-key", default="dummy")
    p.add_argument("--partner-model", required=True)
    p.add_argument("--reference-url", required=True,
                   help="BF16 (or chosen-precision) reference endpoint URL")
    p.add_argument("--reference-key", default="dummy")
    p.add_argument("--reference-model", required=True)
    p.add_argument("--n", type=int, default=20,
                   help="Number of prompt pairs")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--input-tokens", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--output", default="out/precision_sleuth.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
