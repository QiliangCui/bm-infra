"""
Speculative decoding / MTP / Eagle / Medusa / Lookahead detection.

Three signals, any of which is sufficient on its own:

  1. ENTROPY: identical I/O shapes but different output entropy.
     Low entropy (repeat phrase) vs high entropy (random continuation).
     A >20% tok/s gap means multi-token decoding is amortizing weight
     reads across tokens, and acceptance collapses on high-entropy content.

  2. TOKENS-PER-CHUNK: count tokens emitted in each SSE chunk via the
     tokenizer. Pure autoregressive = always 1. Multi-token = variable
     (1, 4, 2, 5, 1, 3, ...). Direct evidence; no inference.

  3. SAWTOOTH: inter-chunk intervals. Pure autoregressive = roughly flat.
     Multi-token = alternating bursts and pauses as draft/verify cycles
     run.

Usage:
  python -m probes.probe_spec_decoding --n 8 --max-tokens 512 \
      --output out/spec_decoding.json
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
from pathlib import Path

import httpx

from common import (
    Endpoint,
    StreamResult,
    get_tokenizer,
    high_entropy_prompt,
    repeat_phrase_prompt,
    stream_chat,
    summary_stats,
    warmup,
    write_json,
)


async def run_entropy_pair(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    max_tokens: int,
    seed: int,
) -> tuple[StreamResult, StreamResult]:
    """Run one low-entropy and one high-entropy request at fixed output length."""
    low = await stream_chat(
        client,
        endpoint,
        repeat_phrase_prompt(target_output_tokens=max_tokens),
        max_tokens=max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    high = await stream_chat(
        client,
        endpoint,
        high_entropy_prompt(seed=seed),
        max_tokens=max_tokens,
        temperature=2.0,
        top_p=1.0,
        seed=seed,
        ignore_eos=True,
    )
    return low, high


def analyze_chunks(r: StreamResult, tokenize: bool) -> dict:
    """Pull out tokens-per-chunk and inter-chunk intervals."""
    intervals = r.inter_chunk_intervals()
    chunks_n = len(r.events)

    tokens_per_chunk: list[int] = []
    if tokenize and r.events:
        tok = get_tokenizer()
        for ev in r.events:
            try:
                tokens_per_chunk.append(tok.count(ev.text))
            except Exception:
                tokens_per_chunk.append(-1)  # mark as unknown

    return {
        "chunks_n": chunks_n,
        "completion_tokens": r.completion_tokens,
        "ttft_s": r.ttft,
        "decode_s": r.decode_time,
        "tpot_s": r.tpot_s,
        "decode_toks_per_s": (
            (r.completion_tokens - 1) / r.decode_time
            if (r.completion_tokens and r.decode_time
                and r.completion_tokens > 1)
            else None
        ),
        "interval_stats": summary_stats(intervals),
        "tokens_per_chunk": tokens_per_chunk,
        "tokens_per_chunk_stats": (
            summary_stats([float(x) for x in tokens_per_chunk if x >= 0])
            if tokens_per_chunk else {}
        ),
    }


async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[spec] endpoint={endpoint.base_url} model={endpoint.model}")

    print("[spec] warmup...")
    await warmup(endpoint, n=args.warmup)

    pairs = []
    async with httpx.AsyncClient() as client:
        for i in range(args.n):
            print(f"[spec] pair {i + 1}/{args.n}: low-entropy run...")
            low, high = await run_entropy_pair(
                client, endpoint, max_tokens=args.max_tokens, seed=3000 + i
            )
            pairs.append((low, high))
            lt = low.decode_time or 0
            ht = high.decode_time or 0
            lt_toks = low.completion_tokens or 0
            ht_toks = high.completion_tokens or 0
            print(f"  low : decode={lt:.2f}s toks={lt_toks} "
                  f"chunks={len(low.events)}")
            print(f"  high: decode={ht:.2f}s toks={ht_toks} "
                  f"chunks={len(high.events)}")

    # Analyze
    low_decode_tps = []
    high_decode_tps = []
    low_analyses = []
    high_analyses = []

    for low, high in pairs:
        la = analyze_chunks(low, tokenize=not args.no_tokenize)
        ha = analyze_chunks(high, tokenize=not args.no_tokenize)
        low_analyses.append(la)
        high_analyses.append(ha)
        if la["decode_toks_per_s"] is not None:
            low_decode_tps.append(la["decode_toks_per_s"])
        if ha["decode_toks_per_s"] is not None:
            high_decode_tps.append(ha["decode_toks_per_s"])

    # Entropy verdict
    verdicts = []
    if low_decode_tps and high_decode_tps:
        low_mean = statistics.fmean(low_decode_tps)
        high_mean = statistics.fmean(high_decode_tps)
        ratio = low_mean / high_mean if high_mean > 0 else float("inf")
        print(f"\n=== ENTROPY PROBE ===")
        print(f"  low-entropy  decode tok/s/user (mean): {low_mean:.1f}")
        print(f"  high-entropy decode tok/s/user (mean): {high_mean:.1f}")
        print(f"  ratio (low / high): {ratio:.2f}x")
        if ratio > 1.5:
            verdicts.append(
                f"MULTI-TOKEN DECODING DETECTED (ratio={ratio:.2f}x). "
                "Low-entropy decode is significantly faster than high-entropy "
                "decode at fixed I/O shapes. This is the signature of SD/MTP/"
                "Eagle/Medusa/Lookahead. Demand that the partner disable it "
                "and re-run for an apples-to-apples baseline."
            )
        elif ratio > 1.2:
            verdicts.append(
                f"BORDERLINE entropy signal (ratio={ratio:.2f}x). "
                "Could be multi-token decoding with low average acceptance, "
                "or could be content-dependent kernel behavior. Cross-check "
                "with tokens-per-chunk."
            )
        else:
            verdicts.append(
                f"No entropy signal (ratio={ratio:.2f}x). "
                "Multi-token decoding either not active, or active with low "
                "acceptance even on low-entropy prompts."
            )

    # Tokens-per-chunk verdict
    if not args.no_tokenize:
        # If ANY chunk has >1 token at any point, that's diagnostic.
        max_tpc_low = max(
            (max(la["tokens_per_chunk"]) for la in low_analyses
             if la["tokens_per_chunk"]),
            default=0,
        )
        max_tpc_high = max(
            (max(ha["tokens_per_chunk"]) for ha in high_analyses
             if ha["tokens_per_chunk"]),
            default=0,
        )
        mean_tpc_low = statistics.fmean(
            [la["tokens_per_chunk_stats"].get("mean", 1.0)
             for la in low_analyses if la["tokens_per_chunk_stats"]]
            or [1.0]
        )
        mean_tpc_high = statistics.fmean(
            [ha["tokens_per_chunk_stats"].get("mean", 1.0)
             for ha in high_analyses if ha["tokens_per_chunk_stats"]]
            or [1.0]
        )

        print(f"\n=== TOKENS PER CHUNK ===")
        print(f"  low-entropy:  mean={mean_tpc_low:.2f}  max={max_tpc_low}")
        print(f"  high-entropy: mean={mean_tpc_high:.2f}  max={max_tpc_high}")
        if max_tpc_low > 1 and mean_tpc_low > 1.2:
            verdicts.append(
                f"TOKENS-PER-CHUNK > 1 ON LOW-ENTROPY (mean={mean_tpc_low:.2f}, "
                f"max={max_tpc_low}). This is direct evidence of multi-token "
                "decoding -- the server emits multiple tokens per forward pass."
            )
        # Network proxies sometimes coalesce chunks, so a single elevated
        # value isn't enough -- we need the entropy contrast.
        if max_tpc_low > 1 and max_tpc_high == 1:
            verdicts.append(
                "Tokens-per-chunk >1 only on low-entropy prompts. "
                "Almost certainly SD/MTP with content-dependent acceptance."
            )

    print("\n=== VERDICTS ===")
    for v in verdicts:
        print(f"  - {v}")
    if not verdicts:
        print("  (no signals)")

    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "n_pairs": args.n,
            "max_tokens": args.max_tokens,
            "low_analyses": low_analyses,
            "high_analyses": high_analyses,
            "low_decode_tps": low_decode_tps,
            "high_decode_tps": high_decode_tps,
            "verdicts": verdicts,
        },
    )
    print(f"\n[spec] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--n", type=int, default=8,
                   help="Number of (low-entropy, high-entropy) pairs")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-tokenize", action="store_true",
                   help="Skip tokens-per-chunk analysis (faster, less signal)")
    p.add_argument("--output", default="out/spec_decoding.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
