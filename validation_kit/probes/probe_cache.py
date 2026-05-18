"""
Cache detection probe.

Three sub-tests:
  1. Identical request repeated N times -> latency histogram.
     Bimodal / fast-on-repeat = response cache.
  2. Same-prefix requests with different suffixes -> if much faster than
     fully-random prefixes, prefix/KV cache is hitting.
  3. Fully-random prefix on every request -> baseline (no caching possible).

Usage:
  export PARTNER_BASE_URL=https://partner.example.com/v1
  export PARTNER_API_KEY=...
  export PARTNER_MODEL=gpt-oss-120b
  python -m probes.probe_cache --n 30 --output out/cache.json
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

from common import (
    Endpoint,
    random_prompt,
    stream_chat,
    summary_stats,
    warmup,
    write_json,
)


SHARED_PREFIX = (
    "You are an assistant. Read the following passage carefully and then "
    "answer questions about it. Passage: "
    + random_prompt(700, seed=42)
    + "\n\nQuestion: "
)


async def run_one(
    client: httpx.AsyncClient, endpoint: Endpoint, prompt: str, max_tokens: int
) -> dict[str, float | None]:
    """Run one request, return latency dict."""
    r = await stream_chat(
        client, endpoint, prompt, max_tokens=max_tokens, temperature=0.0
    )
    return {
        "ttft_s": r.ttft,
        "total_s": r.total_time,
        "completion_tokens": r.completion_tokens,
        "error": r.error,
    }


async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[cache] endpoint={endpoint.base_url} model={endpoint.model}")

    print("[cache] warmup...")
    await warmup(endpoint, n=args.warmup)

    async with httpx.AsyncClient() as client:
        # Test 1: identical request, repeated. Cache hit -> low latency.
        identical_prompt = (
            "What is 17 times 23? Just give me the number, nothing else."
        )
        print(f"[cache] T1 identical x{args.n}...")
        t1 = []
        for i in range(args.n):
            r = await run_one(client, endpoint, identical_prompt, args.max_tokens)
            t1.append(r)
            print(f"  [{i + 1:02d}/{args.n}] ttft={r['ttft_s']:.3f}s "
                  f"total={r['total_s']:.3f}s")

        # Test 2: shared prefix, varying suffix. Prefix cache hit -> fast.
        print(f"[cache] T2 shared-prefix x{args.n}...")
        t2 = []
        for i in range(args.n):
            suffix = f"Question {i}: What is {i * 7 + 11} times {i + 3}?"
            prompt = SHARED_PREFIX + suffix
            r = await run_one(client, endpoint, prompt, args.max_tokens)
            t2.append(r)
            print(f"  [{i + 1:02d}/{args.n}] ttft={r['ttft_s']:.3f}s")

        # Test 3: fully-random prefix each time. No caching possible.
        print(f"[cache] T3 random-prefix x{args.n}...")
        t3 = []
        for i in range(args.n):
            prompt = random_prompt(args.prefix_tokens, seed=2000 + i)
            r = await run_one(client, endpoint, prompt, args.max_tokens)
            t3.append(r)
            print(f"  [{i + 1:02d}/{args.n}] ttft={r['ttft_s']:.3f}s")

    # Interpret
    def ttft_list(rs: list) -> list[float]:
        return [r["ttft_s"] for r in rs if r["ttft_s"] is not None]

    s1 = summary_stats(ttft_list(t1))
    s2 = summary_stats(ttft_list(t2))
    s3 = summary_stats(ttft_list(t3))

    print("\n=== TTFT (seconds) ===")
    for name, s in [("identical", s1), ("shared_prefix", s2), ("random_prefix", s3)]:
        print(f"  {name:<14} n={s['n']:>3} mean={s['mean']:.3f} "
              f"p50={s['p50']:.3f} p95={s['p95']:.3f} "
              f"min={s['min']:.3f} max={s['max']:.3f}")

    # Verdicts
    verdicts = []
    if s1["mean"] > 0 and s3["mean"] > 0:
        ratio_identical = s3["mean"] / s1["mean"]
        ratio_prefix = s3["mean"] / s2["mean"] if s2["mean"] > 0 else 1.0

        if ratio_identical > 1.8:
            verdicts.append(
                f"RESPONSE CACHE LIKELY: identical-request mean TTFT is "
                f"{ratio_identical:.1f}x faster than random-prompt baseline. "
                "Re-run with cache-busting confirmed off, or refuse the result."
            )
        if ratio_prefix > 1.5 and ratio_identical < 1.8:
            verdicts.append(
                f"PREFIX CACHE LIKELY: shared-prefix mean TTFT is "
                f"{ratio_prefix:.1f}x faster than random-prefix baseline. "
                "If you're benchmarking, ensure every request uses a unique prefix."
            )
        if ratio_identical < 1.3 and ratio_prefix < 1.3:
            verdicts.append("NO STRONG CACHE EVIDENCE in TTFT.")

    # Also: bimodal check on identical requests (one slow miss + many fast hits)
    t1_ttfts = sorted(ttft_list(t1))
    if len(t1_ttfts) >= 5:
        fast_half = t1_ttfts[: len(t1_ttfts) // 2]
        slow_half = t1_ttfts[len(t1_ttfts) // 2:]
        if slow_half and fast_half:
            sep = (sum(slow_half) / len(slow_half)) / (sum(fast_half) / len(fast_half))
            if sep > 2.5:
                verdicts.append(
                    f"BIMODAL identical-request distribution (slow/fast = {sep:.1f}x): "
                    "consistent with a response cache that misses on first request."
                )

    print("\n=== VERDICTS ===")
    for v in verdicts:
        print(f"  - {v}")
    if not verdicts:
        print("  (no signals)")

    write_json(
        Path(args.output),
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "n": args.n,
            "summary": {
                "identical": s1, "shared_prefix": s2, "random_prefix": s3,
            },
            "raw": {"identical": t1, "shared_prefix": t2, "random_prefix": t3},
            "verdicts": verdicts,
        },
    )
    print(f"\n[cache] wrote {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--n", type=int, default=30,
                   help="Requests per sub-test")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=32,
                   help="Output length for cache probe; short keeps cost down")
    p.add_argument("--prefix-tokens", type=int, default=700,
                   help="Random-prefix length for T3 (and base for T2)")
    p.add_argument("--output", default="out/cache.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
