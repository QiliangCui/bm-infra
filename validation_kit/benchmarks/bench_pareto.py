"""
Throughput-latency Pareto frontier sweep.

Sweeps client-side concurrency, measuring:
  - p50/p95/p99 TTFT
  - p50/p95/p99 TPOT (time per output token)
  - aggregate decode throughput (tok/s across all concurrent requests)
  - request success rate

Output is the curve you can plot or compare against InferenceMAX. Use this
to find the partner's "useful throughput at SLA" and to compare apples-to-
apples with B200 published results.

Usage:
  python -m benchmarks.bench_pareto \
      --concurrencies 1 4 8 16 32 64 128 \
      --input-tokens 8000 --output-tokens 1024 \
      --duration 90 \
      --output out/pareto.json
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
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


async def _one_request(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    prompt: str,
    max_tokens: int,
) -> dict | None:
    """Run one streaming request, return per-request metrics."""
    r = await stream_chat(
        client, endpoint, prompt,
        max_tokens=max_tokens, temperature=0.0, ignore_eos=True
    )
    if r.error or not r.completion_tokens or not r.decode_time:
        return None
    return {
        "ttft_s": r.ttft,
        "decode_s": r.decode_time,
        "total_s": r.total_time,
        "completion_tokens": r.completion_tokens,
        "tpot_s": r.tpot_s,
    }


async def run_at_concurrency(
    endpoint: Endpoint,
    concurrency: int,
    duration_s: float,
    input_tokens: int,
    max_tokens: int,
) -> dict:
    """Sustain `concurrency` in-flight requests for `duration_s` seconds."""
    print(f"  [pareto] c={concurrency}: sustaining for {duration_s:.0f}s")

    results: list[dict] = []
    seed_counter = [6000]
    stop_at = time.perf_counter() + duration_s
    t_start_wall = time.perf_counter()

    async def worker(worker_id: int) -> None:
        async with httpx.AsyncClient() as client:
            while time.perf_counter() < stop_at:
                seed_counter[0] += 1
                prompt = random_prompt(input_tokens, seed=seed_counter[0])
                r = await _one_request(client, endpoint, prompt, max_tokens)
                if r is not None:
                    results.append(r)

    workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
    await asyncio.gather(*workers)

    wall_time = time.perf_counter() - t_start_wall
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    agg_tok_per_s = total_completion_tokens / wall_time if wall_time > 0 else 0.0

    ttft = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    tpot = [r["tpot_s"] for r in results if r["tpot_s"] is not None]
    per_user_tps = [
        (r["completion_tokens"] - 1) / r["decode_s"]
        for r in results
        if r["decode_s"] and r["completion_tokens"] and r["completion_tokens"] > 1
    ]

    return {
        "concurrency": concurrency,
        "duration_s": wall_time,
        "completed_requests": len(results),
        "total_completion_tokens": total_completion_tokens,
        "aggregate_tok_per_s": agg_tok_per_s,
        "ttft_stats": summary_stats(ttft),
        "tpot_stats": summary_stats(tpot),
        "per_user_decode_tps_stats": summary_stats(per_user_tps),
        "mean_per_user_decode_tps": (
            statistics.fmean(per_user_tps) if per_user_tps else 0.0
        ),
    }


async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[pareto] endpoint={endpoint.base_url} model={endpoint.model}")
    print(f"[pareto] sweep concurrencies={args.concurrencies}")

    print("[pareto] warmup...")
    await warmup(endpoint, n=args.warmup)

    sweep = []
    for c in args.concurrencies:
        point = await run_at_concurrency(
            endpoint,
            concurrency=c,
            duration_s=args.duration,
            input_tokens=args.input_tokens,
            max_tokens=args.output_tokens,
        )
        sweep.append(point)
        print(f"    c={c:>4} agg={point['aggregate_tok_per_s']:>7.0f} tok/s  "
              f"p95 TPOT={point['tpot_stats']['p95']*1000:>6.1f}ms  "
              f"per-user dec={point['mean_per_user_decode_tps']:>5.1f} tok/s")

    # SLA-based summary: throughput at fixed latency SLA
    print(f"\n=== USEFUL THROUGHPUT AT SLA ===")
    sla_summaries = {}
    for sla_ms in args.sla_p95_tpot_ms:
        sla_s = sla_ms / 1000.0
        # Find max aggregate throughput where p95 TPOT <= SLA
        feasible = [p for p in sweep
                    if p["tpot_stats"].get("p95", float("inf")) <= sla_s]
        if feasible:
            best = max(feasible, key=lambda p: p["aggregate_tok_per_s"])
            print(f"  p95 TPOT <= {sla_ms}ms: "
                  f"max aggregate = {best['aggregate_tok_per_s']:.0f} tok/s "
                  f"at concurrency={best['concurrency']}")
            sla_summaries[f"p95_tpot_le_{sla_ms}ms"] = {
                "max_aggregate_tok_per_s": best["aggregate_tok_per_s"],
                "concurrency": best["concurrency"],
                "mean_per_user_decode_tps": best["mean_per_user_decode_tps"],
            }
        else:
            print(f"  p95 TPOT <= {sla_ms}ms: NO concurrency met the SLA")
            sla_summaries[f"p95_tpot_le_{sla_ms}ms"] = None

    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "duration_s_per_point": args.duration,
            "sweep": sweep,
            "sla_summaries": sla_summaries,
        },
    )
    print(f"\n[pareto] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--concurrencies", type=int, nargs="+",
                   default=[1, 4, 8, 16, 32, 64, 128])
    p.add_argument("--duration", type=float, default=90.0,
                   help="Seconds to sustain at each concurrency (after warmup)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--input-tokens", type=int, default=8000)
    p.add_argument("--output-tokens", type=int, default=1024)
    p.add_argument("--sla-p95-tpot-ms", type=int, nargs="+",
                   default=[50, 100, 300],
                   help="SLA targets for per-token p95 latency")
    p.add_argument("--output", default="out/pareto.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
