"""
Roofline / HBM-bandwidth ceiling sanity check.

Measures observed batch=1 / single-user decode tok/s with random-prompt
cache-busting, then compares against the HBM-bandwidth ceiling for the
claimed hardware. If observed > ceiling, the result is not physically
achievable without multi-token amortization or caching.

Bytes-per-token assumptions (gpt-oss-120b, 5.1B active params):
  - default: 5.8 GB/token  (BF16 non-MoE + MXFP4 MoE + FP8 KV @ 8K ctx)
  - aggressive: 3.8 GB/token  (FP8 non-MoE + MXFP4 MoE)
  - heroic: 2.9 GB/token  (uniform ~4-bit everywhere)

Hardware HBM bandwidth:
  - Ironwood: 7.37 TB/s/chip
  - B200:     8.0 TB/s/chip

Usage:
  python -m probes.probe_roofline \
      --hardware ironwood --chip-count 8 \
      --input-tokens 8000 --output-tokens 1024 \
      --output out/roofline.json
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
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


HBM_BW = {
    # TB/s per chip
    "ironwood": 7.37e12,
    "tpu7x": 7.37e12,
    "b200": 8.0e12,
    "b300": 8.0e12,   # same HBM as B200, more compute
    "h200": 4.8e12,
    "h100": 3.35e12,
    "mi300x": 5.3e12,
    "mi355x": 5.3e12,
}

# Bytes per token for gpt-oss-120b decode at 8K context.
# These are best-effort estimates; precise values depend on KV precision
# and the partner's actual quantization config.
BYTES_PER_TOKEN = {
    "default": 5.8e9,       # BF16 non-MoE + MXFP4 MoE + FP8 KV
    "aggressive": 3.8e9,    # FP8 non-MoE + MXFP4 MoE
    "heroic": 2.9e9,        # uniform ~4-bit
}


def ceiling_tps(hardware: str, chip_count: int, precision: str) -> float:
    """Max tok/s/user achievable at the given hardware/precision/TP."""
    if hardware not in HBM_BW:
        raise ValueError(f"Unknown hardware {hardware!r}, known: {list(HBM_BW)}")
    if precision not in BYTES_PER_TOKEN:
        raise ValueError(f"Unknown precision {precision!r}, "
                         f"known: {list(BYTES_PER_TOKEN)}")
    return (HBM_BW[hardware] * chip_count) / BYTES_PER_TOKEN[precision]


def implied_mbu(
    observed_tps: float, hardware: str, chip_count: int, precision: str
) -> float:
    """What fraction of HBM bandwidth must be hitting to achieve this?"""
    return observed_tps / ceiling_tps(hardware, chip_count, precision)


async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[roofline] endpoint={endpoint.base_url} model={endpoint.model}")
    print(f"[roofline] hardware={args.hardware} chips={args.chip_count} "
          f"precision={args.precision}")

    # Print the theoretical ceilings up front
    print("\n=== THEORETICAL CEILINGS (tok/s/user, single user) ===")
    for prec in BYTES_PER_TOKEN:
        c = ceiling_tps(args.hardware, args.chip_count, prec)
        print(f"  {prec:<11} ({BYTES_PER_TOKEN[prec]/1e9:>4.1f} GB/tok): "
              f"{c:>8.0f} tok/s @ 100% MBU  /  {c*0.7:>7.0f} @ 70% MBU")

    print(f"\n[roofline] warmup...")
    await warmup(endpoint, n=args.warmup)

    # Measured: batch=1, random prompts (no cache possible), fixed I/O length
    print(f"[roofline] {args.n} batch=1 runs at "
          f"{args.input_tokens}-token input, {args.output_tokens}-token output")
    measured = []
    async with httpx.AsyncClient() as client:
        for i in range(args.n):
            prompt = random_prompt(args.input_tokens, seed=5000 + i)
            r = await stream_chat(
                client, endpoint, prompt,
                max_tokens=args.output_tokens,
                temperature=0.0,
                ignore_eos=True,
            )
            if r.error:
                print(f"  [{i + 1:02d}] ERROR: {r.error}")
                continue
            if not r.decode_time or not r.completion_tokens:
                print(f"  [{i + 1:02d}] no decode time / token count, skipping")
                continue
            tps = (r.completion_tokens - 1) / r.decode_time
            measured.append({
                "ttft_s": r.ttft,
                "decode_s": r.decode_time,
                "completion_tokens": r.completion_tokens,
                "decode_tps": tps,
            })
            print(f"  [{i + 1:02d}/{args.n}] ttft={r.ttft:.2f}s "
                  f"decode={r.decode_time:.2f}s "
                  f"toks={r.completion_tokens} "
                  f"tps={tps:.1f}")

    if not measured:
        print("\n[roofline] no successful runs; aborting analysis")
        return

    decode_tps = [m["decode_tps"] for m in measured]
    stats = summary_stats(decode_tps)
    print(f"\n=== MEASURED batch=1 decode tok/s/user ===")
    print(f"  n={stats['n']} mean={stats['mean']:.1f} p50={stats['p50']:.1f} "
          f"p95={stats['p95']:.1f} max={stats['max']:.1f}")

    # Implied MBU at each precision assumption
    print(f"\n=== IMPLIED MBU (mean measured tok/s/user vs ceiling) ===")
    verdicts = []
    for prec in BYTES_PER_TOKEN:
        mbu = implied_mbu(stats["mean"], args.hardware, args.chip_count, prec)
        flag = ""
        if mbu > 1.0:
            flag = " <<< EXCEEDS PHYSICAL CEILING"
            verdicts.append(
                f"At {prec} precision: implied MBU = {mbu:.2f} > 1.0. "
                f"Measured {stats['mean']:.0f} tok/s/user exceeds the HBM-"
                f"bandwidth ceiling for {args.chip_count}x {args.hardware}. "
                "This is not physically achievable without multi-token "
                "decoding or caching."
            )
        elif mbu > 0.8:
            flag = " <<< heroic, demand evidence"
        print(f"  {prec:<11}: MBU={mbu:.2f}{flag}")

    # Also report at p95 to catch best-case-only claims
    mbu_p95_default = implied_mbu(
        stats["p95"], args.hardware, args.chip_count, args.precision
    )
    print(f"\n  At p95 ({stats['p95']:.0f} tok/s) and '{args.precision}' precision: "
          f"MBU={mbu_p95_default:.2f}")
    if mbu_p95_default > 1.0:
        verdicts.append(
            f"Even at p95 with the partner's claimed precision "
            f"({args.precision}), implied MBU = {mbu_p95_default:.2f} > 1.0. "
            "The fastest measured runs are not physically achievable."
        )

    print("\n=== VERDICTS ===")
    for v in verdicts:
        print(f"  - {v}")
    if not verdicts:
        print("  (no signals)")
        print(f"\n  Measured tok/s/user is within physical bounds for "
              f"{args.chip_count}x {args.hardware} at {args.precision} precision. "
              "Does not by itself confirm the partner's claim -- only that the "
              "bandwidth math closes. Combine with spec-decoding and cache probes.")

    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "hardware": args.hardware,
            "chip_count": args.chip_count,
            "precision": args.precision,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "ceilings": {
                prec: ceiling_tps(args.hardware, args.chip_count, prec)
                for prec in BYTES_PER_TOKEN
            },
            "measured": measured,
            "stats": stats,
            "implied_mbu": {
                prec: implied_mbu(stats["mean"], args.hardware, args.chip_count, prec)
                for prec in BYTES_PER_TOKEN
            },
            "verdicts": verdicts,
        },
    )
    print(f"\n[roofline] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--hardware", default="ironwood",
                   choices=list(HBM_BW.keys()))
    p.add_argument("--chip-count", type=int, required=True,
                   help="Number of chips the partner says they're using")
    p.add_argument("--precision", default="default",
                   choices=list(BYTES_PER_TOKEN.keys()),
                   help="Precision assumption for the partner's reported config")
    p.add_argument("--input-tokens", type=int, default=8000,
                   help="Match partner's workload shape (8K input for 8K/1K)")
    p.add_argument("--output-tokens", type=int, default=1024)
    p.add_argument("--output", default="out/roofline.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
