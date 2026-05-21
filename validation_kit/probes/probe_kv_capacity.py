"""
Test 1: KV Cache Capacity & Memory Allocation Probe.

Fires simultaneous concurrent long-context prompts (scaling concurrency)
to identify at what point prefill queue-length or decode slot boundaries 
degrade latency exponentially, revealing HBM memory bounds.
"""
import argparse
import asyncio
import time
from pathlib import Path
import httpx

from common import Endpoint, random_prompt, stream_chat, summary_stats, warmup, write_json

async def run_concurrency_step(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    concurrency: int,
    input_tokens: int,
    output_tokens: int
) -> dict:
    """Fires concurrent requests simultaneously and gathers metrics."""
    print(f"[kv_capacity] Running concurrency={concurrency}...")
    
    # Generate unique prompts to bust prefix/response cache
    prompts = [random_prompt(input_tokens, seed=10000 + concurrency + i) for i in range(concurrency)]
    
    tasks = []
    for i in range(concurrency):
        tasks.append(stream_chat(
            client, endpoint, prompts[i],
            max_tokens=output_tokens,
            temperature=0.0,
            ignore_eos=True
        ))
        
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.perf_counter() - start_time
    
    # Parse metrics
    completed = []
    errors = []
    ttfts = []
    tpots = []
    
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
        elif r.error:
            errors.append(r.error)
        else:
            completed.append(r)
            if r.ttft is not None:
                ttfts.append(r.ttft)
            if r.decode_time and r.completion_tokens and r.completion_tokens > 1:
                tpot = r.decode_time / (r.completion_tokens - 1)
                tpots.append(tpot)
                
    success_rate = len(completed) / concurrency if concurrency > 0 else 0.0
    
    return {
        "concurrency": concurrency,
        "duration_s": duration,
        "success_rate": success_rate,
        "errors_count": len(errors),
        "errors": errors[:5],  # save sample errors
        "ttft_stats": summary_stats(ttfts) if ttfts else {},
        "tpot_stats": summary_stats(tpots) if tpots else {},
        "total_tokens_generated": sum(r.completion_tokens for r in completed if r.completion_tokens)
    }

async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[kv_capacity] endpoint={endpoint.base_url} model={endpoint.model}")
    print(f"[kv_capacity] context_len={args.input_tokens} input + {args.output_tokens} output")
    
    print("[kv_capacity] Warming up...")
    await warmup(endpoint, n=3)
    
    concurrencies = [int(x) for x in args.concurrencies.split()]
    results = []
    
    # httpx client configured for high concurrency
    limits = httpx.Limits(max_keepalive_connections=500, max_connections=1000)
    async with httpx.AsyncClient(limits=limits, timeout=180.0) as client:
        for c in concurrencies:
            res = await run_concurrency_step(
                client, endpoint, c, args.input_tokens, args.output_tokens
            )
            results.append(res)
            
            # Log progress to stdout
            ttft_p50 = res["ttft_stats"].get("p50", 0.0)
            tpot_p95 = res["tpot_stats"].get("p95", 0.0) * 1000.0 if res["tpot_stats"] else 0.0
            print(f"  c={c:<4} | success={res['success_rate']:>6.1%} | "
                  f"p50 TTFT={ttft_p50:.2f}s | p95 TPOT={tpot_p95:.1f}ms | "
                  f"total_gen_toks={res['total_tokens_generated']}")
            
            # Grace period to cool down server buffers
            await asyncio.sleep(10.0)
            
            # If success rate collapses completely, stop to prevent server lockout
            if res["success_rate"] < 0.1:
                print("[kv_capacity] Success rate collapsed below 10%; aborting further steps.")
                break
                
    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "steps": results
        }
    )
    print(f"\n[kv_capacity] Wrote complete raw logs to {out_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="KV Cache Capacity Black-Box Probe")
    p.add_argument("--concurrencies", default="4 8 16 32 64 128 256",
                   help="Whitespace separated list of concurrencies to sweep")
    p.add_argument("--input-tokens", type=int, default=8000)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--output", default="out/probe_kv_capacity.json")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
