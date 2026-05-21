"""
Test 2: Prefill Compute Scaling Diagnostic Probe.

Measures TTFT across scaling input prompt lengths under single-user
concurrency (concurrency=1) to detect Chunked Prefill activation
and map prefill processing MFU.
"""
import argparse
import asyncio
import time
from pathlib import Path
import httpx

from common import Endpoint, random_prompt, stream_chat, summary_stats, warmup, write_json

async def run_prefill_step(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    input_tokens: int,
    n_runs: int
) -> dict:
    """Measures TTFT across multiple sequential runs for a fixed input length."""
    print(f"[prefill_scaling] Measuring input_tokens={input_tokens}...")
    
    ttfts = []
    durations = []
    completed_count = 0
    
    for i in range(n_runs):
        # Cache-busting unique prompt per run
        prompt = random_prompt(input_tokens, seed=20000 + input_tokens + i)
        
        start = time.perf_counter()
        r = await stream_chat(
            client, endpoint, prompt,
            max_tokens=1,  # strictly 1 output token to isolate prefill compute
            temperature=0.0,
            ignore_eos=True
        )
        dur = time.perf_counter() - start
        
        if not r.error and r.ttft is not None:
            ttfts.append(r.ttft)
            durations.append(dur)
            completed_count += 1
            
        # Small gap to clear queue
        await asyncio.sleep(1.5)
        
    if not ttfts:
        return {"input_tokens": input_tokens, "error": "No successful runs"}
        
    mean_ttft = sum(ttfts) / len(ttfts)
    prefill_tps = input_tokens / mean_ttft if mean_ttft > 0 else 0.0
    
    return {
        "input_tokens": input_tokens,
        "runs": completed_count,
        "ttft_stats": summary_stats(ttfts),
        "duration_stats": summary_stats(durations),
        "mean_prefill_tps": prefill_tps
    }

async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[prefill_scaling] endpoint={endpoint.base_url} model={endpoint.model}")
    
    print("[prefill_scaling] Warming up...")
    await warmup(endpoint, n=3)
    
    lengths = [int(x) for x in args.lengths.split()]
    results = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for length in lengths:
            res = await run_prefill_step(client, endpoint, length, args.n)
            results.append(res)
            
            if "error" in res:
                print(f"  input={length:<6} | ERROR: {res['error']}")
                continue
                
            mean_t = res["ttft_stats"]["mean"]
            p95_t = res["ttft_stats"]["p95"]
            print(f"  input={length:<6} | mean_TTFT={mean_t:.3f}s | "
                  f"p95_TTFT={p95_t:.3f}s | prefill_speed={res['mean_prefill_tps']:.1f} tok/s")
            
            # Cool down JAX compilation cache bounds
            await asyncio.sleep(5.0)
            
    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "n_runs_per_step": args.n,
            "steps": results
        }
    )
    print(f"\n[prefill_scaling] Wrote complete raw logs to {out_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="Prefill Compute Scaling Diagnostic Probe")
    p.add_argument("--lengths", default="128 256 512 1024 2048 4096 8192",
                   help="Whitespace separated list of prompt lengths to test")
    p.add_argument("--n", type=int, default=5,
                   help="Number of runs per prompt length step")
    p.add_argument("--output", default="out/probe_prefill_scaling.json")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
