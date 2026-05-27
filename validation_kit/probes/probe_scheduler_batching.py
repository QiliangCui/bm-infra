"""
Test 3: Scheduler Batching & Queueing Policy Probe.

Spawns a long-running background generation stream and fires short
high-priority requests at precise 100ms offsets to verify if the
scheduler implements Continuous Batching (iteration-level) or Static batching.
"""
import argparse
import asyncio
import time
from pathlib import Path
import httpx

from common import Endpoint, random_prompt, stream_chat, summary_stats, warmup, write_json

async def run_background_load(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    input_tokens: int,
    output_tokens: int
) -> dict:
    """Launches a heavy, long-running generation task to keep the scheduler occupied."""
    prompt = random_prompt(input_tokens, seed=30000)
    print(f"[scheduler] Spawning background request (input={input_tokens}, output={output_tokens})...")
    
    start = time.perf_counter()
    r = await stream_chat(
        client, endpoint, prompt,
        max_tokens=output_tokens,
        temperature=0.0,
        ignore_eos=True
    )
    dur = time.perf_counter() - start
    
    status = "success" if not r.error else f"error: {r.error}"
    print(f"[scheduler] Background request completed with status: {status}")
    
    return {
        "status": status,
        "duration_s": dur,
        "ttft_s": r.ttft,
        "completion_tokens": r.completion_tokens
    }

async def run_probe_request(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    delay_ms: int,
    output_tokens: int
) -> dict:
    """Waits for delay_ms then shoots a short probe sequence."""
    await asyncio.sleep(delay_ms / 1000.0)
    
    # Short prompt to prevent overlap prefill bottlenecks
    prompt = "Say hello"
    print(f"  [scheduler] Launching probe request at +{delay_ms}ms offset...")
    
    start = time.perf_counter()
    r = await stream_chat(
        client, endpoint, prompt,
        max_tokens=output_tokens,
        temperature=0.0,
        ignore_eos=True
    )
    dur = time.perf_counter() - start
    
    return {
        "offset_ms": delay_ms,
        "duration_s": dur,
        "ttft_s": r.ttft,
        "completion_tokens": r.completion_tokens,
        "error": r.error
    }

async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[scheduler] endpoint={endpoint.base_url} model={endpoint.model}")
    
    print("[scheduler] Warming up...")
    await warmup(endpoint, n=3)
    
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(limits=limits, timeout=180.0) as client:
        # 1. Run background generator
        bg_task = asyncio.create_task(
            run_background_load(client, endpoint, args.bg_input, args.bg_output)
        )
        
        # Give background request 500ms to clear prefill and settle into decode iteration
        await asyncio.sleep(0.5)
        
        # 2. Fire short probe tasks at sequential offsets
        probe_offsets = [100, 300, 600, 1000, 1500]
        probe_tasks = []
        for offset in probe_offsets:
            probe_tasks.append(asyncio.create_task(
                run_probe_request(client, endpoint, offset, args.probe_output)
            ))
            
        # Wait for everything to complete
        print("[scheduler] Waiting for scheduler tasks to finish...")
        bg_result = await bg_task
        probe_results = await asyncio.gather(*probe_tasks)
        
    # 3. Verify scheduler policy
    continuous_batching_signals = []
    static_batching_signals = []
    
    print("\n=== PROBE LATENCY EVALUATION ===")
    for pr in probe_results:
        if pr["error"]:
            print(f"  Probe Offset +{pr['offset_ms']:>4}ms | ERROR: {pr['error']}")
            continue
            
        print(f"  Probe Offset +{pr['offset_ms']:>4}ms | TTFT={pr['ttft_s']:.3f}s | "
              f"Total Duration={pr['duration_s']:.3f}s | Gen Tokens={pr['completion_tokens']}")
        
        # If a probe request completes inside 1.5 seconds while the background 
        # request took much longer, it must have been scheduled concurrently.
        if pr["duration_s"] < 1.5:
            continuous_batching_signals.append(pr)
        else:
            static_batching_signals.append(pr)
            
    print("\n=== SCHEDULER ANALYSIS ===")
    verdict = ""
    if len(continuous_batching_signals) >= 3:
        verdict = ("CONTINUOUS BATCHING DETECTED. Short requests were successfully "
                   "interleaved at the iteration level during active generation, "
                   "retaining sub-second latency.")
    else:
        verdict = ("STATIC BATCHING OR HIGH PREEMPTION DETECTED. Short requests "
                   "were serialized and forced to wait in the queue until the "
                   "background workload completed.")
    print(f"  Verdict: {verdict}")
    
    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "background_job": bg_result,
            "probes": probe_results,
            "verdict": verdict
        }
    )
    print(f"\n[scheduler] Wrote complete raw logs to {out_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="Scheduler Batching & Queueing Policy Probe")
    p.add_argument("--bg-input", type=int, default=4000,
                   help="Input length for background workload")
    p.add_argument("--bg-output", type=int, default=512,
                   help="Output length for background workload")
    p.add_argument("--probe-output", type=int, default=10,
                   help="Output length for probe requests")
    p.add_argument("--output", default="out/probe_scheduler_batching.json")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
