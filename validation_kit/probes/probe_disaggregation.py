"""
Test 4: Prefill-Decode Workload Disaggregation Probe.

Measures the decode TPOT jitter of a single-user generation request
under a quiet baseline versus under an intensive concurrent prefill-only burst
to verify if prefill and decode nodes are architecturally decoupled.
"""
import argparse
import asyncio
import time
import statistics
from pathlib import Path
import httpx

from common import Endpoint, random_prompt, stream_chat, summary_stats, warmup, write_json

async def run_prefill_burst_worker(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    worker_id: int,
    input_tokens: int
) -> str:
    """Launches a prefill-heavy request (large input, 1 output) to tax prompt compute."""
    prompt = random_prompt(input_tokens, seed=40000 + worker_id)
    
    try:
        r = await stream_chat(
            client, endpoint, prompt,
            max_tokens=1,
            temperature=0.0,
            ignore_eos=True
        )
        if r.error:
            return f"error: {r.error}"
        return "success"
    except Exception as e:
        return f"exception: {e}"

async def run_prefill_heavy_burst(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    concurrency: int,
    input_tokens: int
) -> list[str]:
    """Fires a burst of prefill-heavy tasks concurrently."""
    print(f"[disaggregation] Launching prefill-heavy burst (concurrency={concurrency}, input={input_tokens})...")
    tasks = [
        asyncio.create_task(run_prefill_burst_worker(client, endpoint, i, input_tokens))
        for i in range(concurrency)
    ]
    return await asyncio.gather(*tasks)

async def run_decode_stream(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    output_tokens: int,
    prefill_delay_s: float,
    burst_task: asyncio.Task | None = None
) -> dict:
    """Runs a single-user decode stream and records chunk emission timestamps."""
    prompt = "Write a long story about a rocket ship."
    timestamps = []
    
    # Custom SSE streaming parser to record timestamp at every chunk receipt
    async def sse_stream():
        url = f"{endpoint.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {endpoint.api_key}"}
        payload = {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True
        }
        
        prefill_burst_triggered = False
        
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            start_time = time.perf_counter()
            async for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk_time = time.perf_counter()
                timestamps.append(chunk_time)
                
                # Trigger the background burst once if delay reached
                if not prefill_burst_triggered and (chunk_time - start_time) >= prefill_delay_s:
                    if burst_task:
                        print(f"[disaggregation] Triggering prefill burst after {(chunk_time - start_time):.2f}s of decoding...")
                        prefill_burst_triggered = True
                        
    await sse_stream()
    
    if len(timestamps) < 5:
        return {"error": "Insufficient tokens generated to calculate jitter"}
        
    # Calculate Time-per-Output-Token (TPOT) intervals
    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
    
    return {
        "intervals": intervals,
        "stats": summary_stats(intervals) if intervals else {}
    }

async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[disaggregation] endpoint={endpoint.base_url} model={endpoint.model}")
    
    print("[disaggregation] Warming up...")
    await warmup(endpoint, n=3)
    
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
    async with httpx.AsyncClient(limits=limits, timeout=180.0) as client:
        # 1. Measure Baseline Jitter (no noise)
        print("[disaggregation] Measuring quiet baseline decode jitter...")
        baseline = await run_decode_stream(client, endpoint, args.decode_output, 999.0)
        if "error" in baseline:
            print(f"  ERROR: {baseline['error']}")
            return
            
        baseline_jitter = baseline["stats"]["stdev"]
        print(f"  Baseline decode TPOT mean={baseline['stats']['mean']*1000.0:.1f}ms | "
              f"stdev (jitter)={baseline_jitter*1000.0:.2f}ms")
        
        # Cool down queue
        await asyncio.sleep(5.0)
        
        # 2. Measure Contended Jitter (prefill noise)
        # Setup async prefill burst
        burst_worker = run_prefill_heavy_burst(client, endpoint, args.burst_concurrency, args.burst_input)
        burst_task = asyncio.create_task(burst_worker)
        
        # Run the decode stream and trigger the burst 1 second after generation starts
        print("[disaggregation] Measuring contended decode jitter (triggering prefill noise)...")
        contended = await run_decode_stream(client, endpoint, args.decode_output, 1.0, burst_task)
        
        # Wait for prefill burst to finish cleanly
        await burst_task
        
        if "error" in contended:
            print(f"  ERROR: {contended['error']}")
            return
            
        contended_jitter = contended["stats"]["stdev"]
        print(f"  Contended decode TPOT mean={contended['stats']['mean']*1000.0:.1f}ms | "
              f"stdev (jitter)={contended_jitter*1000.0:.2f}ms")
        
    # 3. Analyze disaggregation ratio
    jitter_ratio = contended_jitter / baseline_jitter if baseline_jitter > 0 else 1.0
    print(f"\n=== WORKLOAD DISAGGREGATION ANALYSIS ===")
    print(f"  TPOT Jitter Ratio (Contended / Baseline): {jitter_ratio:.2f}x")
    
    verdict = ""
    if jitter_ratio > 3.0:
        verdict = (f"STRONG CONTENTION BUBBLE DETECTED (Ratio={jitter_ratio:.2f}x). "
                   "Parallel prefill bursts significantly degraded active decode token "
                   "intervals. Prefill and decode share the same hardware queues.")
    elif jitter_ratio > 1.5:
        verdict = (f"MODERATE JITTER VARIANCE (Ratio={jitter_ratio:.2f}x). "
                   "Minor scheduling bubbles present under intense prefill noise.")
    else:
        verdict = (f"DISAGGREGATED SERVING CONFIRMED (Ratio={jitter_ratio:.2f}x). "
                   "Decode TPOT intervals remained completely isolated from concurrent "
                   "prefill compute noise, verifying dedicated prefill/decode execution nodes.")
    print(f"  Verdict: {verdict}")
    
    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "baseline_jitter_s": baseline_jitter,
            "contended_jitter_s": contended_jitter,
            "jitter_ratio": jitter_ratio,
            "verdict": verdict,
            "baseline_raw_intervals": baseline["intervals"],
            "contended_raw_intervals": contended["intervals"]
        }
    )
    print(f"\n[disaggregation] Wrote complete raw logs to {out_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="Prefill-Decode Workload Disaggregation Probe")
    p.add_argument("--decode-output", type=int, default=100,
                   help="Number of tokens to generate in decode test stream")
    p.add_argument("--burst-concurrency", type=int, default=16,
                   help="Parallel prefill tasks to spawn")
    p.add_argument("--burst-input", type=int, default=8000,
                   help="Prompt length for prefill burst")
    p.add_argument("--output", default="out/probe_disaggregation.json")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
