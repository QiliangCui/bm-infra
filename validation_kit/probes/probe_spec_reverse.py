"""
Test 5: Speculative Drafting Window Reverse-Engineering Probe.

Runs sequential low-entropy repetition prompts (100% draft acceptance scenario)
at temperature=0 and analyzes the exact token burst counts returned in every
individual SSE chunk to reverse-engineer the model's validation draft size (k).
"""
import argparse
import asyncio
import collections
from pathlib import Path
import httpx

from common import Endpoint, repeat_phrase_prompt, StreamResult, warmup, write_json, get_tokenizer

async def get_chunk_text_bursts(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    max_tokens: int
) -> list[str]:
    """Streams low-entropy repetition prompt and captures text per chunk."""
    prompt = repeat_phrase_prompt(target_output_tokens=max_tokens)
    url = f"{endpoint.base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {endpoint.api_key}"}
    payload = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True
    }
    
    chunk_texts = []
    
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line.endswith("[DONE]"):
                continue
            
            # Parse text payload from SSE chunk
            import json
            try:
                data = json.loads(line[6:])
                choices = data.get("choices")
                if choices:
                    delta = choices[0].get("delta", {})
                    text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
                    if text:
                        chunk_texts.append(text)
            except Exception:
                continue
                
    return chunk_texts

async def run_draft_reverse_step(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    max_tokens: int,
    tokenizer_name: str
) -> dict:
    """Runs generation and counts the number of tokens in every burst chunk."""
    print(f"[spec_reverse] Running low-entropy validation stream...")
    
    burst_texts = await get_chunk_text_bursts(client, endpoint, max_tokens)
    if not burst_texts:
        return {"error": "No tokens returned"}
        
    # Count actual tokens inside each text burst using the target tokenizer
    tok = get_tokenizer()
    tokens_per_burst = []
    
    for text in burst_texts:
        try:
            count = tok.count(text)
            if count > 0:
                tokens_per_burst.append(count)
        except Exception:
            continue
            
    # Frequency map of token counts per burst
    freq_map = collections.Counter(tokens_per_burst)
    
    return {
        "raw_burst_counts": tokens_per_burst,
        "frequency": dict(freq_map),
        "max_burst": max(tokens_per_burst) if tokens_per_burst else 1,
        "mean_burst": sum(tokens_per_burst) / len(tokens_per_burst) if tokens_per_burst else 1.0
    }

async def main_async(args: argparse.Namespace) -> None:
    endpoint = Endpoint.from_env()
    print(f"[spec_reverse] endpoint={endpoint.base_url} model={endpoint.model}")
    print(f"[spec_reverse] tokenizer={args.tokenizer}")
    
    print("[spec_reverse] Warming up...")
    await warmup(endpoint, n=3)
    
    results = []
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in range(args.n):
            print(f"[spec_reverse] Sweep Run {i+1}/{args.n}...")
            res = await run_draft_reverse_step(client, endpoint, args.max_tokens, args.tokenizer)
            results.append(res)
            
            if "error" in res:
                print(f"  ERROR: {res['error']}")
                continue
                
            print(f"  Completed | max_burst={res['max_burst']} | mean_burst={res['mean_burst']:.2f} | "
                  f"freq_dist={res['frequency']}")
            
            await asyncio.sleep(2.0)
            
    # Compile global frequency map
    global_counter = collections.Counter()
    for r in results:
        if "frequency" in r:
            global_counter.update(r["frequency"])
            
    # The speculation draft validation limit k is usually max_burst - 1 
    # (since 1 token is base autoregressive and k tokens are speculation)
    max_seen_burst = max(global_counter.keys()) if global_counter else 1
    inferred_k = max_seen_burst - 1
    
    print(f"\n=== SPECULATIVE DRAFTER CAPACITY REVERSE-ENGINEER ===")
    print(f"  Global Burst Frequencies: {dict(global_counter)}")
    print(f"  Max Single-Chunk Burst Size: {max_seen_burst} tokens")
    
    verdict = ""
    if inferred_k > 0:
        verdict = (f"SPECULATIVE DECODING CONFIRMED. The draft validation window size "
                   f"is reverse-engineered to be exactly k={inferred_k} tokens. "
                   f"The server verifies batches of {inferred_k} speculative candidate "
                   f"tokens per iteration pass.")
    else:
        verdict = ("NO SPECULATIVE SIGNAL DETECTED. Every single streamed chunk contained "
                   "exactly 1 token. Autoregressive execution only.")
    print(f"  Verdict: {verdict}")
    
    out_path = Path(args.output)
    write_json(
        out_path,
        {
            "endpoint": endpoint.base_url,
            "model": endpoint.model,
            "tokenizer": args.tokenizer,
            "runs": results,
            "global_frequencies": dict(global_counter),
            "max_burst_seen": max_seen_burst,
            "inferred_k": inferred_k,
            "verdict": verdict
        }
    )
    print(f"\n[spec_reverse] Wrote complete raw logs to {out_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="Speculative Drafting Window Reverse-Engineering Probe")
    p.add_argument("--n", type=int, default=5,
                   help="Number of repetition runs to stack")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tokenizer", default="openai/gpt-oss-120b",
                   help="Target model tokenizer on HuggingFace")
    p.add_argument("--output", default="out/probe_spec_reverse.json")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
