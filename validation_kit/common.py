"""
Shared utilities for the partner validation kit.

All probes use an OpenAI-compatible streaming client. Works against vLLM,
SGLang, TensorRT-LLM Serve, and any vendor API speaking the OpenAI protocol.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    """OpenAI-compatible endpoint config."""
    base_url: str
    api_key: str = "dummy"
    model: str = "gpt-oss-120b"

    @classmethod
    def from_env(cls) -> "Endpoint":
        return cls(
            base_url=os.environ.get("PARTNER_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("PARTNER_API_KEY", "dummy"),
            model=os.environ.get("PARTNER_MODEL", "gpt-oss-120b"),
        )


# ---------------------------------------------------------------------------
# Streaming client
# ---------------------------------------------------------------------------

@dataclass
class ChunkEvent:
    """One SSE delta event."""
    t_relative: float       # seconds since request start
    text: str               # delta content (may be multi-token)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class StreamResult:
    """Result of one streaming generation."""
    t_start: float
    t_first_chunk: float | None
    t_last_chunk: float | None
    events: list[ChunkEvent] = field(default_factory=list)
    full_text: str = ""
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    error: str | None = None

    @property
    def ttft(self) -> float | None:
        """Time to first token (seconds), if available."""
        return self.t_first_chunk

    @property
    def total_time(self) -> float | None:
        """End-to-end seconds."""
        if self.t_last_chunk is None:
            return None
        return self.t_last_chunk

    @property
    def decode_time(self) -> float | None:
        """Decode-only seconds (excludes prefill / TTFT)."""
        if self.t_first_chunk is None or self.t_last_chunk is None:
            return None
        return max(self.t_last_chunk - self.t_first_chunk, 1e-9)

    @property
    def tpot_s(self) -> float | None:
        """Time per output token (seconds)."""
        if self.decode_time is None or not self.completion_tokens:
            return None
        if self.completion_tokens <= 1:
            return None
        return self.decode_time / (self.completion_tokens - 1)

    def inter_chunk_intervals(self) -> list[float]:
        """Inter-arrival times between chunks (seconds)."""
        if len(self.events) < 2:
            return []
        return [
            self.events[i].t_relative - self.events[i - 1].t_relative
            for i in range(1, len(self.events))
        ]


async def stream_chat(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    prompt: str,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
    ignore_eos: bool = True,
    extra_body: dict[str, Any] | None = None,
) -> StreamResult:
    """
    Issue one streaming chat completion. Records per-chunk timestamps.

    ignore_eos defaults to True for benchmarking: keeps output length fixed
    so we measure compute, not the model's choice of when to stop.
    """
    body: dict[str, Any] = {
        "model": endpoint.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None and os.environ.get("VK_DROP_SEED", "0") != "1":
        body["seed"] = seed
    if ignore_eos:
        # vLLM / SGLang accept this; some servers ignore. Harmless if ignored.
        body["ignore_eos"] = True
    if extra_body:
        body.update(extra_body)

    headers = {
        "Authorization": f"Bearer {endpoint.api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    result = StreamResult(t_start=t0, t_first_chunk=None, t_last_chunk=None)

    try:
        async with client.stream(
            "POST",
            f"{endpoint.base_url.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                err_body = await resp.aread()
                result.error = f"HTTP {resp.status_code}: {err_body[:500]!r}"
                return result

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: "):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter() - t0
                usage = obj.get("usage")
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens")
                    result.completion_tokens = usage.get("completion_tokens")

                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content") or ""
                finish_reason = choice.get("finish_reason")

                if text:
                    if result.t_first_chunk is None:
                        result.t_first_chunk = now
                    result.t_last_chunk = now
                    result.events.append(
                        ChunkEvent(
                            t_relative=now,
                            text=text,
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                    )
                    result.full_text += text
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        result.error = f"{type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# Random-token prompt generation (for cache-busting and roofline runs)
# ---------------------------------------------------------------------------

# A vocabulary of common English words. Used to build pseudo-random prompts
# that survive tokenization without collapsing into a small set of subwords.
_WORDS = (
    "the quick brown fox jumps over a lazy dog while computing partial sums "
    "from tensor cores attached to high bandwidth memory across chips in a "
    "pod configuration with explicit collective operations and overlapping "
    "communication with computation to maintain throughput under load on "
    "modern accelerators serving large language models in production "
    "environments where latency budgets matter as much as raw throughput "
    "and where the scheduler must balance prefill and decode work fairly "
    "between concurrent requests without starving long generations or "
    "blocking short ones behind expensive ones in the queue"
).split()


def random_prompt(n_tokens_approx: int, seed: int | None = None) -> str:
    """
    Generate a roughly-N-token prompt of random English words.

    Used for cache-busting: each prompt has a unique prefix so no prefix
    cache can hit. Token count is approximate (1 word ≈ 1.3 tokens for
    most tokenizers; the harness should resample if it needs an exact count).
    """
    rng = random.Random(seed)
    # Approx 1 token per 0.95 words for English BPE (gpt-oss tokenizer is very efficient)
    n_words = max(1, int(n_tokens_approx / 0.95))
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def random_prompt_in_tokens(
    target_prompt_tokens: int,
    seed: int | None = None,
    tolerance: int = 4,
    max_iter: int = 20,
) -> str:
    """Generate a random prompt that produces exactly `target_prompt_tokens`
    (within `tolerance`) when wrapped as a single user message via the chat
    template.

    Use this when you need the server's `usage.prompt_tokens` to land on a
    specific value -- e.g., to control where `prompt_tokens + max_tokens` sits
    relative to a server's `max_model_len`. `random_prompt()` is content-token
    accurate but does not account for chat-template overhead (~600+ tokens for
    gpt-oss harmony), so the server-reported prompt_tokens is systematically
    larger than the requested value.

    Requires `transformers` installed and `PARTNER_TOKENIZER` (or default
    `openai/gpt-oss-120b`) loadable. Assumes the server uses the same chat
    template as the HF tokenizer.
    """
    tok = get_tokenizer()
    n_words = max(1, target_prompt_tokens)
    text = ""
    for _ in range(max_iter):
        rng = random.Random(seed)
        text = " ".join(rng.choice(_WORDS) for _ in range(n_words))
        actual = tok.chat_token_count(text)
        diff = target_prompt_tokens - actual
        if abs(diff) <= tolerance:
            return text
        n_words = max(1, n_words + diff)
    return text


def repeat_phrase_prompt(target_output_tokens: int = 256) -> str:
    """
    Low-entropy prompt: asks the model to repeat a phrase many times.

    Used for the entropy probe: if multi-token decoding is active, draft
    acceptance will be near 100% on this prompt because every next token
    is trivially predictable.
    """
    return (
        f"Please write the phrase 'the cat sat on the mat' exactly "
        f"{target_output_tokens // 6 + 1} times in a row, separated by single "
        "spaces, with no other text, numbering, or punctuation."
    )


def high_entropy_prompt(seed: int | None = None) -> str:
    """
    High-entropy continuation prompt: asks for random-looking content.

    Used for the entropy probe paired with high temperature: draft
    acceptance collapses toward 1/k, so multi-token decoding becomes
    much slower per token than for the low-entropy prompt.
    """
    rng = random.Random(seed)
    seed_text = " ".join(rng.choice(_WORDS) for _ in range(40))
    return (
        "Continue the following text with completely unpredictable, "
        "creative, surprising content. Avoid common phrases and structures.\n\n"
        + seed_text
    )


# ---------------------------------------------------------------------------
# Tokenizer for tokens-per-chunk analysis
# ---------------------------------------------------------------------------

class _LazyTokenizer:
    """Loads transformers tokenizer once, on first use."""

    def __init__(self, name_or_path: str | None = None) -> None:
        self._name = name_or_path or os.environ.get(
            "PARTNER_TOKENIZER", "openai/gpt-oss-120b"
        )
        self._tok: Any | None = None

    def _load(self) -> Any:
        if self._tok is None:
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "transformers not installed. `pip install transformers` "
                    "or skip token-level analysis."
                ) from e
            self._tok = AutoTokenizer.from_pretrained(self._name)
        return self._tok

    def count(self, text: str) -> int:
        tok = self._load()
        return len(tok.encode(text, add_special_tokens=False))

    def chat_token_count(self, prompt: str, role: str = "user") -> int:
        """Token count after wrapping in a single-message chat template.

        Matches what a server reports as `usage.prompt_tokens` (assuming the
        server uses the same chat template as the HF tokenizer).

        Note: we render with `tokenize=False` and then encode separately,
        because some chat templates (notably gpt-oss harmony) return only the
        2-token generation prompt when called with `tokenize=True`.
        """
        tok = self._load()
        rendered = tok.apply_chat_template(
            [{"role": role, "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tok.encode(rendered, add_special_tokens=False))


_tokenizer_singleton: _LazyTokenizer | None = None


def get_tokenizer() -> _LazyTokenizer:
    global _tokenizer_singleton
    if _tokenizer_singleton is None:
        _tokenizer_singleton = _LazyTokenizer()
    return _tokenizer_singleton


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def percentiles(xs: list[float], ps: list[float]) -> dict[str, float]:
    """Compute percentiles. Returns dict like {'p50': ..., 'p95': ...}."""
    if not xs:
        return {f"p{int(p * 100)}": float("nan") for p in ps}
    s = sorted(xs)
    out = {}
    for p in ps:
        idx = min(len(s) - 1, int(p * len(s)))
        out[f"p{int(p * 100)}"] = s[idx]
    return out


def summary_stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "stdev": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
        **percentiles(xs, [0.5, 0.9, 0.95, 0.99]),
    }


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

async def warmup(endpoint: Endpoint, n: int = 5, max_tokens: int = 64) -> None:
    """
    Issue a few dummy requests so JIT/compilation/autotuning is past us
    before the measured phase begins.
    """
    async with httpx.AsyncClient() as client:
        for i in range(n):
            await stream_chat(
                client,
                endpoint,
                random_prompt(64, seed=10_000 + i),
                max_tokens=max_tokens,
                temperature=0.0,
            )
