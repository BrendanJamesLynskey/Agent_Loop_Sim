"""Token, cache, cost and latency accounting for each model call.

**Prices** are list prices in US dollars per million tokens, copied from the providers'
public pricing pages on the date in ``accessed``. They are **illustrative**: prices change,
and the token counts here come from Qwen2.5's tokenizer, not from these providers'
tokenizers, so a cost is "what this many tokens would cost at that price", not a quote.

**Prompt cache.** A provider keeps the KV cache of recent prompt prefixes. A call whose
prompt starts with a cached prefix pays the cache-read price for those tokens. The model
here: every prompt (of at least ``min_tokens``) is written to the cache; a later prompt's
cached tokens are its longest common token prefix with any live entry, rounded down to a
multiple of ``block``; an entry lives ``ttl_ms`` after it was last written or read. Cache
writes cost ``cache_write`` per million tokens (1.25x input for a 5-minute write on the
Claude API and on OpenAI's GPT-5.6 and later; no premium where a provider charges none).
A read costs ``cache_read``, which is ``read_multiplier`` (rho) times the input price: 0.1 on
most models, 0.05 on Claude Opus 5.5, Claude Sonnet 5.5 and GPT-6.1 Sol, 0.025 on Claude
Fable 5.1 (since 1.12). ``generation`` groups the models for the sites' price switch; a
model listed with no cached-input price (``cache_listed`` false) is modelled with no cache.

**Thinking (reasoning) tokens** (since 1.12): a policy may set ``thinking: {"tokens": n}``;
every acting model call then also generates ``n`` reasoning tokens before its answer, billed
at the output price and decoded at the decode rate, and not added to the context. (Claude's
4.6-and-later models keep earlier turns' thinking in the context and bill it as input; that
part is not modelled.) The number is a fixed, illustrative budget: real models choose how
much to think on each call.

**Latency** is the simple model of the LLM Inference Explained site: time to first token =
a fixed overhead + the uncached prompt tokens / prefill rate; then the output tokens at the
decode rate. Cached prompt tokens are taken as free to prefill.
"""
from __future__ import annotations

from typing import Any

PRICES: dict[str, dict[str, Any]] = {
    "claude-sonnet-4.6": {
        "label": "Claude Sonnet 4.6 (API list price)",
        "generation": "Claude 4.x",
        "input": 3,
        "output": 15,
        "cache_read": 0.3,
        "cache_write": 3.75,
        "read_multiplier": 0.1,
        "min_tokens": 1024,
        "block": 1,
        "ttl_ms": 300000,
        "cache_listed": True,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "source_cache": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
        "accessed": "2026-10-16",
    },
    "claude-haiku-4.5": {
        "label": "Claude Haiku 4.5 (API list price)",
        "generation": "Claude 4.x",
        "input": 1,
        "output": 5,
        "cache_read": 0.1,
        "cache_write": 1.25,
        "read_multiplier": 0.1,
        "min_tokens": 4096,
        "block": 1,
        "ttl_ms": 300000,
        "cache_listed": True,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "source_cache": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
        "accessed": "2026-10-16",
    },
    "gpt-5-mini": {
        "label": "GPT-5 mini (API list price, standard tier)",
        "generation": "GPT-5",
        "input": 0.25,
        "output": 2,
        "cache_read": 0.025,
        "cache_write": 0.25,
        "read_multiplier": 0.1,
        "min_tokens": 1024,
        "block": 128,
        "ttl_ms": 1800000,
        "cache_listed": True,
        "source": "https://developers.openai.com/api/docs/pricing",
        "source_cache": "https://developers.openai.com/api/docs/guides/prompt-caching",
        "accessed": "2026-10-16",
    },
    "claude-opus-5.5": {
        "label": "Claude Opus 5.5 (API list price)",
        "generation": "Claude 5.x",
        "input": 4,
        "output": 20,
        "cache_read": 0.2,
        "cache_write": 5,
        "read_multiplier": 0.05,
        "min_tokens": 512,
        "block": 1,
        "ttl_ms": 300000,
        "cache_listed": True,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "source_cache": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
        "accessed": "2026-10-16",
    },
    "claude-sonnet-5.5": {
        "label": "Claude Sonnet 5.5 (API list price)",
        "generation": "Claude 5.x",
        "input": 2,
        "output": 10,
        "cache_read": 0.1,
        "cache_write": 2.5,
        "read_multiplier": 0.05,
        "min_tokens": 512,
        "block": 1,
        "ttl_ms": 300000,
        "cache_listed": True,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "source_cache": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
        "accessed": "2026-10-16",
    },
    "claude-haiku-5.5": {
        "label": "Claude Haiku 5.5 (API list price, prompts up to 100,000 tokens)",
        "generation": "Claude 5.x",
        "input": 0.1,
        "output": 0.5,
        "cache_read": 0.01,
        "cache_write": 0.125,
        "read_multiplier": 0.1,
        "min_tokens": 512,
        "block": 1,
        "ttl_ms": 300000,
        "cache_listed": True,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "source_cache": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
        "accessed": "2026-10-16",
    },
    "claude-fable-5.1": {
        "label": "Claude Fable 5.1 (API list price)",
        "generation": "Claude 5.x",
        "input": 10,
        "output": 50,
        "cache_read": 0.25,
        "cache_write": 12.5,
        "read_multiplier": 0.025,
        "min_tokens": 512,
        "block": 1,
        "ttl_ms": 300000,
        "cache_listed": True,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing",
        "source_cache": "https://platform.claude.com/docs/en/build-with-claude/prompt-caching",
        "accessed": "2026-10-16",
    },
    "gpt-6.1-sol": {
        "label": "GPT-6.1 Sol (API list price, standard tier, short context)",
        "generation": "GPT-6",
        "input": 2,
        "output": 10,
        "cache_read": 0.1,
        "cache_write": 2.5,
        "read_multiplier": 0.05,
        "min_tokens": 1024,
        "block": 1,
        "ttl_ms": 1800000,
        "cache_listed": True,
        "source": "https://developers.openai.com/api/docs/pricing",
        "source_cache": "https://developers.openai.com/api/docs/guides/prompt-caching",
        "accessed": "2026-10-16",
    },
    "gpt-6-luna": {
        "label": "GPT-6 Luna (API list price, standard tier, short context)",
        "generation": "GPT-6",
        "input": 0.1,
        "output": 0.5,
        "cache_read": 0.01,
        "cache_write": 0.125,
        "read_multiplier": 0.1,
        "min_tokens": 1024,
        "block": 1,
        "ttl_ms": 1800000,
        "cache_listed": True,
        "source": "https://developers.openai.com/api/docs/pricing",
        "source_cache": "https://developers.openai.com/api/docs/guides/prompt-caching",
        "accessed": "2026-10-16",
    },
    "gpt-6-astra": {
        "label": "GPT-6 Astra (API list price, standard tier, short context)",
        "generation": "GPT-6",
        "input": 10,
        "output": 50,
        "cache_read": 1,
        "cache_write": 12.5,
        "read_multiplier": 0.1,
        "min_tokens": 1024,
        "block": 1,
        "ttl_ms": 1800000,
        "cache_listed": True,
        "source": "https://developers.openai.com/api/docs/pricing",
        "source_cache": "https://developers.openai.com/api/docs/guides/prompt-caching",
        "accessed": "2026-10-16",
    },
    "gpt-oss-120b": {
        "label": "gpt-oss-120b, open weights (Together AI serverless list price)",
        "generation": "open weights",
        "input": 0.15,
        "output": 0.6,
        "cache_read": 0.15,
        "cache_write": 0.15,
        "read_multiplier": 1,
        "min_tokens": 1000000000,
        "block": 1,
        "ttl_ms": 0,
        "cache_listed": False,
        "source": "https://www.together.ai/pricing",
        "source_cache": "https://www.together.ai/pricing",
        "accessed": "2026-10-16",
    },
    "local": {
        "label": "Local model (no per-token price)",
        "generation": "local",
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "read_multiplier": 0,
        "min_tokens": 0,
        "block": 1,
        "ttl_ms": 1e18,
        "cache_listed": True,
        "source": "",
        "source_cache": "",
        "accessed": "2026-10-16",
    },
}

LATENCY: dict[str, dict[str, Any]] = {
    "hosted": {
        "label": "Hosted API (illustrative)",
        "overhead_ms": 400,
        "prefill_tps": 5000,
        "decode_tps": 80,
        "source": "illustrative round numbers; see the LLM Inference Explained site for where they come from",
    },
    "local-i7": {
        "label": "Qwen2.5-1.5B Q4_K_M, llama.cpp, i7-3770 CPU (measured)",
        "overhead_ms": 50,
        "prefill_tps": 42.7,
        "decode_tps": 16.5,
        "source": "prefill and decode rates: medians of the recorded traces' llama.cpp timings (traces/*.json; "
        "prefill over calls with at least 100 uncached prompt tokens); the 50 ms overhead is assumed",
    },
}


def lcp(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PromptCache:
    """The provider-side prefix cache of one run (see the module docstring)."""

    def __init__(self, enabled: bool, min_tokens: int, block: int, ttl_ms: float) -> None:
        self.enabled = enabled
        self.min_tokens = min_tokens
        self.block = block
        self.ttl_ms = ttl_ms
        self.entries: list[dict[str, Any]] = []  # {"ids": [...], "t": last use}

    def lookup(self, ids: list[int], now: float) -> int:
        """Cached tokens for a prompt sent at ``now``; refreshes the entry it reads."""
        if not self.enabled:
            return 0
        self.entries = [e for e in self.entries if now - e["t"] <= self.ttl_ms]
        best = 0
        best_entry = None
        for e in self.entries:
            n = lcp(e["ids"], ids)
            if n > best:
                best = n
                best_entry = e
        cached = (best // self.block) * self.block
        if cached < self.min_tokens:
            return 0
        if best_entry is not None:
            best_entry["t"] = now
        return cached

    def store(self, ids: list[int], now: float) -> bool:
        """Write a prompt to the cache; False if caching is off or the prompt is too short."""
        if not self.enabled or len(ids) < self.min_tokens:
            return False
        self.entries.append({"ids": ids, "t": now})
        return True


def call_cost(price: dict[str, Any], input_tokens: int, cached: int, stored: bool, output_tokens: int) -> float:
    """US dollars for one call. The cached prefix pays the cache-read price; if the prompt
    is written to the cache, the rest of it pays the cache-write price, else the input price."""
    fresh = input_tokens - cached
    rate = price["cache_write"] if stored else price["input"]
    return (cached * price["cache_read"] + fresh * rate + output_tokens * price["output"]) / 1e6


def call_latency(profile: dict[str, Any], input_tokens: int, cached: int, output_tokens: int) -> tuple[float, float]:
    """(time to first token, total duration) of one call, in milliseconds."""
    ttft = profile["overhead_ms"] + (input_tokens - cached) * 1000 / profile["prefill_tps"]
    return ttft, ttft + output_tokens * 1000 / profile["decode_tps"]
