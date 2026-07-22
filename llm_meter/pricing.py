"""
Token pricing table.

All prices are in USD per 1,000 tokens.

Each model entry carries ``input`` and ``output`` prices and *may* carry
``cache_read`` and ``cache_write`` prices for prompt-cache tokens. A missing
cache key means "not priced" — cache tokens billed against that model are
reported as *unpriced* rather than silently costed at $0 (see
:func:`compute_cost` and CostTracker's unpriced-token bucket).

Cache prices below follow each provider's published multipliers:
  * Anthropic  — cache write = 1.25x input, cache read = 0.1x input.
  * OpenAI     — cached input read = 0.5x input; cache writes are not billed,
                 so ``cache_write`` is set to 0.0 (priced-and-free, which is
                 deliberately different from a *missing* key / unpriced).

Prices drift — providers change them without notice. Verify against the
provider pricing pages before trusting these numbers for real invoicing.
"""

from dataclasses import dataclass
from typing import Dict, Optional

# Pricing per 1K tokens. Required keys: "input", "output".
# Optional keys: "cache_read", "cache_write" (prompt-cache tokens).
PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI — cached input read = 0.5x input; cache writes are not billed (0.0).
    "gpt-4o":                  {"input": 0.005,    "output": 0.015,
                                "cache_read": 0.0025,   "cache_write": 0.0},
    "gpt-4o-mini":             {"input": 0.00015,  "output": 0.0006,
                                "cache_read": 0.000075, "cache_write": 0.0},
    "gpt-4-turbo":             {"input": 0.010,    "output": 0.030},
    "gpt-4":                   {"input": 0.030,    "output": 0.060},
    "gpt-3.5-turbo":           {"input": 0.0005,   "output": 0.0015},
    "o1":                      {"input": 0.015,    "output": 0.060},
    "o1-mini":                 {"input": 0.003,    "output": 0.012},
    # Anthropic — cache read = 0.1x input, cache write = 1.25x input.
    "claude-3-5-sonnet":       {"input": 0.003,    "output": 0.015,
                                "cache_read": 0.0003,    "cache_write": 0.00375},
    "claude-3-5-haiku":        {"input": 0.00080,  "output": 0.00400,
                                "cache_read": 0.00008,   "cache_write": 0.00100},
    "claude-3-opus":           {"input": 0.015,    "output": 0.075,
                                "cache_read": 0.0015,    "cache_write": 0.01875},
    "claude-3-sonnet":         {"input": 0.003,    "output": 0.015,
                                "cache_read": 0.0003,    "cache_write": 0.00375},
    "claude-3-haiku":          {"input": 0.00025,  "output": 0.00125,
                                "cache_read": 0.000025,  "cache_write": 0.0003125},
    # Google
    "gemini-1.5-pro":          {"input": 0.00125,  "output": 0.005},
    "gemini-1.5-flash":        {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash":        {"input": 0.000100, "output": 0.0004},
    # Moonshot / Kimi
    "kimi-k2":                 {"input": 0.010,    "output": 0.030},
    "kimi-k2.5":               {"input": 0.010,    "output": 0.030},
    # Meta / open weights (self-hosted approximate cost)
    "llama-3.1-70b":           {"input": 0.0008,   "output": 0.0008},
    "llama-3.1-8b":            {"input": 0.0002,   "output": 0.0002},
    # Fallback
    "default":                 {"input": 0.010,    "output": 0.030},
}


@dataclass
class CostBreakdown:
    """Per-call cost split across input, output, and prompt-cache tokens.

    ``unpriced_cache_read_tokens`` / ``unpriced_cache_write_tokens`` count cache
    tokens billed against a model that has no cache price defined. They are
    *not* folded into any cost figure — they are reported so a tiny metered
    bill can never hide a large real one.
    """
    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_write_cost: float
    total_cost: float
    unpriced_cache_read_tokens: int = 0
    unpriced_cache_write_tokens: int = 0

    @property
    def unpriced_tokens(self) -> int:
        return self.unpriced_cache_read_tokens + self.unpriced_cache_write_tokens


def add_model_pricing(
    model: str,
    input_per_1k: float,
    output_per_1k: float,
    cache_read_per_1k: Optional[float] = None,
    cache_write_per_1k: Optional[float] = None,
) -> None:
    """Register pricing for a custom or new model.

    Args:
        model:               Model identifier (must match what you pass to
                             CostTracker.record).
        input_per_1k:        Price per 1,000 input (prompt) tokens in USD.
        output_per_1k:       Price per 1,000 output (completion) tokens in USD.
        cache_read_per_1k:   Optional price per 1,000 cache-read tokens. Omit
                             (or None) to leave cache reads unpriced.
        cache_write_per_1k:  Optional price per 1,000 cache-write tokens. Omit
                             (or None) to leave cache writes unpriced.

    Pass ``0.0`` (not None) for a cache price that is deliberately free — that
    is priced-and-free and will not show up as unpriced usage.
    """
    entry: Dict[str, float] = {"input": input_per_1k, "output": output_per_1k}
    if cache_read_per_1k is not None:
        entry["cache_read"] = cache_read_per_1k
    if cache_write_per_1k is not None:
        entry["cache_write"] = cache_write_per_1k
    PRICING[model] = entry


def resolve_pricing(model: str) -> Dict[str, float]:
    """Return the pricing entry for ``model``, falling back to prefix match then
    the ``default`` entry."""
    return PRICING.get(model) or PRICING.get(
        next((k for k in PRICING if model.startswith(k)), "default"),
        PRICING["default"],
    )


def compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> CostBreakdown:
    """Compute a full :class:`CostBreakdown` for a single request.

    Cache tokens billed against a model with no matching cache price are placed
    in the unpriced buckets instead of being costed at $0. Falls back to
    ``default`` pricing when the model is unknown.
    """
    pricing = resolve_pricing(model)

    input_cost = (prompt_tokens / 1_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000) * pricing["output"]

    cache_read_cost = 0.0
    unpriced_read = 0
    if cache_read_tokens:
        if "cache_read" in pricing:
            cache_read_cost = (cache_read_tokens / 1_000) * pricing["cache_read"]
        else:
            unpriced_read = cache_read_tokens

    cache_write_cost = 0.0
    unpriced_write = 0
    if cache_write_tokens:
        if "cache_write" in pricing:
            cache_write_cost = (cache_write_tokens / 1_000) * pricing["cache_write"]
        else:
            unpriced_write = cache_write_tokens

    total = input_cost + output_cost + cache_read_cost + cache_write_cost
    return CostBreakdown(
        input_cost=round(input_cost, 8),
        output_cost=round(output_cost, 8),
        cache_read_cost=round(cache_read_cost, 8),
        cache_write_cost=round(cache_write_cost, 8),
        total_cost=round(total, 8),
        unpriced_cache_read_tokens=unpriced_read,
        unpriced_cache_write_tokens=unpriced_write,
    )


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate input+output cost in USD for a given token usage.

    Falls back to "default" pricing if the model is not in the table. For cache
    token accounting use :func:`compute_cost`.
    """
    return compute_cost(model, prompt_tokens, completion_tokens).total_cost
