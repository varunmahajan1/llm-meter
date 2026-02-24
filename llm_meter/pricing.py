"""
Token pricing table.

All prices are in USD per 1,000 tokens.
Update as providers change their pricing.
"""

from typing import Dict

# Pricing per 1K tokens: {"input": float, "output": float}
PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o":                  {"input": 0.005,    "output": 0.015},
    "gpt-4o-mini":             {"input": 0.00015,  "output": 0.0006},
    "gpt-4-turbo":             {"input": 0.010,    "output": 0.030},
    "gpt-4":                   {"input": 0.030,    "output": 0.060},
    "gpt-3.5-turbo":           {"input": 0.0005,   "output": 0.0015},
    "o1":                      {"input": 0.015,    "output": 0.060},
    "o1-mini":                 {"input": 0.003,    "output": 0.012},
    # Anthropic
    "claude-3-5-sonnet":       {"input": 0.003,    "output": 0.015},
    "claude-3-5-haiku":        {"input": 0.00080,  "output": 0.00400},
    "claude-3-opus":           {"input": 0.015,    "output": 0.075},
    "claude-3-sonnet":         {"input": 0.003,    "output": 0.015},
    "claude-3-haiku":          {"input": 0.00025,  "output": 0.00125},
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


def add_model_pricing(model: str, input_per_1k: float, output_per_1k: float) -> None:
    """Register pricing for a custom or new model.

    Args:
        model:           Model identifier (must match what you pass to CostTracker.record).
        input_per_1k:    Price per 1,000 input (prompt) tokens in USD.
        output_per_1k:   Price per 1,000 output (completion) tokens in USD.
    """
    PRICING[model] = {"input": input_per_1k, "output": output_per_1k}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD for a given token usage.

    Falls back to "default" pricing if the model is not in the table.
    """
    pricing = PRICING.get(model) or PRICING.get(
        next((k for k in PRICING if model.startswith(k)), "default"),
        PRICING["default"],
    )
    input_cost = (prompt_tokens / 1_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000) * pricing["output"]
    return round(input_cost + output_cost, 8)
