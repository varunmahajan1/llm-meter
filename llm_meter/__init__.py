"""
llm-meter — LLM token cost tracking and rate limiting.

Usage:
    from llm_meter import CostTracker, RateLimiter

    tracker = CostTracker()
    tracker.record(user_id="u1", session_id="s1", model="gpt-4o",
                   prompt_tokens=500, completion_tokens=200)

    report = tracker.session_report("u1", "s1")
    print(f"${report['total_cost_usd']:.4f} used ({report['percentage_used']:.1f}%)")

    limiter = RateLimiter()
    allowed, info = limiter.check("user-123", max_requests=60, window_seconds=60)
    if not allowed:
        raise Exception(f"Rate limit hit. Retry in {info['retry_after']}s")
"""

from .tracker import (
    CostTracker,
    TokenUsage,
    CostAlert,
    LimitExceeded,
    UnpricedEvent,
    DriftReport,
)
from .limiter import RateLimiter
from .pricing import PRICING, add_model_pricing, compute_cost, CostBreakdown

__all__ = [
    "CostTracker",
    "TokenUsage",
    "CostAlert",
    "LimitExceeded",
    "UnpricedEvent",
    "DriftReport",
    "RateLimiter",
    "PRICING",
    "add_model_pricing",
    "compute_cost",
    "CostBreakdown",
]

__version__ = "0.2.0"
