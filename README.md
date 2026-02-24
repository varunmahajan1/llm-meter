# llm-meter

**LLM token cost tracking and rate limiting — works with Redis or in-memory.**

```bash
pip install llm-meter
pip install llm-meter[redis]   # for Redis backend
```

---

## Why

Every LLM application needs to know how much it's spending and how fast requests are coming in. `llm-meter` gives you:

- **Per-session and per-day token accounting** with USD cost calculation
- **Configurable limits** with threshold alerts (at 80%) and hard limits
- **Sliding-window rate limiting** using Redis sorted sets or in-memory lists
- **Zero-config dual backend** — in-memory in dev, Redis in production, no code changes

---

## Cost Tracker

```python
from llm_meter import CostTracker

tracker = CostTracker(
    redis_url="redis://localhost:6379/0",  # optional; defaults to in-memory
    session_token_limit=100_000,
    daily_token_limit=500_000,
    warn_threshold_pct=80.0,
)

# Record a request
usage = tracker.record(
    user_id="user-123",
    session_id="sess-abc",
    model="gpt-4o",
    prompt_tokens=800,
    completion_tokens=300,
)
print(f"This call: {usage.total_tokens} tokens, ${usage.cost_usd:.6f}")

# Check limits
allowed, alert = tracker.check("user-123", "sess-abc")
if not allowed:
    raise Exception(f"Limit exceeded: {alert.message}")
if alert and alert.alert_type == "threshold":
    logger.warning("Usage at %.1f%%", alert.percentage)

# Or raise automatically
from llm_meter import LimitExceeded
try:
    tracker.check_and_raise("user-123", "sess-abc")
except LimitExceeded as e:
    return {"error": str(e), "limit_type": e.alert.alert_type}

# Reports
session_report = tracker.session_report("user-123", "sess-abc")
daily_report = tracker.daily_report("user-123")
```

### Session report shape

```python
{
    "session_id": "sess-abc",
    "user_id": "user-123",
    "total_tokens": 12000,
    "total_cost_usd": 0.000720,
    "requests": 8,
    "models": {"gpt-4o": 9000, "gpt-4o-mini": 3000},
    "limit": 100000,
    "percentage_used": 12.0,
}
```

---

## Rate Limiter

```python
from llm_meter import RateLimiter

limiter = RateLimiter(redis_url="redis://localhost:6379/0")  # optional

# Check
allowed, info = limiter.check(
    identifier="user-123",
    max_requests=60,
    window_seconds=60,
)
if not allowed:
    raise Exception(f"Rate limited. Retry in {info['retry_after']}s")

# info dict:
# {
#   "allowed": False,
#   "current_requests": 60,
#   "max_requests": 60,
#   "window_seconds": 60,
#   "remaining": 0,
#   "retry_after": 60,
# }

# Quick boolean check
if not limiter.is_allowed("user-123"):
    return 429
```

---

## Pricing table

Built-in pricing for all major providers. Prices are USD per 1,000 tokens:

| Model | Input | Output |
|---|---|---|
| gpt-4o | $0.005 | $0.015 |
| gpt-4o-mini | $0.00015 | $0.0006 |
| claude-3-5-sonnet | $0.003 | $0.015 |
| claude-3-5-haiku | $0.0008 | $0.004 |
| gemini-1.5-pro | $0.00125 | $0.005 |
| kimi-k2.5 | $0.010 | $0.030 |
| *(and more)* | | |

### Add custom pricing

```python
from llm_meter import add_model_pricing

add_model_pricing("my-fine-tuned-model", input_per_1k=0.002, output_per_1k=0.008)
```

---

## Backend selection

| Condition | Backend used |
|---|---|
| `redis_url` not set | In-memory (no dependencies) |
| `redis_url` set, Redis reachable | Redis |
| `redis_url` set, Redis unreachable | Automatic in-memory fallback |

Both backends produce identical results. The in-memory backend does not enforce TTLs (all data lives until the process exits), which is fine for development.

---

## Integration example (FastAPI)

```python
from fastapi import FastAPI, HTTPException, Depends
from llm_meter import CostTracker, RateLimiter, LimitExceeded

app = FastAPI()
tracker = CostTracker(redis_url=REDIS_URL)
limiter = RateLimiter(redis_url=REDIS_URL)

@app.post("/chat")
async def chat(request: ChatRequest, user_id: str = Depends(get_user_id)):
    # Rate limit
    allowed, info = limiter.check(user_id, max_requests=10, window_seconds=60)
    if not allowed:
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": str(info["retry_after"])},
            detail="Too many requests",
        )

    # Cost limit
    try:
        tracker.check_and_raise(user_id, request.session_id)
    except LimitExceeded as e:
        raise HTTPException(status_code=402, detail=str(e))

    # Call LLM
    response = await call_llm(request.messages)

    # Record usage
    tracker.record(
        user_id=user_id,
        session_id=request.session_id,
        model=response.model,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )

    return response
```

---

## License

MIT
