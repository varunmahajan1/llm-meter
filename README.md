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
- **Prompt-cache token accounting** — cache-read and cache-write tokens priced separately
- **Unpriced-token detection** — cache tokens with no defined price are bucketed and surfaced, never silently costed at $0
- **Billing drift reconciliation** — compare metered spend against the real invoice
- **Configurable limits** with threshold alerts (at 80%) and hard limits
- **Sliding-window rate limiting** using Redis sorted sets or in-memory lists
- **Zero-config dual backend** — in-memory in dev, Redis in production, no code changes

---

## Cache token accounting

> **War story.** Metering on a production AI assistant I run reported a monthly
> bill an *order of magnitude* below the real invoice — a couple of dollars
> metered against a several-hundred-dollar charge. The cause: prompt-cache
> read/write tokens (which dwarfed the base input tokens) had no price in the
> table, so they were costed at $0, and a few call paths weren't metered at all.
> The meter looked healthy while the spend ran away. This release makes unpriced
> usage impossible to miss.

Modern providers bill prompt-cache tokens on a separate line from base input:

- **Anthropic** — cache write = 1.25x input, cache read = 0.1x input.
- **OpenAI** (gpt-4o / gpt-4o-mini class) — cached input read = 0.5x input; cache
  writes are not billed (priced at `0.0`).

Pass the counts your provider reports as separate `cache_read_tokens` /
`cache_write_tokens` (Anthropic-style — counts distinct from base input, not a
subset of it):

```python
usage = tracker.record(
    user_id="user-123",
    session_id="sess-abc",
    model="claude-3-5-sonnet",
    prompt_tokens=1_000,
    completion_tokens=800,
    cache_read_tokens=40_000,    # priced at 0.1x input
    cache_write_tokens=12_000,   # priced at 1.25x input
)
print(usage.cost_usd, usage.cache_read_cost, usage.cache_write_cost)
```

Reports break the cache costs and token counts out:

```python
report = tracker.session_report("user-123", "sess-abc")
# {
#   ...
#   "cache_read_tokens": 40000,
#   "cache_write_tokens": 12000,
#   "cache_read_cost_usd": 0.012,
#   "cache_write_cost_usd": 0.045,
#   "unpriced_tokens": 0,
#   "unpriced_models": {},
# }
```

### Unpriced tokens are surfaced, never hidden

If cache tokens are recorded for a model that has **no** cache price defined,
they are counted in an `unpriced_tokens` bucket (with the offending models) and
reported everywhere — instead of quietly becoming $0. Register a callback to be
alerted the moment it happens:

```python
def alert_unpriced(event):
    log.warning(
        "Unpriced cache tokens: %d on %s — metered cost is understated",
        event.total_unpriced_tokens, event.model,
    )

tracker = CostTracker(on_unpriced=alert_unpriced)
tracker.record("u", "s", "gemini-1.5-pro", 100, 50, cache_read_tokens=40_000)
# alert_unpriced fires: gemini-1.5-pro has no cache price in the table
```

Add cache pricing for your own models (both fields optional and backward
compatible — pass `0.0` for a cache tier that is deliberately free):

```python
from llm_meter import add_model_pricing

add_model_pricing(
    "my-model",
    input_per_1k=0.002,
    output_per_1k=0.008,
    cache_read_per_1k=0.0002,
    cache_write_per_1k=0.0025,
)
```

---

## Billing drift detection

Even a well-instrumented meter should be checked against the real invoice.
`reconcile` compares what you metered for a period against what the provider
actually billed and flags drift over a configurable threshold:

```python
from llm_meter import CostTracker

report = CostTracker.reconcile(
    actual_billed_usd=450.00,   # from the provider invoice
    metered_usd=9.59,           # what llm-meter reported for the period
    drift_threshold=0.25,       # flag when >25% off
)
# DriftReport(
#   metered_usd=9.59,
#   actual_billed_usd=450.0,
#   drift_usd=440.41,           # positive = meter under-reported
#   drift_ratio=45.9,           # the meter was ~46x too low
#   threshold=0.25,
#   over_threshold=True,
# )
if report.over_threshold:
    page_oncall(report)
```

`drift_ratio` is relative to the metered figure, so `0.02` means "invoice was 2%
higher than metered" and `45.9` means "the meter reported ~46x too little" — the
exact failure mode from the war story above.

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
    "cache_read_tokens": 0,
    "cache_write_tokens": 0,
    "cache_read_cost_usd": 0.0,
    "cache_write_cost_usd": 0.0,
    "unpriced_tokens": 0,
    "unpriced_models": {},
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

Built-in pricing for all major providers. Prices are USD per 1,000 tokens.
Cache columns are blank where the provider has no published cache tier in the
table — recording cache tokens against those models surfaces them as *unpriced*
(see [Cache token accounting](#cache-token-accounting)):

| Model | Input | Output | Cache read | Cache write |
|---|---|---|---|---|
| gpt-4o | $0.005 | $0.015 | $0.0025 | $0.0 (free) |
| gpt-4o-mini | $0.00015 | $0.0006 | $0.000075 | $0.0 (free) |
| claude-3-5-sonnet | $0.003 | $0.015 | $0.0003 | $0.00375 |
| claude-3-5-haiku | $0.0008 | $0.004 | $0.00008 | $0.001 |
| claude-3-opus | $0.015 | $0.075 | $0.0015 | $0.01875 |
| gemini-1.5-pro | $0.00125 | $0.005 | — | — |
| kimi-k2.5 | $0.010 | $0.030 | — | — |
| *(and more)* | | | | |

> Prices drift — providers change them without notice. Verify against the
> provider pricing pages before trusting these numbers for real invoicing.

### Add custom pricing

```python
from llm_meter import add_model_pricing

# input/output only (cache tokens for this model stay unpriced)
add_model_pricing("my-fine-tuned-model", input_per_1k=0.002, output_per_1k=0.008)

# with cache tiers
add_model_pricing(
    "my-cached-model",
    input_per_1k=0.002,
    output_per_1k=0.008,
    cache_read_per_1k=0.0002,
    cache_write_per_1k=0.0025,
)
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

## Related projects

Sibling libraries for building reliable LLM applications:

- [agent-runtime](https://github.com/varunmahajan1/agent-runtime) — agent execution loop and tool orchestration
- [agent-stream](https://github.com/varunmahajan1/agent-stream) — streaming primitives for agent output
- [promptshield](https://github.com/varunmahajan1/promptshield) — prompt-injection and jailbreak defense
- [llm-failover](https://github.com/varunmahajan1/llm-failover) — multi-provider failover and retries
- [ssrfguard](https://github.com/varunmahajan1/ssrfguard) — SSRF protection for tool/URL fetching
- [timeanchor](https://github.com/varunmahajan1/timeanchor) — time-grounding helpers for LLM prompts
- [channelfmt](https://github.com/varunmahajan1/channelfmt) — per-channel message formatting

---

## License

MIT
