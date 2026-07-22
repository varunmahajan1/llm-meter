"""
Per-session and per-day token cost tracking.

Supports Redis (production) or in-memory (dev/test) backends.
The backend is selected transparently — no code changes needed.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from .pricing import compute_cost

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """Record of a single LLM request.

    Cache tokens are counted separately from ``prompt_tokens`` /
    ``completion_tokens`` (Anthropic-style: the provider reports cache-read and
    cache-write counts distinct from the base input count). ``total_tokens``
    therefore sums all four buckets. ``unpriced_tokens`` counts cache tokens the
    model had no cache price for — those are surfaced, never costed at $0.
    """
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    unpriced_tokens: int = 0
    unpriced_cache_read_tokens: int = 0
    unpriced_cache_write_tokens: int = 0

    @classmethod
    def create(
        cls,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> "TokenUsage":
        cost = compute_cost(
            model, prompt_tokens, completion_tokens,
            cache_read_tokens, cache_write_tokens,
        )
        return cls(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                prompt_tokens + completion_tokens
                + cache_read_tokens + cache_write_tokens
            ),
            cost_usd=cost.total_cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_cost=cost.cache_read_cost,
            cache_write_cost=cost.cache_write_cost,
            unpriced_tokens=cost.unpriced_tokens,
            unpriced_cache_read_tokens=cost.unpriced_cache_read_tokens,
            unpriced_cache_write_tokens=cost.unpriced_cache_write_tokens,
        )


@dataclass
class UnpricedEvent:
    """Emitted when cache tokens are recorded for a model with no cache price.

    This is the failure mode where a meter under-reports by an order of
    magnitude: cache tokens dwarf base input tokens, and if they cost $0 the
    reported bill looks tiny next to the real invoice.
    """
    model: str
    cache_read_tokens: int
    cache_write_tokens: int
    total_unpriced_tokens: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class DriftReport:
    """Result of reconciling metered spend against a real invoice."""
    metered_usd: float
    actual_billed_usd: float
    drift_usd: float          # actual - metered (positive = meter under-reported)
    drift_ratio: float        # drift_usd / metered_usd (how far off the meter is)
    threshold: float
    over_threshold: bool


@dataclass
class CostAlert:
    """Alert when usage nears or exceeds a limit."""
    alert_type: str         # "threshold" | "limit_exceeded"
    message: str
    current_usage: int      # tokens
    limit: int
    percentage: float


class LimitExceeded(Exception):
    """Raised by CostTracker.check_and_raise when a limit is exceeded."""
    def __init__(self, alert: CostAlert) -> None:
        super().__init__(alert.message)
        self.alert = alert


# ── Backend abstraction ───────────────────────────────────────────────────────

class _Store:
    """Minimal key-value store interface used by CostTracker."""

    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None: ...
    def incr(self, key: str, amount: int = 1) -> int: ...


class _RedisStore(_Store):
    def __init__(self, redis_url: str) -> None:
        self._r = _redis_lib.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Optional[str]:
        try:
            return self._r.get(key)
        except Exception as e:
            logger.error("Redis GET error: %s", e)
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        try:
            if ttl:
                self._r.setex(key, ttl, value)
            else:
                self._r.set(key, value)
        except Exception as e:
            logger.error("Redis SET error: %s", e)

    def incr(self, key: str, amount: int = 1) -> int:
        try:
            return self._r.incrby(key, amount)
        except Exception as e:
            logger.error("Redis INCR error: %s", e)
            return 0


class _MemoryStore(_Store):
    def __init__(self) -> None:
        self._data: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        self._data[key] = value  # TTL not enforced in memory store

    def incr(self, key: str, amount: int = 1) -> int:
        current = int(self._data.get(key, 0))
        current += amount
        self._data[key] = str(current)
        return current


# ── CostTracker ───────────────────────────────────────────────────────────────

class CostTracker:
    """Track token costs per session and per day.

    Usage::

        tracker = CostTracker()  # in-memory by default

        # With Redis:
        tracker = CostTracker(redis_url="redis://localhost:6379/0")

        # Record a request
        usage = tracker.record(
            user_id="user-123",
            session_id="sess-abc",
            model="gpt-4o",
            prompt_tokens=800,
            completion_tokens=300,
        )
        print(f"This call cost ${usage.cost_usd:.6f}")

        # Check limits (returns alert or None)
        allowed, alert = tracker.check("user-123", "sess-abc")

        # Or raise automatically
        tracker.check_and_raise("user-123", "sess-abc")

        # Reports
        report = tracker.session_report("user-123", "sess-abc")
        daily  = tracker.daily_report("user-123")
    """

    SESSION_TTL = 7 * 24 * 3600  # 7 days
    DAILY_TTL   = 2 * 24 * 3600  # 2 days

    def __init__(
        self,
        redis_url: Optional[str] = None,
        session_token_limit: int = 100_000,
        daily_token_limit: int = 500_000,
        warn_threshold_pct: float = 80.0,
        key_prefix: str = "llm-meter",
        on_unpriced: Optional[Callable[["UnpricedEvent"], None]] = None,
    ) -> None:
        """
        Args:
            redis_url:            Redis connection URL. If None or Redis is unavailable,
                                  falls back to in-memory storage.
            session_token_limit:  Max tokens allowed per session.
            daily_token_limit:    Max tokens allowed per user per day.
            warn_threshold_pct:   Percentage (0-100) at which a threshold alert is issued.
            key_prefix:           Prefix for all storage keys.
            on_unpriced:          Optional callback invoked with an
                                  :class:`UnpricedEvent` whenever cache tokens are
                                  recorded for a model with no cache price. Use it
                                  to log / alert so unpriced usage is never missed.
        """
        self.session_limit = session_token_limit
        self.daily_limit = daily_token_limit
        self.warn_pct = warn_threshold_pct
        self._prefix = key_prefix
        self._on_unpriced = on_unpriced
        self._store = self._make_store(redis_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        user_id: str,
        session_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> TokenUsage:
        """Record token usage for a single LLM request.

        ``cache_read_tokens`` / ``cache_write_tokens`` are prompt-cache token
        counts reported separately from the base prompt/completion counts. If
        the model has no cache price they are bucketed as *unpriced* (surfaced
        in every report and via the ``on_unpriced`` callback) rather than
        silently costed at $0.

        Returns:
            :class:`TokenUsage` with cost and token counts.
        """
        usage = TokenUsage.create(
            model, prompt_tokens, completion_tokens,
            cache_read_tokens, cache_write_tokens,
        )
        total = usage.total_tokens

        # Update session record
        sess_key = self._sess_key(user_id, session_id)
        sess = self._load_sess(sess_key)
        sess["total_tokens"] += total
        sess["total_cost"] += usage.cost_usd
        sess["requests"] += 1
        sess["models"][model] = sess["models"].get(model, 0) + total
        sess["cache_read_tokens"] += cache_read_tokens
        sess["cache_write_tokens"] += cache_write_tokens
        sess["cache_read_cost"] += usage.cache_read_cost
        sess["cache_write_cost"] += usage.cache_write_cost
        if usage.unpriced_tokens:
            sess["unpriced_tokens"] += usage.unpriced_tokens
            sess["unpriced_models"][model] = (
                sess["unpriced_models"].get(model, 0) + usage.unpriced_tokens
            )
        self._store.set(sess_key, json.dumps(sess), ttl=self.SESSION_TTL)

        # Increment daily counters
        day = self._day_key(user_id)
        self._store.incr(f"{day}:tokens", total)
        self._store.incr(f"{day}:requests", 1)
        if cache_read_tokens:
            self._store.incr(f"{day}:cache_read_tokens", cache_read_tokens)
        if cache_write_tokens:
            self._store.incr(f"{day}:cache_write_tokens", cache_write_tokens)
        if usage.unpriced_tokens:
            self._store.incr(f"{day}:unpriced_tokens", usage.unpriced_tokens)

        logger.debug(
            "llm-meter: user=%s session=%s model=%s tokens=%d cost=$%.6f "
            "cache_read=%d cache_write=%d unpriced=%d",
            user_id, session_id, model, total, usage.cost_usd,
            cache_read_tokens, cache_write_tokens, usage.unpriced_tokens,
        )

        if usage.unpriced_tokens:
            logger.warning(
                "llm-meter: %d unpriced cache tokens for model %r "
                "(no cache price defined) — metered cost excludes them",
                usage.unpriced_tokens, model,
            )
            self._emit_unpriced(UnpricedEvent(
                model=model,
                cache_read_tokens=usage.unpriced_cache_read_tokens,
                cache_write_tokens=usage.unpriced_cache_write_tokens,
                total_unpriced_tokens=usage.unpriced_tokens,
                user_id=user_id,
                session_id=session_id,
            ))
        return usage

    def _emit_unpriced(self, event: "UnpricedEvent") -> None:
        if self._on_unpriced is None:
            return
        try:
            self._on_unpriced(event)
        except Exception as e:  # a bad callback must never break metering
            logger.error("llm-meter: on_unpriced callback raised: %s", e)

    def check(
        self, user_id: str, session_id: str
    ) -> Tuple[bool, Optional[CostAlert]]:
        """Check whether the user is within limits.

        Returns:
            ``(allowed, alert)`` — alert is non-None for both limit_exceeded and threshold.
        """
        sess = self._load_sess(self._sess_key(user_id, session_id))
        sess_tokens = sess["total_tokens"]

        if sess_tokens >= self.session_limit:
            return False, CostAlert(
                alert_type="limit_exceeded",
                message=f"Session token limit reached ({sess_tokens}/{self.session_limit})",
                current_usage=sess_tokens,
                limit=self.session_limit,
                percentage=100.0,
            )

        day = self._day_key(user_id)
        daily_tokens = int(self._store.get(f"{day}:tokens") or 0)

        if daily_tokens >= self.daily_limit:
            return False, CostAlert(
                alert_type="limit_exceeded",
                message=f"Daily token limit reached ({daily_tokens}/{self.daily_limit})",
                current_usage=daily_tokens,
                limit=self.daily_limit,
                percentage=100.0,
            )

        # Threshold warnings
        sess_pct = (sess_tokens / self.session_limit) * 100
        if sess_pct >= self.warn_pct:
            return True, CostAlert(
                alert_type="threshold",
                message=f"Session usage at {sess_pct:.1f}%",
                current_usage=sess_tokens,
                limit=self.session_limit,
                percentage=sess_pct,
            )

        daily_pct = (daily_tokens / self.daily_limit) * 100
        if daily_pct >= self.warn_pct:
            return True, CostAlert(
                alert_type="threshold",
                message=f"Daily usage at {daily_pct:.1f}%",
                current_usage=daily_tokens,
                limit=self.daily_limit,
                percentage=daily_pct,
            )

        return True, None

    def check_and_raise(self, user_id: str, session_id: str) -> None:
        """Like :meth:`check` but raises :exc:`LimitExceeded` when not allowed."""
        allowed, alert = self.check(user_id, session_id)
        if not allowed and alert:
            raise LimitExceeded(alert)

    def session_report(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Return a usage report for a session."""
        sess = self._load_sess(self._sess_key(user_id, session_id))
        tokens = sess["total_tokens"]
        return {
            "session_id": session_id,
            "user_id": user_id,
            "total_tokens": tokens,
            "total_cost_usd": round(sess["total_cost"], 8),
            "requests": sess["requests"],
            "models": sess["models"],
            "cache_read_tokens": sess["cache_read_tokens"],
            "cache_write_tokens": sess["cache_write_tokens"],
            "cache_read_cost_usd": round(sess["cache_read_cost"], 8),
            "cache_write_cost_usd": round(sess["cache_write_cost"], 8),
            "unpriced_tokens": sess["unpriced_tokens"],
            "unpriced_models": sess["unpriced_models"],
            "limit": self.session_limit,
            "percentage_used": round((tokens / self.session_limit) * 100, 2),
        }

    def daily_report(self, user_id: str) -> Dict[str, Any]:
        """Return a daily usage report for a user."""
        day = self._day_key(user_id)
        tokens = int(self._store.get(f"{day}:tokens") or 0)
        requests = int(self._store.get(f"{day}:requests") or 0)
        return {
            "user_id": user_id,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_tokens": tokens,
            "requests": requests,
            "cache_read_tokens": int(self._store.get(f"{day}:cache_read_tokens") or 0),
            "cache_write_tokens": int(self._store.get(f"{day}:cache_write_tokens") or 0),
            "unpriced_tokens": int(self._store.get(f"{day}:unpriced_tokens") or 0),
            "limit": self.daily_limit,
            "percentage_used": round((tokens / self.daily_limit) * 100, 2),
        }

    @staticmethod
    def reconcile(
        actual_billed_usd: float,
        metered_usd: float,
        drift_threshold: float = 0.25,
    ) -> "DriftReport":
        """Compare metered spend against a real provider invoice for a period.

        This is the safety net for the unpriced-token failure mode: even a
        perfectly bucketed meter is worth double-checking against the invoice.

        Args:
            actual_billed_usd:  The real amount the provider billed.
            metered_usd:        The total this meter reported for the same period.
            drift_threshold:    Fractional drift (relative to ``metered_usd``)
                                above which ``over_threshold`` is True. Default
                                0.25 = flag when the invoice is >25% off the meter.

        Returns:
            :class:`DriftReport`.
        """
        drift_usd = actual_billed_usd - metered_usd
        if metered_usd > 0:
            drift_ratio = drift_usd / metered_usd
        else:
            drift_ratio = float("inf") if actual_billed_usd > 0 else 0.0
        return DriftReport(
            metered_usd=round(metered_usd, 8),
            actual_billed_usd=round(actual_billed_usd, 8),
            drift_usd=round(drift_usd, 8),
            drift_ratio=drift_ratio,
            threshold=drift_threshold,
            over_threshold=abs(drift_ratio) > drift_threshold,
        )

    def reset_session(self, user_id: str, session_id: str) -> None:
        """Clear cost data for a session (e.g. when session is deleted)."""
        key = self._sess_key(user_id, session_id)
        self._store.set(key, json.dumps(self._empty_sess()), ttl=self.SESSION_TTL)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_store(self, redis_url: Optional[str]) -> _Store:
        if redis_url and _REDIS_AVAILABLE:
            try:
                store = _RedisStore(redis_url)
                store._r.ping()
                logger.info("llm-meter: using Redis backend (%s)", redis_url)
                return store
            except Exception as e:
                logger.warning("llm-meter: Redis unavailable (%s), falling back to in-memory", e)
        return _MemoryStore()

    def _sess_key(self, user_id: str, session_id: str) -> str:
        return f"{self._prefix}:sess:{user_id}:{session_id}"

    def _day_key(self, user_id: str) -> str:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return f"{self._prefix}:day:{user_id}:{date}"

    def _empty_sess(self) -> Dict[str, Any]:
        return {
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests": 0,
            "models": {},
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_read_cost": 0.0,
            "cache_write_cost": 0.0,
            "unpriced_tokens": 0,
            "unpriced_models": {},
        }

    def _load_sess(self, key: str) -> Dict[str, Any]:
        sess = self._empty_sess()
        raw = self._store.get(key)
        if raw:
            try:
                # Merge onto defaults so records written by older versions
                # (missing the cache keys) load without KeyErrors.
                sess.update(json.loads(raw))
            except json.JSONDecodeError:
                pass
        return sess
