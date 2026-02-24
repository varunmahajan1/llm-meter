"""
Per-session and per-day token cost tracking.

Supports Redis (production) or in-memory (dev/test) backends.
The backend is selected transparently — no code changes needed.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .pricing import calculate_cost

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """Record of a single LLM request."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: str

    @classmethod
    def create(cls, model: str, prompt_tokens: int, completion_tokens: int) -> "TokenUsage":
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
        return cls(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


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
    ) -> None:
        """
        Args:
            redis_url:            Redis connection URL. If None or Redis is unavailable,
                                  falls back to in-memory storage.
            session_token_limit:  Max tokens allowed per session.
            daily_token_limit:    Max tokens allowed per user per day.
            warn_threshold_pct:   Percentage (0-100) at which a threshold alert is issued.
            key_prefix:           Prefix for all storage keys.
        """
        self.session_limit = session_token_limit
        self.daily_limit = daily_token_limit
        self.warn_pct = warn_threshold_pct
        self._prefix = key_prefix
        self._store = self._make_store(redis_url)

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        user_id: str,
        session_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TokenUsage:
        """Record token usage for a single LLM request.

        Returns:
            :class:`TokenUsage` with cost and token counts.
        """
        usage = TokenUsage.create(model, prompt_tokens, completion_tokens)
        total = usage.total_tokens

        # Update session record
        sess_key = self._sess_key(user_id, session_id)
        sess = self._load_sess(sess_key)
        sess["total_tokens"] += total
        sess["total_cost"] += usage.cost_usd
        sess["requests"] += 1
        sess["models"][model] = sess["models"].get(model, 0) + total
        self._store.set(sess_key, json.dumps(sess), ttl=self.SESSION_TTL)

        # Increment daily counters
        day = self._day_key(user_id)
        self._store.incr(f"{day}:tokens", total)
        self._store.incr(f"{day}:requests", 1)

        logger.debug(
            "llm-meter: user=%s session=%s model=%s tokens=%d cost=$%.6f",
            user_id, session_id, model, total, usage.cost_usd,
        )
        return usage

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
            "limit": self.daily_limit,
            "percentage_used": round((tokens / self.daily_limit) * 100, 2),
        }

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
        return {"total_tokens": 0, "total_cost": 0.0, "requests": 0, "models": {}}

    def _load_sess(self, key: str) -> Dict[str, Any]:
        raw = self._store.get(key)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return self._empty_sess()
