"""
Sliding-window rate limiter.

Uses Redis sorted sets in production for accurate distributed rate limiting.
Falls back to an in-memory list for dev/test — no code changes needed.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RateLimiter:
    """Sliding-window rate limiter.

    Usage::

        limiter = RateLimiter()  # in-memory

        # With Redis:
        limiter = RateLimiter(redis_url="redis://localhost:6379/0")

        # Check
        allowed, info = limiter.check("user-123", max_requests=60, window_seconds=60)
        if not allowed:
            raise Exception(f"Rate limit. Retry in {info['retry_after']}s")

        # Convenience
        if not limiter.is_allowed("user-123"):
            return 429

    ``info`` dict keys:
    - ``allowed``          bool
    - ``current_requests`` int — requests in current window
    - ``max_requests``     int
    - ``window_seconds``   int
    - ``remaining``        int — requests left before limit
    - ``retry_after``      int — seconds to wait (0 if allowed)
    """

    KEY_PREFIX = "llm-meter:rl"

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis: Any = None
        self._memory: Dict[str, list] = {}

        if redis_url and _REDIS_AVAILABLE:
            try:
                self._redis = _redis_lib.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("llm-meter RateLimiter: using Redis backend")
            except Exception as e:
                logger.warning("llm-meter RateLimiter: Redis unavailable (%s), using in-memory", e)
                self._redis = None

    def check(
        self,
        identifier: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check whether ``identifier`` is within the rate limit.

        Args:
            identifier:      Any string key (user ID, IP address, API key, etc.)
            max_requests:    Maximum allowed requests in ``window_seconds``.
            window_seconds:  Sliding window duration in seconds.

        Returns:
            ``(allowed, info)`` — see class docstring for ``info`` keys.
        """
        key = f"{self.KEY_PREFIX}:{identifier}:{max_requests}:{window_seconds}"
        now = time.time()
        window_start = now - window_seconds

        if self._redis:
            current_count = self._redis_check(key, now, window_start, window_seconds)
        else:
            current_count = self._memory_check(key, now, window_start)

        allowed = current_count < max_requests
        return allowed, {
            "allowed": allowed,
            "current_requests": current_count,
            "max_requests": max_requests,
            "window_seconds": window_seconds,
            "remaining": max(0, max_requests - current_count - 1),
            "retry_after": window_seconds if not allowed else 0,
        }

    def is_allowed(
        self,
        identifier: str,
        max_requests: int = 60,
        window_seconds: int = 60,
    ) -> bool:
        """Convenience wrapper — returns bool only."""
        allowed, _ = self.check(identifier, max_requests, window_seconds)
        return allowed

    # ── Backends ──────────────────────────────────────────────────────────────

    def _redis_check(
        self, key: str, now: float, window_start: float, window_seconds: int
    ) -> int:
        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)   # remove expired entries
            pipe.zcard(key)                                # count remaining
            pipe.zadd(key, {str(now): now})               # add current request
            pipe.expire(key, window_seconds)
            results = pipe.execute()
            return results[1]  # count before adding current request
        except Exception as e:
            logger.error("RateLimiter Redis error: %s", e)
            return 0

    def _memory_check(self, key: str, now: float, window_start: float) -> int:
        if key not in self._memory:
            self._memory[key] = []
        # Evict expired entries
        self._memory[key] = [ts for ts in self._memory[key] if ts > window_start]
        count = len(self._memory[key])
        self._memory[key].append(now)
        return count
