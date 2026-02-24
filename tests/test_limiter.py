"""Tests for RateLimiter."""

import time
import pytest
from llm_meter import RateLimiter


@pytest.fixture
def limiter():
    return RateLimiter()


class TestRateLimiter:
    def test_allows_under_limit(self, limiter):
        for _ in range(5):
            allowed, info = limiter.check("u1", max_requests=10, window_seconds=60)
            assert allowed
            assert info["remaining"] > 0

    def test_blocks_at_limit(self, limiter):
        for i in range(5):
            limiter.check(f"u2-unique-{i}", max_requests=5, window_seconds=60)

        ident = "u2-limit-test"
        for _ in range(5):
            limiter.check(ident, max_requests=5, window_seconds=60)
        allowed, info = limiter.check(ident, max_requests=5, window_seconds=60)
        assert not allowed
        assert info["retry_after"] > 0

    def test_different_identifiers_independent(self, limiter):
        for _ in range(4):
            limiter.check("ua", max_requests=5, window_seconds=60)

        allowed, _ = limiter.check("ub", max_requests=5, window_seconds=60)
        assert allowed

    def test_info_dict_shape(self, limiter):
        allowed, info = limiter.check("u3", max_requests=10, window_seconds=30)
        assert "allowed" in info
        assert "current_requests" in info
        assert "max_requests" in info
        assert "window_seconds" in info
        assert "remaining" in info
        assert "retry_after" in info
        assert info["max_requests"] == 10
        assert info["window_seconds"] == 30

    def test_is_allowed_bool(self, limiter):
        assert limiter.is_allowed("u4", max_requests=10, window_seconds=60)

    def test_sliding_window_clears(self, limiter):
        ident = "u5-sliding"
        for _ in range(3):
            limiter.check(ident, max_requests=3, window_seconds=1)

        allowed, _ = limiter.check(ident, max_requests=3, window_seconds=1)
        assert not allowed

        time.sleep(1.1)
        allowed, _ = limiter.check(ident, max_requests=3, window_seconds=1)
        assert allowed, "Window should have slid past old requests"
