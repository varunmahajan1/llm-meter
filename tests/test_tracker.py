"""Tests for CostTracker."""

import pytest
from llm_meter import CostTracker, LimitExceeded


@pytest.fixture
def tracker():
    return CostTracker(session_token_limit=10_000, daily_token_limit=50_000)


class TestRecord:
    def test_records_usage(self, tracker):
        usage = tracker.record("u1", "s1", "gpt-4o", 500, 200)
        assert usage.total_tokens == 700
        assert usage.cost_usd > 0
        assert usage.model == "gpt-4o"

    def test_accumulates_across_calls(self, tracker):
        tracker.record("u1", "s2", "gpt-4o", 1000, 500)
        tracker.record("u1", "s2", "gpt-4o", 1000, 500)
        report = tracker.session_report("u1", "s2")
        assert report["total_tokens"] == 3000
        assert report["requests"] == 2

    def test_tracks_model_breakdown(self, tracker):
        tracker.record("u1", "s3", "gpt-4o", 100, 50)
        tracker.record("u1", "s3", "gpt-4o-mini", 200, 100)
        report = tracker.session_report("u1", "s3")
        assert "gpt-4o" in report["models"]
        assert "gpt-4o-mini" in report["models"]


class TestLimits:
    def test_allows_under_limit(self, tracker):
        tracker.record("u2", "s4", "gpt-4o-mini", 100, 50)
        allowed, alert = tracker.check("u2", "s4")
        assert allowed
        assert alert is None

    def test_blocks_over_session_limit(self):
        tracker = CostTracker(session_token_limit=100, daily_token_limit=50_000)
        tracker.record("u3", "s5", "gpt-4o-mini", 80, 30)
        allowed, alert = tracker.check("u3", "s5")
        assert not allowed
        assert alert is not None
        assert alert.alert_type == "limit_exceeded"

    def test_check_and_raise(self):
        tracker = CostTracker(session_token_limit=100, daily_token_limit=50_000)
        tracker.record("u4", "s6", "gpt-4o-mini", 80, 30)
        with pytest.raises(LimitExceeded) as exc_info:
            tracker.check_and_raise("u4", "s6")
        assert exc_info.value.alert.alert_type == "limit_exceeded"

    def test_warn_at_threshold(self):
        tracker = CostTracker(
            session_token_limit=10_000,
            daily_token_limit=50_000,
            warn_threshold_pct=50.0,
        )
        tracker.record("u5", "s7", "gpt-4o-mini", 4000, 2000)  # 6000 tokens = 60%
        allowed, alert = tracker.check("u5", "s7")
        assert allowed  # not blocked
        assert alert is not None
        assert alert.alert_type == "threshold"


class TestReports:
    def test_session_report_shape(self, tracker):
        tracker.record("u6", "s8", "gpt-4o", 500, 200)
        report = tracker.session_report("u6", "s8")
        assert "total_tokens" in report
        assert "total_cost_usd" in report
        assert "requests" in report
        assert "models" in report
        assert "limit" in report
        assert "percentage_used" in report

    def test_daily_report_shape(self, tracker):
        tracker.record("u7", "s9", "gpt-4o", 100, 50)
        report = tracker.daily_report("u7")
        assert "total_tokens" in report
        assert "requests" in report
        assert "date" in report

    def test_reset_session(self, tracker):
        tracker.record("u8", "s10", "gpt-4o", 500, 200)
        tracker.reset_session("u8", "s10")
        report = tracker.session_report("u8", "s10")
        assert report["total_tokens"] == 0


class TestPricingModels:
    def test_unknown_model_uses_default(self, tracker):
        usage = tracker.record("u9", "s11", "unknown-model-xyz", 1000, 500)
        assert usage.cost_usd > 0

    def test_cost_is_positive(self, tracker):
        for model in ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", "kimi-k2.5"]:
            usage = tracker.record("u10", f"s-{model}", model, 1000, 500)
            assert usage.cost_usd > 0, f"Expected positive cost for {model}"
