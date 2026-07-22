"""Tests for cache-token accounting, unpriced detection, and billing drift."""

import pytest

from llm_meter import (
    CostTracker,
    UnpricedEvent,
    DriftReport,
    compute_cost,
    add_model_pricing,
    PRICING,
)


class TestCacheCostMath:
    def test_anthropic_cache_read_is_10pct_of_input(self):
        # claude-3-5-sonnet: input $0.003/1k -> cache read $0.0003/1k
        b = compute_cost("claude-3-5-sonnet", 0, 0, cache_read_tokens=1_000)
        assert b.cache_read_cost == pytest.approx(0.0003)
        assert b.cache_write_cost == 0.0
        assert b.total_cost == pytest.approx(0.0003)

    def test_anthropic_cache_write_is_125pct_of_input(self):
        # claude-3-5-sonnet: input $0.003/1k -> cache write $0.00375/1k
        b = compute_cost("claude-3-5-sonnet", 0, 0, cache_write_tokens=1_000)
        assert b.cache_write_cost == pytest.approx(0.00375)

    def test_openai_cache_read_is_half_input_and_write_free(self):
        # gpt-4o: input $0.005/1k -> cache read $0.0025/1k, cache write free
        b = compute_cost("gpt-4o", 0, 0, cache_read_tokens=1_000, cache_write_tokens=1_000)
        assert b.cache_read_cost == pytest.approx(0.0025)
        assert b.cache_write_cost == 0.0  # priced-and-free, not unpriced
        assert b.unpriced_tokens == 0

    def test_full_breakdown_sums_to_total(self):
        b = compute_cost(
            "claude-3-5-sonnet",
            prompt_tokens=1_000,
            completion_tokens=1_000,
            cache_read_tokens=1_000,
            cache_write_tokens=1_000,
        )
        expected = 0.003 + 0.015 + 0.0003 + 0.00375
        assert b.total_cost == pytest.approx(expected)

    def test_record_includes_cache_cost_in_total(self):
        tracker = CostTracker()
        usage = tracker.record(
            "u", "s", "claude-3-5-sonnet",
            prompt_tokens=1_000, completion_tokens=0,
            cache_write_tokens=10_000,
        )
        # base input 0.003 + cache write 0.0375
        assert usage.cost_usd == pytest.approx(0.003 + 0.0375)
        assert usage.cache_write_tokens == 10_000
        assert usage.total_tokens == 11_000  # cache tokens counted separately


class TestUnpricedDetection:
    def test_unpriced_when_model_has_no_cache_price(self):
        # gemini-1.5-pro has no cache pricing defined
        b = compute_cost("gemini-1.5-pro", 100, 50, cache_read_tokens=5_000)
        assert b.cache_read_cost == 0.0
        assert b.unpriced_cache_read_tokens == 5_000
        assert b.unpriced_tokens == 5_000

    def test_unpriced_split_by_kind(self):
        # openai gpt-4-turbo has neither cache_read nor cache_write
        b = compute_cost("gpt-4-turbo", 0, 0, cache_read_tokens=100, cache_write_tokens=200)
        assert b.unpriced_cache_read_tokens == 100
        assert b.unpriced_cache_write_tokens == 200
        assert b.unpriced_tokens == 300

    def test_unpriced_bucketed_in_reports(self):
        tracker = CostTracker()
        tracker.record("u", "s", "gemini-1.5-pro", 100, 50, cache_read_tokens=9_000)
        report = tracker.session_report("u", "s")
        assert report["unpriced_tokens"] == 9_000
        assert report["unpriced_models"]["gemini-1.5-pro"] == 9_000

        daily = tracker.daily_report("u")
        assert daily["unpriced_tokens"] == 9_000

    def test_on_unpriced_callback_fires(self):
        events = []
        tracker = CostTracker(on_unpriced=events.append)
        tracker.record("u", "s", "gemini-1.5-pro", 0, 0, cache_write_tokens=1_234)

        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, UnpricedEvent)
        assert ev.model == "gemini-1.5-pro"
        assert ev.cache_write_tokens == 1_234
        assert ev.cache_read_tokens == 0
        assert ev.total_unpriced_tokens == 1_234
        assert ev.user_id == "u"
        assert ev.session_id == "s"

    def test_callback_not_fired_for_priced_cache(self):
        events = []
        tracker = CostTracker(on_unpriced=events.append)
        tracker.record("u", "s", "claude-3-5-sonnet", 0, 0, cache_read_tokens=1_000)
        assert events == []

    def test_bad_callback_does_not_break_recording(self):
        def boom(_ev):
            raise RuntimeError("callback failed")

        tracker = CostTracker(on_unpriced=boom)
        usage = tracker.record("u", "s", "gemini-1.5-pro", 0, 0, cache_read_tokens=500)
        assert usage.unpriced_tokens == 500  # recording still succeeded


class TestReconcile:
    def test_meter_underreports_by_order_of_magnitude(self):
        report = CostTracker.reconcile(actual_billed_usd=450.0, metered_usd=9.59)
        assert isinstance(report, DriftReport)
        assert report.drift_usd == pytest.approx(440.41)
        assert report.drift_ratio == pytest.approx(440.41 / 9.59)
        assert report.over_threshold is True

    def test_within_threshold_not_flagged(self):
        report = CostTracker.reconcile(
            actual_billed_usd=102.0, metered_usd=100.0, drift_threshold=0.25
        )
        assert report.drift_usd == pytest.approx(2.0)
        assert report.drift_ratio == pytest.approx(0.02)
        assert report.over_threshold is False

    def test_zero_metered_with_real_bill_is_infinite_drift(self):
        report = CostTracker.reconcile(actual_billed_usd=50.0, metered_usd=0.0)
        assert report.drift_ratio == float("inf")
        assert report.over_threshold is True

    def test_zero_both_no_drift(self):
        report = CostTracker.reconcile(actual_billed_usd=0.0, metered_usd=0.0)
        assert report.drift_ratio == 0.0
        assert report.over_threshold is False


class TestBackwardCompat:
    def test_record_without_cache_args(self):
        tracker = CostTracker()
        usage = tracker.record("u", "s", "gpt-4o", 500, 200)
        assert usage.total_tokens == 700
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0
        assert usage.unpriced_tokens == 0
        assert usage.cost_usd == pytest.approx(0.005 * 0.5 + 0.015 * 0.2)

    def test_session_report_has_cache_fields_defaulting_to_zero(self):
        tracker = CostTracker()
        tracker.record("u", "s", "gpt-4o", 500, 200)
        report = tracker.session_report("u", "s")
        assert report["cache_read_tokens"] == 0
        assert report["cache_write_tokens"] == 0
        assert report["cache_read_cost_usd"] == 0.0
        assert report["cache_write_cost_usd"] == 0.0
        assert report["unpriced_tokens"] == 0
        assert report["unpriced_models"] == {}

    def test_add_model_pricing_backward_compatible(self):
        add_model_pricing("bc-model-no-cache", input_per_1k=0.002, output_per_1k=0.008)
        assert "cache_read" not in PRICING["bc-model-no-cache"]
        b = compute_cost("bc-model-no-cache", 1_000, 1_000, cache_read_tokens=1_000)
        assert b.unpriced_cache_read_tokens == 1_000  # cache left unpriced

    def test_add_model_pricing_with_cache(self):
        add_model_pricing(
            "bc-model-cache", input_per_1k=0.002, output_per_1k=0.008,
            cache_read_per_1k=0.0002, cache_write_per_1k=0.0025,
        )
        b = compute_cost("bc-model-cache", 0, 0, cache_read_tokens=1_000, cache_write_tokens=1_000)
        assert b.cache_read_cost == pytest.approx(0.0002)
        assert b.cache_write_cost == pytest.approx(0.0025)
        assert b.unpriced_tokens == 0
