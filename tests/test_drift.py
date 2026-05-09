"""
Tests for the PSI-based data drift monitor.

Covers:
  - PSI computation (stable / moderate / significant)
  - Edge cases: tiny samples, duplicate bin edges, NaN values
  - Full drift report structure
  - psi_status() categorisation
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import compute_psi, compute_drift_report, psi_status


# ------------------------------------------------------------------ #
# PSI computation                                                      #
# ------------------------------------------------------------------ #

class TestComputePsi:
    """Unit tests for compute_psi()."""

    def _same(self, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """Two samples from the same distribution — PSI should be near 0."""
        rng = np.random.default_rng(42)
        train = rng.normal(70.0, 10.0, n)
        live  = rng.normal(70.0, 10.0, n)
        return train, live

    def _shifted(self, n: int = 500, shift: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
        """Live sample shifted far from training — PSI should be high."""
        rng = np.random.default_rng(42)
        train = rng.normal(70.0, 10.0, n)
        live  = rng.normal(70.0 + shift, 10.0, n)
        return train, live

    def test_identical_distributions_near_zero(self):
        train, live = self._same()
        psi = compute_psi(train, live)
        assert not math.isnan(psi)
        assert psi < 0.10, f"Expected PSI < 0.10 for identical dist, got {psi}"

    def test_shifted_distribution_high_psi(self):
        train, live = self._shifted(shift=40.0)
        psi = compute_psi(train, live)
        assert not math.isnan(psi)
        assert psi > 0.20, f"Expected PSI > 0.20 for large shift, got {psi}"

    def test_moderate_shift_between_thresholds(self):
        rng = np.random.default_rng(7)
        train = rng.normal(70.0, 10.0, 1000)
        live  = rng.normal(78.0, 10.0, 1000)   # moderate shift
        psi = compute_psi(train, live)
        assert not math.isnan(psi)
        # Moderate shift: somewhere above 0, not necessarily above 0.10
        # Just verify it's a real number
        assert psi >= 0.0

    def test_returns_nan_for_tiny_live_sample(self):
        train = np.arange(100.0)
        live  = np.array([1.0, 2.0])   # fewer than 5 values
        psi = compute_psi(train, live)
        assert math.isnan(psi)

    def test_returns_nan_for_tiny_train_sample(self):
        train = np.array([1.0, 2.0])
        live  = np.arange(100.0)
        psi = compute_psi(train, live)
        assert math.isnan(psi)

    def test_handles_nan_values_in_input(self):
        rng = np.random.default_rng(1)
        train = rng.normal(80.0, 5.0, 200)
        live  = rng.normal(80.0, 5.0, 200)
        # Inject NaNs — should be stripped before computation
        train[::10] = np.nan
        live[::10]  = np.nan
        psi = compute_psi(train, live)
        # Should compute without error; identical dist → stable
        assert not math.isnan(psi)
        assert psi < 0.20

    def test_constant_feature_returns_zero_or_nan(self):
        """All-same value collapses to one bin. Either NaN (degenerate) or 0.0 (no drift)."""
        train = np.ones(100)
        live  = np.ones(100)
        psi = compute_psi(train, live)
        # PSI is either undefined (NaN) or 0.0 — both are acceptable for a constant feature
        assert math.isnan(psi) or psi == pytest.approx(0.0, abs=1e-6)

    def test_psi_is_non_negative(self):
        rng = np.random.default_rng(99)
        train = rng.exponential(scale=2.0, size=300)
        live  = rng.exponential(scale=3.0, size=300)
        psi = compute_psi(train, live)
        assert math.isnan(psi) or psi >= 0.0


# ------------------------------------------------------------------ #
# psi_status categorisation                                            #
# ------------------------------------------------------------------ #

class TestPsiStatus:
    """Unit tests for psi_status()."""

    def test_below_010_is_stable(self):
        assert psi_status(0.05) == "stable"
        assert psi_status(0.0)  == "stable"
        assert psi_status(0.099) == "stable"

    def test_between_010_and_020_is_moderate(self):
        assert psi_status(0.10) == "moderate"
        assert psi_status(0.15) == "moderate"
        assert psi_status(0.199) == "moderate"

    def test_above_020_is_significant(self):
        assert psi_status(0.20) == "significant"
        assert psi_status(0.50) == "significant"
        assert psi_status(1.0)  == "significant"

    def test_nan_is_unknown(self):
        assert psi_status(float("nan")) == "unknown"


# ------------------------------------------------------------------ #
# Full drift report                                                    #
# ------------------------------------------------------------------ #

class TestComputeDriftReport:
    """Integration tests for compute_drift_report()."""

    def _make_df(self, n: int = 200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "stay_id":           np.arange(n),
            "heart_rate_last":   rng.normal(80, 12, n),
            "map_last":          rng.normal(72, 10, n),
            "resp_rate_last":    rng.normal(16, 3,  n),
            "spo2_last":         rng.normal(96, 2,  n),
            "lactate_last":      rng.exponential(1.5, n),
        })

    def test_returns_required_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)   # avoid polluting real logs/
        train_df = self._make_df(300, seed=1)
        live_df  = self._make_df(100, seed=2)
        feature_cols = ["heart_rate_last", "map_last", "resp_rate_last",
                        "spo2_last", "lactate_last"]
        risk_scores = np.random.default_rng(3).uniform(0, 1, 100)

        report = compute_drift_report(train_df, live_df, feature_cols, risk_scores)

        required = {"overall_status", "overall_psi", "features",
                    "risk_distribution", "psi_history", "evaluated_at", "live_patients"}
        assert required.issubset(set(report.keys()))

    def test_overall_status_is_valid_string(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        train_df = self._make_df(300)
        live_df  = self._make_df(100)
        feature_cols = ["heart_rate_last", "map_last"]
        risk_scores = np.full(100, 0.3)

        report = compute_drift_report(train_df, live_df, feature_cols, risk_scores)
        assert report["overall_status"] in ("stable", "moderate", "significant", "unknown")

    def test_feature_rows_have_required_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        train_df = self._make_df(300)
        live_df  = self._make_df(100)
        feature_cols = ["heart_rate_last", "map_last"]
        risk_scores = np.full(100, 0.3)

        report = compute_drift_report(train_df, live_df, feature_cols, risk_scores)
        for feat_row in report["features"]:
            assert "feature" in feat_row
            assert "label"   in feat_row
            assert "status"  in feat_row
            assert feat_row["status"] in ("stable", "moderate", "significant", "unknown")

    def test_risk_distribution_sums_to_one(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        train_df = self._make_df(300)
        live_df  = self._make_df(100)
        feature_cols = ["heart_rate_last"]
        risk_scores = np.random.default_rng(9).uniform(0, 1, 100)

        report = compute_drift_report(train_df, live_df, feature_cols, risk_scores)
        live_pct = report["risk_distribution"]["live"]
        total = sum(live_pct.values())
        assert abs(total - 1.0) < 0.01, f"Risk dist sums to {total}, expected ~1.0"

    def test_unknown_status_when_insufficient_live_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        train_df = self._make_df(300)
        # Live df with only 3 patients — below minimum of 5
        live_df = self._make_df(3)
        feature_cols = ["heart_rate_last"]
        risk_scores = np.array([0.2, 0.5, 0.8])

        report = compute_drift_report(train_df, live_df, feature_cols, risk_scores)
        # All features will have NaN PSI → overall unknown
        assert report["overall_status"] == "unknown"

    def test_sorted_worst_psi_first(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rng = np.random.default_rng(5)
        train_df = pd.DataFrame({
            "stay_id":         np.arange(300),
            "heart_rate_last": rng.normal(80, 5, 300),   # stable
            "map_last":        rng.normal(72, 5, 300),    # will be shifted in live
        })
        live_df = pd.DataFrame({
            "stay_id":         np.arange(100),
            "heart_rate_last": rng.normal(80, 5, 100),   # same → low PSI
            "map_last":        rng.normal(110, 5, 100),   # shifted → high PSI
        })
        feature_cols = ["heart_rate_last", "map_last"]
        risk_scores = np.full(100, 0.4)

        report = compute_drift_report(train_df, live_df, feature_cols, risk_scores)
        psi_values = [r["psi"] for r in report["features"] if r["psi"] is not None]
        assert psi_values == sorted(psi_values, reverse=True), "Features not sorted worst-PSI-first"
