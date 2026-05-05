"""Tests for model loading and prediction."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestPredictPatient:
    def test_predict_returns_valid_risk_score(self):
        from src.model.predict import predict_patient
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        artifact = {"model": mock_model, "feature_cols": ["feat_a", "feat_b"]}

        result = predict_patient({"feat_a": 1.0, "feat_b": 2.0}, artifact)

        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["risk_label"] in ("HIGH", "MODERATE", "LOW")
        assert len(result["feature_vector"]) == 2

    def test_high_risk_label_above_06(self):
        from src.model.predict import predict_patient
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.05, 0.95]])
        artifact = {"model": mock_model, "feature_cols": ["feat_a"]}

        result = predict_patient({"feat_a": 5.0}, artifact)
        assert result["risk_label"] == "HIGH"

    def test_low_risk_label_below_04(self):
        from src.model.predict import predict_patient
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])
        artifact = {"model": mock_model, "feature_cols": ["feat_a"]}

        result = predict_patient({"feat_a": 1.0}, artifact)
        assert result["risk_label"] == "LOW"

    def test_missing_features_filled_with_nan(self):
        from src.model.predict import predict_patient
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        artifact = {"model": mock_model, "feature_cols": ["feat_a", "feat_b", "feat_c"]}

        # Only provide feat_a — b and c should be NaN
        result = predict_patient({"feat_a": 1.0}, artifact)
        assert np.isnan(result["feature_vector"][1])
        assert np.isnan(result["feature_vector"][2])


class TestPredictBatch:
    def test_batch_adds_risk_columns(self):
        from src.model.predict import predict_batch
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        artifact = {"model": mock_model, "feature_cols": ["feat_a"]}

        df = pd.DataFrame({"stay_id": [1, 2], "feat_a": [1.0, 2.0]})
        result = predict_batch(df, artifact)

        assert "risk_score" in result.columns
        assert "risk_label" in result.columns
        assert len(result) == 2

    def test_batch_risk_scores_between_0_and_1(self):
        from src.model.predict import predict_batch
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.random.random((10, 2))
        artifact = {"model": mock_model, "feature_cols": ["feat_a"]}

        df = pd.DataFrame({"feat_a": np.random.random(10)})
        result = predict_batch(df, artifact)

        assert (result["risk_score"] >= 0).all()
        assert (result["risk_score"] <= 1).all()
