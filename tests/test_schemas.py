"""Tests for pydantic input/output schemas."""
import pytest
from pydantic import ValidationError
from src.schemas import VitalSigns, LabValues, SepsisRiskOutput, FeatureContribution, FHIRObservation


class TestVitalSigns:
    def test_valid_vitals(self):
        v = VitalSigns(heart_rate=80, map=70, resp_rate=16, temperature_c=37.0, spo2=98)
        assert v.heart_rate == 80

    def test_map_auto_computed_from_sbp_dbp(self):
        v = VitalSigns(sbp=120, dbp=80)
        assert v.map == pytest.approx(93.3, abs=0.2)

    def test_spo2_above_100_rejected(self):
        with pytest.raises(ValidationError):
            VitalSigns(spo2=101)

    def test_heart_rate_above_300_rejected(self):
        with pytest.raises(ValidationError):
            VitalSigns(heart_rate=350)

    def test_all_none_is_valid(self):
        v = VitalSigns()
        assert v.heart_rate is None

    def test_gcs_bounds(self):
        with pytest.raises(ValidationError):
            VitalSigns(gcs_total=2)   # min is 3
        with pytest.raises(ValidationError):
            VitalSigns(gcs_total=16)  # max is 15


class TestLabValues:
    def test_valid_labs(self):
        labs = LabValues(lactate=2.5, wbc=12.0, creatinine=1.8, platelets=180)
        assert labs.lactate == 2.5

    def test_potassium_out_of_range(self):
        with pytest.raises(ValidationError):
            LabValues(potassium=0.5)

    def test_all_none_valid(self):
        labs = LabValues()
        assert labs.wbc is None


class TestSepsisRiskOutput:
    def test_high_risk_output(self):
        feat = FeatureContribution(
            feature="lactate_last", label="Lactate (last)",
            value=4.5, unit="mmol/L", shap_value=0.18,
            direction="increases_risk"
        )
        out = SepsisRiskOutput(
            stay_id="12345", risk_score=0.75, risk_label="HIGH",
            top_features=[feat]
        )
        assert out.requires_immediate_action is True
        assert out.alert_color == "red"

    def test_risk_score_bounds(self):
        with pytest.raises(ValidationError):
            SepsisRiskOutput(stay_id="x", risk_score=1.5, risk_label="HIGH", top_features=[])

    def test_low_risk(self):
        out = SepsisRiskOutput(
            stay_id="x", risk_score=0.2, risk_label="LOW", top_features=[]
        )
        assert out.requires_immediate_action is False


class TestFHIRObservation:
    def test_valid_loinc(self):
        obs = FHIRObservation(
            status="final",
            subject_reference="Patient/123",
            loinc_code="2524-7",
            display="Lactate",
            value_quantity=3.2,
            value_unit="mmol/L",
        )
        assert obs.loinc_code == "2524-7"

    def test_invalid_loinc_format(self):
        with pytest.raises(ValidationError):
            FHIRObservation(
                status="final",
                subject_reference="Patient/123",
                loinc_code="INVALID",
                display="Something",
            )
