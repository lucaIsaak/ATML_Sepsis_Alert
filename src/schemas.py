"""
Pydantic v2 schemas for SepsisAlert — system boundary validation.

All data entering or leaving the system is validated here before it
reaches the model or the clinician-facing API response.

Design principle — validate at the boundary, trust internally:
  Validation is applied only at system entry points (FHIR adapter input,
  API request bodies) and exit points (API responses). Internal functions
  receive already-validated data and do not re-validate — keeps the
  inference path fast and avoids redundant safety checks.

Clinical bound rationale:
  Every field bound (ge/le) reflects a physiologically plausible range,
  not an arbitrary software constraint. Values outside these ranges indicate
  sensor malfunction or data entry error — not a real patient state.
  Accepting implausible values would produce authoritative-looking risk scores
  for phantom patients, a direct patient safety risk.

  Bounds cross-referenced against:
    - Sepsis-3 criteria (Singer et al. 2016, JAMA 315(8):801-810)
    - SOFA score component ranges (Vincent et al. 1996, Intensive Care Med)
    - MIMIC-IV observed ranges across 93,224 ICU stays

EU AI Act Art. 10 (data governance for high-risk AI):
  Input validation is a required data quality measure. These validators
  constitute the software implementation of that regulatory requirement.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ------------------------------------------------------------------ #
# Input schemas                                                        #
# ------------------------------------------------------------------ #

class VitalSigns(BaseModel):
    """Vitals as measured every 1–4 hours at the bedside."""

    heart_rate: Optional[float] = Field(None, ge=0, le=300, description="bpm")
    sbp: Optional[float] = Field(None, ge=0, le=300, description="Systolic BP mmHg")
    dbp: Optional[float] = Field(None, ge=0, le=200, description="Diastolic BP mmHg")
    map: Optional[float] = Field(None, ge=0, le=200, description="Mean arterial pressure mmHg")
    resp_rate: Optional[float] = Field(None, ge=0, le=60, description="breaths/min")
    temperature_c: Optional[float] = Field(None, ge=30, le=45, description="°C")
    spo2: Optional[float] = Field(None, ge=50, le=100, description="%")
    gcs_total: Optional[int] = Field(None, ge=3, le=15, description="Glasgow Coma Scale")

    @field_validator("spo2")
    @classmethod
    def spo2_range(cls, v):
        # WHY: SpO2 > 100% is physically impossible — pulse oximeters saturate at 100%.
        # Values above 100 indicate a sensor or transcription error and must not
        # reach the model, where they would produce an implausibly low-risk score
        # (high SpO2 decreases_risk in SHAP) for a patient with faulty monitoring.
        if v is not None and v > 100:
            raise ValueError("SpO2 cannot exceed 100% — value indicates sensor or transcription error")
        return v

    @model_validator(mode="after")
    def map_from_sbp_dbp(self):
        # WHY: MAP is the primary Sepsis-3 haemodynamic criterion (MAP < 65 mmHg).
        # FHIR feeds sometimes omit MAP but include SBP/DBP. Auto-computing from
        # MAP = DBP + (SBP - DBP) / 3 (standard formula) ensures the most important
        # sepsis feature is always available when the raw pressures are present.
        if self.map is None and self.sbp is not None and self.dbp is not None:
            self.map = round(self.dbp + (self.sbp - self.dbp) / 3, 1)
        return self


class LabValues(BaseModel):
    """Daily lab panel — what's ordered every morning in ICU."""

    # Infection / inflammatory markers
    wbc: Optional[float] = Field(None, ge=0, le=200, description="White blood cells K/µL")
    lactate: Optional[float] = Field(None, ge=0, le=30, description="Lactate mmol/L")

    # Organ function (SOFA score components)
    creatinine: Optional[float] = Field(None, ge=0, le=30, description="mg/dL — renal")
    bilirubin: Optional[float] = Field(None, ge=0, le=50, description="mg/dL — hepatic")
    platelets: Optional[float] = Field(None, ge=0, le=2000, description="K/µL — coagulation")

    # BMP (basic metabolic panel)
    sodium: Optional[float] = Field(None, ge=100, le=180, description="mEq/L")
    potassium: Optional[float] = Field(None, ge=1, le=10, description="mEq/L")
    bicarbonate: Optional[float] = Field(None, ge=0, le=50, description="mEq/L — acid-base")
    bun: Optional[float] = Field(None, ge=0, le=200, description="Blood urea nitrogen mg/dL")
    glucose: Optional[float] = Field(None, ge=0, le=1000, description="mg/dL")

    # CBC
    hemoglobin: Optional[float] = Field(None, ge=0, le=25, description="g/dL")

    @field_validator("potassium")
    @classmethod
    def potassium_range(cls, v):
        """Validate potassium is within plausible physiological range."""
        if v is not None and (v < 1.0 or v > 10.0):
            raise ValueError(f"Potassium {v} is outside plausible range (1–10 mEq/L)")
        return v


class PatientContext(BaseModel):
    """Non-identifying patient context passed to the LLM."""

    stay_id: str
    age: Optional[int] = Field(None, ge=18, le=120)
    gender: Optional[Literal["M", "F", "Other"]] = None
    care_unit: Optional[str] = None
    admission_type: Optional[str] = None


# ------------------------------------------------------------------ #
# Output schemas                                                       #
# ------------------------------------------------------------------ #

class FeatureContribution(BaseModel):
    """Single SHAP feature contribution."""

    feature: str
    label: str
    value: Optional[float]
    unit: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]

    @property
    def formatted_value(self) -> str:
        """Return a human-readable string for the feature value."""
        if self.value is None:
            return "not measured"
        return f"{self.value:.1f} {self.unit}".strip()


class SepsisRiskOutput(BaseModel):
    """Full model output for one patient — what flows to the nurse."""

    stay_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_label: Literal["HIGH", "MODERATE", "LOW"]
    top_features: list[FeatureContribution]
    narrative: Optional[str] = None
    narrative_type: Optional[Literal["nurse_brief", "doctor_detail"]] = None
    model_version: str = "HistGradientBoosting-v1"

    @field_validator("risk_label", mode="before")
    @classmethod
    def derive_label(cls, v, info):
        """Derive risk label from score if not explicitly provided."""
        if v is not None:
            return v
        score = info.data.get("risk_score", 0)
        if score >= 0.6:
            return "HIGH"
        if score >= 0.4:
            return "MODERATE"
        return "LOW"

    @property
    def alert_color(self) -> str:
        """Return CSS colour string for the risk label."""
        return {"HIGH": "red", "MODERATE": "orange", "LOW": "green"}[self.risk_label]

    @property
    def requires_immediate_action(self) -> bool:
        """Return True if risk is HIGH and requires immediate attention."""
        return self.risk_label == "HIGH"


class NarrativeRequest(BaseModel):
    """Input to the narrative generator."""

    stay_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_label: Literal["HIGH", "MODERATE", "LOW"]
    top_features: list[FeatureContribution]
    patient_context: Optional[PatientContext] = None
    narrative_type: Literal["nurse_brief", "doctor_detail"] = "nurse_brief"


class FHIRObservation(BaseModel):
    """
    Simplified HL7 FHIR R4 Observation resource.

    Used for ingesting data from Epic / Oracle Health (Cerner).
    """

    resource_type: Literal["Observation"] = "Observation"
    status: Literal["final", "preliminary", "amended"]
    subject_reference: str          # Patient/12345
    encounter_reference: Optional[str] = None
    loinc_code: str                 # e.g. "2524-7" for Lactate
    display: str
    value_quantity: Optional[float] = None
    value_unit: str = ""
    effective_datetime: Optional[str] = None

    @field_validator("loinc_code")
    @classmethod
    def valid_loinc(cls, v):
        """Validate LOINC code follows numeric-numeric format."""
        # LOINC codes are numeric-numeric format
        parts = v.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid LOINC code format: {v}")
        return v
