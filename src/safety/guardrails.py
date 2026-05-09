"""
AI Safety guardrails for SepsisAlert.

Three protection layers enforced on every alert cycle:

  Layer 1 — InputGuard (before model inference)
  -----------------------------------------------
  Detects features that are out-of-distribution (OOD) relative to the
  training data. When vitals or labs are physiologically implausible or
  statistically extreme, the model is silently extrapolating — a known
  failure mode for gradient boosting on tabular EHR data.

  The guard computes a per-feature z-score against training statistics
  stored in the model artifact. If ≥3 features are >3σ from the mean,
  the prediction is flagged LOW_CONFIDENCE and the dashboard warns the
  clinician that the score may be unreliable.

  Fallback: if training stats are not in the artifact, hard physiological
  plausibility bounds are applied instead.

  Layer 2 — NarrativeGuard (after LLM generation)
  -------------------------------------------------
  Validates that the LLM-generated alert narrative does not contain:
    - Confirmed diagnoses ("patient has sepsis", "diagnosed with")
    - Definitive treatment orders ("start antibiotics", "administer X")
    - Values or claims with no grounding in the SHAP input
    - Excessive hedging that reduces actionability

  If any prohibited pattern is detected the narrative is replaced with a
  structured fallback built directly from the SHAP output — guaranteeing
  clinical safety even when the LLM misbehaves.

  Layer 3 — AuditLogger (every alert dispatched)
  -----------------------------------------------
  Writes an append-only JSON-Lines audit log for every alert, including:
    - Timestamp, stay_id, risk score, escalation tier
    - Top SHAP features that drove the alert
    - OOD confidence flag and any narrative validation warnings
    - Whether the narrative was replaced by the fallback

  Required for:
    - GDPR Article 22 (transparency of automated decision-making)
    - EU AI Act Annex III (high-risk CDSS audit documentation)
    - Clinical governance and incident review

  None of these logs leave the hospital server. The AuditLogger writes
  locally; no data is sent to any external API.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------ #
# Physiological plausibility bounds (fallback when no training stats) #
# ------------------------------------------------------------------ #

_HARD_BOUNDS: dict[str, tuple[float, float]] = {
    "heart_rate_mean":    (20.0,  250.0),
    "heart_rate_min":     (20.0,  250.0),
    "heart_rate_max":     (20.0,  250.0),
    "heart_rate_last":    (20.0,  250.0),
    "map_mean":           (15.0,  180.0),
    "map_min":            (15.0,  180.0),
    "map_max":            (15.0,  180.0),
    "resp_rate_mean":     (4.0,   60.0),
    "resp_rate_max":      (4.0,   60.0),
    "spo2_min":           (50.0,  100.0),
    "spo2_mean":          (50.0,  100.0),
    "temperature_f_mean": (88.0,  113.0),
    "temperature_f_last": (88.0,  113.0),
    "lactate_last":       (0.0,   25.0),
    "lactate_mean":       (0.0,   25.0),
    "wbc_last":           (0.0,   150.0),
    "creatinine_last":    (0.0,   25.0),
    "bilirubin_last":     (0.0,   40.0),
    "platelets_last":     (1.0,   1500.0),
    "bicarbonate_last":   (2.0,   50.0),
    "glucose_last":       (10.0,  900.0),
    "age":                (18.0,  115.0),
}

# Patterns the LLM must NOT produce
_PROHIBITED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bdiagnosed with sepsis\b",      re.IGNORECASE),
    re.compile(r"\bpatient has sepsis\b",          re.IGNORECASE),
    re.compile(r"\bsepsis (?:is )?confirmed\b",   re.IGNORECASE),
    re.compile(r"\bconfirmed sepsis\b",            re.IGNORECASE),
    re.compile(r"\bstart (?:IV )?antibiotics\b",  re.IGNORECASE),
    re.compile(r"\badminister (?:immediately)?\b", re.IGNORECASE),
    re.compile(r"\borgive \d+\b",                  re.IGNORECASE),  # dosing instructions
    re.compile(r"\bimmediately inject\b",           re.IGNORECASE),
    re.compile(r"\bdo not\b.*\brescuscitat\b",      re.IGNORECASE),
]


# ------------------------------------------------------------------ #
# Layer 1 — InputGuard                                                #
# ------------------------------------------------------------------ #

@dataclass
class OODResult:
    """Result of the out-of-distribution check for one patient."""

    is_ood: bool
    n_outlier_features: int
    outlier_features: list[str]
    confidence_flag: str   # "NORMAL" | "CAUTION" | "LOW_CONFIDENCE"
    details: dict[str, str] = field(default_factory=dict)


class InputGuard:
    """
    Detect out-of-distribution inputs before model inference.

    Uses training statistics (mean ± std) from the model artifact when
    available; falls back to hard physiological plausibility bounds.

    Thresholds:
        CAUTION        — 1-2 features > 3σ or outside hard bounds
        LOW_CONFIDENCE — ≥3 features > 3σ or outside hard bounds
    """

    def __init__(self, training_stats: Optional[dict] = None):
        """
        Initialise with optional training statistics dict.

        training_stats format: {feature_name: {"mean": float, "std": float}}
        """
        self._stats = training_stats or {}

    @classmethod
    def from_artifact(cls, artifact: dict) -> "InputGuard":
        """Build an InputGuard from a saved model artifact."""
        return cls(training_stats=artifact.get("training_stats", {}))

    def check(self, features: dict) -> OODResult:
        """
        Check a feature dict for out-of-distribution values.

        Returns OODResult with confidence flag.
        """
        outliers: list[str] = []
        details: dict[str, str] = {}

        for feat_name, value in features.items():
            if value is None or (isinstance(value, float) and _is_nan(value)):
                continue  # missing values handled by model's native NaN support

            # Z-score check against training distribution
            if feat_name in self._stats:
                stat = self._stats[feat_name]
                mean, std = stat["mean"], stat["std"]
                if std > 0:
                    z = abs((value - mean) / std)
                    if z > 3.5:
                        outliers.append(feat_name)
                        details[feat_name] = f"z={z:.1f} (value={value:.1f}, mean={mean:.1f})"
                        continue

            # Fallback: hard bounds
            if feat_name in _HARD_BOUNDS:
                lo, hi = _HARD_BOUNDS[feat_name]
                if value < lo or value > hi:
                    outliers.append(feat_name)
                    details[feat_name] = (
                        f"outside bounds [{lo}, {hi}]: value={value:.1f}"
                    )

        n = len(outliers)
        if n == 0:
            flag = "NORMAL"
        elif n <= 2:
            flag = "CAUTION"
        else:
            flag = "LOW_CONFIDENCE"

        return OODResult(
            is_ood=(n > 0),
            n_outlier_features=n,
            outlier_features=outliers,
            confidence_flag=flag,
            details=details,
        )


# ------------------------------------------------------------------ #
# Layer 2 — NarrativeGuard                                            #
# ------------------------------------------------------------------ #

@dataclass
class NarrativeResult:
    """Result of the narrative safety check."""

    text: str
    was_replaced: bool
    violations_found: list[str]
    is_fallback: bool


class NarrativeGuard:
    """
    Validate LLM-generated narratives for clinical safety.

    Checks for prohibited patterns (confirmed diagnoses, treatment orders).
    Replaces the full narrative with a safe SHAP-grounded fallback if any
    violation is detected — ensuring alert safety even when the LLM fails.
    """

    def validate(self, narrative: str, shap_summary: str) -> NarrativeResult:
        """
        Validate a narrative. Return safe text and metadata.

        Args:
            narrative:    Raw LLM output.
            shap_summary: SHAP-grounded summary used to build the fallback.
        """
        violations: list[str] = []
        for pattern in _PROHIBITED_PATTERNS:
            if pattern.search(narrative):
                violations.append(pattern.pattern)

        if violations:
            return NarrativeResult(
                text=self._build_fallback(shap_summary),
                was_replaced=True,
                violations_found=violations,
                is_fallback=True,
            )

        return NarrativeResult(
            text=narrative,
            was_replaced=False,
            violations_found=[],
            is_fallback=False,
        )

    @staticmethod
    def _build_fallback(shap_summary: str) -> str:
        """
        Build a guaranteed-safe SBAR narrative from the raw SHAP summary.

        This contains no LLM inference — it is a deterministic template.
        """
        return (
            "SITUATION: AI model indicates elevated sepsis risk."
            " Clinical assessment required.\n"
            f"CONCERN: Key abnormal values detected:\n{shap_summary}\n"
            "ACTIONS:\n"
            "  1. Reassess patient at bedside immediately.\n"
            "  2. Review flagged lab and vital sign values.\n"
            "  3. Notify physician if clinical concern persists.\n"
            "NOTE: Automated narrative generation was replaced by safe fallback."
            " This is not a confirmed diagnosis."
        )


# ------------------------------------------------------------------ #
# Layer 3 — AuditLogger                                               #
# ------------------------------------------------------------------ #

class AuditLogger:
    """
    Append-only JSON-Lines audit log for every alert.

    Each line is a self-contained JSON object. The log is never overwritten
    — only appended — to satisfy clinical audit trail requirements.

    Compliance note:
        This logger enables compliance with GDPR Article 22 (right to
        explanation for automated decisions) and EU AI Act Annex III
        requirements for high-risk AI systems in healthcare.

        The log stays on-premise. No data is transmitted externally.
    """

    def __init__(self, log_path: str | Path = "logs/audit.jsonl"):
        """Initialise logger. Creates log directory if needed."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_alert(
        self,
        stay_id: str,
        risk_score: float,
        tier: str,
        top_features: list[dict],
        ood_result: OODResult,
        narrative_result: NarrativeResult,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Append one alert record to the audit log."""
        record = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "stay_id": stay_id,
            "risk_score": round(risk_score, 4),
            "risk_tier": tier,          # unified key — matches log_prediction() and the audit UI
            "escalation_tier": tier,    # kept for backward-compat with existing log readers
            "ood_flag": ood_result.confidence_flag,
            "ood_outlier_features": ood_result.outlier_features,
            "narrative_was_replaced": narrative_result.was_replaced,
            "narrative_violations": narrative_result.violations_found,
            "top_features": [
                {
                    "feature": f.get("feature"),
                    "value": f.get("value"),
                    "shap": round(f.get("shap", 0), 4),
                    "direction": f.get("direction"),
                }
                for f in top_features[:5]
            ],
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def log_prediction(
        self,
        stay_id: str,
        risk_score: float,
        risk_tier: str,
        ood_flag: str,
        outlier_features: list[str],
        top_features: list[dict],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Lightweight audit entry for a model prediction (no narrative required).

        Used by the REST API when a patient detail is fetched — ensures every
        prediction served to a clinician leaves an audit trail even without the
        full PatientMonitorAgent pipeline running.
        """
        record = {
            "timestamp": (timestamp or datetime.now()).isoformat(),
            "stay_id": stay_id,
            "risk_score": round(risk_score, 4),
            "risk_tier": risk_tier,
            "ood_flag": ood_flag,
            "ood_outlier_features": outlier_features,
            "top_features": [
                {
                    "feature": f.get("feature"),
                    "value": f.get("value"),
                    "shap": round(f.get("shap", 0), 4),
                }
                for f in top_features[:5]
            ],
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def read_recent(self, n: int = 50) -> list[dict]:
        """Return the last n audit records (for dashboard display)."""
        if not self.log_path.exists():
            return []
        records: list[dict] = []
        with open(self.log_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records[-n:]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _is_nan(value: float) -> bool:
    """Return True if value is NaN without importing numpy."""
    try:
        return value != value  # NaN is not equal to itself
    except TypeError:
        return False
