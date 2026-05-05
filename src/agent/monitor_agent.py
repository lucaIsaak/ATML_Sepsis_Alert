"""
PatientMonitorAgent — ReAct-pattern sepsis monitoring agent.

ReAct = Reason + Act loop.
The agent doesn't just run a pipeline — it reasons about each patient's
state, consults memory, and decides what action to take.

Escalation tiers:
  Tier 0 — No alert (risk < threshold)
  Tier 1 — Nurse alert (risk >= 0.4, SBAR narrative)
  Tier 2 — Doctor alert (risk >= 0.6, detailed summary + suggested workup)
  Tier 3 — Critical escalation (risk >= 0.8 OR rapid deterioration)

Per-patient memory:
  - Alert history (timestamp, score, tier, acknowledged)
  - Risk trajectory (is patient improving or deteriorating?)
  - Time since last physician notification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.patient_buffer import BufferRegistry, Observation
from src.explainability.shap_explainer import SHAPExplanation, build_explainer, explain_patient
from src.model.predict import load_model, predict_patient
from src.narrative.ollama_client import OllamaClient


# ------------------------------------------------------------------ #
# Data structures                                                      #
# ------------------------------------------------------------------ #

class EscalationTier(IntEnum):
    """Alert escalation levels for ICU staff notification."""

    NONE     = 0   # risk < 0.4
    NURSE    = 1   # risk 0.4-0.6
    DOCTOR   = 2   # risk 0.6-0.8
    CRITICAL = 3   # risk > 0.8 OR rapid deterioration


@dataclass
class AlertRecord:  # pylint: disable=too-many-instance-attributes
    """A single alert dispatched for a patient at a point in time."""

    stay_id: str
    timestamp: datetime
    risk_score: float
    tier: EscalationTier
    nurse_narrative: Optional[str]
    doctor_narrative: Optional[str]
    top_features: list[dict]
    acknowledged_nurse: bool = False
    acknowledged_doctor: bool = False
    escalated_from: Optional[EscalationTier] = None


@dataclass
class PatientMemory:
    """Per-patient state the agent maintains across time steps."""

    stay_id: str
    risk_history: list[tuple[datetime, float]] = field(default_factory=list)
    alert_history: list[AlertRecord] = field(default_factory=list)
    last_doctor_notification: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None   # avoid alert fatigue

    def record_score(self, timestamp: datetime, score: float) -> None:
        """Append a risk score observation and cap history at 48 entries."""
        self.risk_history.append((timestamp, score))
        if len(self.risk_history) > 48:   # keep last 48 time steps
            self.risk_history = self.risk_history[-48:]

    @property
    def trend(self) -> float:
        """Risk trend over last 3 observations. Positive = worsening."""
        if len(self.risk_history) < 3:
            return 0.0
        recent = [s for _, s in self.risk_history[-3:]]
        return recent[-1] - recent[0]

    @property
    def last_risk_score(self) -> Optional[float]:
        """Return the most recent risk score, or None if no history."""
        if not self.risk_history:
            return None
        return self.risk_history[-1][1]

    @property
    def is_deteriorating_rapidly(self) -> bool:
        """True if risk increased >0.2 in last 3 time steps."""
        return self.trend >= 0.20

    def is_suppressed(self, now: datetime) -> bool:
        """Return True if alert suppression is still active."""
        if self.suppressed_until is None:
            return False
        return now < self.suppressed_until

    def suppress(self, hours: float, now: datetime) -> None:
        """Suppress alerts for the given number of hours from now."""
        self.suppressed_until = now + timedelta(hours=hours)


# ------------------------------------------------------------------ #
# Agent                                                                #
# ------------------------------------------------------------------ #

class PatientMonitorAgent:  # pylint: disable=too-many-instance-attributes
    """
    ReAct-pattern agent for ICU sepsis monitoring.

    The agent loop per patient:
      OBSERVE  -> get latest features from buffer
      THINK    -> reason about risk, trend, memory, escalation tier
      ACT      -> call the right tools for the right tier
      REMEMBER -> update patient memory
    """

    def __init__(self, cfg: dict | None = None):
        """Initialise agent — loads model, narrative client, and registry."""
        if cfg is None:
            with open("config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        self.cfg = cfg

        # Core components
        self.artifact     = load_model(cfg)
        self.narrative    = OllamaClient(cfg)
        self.registry     = BufferRegistry(window_hours=cfg["cohort"]["lookback_window_hours"])

        # SHAP explainer (built lazily on first use)
        self._explainer   = None

        # Per-patient memory
        self._memory: dict[str, PatientMemory] = {}

        # Global alert log (all tiers)
        self.alert_log: list[AlertRecord] = []

        # Thresholds
        self.threshold_nurse    = cfg["agent"]["risk_threshold"]       # 0.4
        self.threshold_doctor   = 0.6
        self.threshold_critical = 0.8
        self.min_rescore_gap_h  = 1.0   # don't re-alert within 1h unless deteriorating

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def push_observation(self, stay_id: str, obs: Observation) -> None:
        """Push a new observation into the patient buffer."""
        self.registry.push(stay_id, obs)

    def run_cycle(self, timestamp: datetime | None = None) -> list[AlertRecord]:
        """
        Run one monitoring cycle over all active patients.

        Called periodically (e.g. every 15-60 minutes in production).
        Returns new alerts generated this cycle.
        """
        if timestamp is None:
            timestamp = datetime.now()

        new_alerts = []
        features_df = self.registry.get_all_features()

        if features_df.empty:
            return new_alerts

        for _, row in features_df.iterrows():
            stay_id = str(row["stay_id"])
            alert = self._process_patient(stay_id, row.to_dict(), timestamp)
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def process_streaming_batch(
        self,
        timestamp: datetime,
        observations: list,
    ) -> list[AlertRecord]:
        """
        Process a batch of streaming observations.

        Accepts output from MIMICStreamSimulator or a live FHIR webhook.
        """
        # Route each observation to the right patient buffer
        for obs in observations:
            stay_id = getattr(obs, "stay_id", None)
            if stay_id:
                self.registry.push(stay_id, obs)

        return self.run_cycle(timestamp)

    def acknowledge(self, stay_id: str, role: str = "nurse") -> None:
        """Mark the latest alert for a patient as acknowledged."""
        mem = self._get_memory(stay_id)
        if mem.alert_history:
            alert = mem.alert_history[-1]
            if role == "nurse":
                alert.acknowledged_nurse = True
            else:
                alert.acknowledged_doctor = True
                mem.last_doctor_notification = datetime.now()

    # ---------------------------------------------------------------- #
    # ReAct core                                                         #
    # ---------------------------------------------------------------- #

    def _process_patient(
        self, stay_id: str, features: dict, timestamp: datetime
    ) -> Optional[AlertRecord]:
        """
        ReAct loop for a single patient.

        Returns an AlertRecord if an alert should be dispatched.
        """
        mem = self._get_memory(stay_id)

        # === OBSERVE ===
        prediction = self._tool_run_model(features)
        risk_score = prediction["risk_score"]

        # === THINK ===
        mem.record_score(timestamp, risk_score)
        tier = self._reason_escalation_tier(risk_score, mem, timestamp)

        # No action needed
        if tier == EscalationTier.NONE:
            return None

        # Suppressed to avoid fatigue
        if mem.is_suppressed(timestamp) and not mem.is_deteriorating_rapidly:
            return None

        # === ACT ===
        explanation = self._tool_explain(
            prediction["feature_vector"],
            prediction["feature_names"],
            risk_score,
            stay_id,
        )

        nurse_narrative = None
        doctor_narrative = None
        patient_context = self._build_context(stay_id)

        if tier >= EscalationTier.NURSE:
            nurse_narrative = self._tool_nurse_alert(explanation, patient_context)

        if tier >= EscalationTier.DOCTOR:
            doctor_narrative = self._tool_doctor_summary(explanation, patient_context)
            mem.last_doctor_notification = timestamp

        if tier == EscalationTier.CRITICAL:
            self._tool_critical_escalation(stay_id, risk_score, mem)

        alert = AlertRecord(
            stay_id=stay_id,
            timestamp=timestamp,
            risk_score=risk_score,
            tier=tier,
            nurse_narrative=nurse_narrative,
            doctor_narrative=doctor_narrative,
            top_features=explanation.top_features,
            escalated_from=(
                EscalationTier(int(tier) - 1) if tier > EscalationTier.NURSE else None
            ),
        )

        # === REMEMBER ===
        mem.alert_history.append(alert)
        self.alert_log.append(alert)

        # Suppress re-alerting for 2h unless critical
        if tier < EscalationTier.CRITICAL:
            mem.suppress(hours=2.0, now=timestamp)

        print(
            f"[{timestamp.strftime('%H:%M')}] "
            f"Stay {stay_id} | {tier.name} | score={risk_score:.3f} | "
            f"trend={mem.trend:+.2f}"
        )

        return alert

    def _reason_escalation_tier(
        self,
        risk_score: float,
        mem: PatientMemory,
        timestamp: datetime,
    ) -> EscalationTier:
        """
        THINK step: decide escalation tier.

        Considers:
        - Current risk score
        - Rate of deterioration
        - Time since last alert
        - Whether physician has been notified recently
        """
        # Rapid deterioration overrides score thresholds
        if mem.is_deteriorating_rapidly and risk_score >= self.threshold_nurse:
            return (
                EscalationTier.CRITICAL
                if risk_score >= self.threshold_doctor
                else EscalationTier.DOCTOR
            )

        if risk_score >= self.threshold_critical:
            return EscalationTier.CRITICAL
        if risk_score >= self.threshold_doctor:
            # Escalate to doctor — but only if not recently notified
            if mem.last_doctor_notification:
                hours_since = (timestamp - mem.last_doctor_notification).total_seconds() / 3600
                if hours_since < 4.0:
                    return EscalationTier.NURSE   # downgrade — doc already knows
            return EscalationTier.DOCTOR
        if risk_score >= self.threshold_nurse:
            return EscalationTier.NURSE

        return EscalationTier.NONE

    # ---------------------------------------------------------------- #
    # Tools                                                              #
    # ---------------------------------------------------------------- #

    def _tool_run_model(self, features: dict) -> dict:
        """Run the sepsis model for one patient and return prediction dict."""
        return predict_patient(features, self.artifact)

    def _tool_explain(
        self,
        feature_vector: np.ndarray,
        feature_names: list[str],
        risk_score: float,
        stay_id: str,
    ) -> SHAPExplanation:
        """Compute SHAP explanation for a single patient."""
        explainer = self._get_explainer()
        return explain_patient(explainer, feature_vector, feature_names, risk_score, stay_id)

    def _tool_nurse_alert(self, explanation: SHAPExplanation, context: str) -> str:
        """Generate a nurse-facing SBAR narrative."""
        return self.narrative.generate_nurse_alert(explanation, context)

    def _tool_doctor_summary(self, explanation: SHAPExplanation, context: str) -> str:
        """Generate a physician-facing clinical summary."""
        return self.narrative.generate_doctor_summary(explanation, context)

    def _tool_critical_escalation(
        self, stay_id: str, risk_score: float, mem: PatientMemory
    ) -> None:
        """
        Critical tier: log for immediate response.

        In production: trigger pager / EHR alert banner.
        """
        print(
            f"[CRITICAL] Stay {stay_id} | score={risk_score:.3f} | "
            f"trend={mem.trend:+.2f} | "
            f"IMMEDIATE INTERVENTION REQUIRED"
        )

    # ---------------------------------------------------------------- #
    # Helpers                                                            #
    # ---------------------------------------------------------------- #

    def _get_memory(self, stay_id: str) -> PatientMemory:
        """Return or initialise per-patient memory."""
        if stay_id not in self._memory:
            self._memory[stay_id] = PatientMemory(stay_id=stay_id)
        return self._memory[stay_id]

    def _get_explainer(self):
        """Build and cache SHAP explainer (lazy initialisation)."""
        if self._explainer is None:
            path = Path(self.cfg["data"]["processed_path"]) / "features.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                background = df[self.artifact["feature_cols"]].dropna().sample(
                    min(100, len(df)), random_state=42
                )
            else:
                background = pd.DataFrame(columns=self.artifact["feature_cols"])
            self._explainer = build_explainer(self.artifact["model"], background)
        return self._explainer

    def _build_context(self, stay_id: str) -> str:
        """Build a one-line patient context string for the LLM prompt."""
        buf = self.registry.get_buffer(stay_id)
        if buf is None:
            return stay_id
        mem = self._memory.get(stay_id)
        trend_str = ""
        if mem and len(mem.risk_history) >= 2:
            trend_str = f" | trend {mem.trend:+.2f}"
        return (
            f"Stay {stay_id} | {buf.hours_of_data:.1f}h data"
            f" | {buf.n_observations} obs{trend_str}"
        )

    def summary(self) -> dict:
        """Summary statistics for dashboard display."""
        tiers = [a.tier for a in self.alert_log]
        return {
            "total_alerts": len(self.alert_log),
            "critical": tiers.count(EscalationTier.CRITICAL),
            "doctor": tiers.count(EscalationTier.DOCTOR),
            "nurse": tiers.count(EscalationTier.NURSE),
            "active_patients": len(self.registry.active_patients),
            "unacknowledged": sum(
                1 for a in self.alert_log if not a.acknowledged_nurse
            ),
        }
