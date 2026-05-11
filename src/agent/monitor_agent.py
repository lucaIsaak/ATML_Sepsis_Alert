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

import threading
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
from src.safety.guardrails import AuditLogger, InputGuard, NarrativeGuard


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

        # Safety guardrails (Layer 1, 2, 3)
        self.input_guard     = InputGuard.from_artifact(self.artifact)
        self.narrative_guard = NarrativeGuard()
        self.audit_logger    = AuditLogger(log_path="logs/audit.jsonl")

        # SHAP explainer (built lazily on first use)
        self._explainer      = None
        self._explainer_lock = threading.Lock()

        # Per-patient memory
        self._memory: dict[str, PatientMemory] = {}

        # Global alert log (all tiers)
        self.alert_log: list[AlertRecord] = []

        # Thresholds
        self.threshold_nurse    = cfg["agent"]["risk_threshold"]       # 0.4
        self.threshold_doctor   = 0.6
        self.threshold_critical = 0.8
        self.min_rescore_gap_h  = 1.0   # don't re-alert within 1h unless deteriorating

    @classmethod
    def from_artifact(cls, artifact: dict, cfg: dict) -> "PatientMonitorAgent":
        """
        Construct the agent from an already-loaded model artifact.

        Used by the FastAPI lifespan to avoid double-loading the model from disk.
        The narrative client (Ollama) is initialised but its availability is
        checked lazily — the agent works fine without Ollama running.
        """
        instance = cls.__new__(cls)
        instance.cfg          = cfg
        instance.artifact     = artifact
        instance.narrative    = OllamaClient(cfg)
        instance.registry     = BufferRegistry(
            window_hours=cfg["cohort"]["lookback_window_hours"]
        )
        instance.input_guard     = InputGuard.from_artifact(artifact)
        instance.narrative_guard = NarrativeGuard()
        instance.audit_logger    = AuditLogger(log_path="logs/audit.jsonl")
        instance._explainer      = None
        instance._explainer_lock = threading.Lock()
        instance._memory         = {}
        instance.alert_log       = []
        instance.threshold_nurse    = cfg["agent"]["risk_threshold"]
        instance.threshold_doctor   = 0.6
        instance.threshold_critical = 0.8
        instance.min_rescore_gap_h  = 1.0
        return instance

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def push_observation(self, stay_id: str, obs: Observation) -> None:
        """Push a new observation into the patient buffer."""
        self.registry.push(stay_id, obs)

    def run_cycle(self, timestamp: datetime | None = None) -> list[AlertRecord]:
        """
        Run one monitoring cycle over all active patients.

        Called periodically (every 60 minutes in production).
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

    def process_from_dataframe(
        self,
        features_df: "pd.DataFrame",
        timestamp: "datetime | None" = None,
    ) -> list[AlertRecord]:
        """
        Run the full ReAct monitoring loop over a pre-computed features DataFrame.

        This is the primary integration point for the FastAPI backend, which
        loads features from features.parquet rather than from live FHIR streams.
        Each row is treated as the current state of one ICU patient.

        Unlike run_cycle() (which reads from BufferRegistry), this method
        accepts an already-aggregated feature DataFrame directly — matching
        the format produced by src.data.features.extract_features().

        Side effects:
            - Writes to logs/audit.jsonl for every alert dispatched.
            - Updates self._memory (per-patient risk history + suppression).
            - Appends to self.alert_log.

        Returns list of AlertRecord for patients that triggered an alert this cycle.
        """
        if timestamp is None:
            timestamp = datetime.now()

        new_alerts: list[AlertRecord] = []
        for _, row in features_df.iterrows():
            if "stay_id" not in row.index:
                raise ValueError("features_df must contain a 'stay_id' column")
            stay_id = str(int(row["stay_id"]))
            try:
                alert = self._process_patient(stay_id, row.to_dict(), timestamp)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[MonitorAgent] Error processing stay {stay_id}: {exc}")
                alert = None
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def update_artifact(self, new_artifact: dict) -> None:
        """
        Hot-swap the model artifact after retraining.

        Updates the model, feature columns, and rebuilds InputGuard.
        SHAP explainer is reset so it is rebuilt on the next request.
        """
        self.artifact     = new_artifact
        self.input_guard  = InputGuard.from_artifact(new_artifact)
        self._explainer   = None   # force rebuild with new model

    def process_streaming_batch(
        self,
        timestamp: datetime,
        observations: list,
    ) -> list[AlertRecord]:
        """
        Process a batch of streaming observations.

        Accepts output from MIMICStreamSimulator or a live FHIR webhook.
        """
        # Route each observation to the right patient buffer using typed push
        # push() auto-routes by item name: vitals → STREAMING, labs → BATCH
        for obs in observations:
            stay_id = getattr(obs, "stay_id", None)
            if stay_id:
                self.registry.push(stay_id, obs)

        return self.run_cycle(timestamp)

    # ---------------------------------------------------------------- #
    # ReAct core                                                         #
    # ---------------------------------------------------------------- #

    def _process_patient(  # pylint: disable=too-many-locals
        self, stay_id: str, features: dict, timestamp: datetime
    ) -> Optional[AlertRecord]:
        """
        ReAct loop for a single patient.

        Returns an AlertRecord if an alert should be dispatched.
        """
        mem = self._get_memory(stay_id)

        # === OBSERVE ===
        # Layer 1: check inputs before running the model
        ood = self.input_guard.check(features)
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
        from src.explainability.shap_explainer import format_for_narrative  # pylint: disable=import-outside-toplevel
        explanation = self._tool_explain(
            prediction["feature_vector"],
            prediction["feature_names"],
            risk_score,
            stay_id,
        )

        nurse_narrative = None
        doctor_narrative = None
        nurse_nar_result = None
        patient_context = self._build_context(stay_id)

        if tier >= EscalationTier.NURSE:
            raw_nurse = self._tool_nurse_alert(explanation, patient_context)
            # Layer 2: validate narrative, replace if unsafe
            nurse_nar_result = self.narrative_guard.validate(
                raw_nurse, format_for_narrative(explanation)
            )
            nurse_narrative = nurse_nar_result.text

        if tier >= EscalationTier.DOCTOR:
            raw_doctor = self._tool_doctor_summary(explanation, patient_context)
            doctor_result = self.narrative_guard.validate(
                raw_doctor, format_for_narrative(explanation)
            )
            doctor_narrative = doctor_result.text
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

        # Layer 3: audit log every dispatched alert
        self.audit_logger.log_alert(
            stay_id=stay_id,
            risk_score=risk_score,
            tier=tier.name,
            top_features=explanation.top_features,
            ood_result=ood,
            narrative_result=nurse_nar_result or self.narrative_guard.validate("", ""),
            timestamp=timestamp,
        )

        # === REMEMBER ===
        mem.alert_history.append(alert)
        if len(mem.alert_history) > 48:
            mem.alert_history = mem.alert_history[-48:]
        self.alert_log.append(alert)
        if len(self.alert_log) > 1000:
            self.alert_log = self.alert_log[-1000:]

        # Suppress re-alerting for 2h unless critical
        if tier < EscalationTier.CRITICAL:
            mem.suppress(hours=2.0, now=timestamp)

        ood_flag = f" [{ood.confidence_flag}]" if ood.confidence_flag != "NORMAL" else ""
        print(
            f"[{timestamp.strftime('%H:%M')}] "
            f"Stay {stay_id} | {tier.name} | score={risk_score:.3f} | "
            f"trend={mem.trend:+.2f}{ood_flag}"
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
        - Rate of deterioration (near-miss rule + rapid-deterioration override)
        - Time since last alert
        - Whether physician has been notified recently
        """
        # Rapid deterioration overrides score thresholds for patients already at nurse tier+
        if mem.is_deteriorating_rapidly and risk_score >= self.threshold_nurse:
            return (
                EscalationTier.CRITICAL
                if risk_score >= self.threshold_doctor
                else EscalationTier.DOCTOR
            )

        # Near-miss rule: patient is below threshold but deteriorating rapidly.
        # Catches patients whose risk is rising fast before it crosses 0.40 —
        # a patient at 0.35 trending +0.20 in 3 steps will cross the threshold
        # within one more cycle; alert the nurse now, not after the fact.
        _near_miss_low = 0.30
        if (
            _near_miss_low <= risk_score < self.threshold_nurse
            and mem.is_deteriorating_rapidly
        ):
            return EscalationTier.NURSE

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

        Production integration point: replace the print below with a call to
        the hospital's pager API or EHR alert banner (e.g. Epic FHIR Task resource).
        The FHIR adapter in src/integrations/fhir_adapter.py provides the
        wire format; the specific endpoint is configured per deployment.
        """
        # TODO (production): POST to hospital pager / EHR critical alert API here
        print(
            f"[CRITICAL] Stay {stay_id} | score={risk_score:.3f} | "
            f"trend={mem.trend:+.2f} | "
            f"IMMEDIATE INTERVENTION REQUIRED — physician acknowledgement required"
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
        """Build and cache SHAP explainer (lazy, thread-safe initialisation)."""
        with self._explainer_lock:
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
