"""
PatientMonitorAgent — the core agentic loop of SepsisAlert.

The agent monitors active ICU patients, runs inference when new data
arrives, generates SHAP explanations and LLM narratives for high-risk
patients, and dispatches alerts to the dashboard.

Tool pattern:
    Each "tool" is a method the agent can call. The agent decides
    WHICH tool to call based on patient state, mirroring a real
    clinical decision workflow.

Usage:
    agent = PatientMonitorAgent()
    alerts = agent.run_cycle(patient_data_df)
"""

import time
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import shap

from src.model.predict import load_model, predict_patient
from src.explainability.shap_explainer import (
    build_explainer, explain_patient, SHAPExplanation
)
from src.narrative.ollama_client import OllamaClient


@dataclass
class Alert:
    """A dispatched sepsis alert for one patient."""
    stay_id: str
    timestamp: datetime
    risk_score: float
    risk_label: str
    top_features: list[dict]
    narrative: str
    acknowledged: bool = False
    patient_context: str = ""


class PatientMonitorAgent:
    """
    Agentic monitoring loop for active ICU patients.

    The agent maintains a state dict of seen patients and only
    re-alerts if risk materially changes (avoids alert fatigue).
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            with open("config.yaml") as f:
                cfg = yaml.safe_load(f)

        self.cfg = cfg
        self.threshold = cfg["agent"]["risk_threshold"]
        self.artifact = load_model(cfg)
        self.narrative_client = OllamaClient(cfg)

        # Build SHAP explainer using a small background dataset
        background = self._load_background()
        self.explainer = build_explainer(
            self.artifact["model"],
            background[self.artifact["feature_cols"]],
        )

        # State: {stay_id: last_risk_score}
        self._patient_state: dict[str, float] = {}
        self._alert_log: list[Alert] = []

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def run_cycle(self, patient_df: pd.DataFrame) -> list[Alert]:
        """
        Run one monitoring cycle over all active patients.

        Args:
            patient_df: DataFrame with one row per active ICU patient.
                        Must have columns matching the model feature set.

        Returns:
            List of new Alert objects generated this cycle.
        """
        new_alerts = []

        for _, row in patient_df.iterrows():
            stay_id = str(row.get("stay_id", "unknown"))
            features = row.to_dict()

            # Tool 1: Run sepsis model
            prediction = self._tool_run_model(features)
            risk_score = prediction["risk_score"]

            # Decide whether to alert
            if not self._should_alert(stay_id, risk_score):
                continue

            # Tool 2: Explain with SHAP
            explanation = self._tool_explain(
                prediction["feature_vector"],
                prediction["feature_names"],
                risk_score,
                stay_id,
            )

            # Tool 3: Generate narrative
            patient_context = self._build_context(row)
            narrative = self._tool_generate_narrative(explanation, patient_context)

            # Tool 4: Dispatch alert
            alert = self._tool_dispatch_alert(explanation, narrative, patient_context)
            new_alerts.append(alert)

            # Update state
            self._patient_state[stay_id] = risk_score

        return new_alerts

    def get_alert_log(self) -> list[Alert]:
        return self._alert_log

    def acknowledge_alert(self, stay_id: str) -> None:
        for alert in self._alert_log:
            if alert.stay_id == stay_id and not alert.acknowledged:
                alert.acknowledged = True
                break

    # ------------------------------------------------------------------ #
    # Tools (internal)                                                      #
    # ------------------------------------------------------------------ #

    def _tool_run_model(self, features: dict) -> dict:
        """Tool: Run LightGBM inference. Returns risk score + feature vector."""
        return predict_patient(features, self.artifact)

    def _tool_explain(
        self,
        feature_vector: np.ndarray,
        feature_names: list[str],
        risk_score: float,
        stay_id: str,
    ) -> SHAPExplanation:
        """Tool: Compute SHAP explanation for this prediction."""
        return explain_patient(
            self.explainer,
            feature_vector,
            feature_names,
            risk_score,
            stay_id,
        )

    def _tool_generate_narrative(
        self,
        explanation: SHAPExplanation,
        patient_context: str,
    ) -> str:
        """Tool: Call local LLM to generate clinical narrative."""
        return self.narrative_client.generate_alert(explanation, patient_context)

    def _tool_dispatch_alert(
        self,
        explanation: SHAPExplanation,
        narrative: str,
        patient_context: str,
    ) -> Alert:
        """Tool: Create and log the alert."""
        label = (
            "HIGH" if explanation.risk_score >= 0.6
            else "MODERATE" if explanation.risk_score >= 0.4
            else "LOW"
        )
        alert = Alert(
            stay_id=explanation.stay_id,
            timestamp=datetime.now(),
            risk_score=explanation.risk_score,
            risk_label=label,
            top_features=explanation.top_features,
            narrative=narrative,
            patient_context=patient_context,
        )
        self._alert_log.append(alert)
        print(f"[ALERT] Stay {alert.stay_id} | {label} ({alert.risk_score:.2f}) | {alert.timestamp}")
        return alert

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    def _should_alert(self, stay_id: str, risk_score: float) -> bool:
        """
        Alert if:
        - Risk exceeds threshold AND
        - Patient not yet alerted, OR risk increased by >0.15 since last alert
        """
        if risk_score < self.threshold:
            return False
        prev = self._patient_state.get(stay_id)
        if prev is None:
            return True
        return (risk_score - prev) >= 0.15

    def _build_context(self, row: pd.Series) -> str:
        """Build a non-identifying patient context string."""
        parts = []
        age = row.get("age")
        gender = row.get("gender_male")
        if age and not np.isnan(float(age)):
            parts.append(f"{int(age)}yo")
        if gender is not None:
            parts.append("M" if gender == 1 else "F")
        unit = row.get("first_careunit", "")
        if unit:
            parts.append(str(unit))
        return " | ".join(parts)

    def _load_background(self, n_samples: int = 200) -> pd.DataFrame:
        """Load a background sample for SHAP explainer."""
        path = Path(self.cfg["data"]["processed_path"]) / "features.parquet"
        if not path.exists():
            # Return empty DataFrame if no data yet (for testing)
            return pd.DataFrame(columns=self.artifact["feature_cols"])
        df = pd.read_parquet(path)
        return df.sample(min(n_samples, len(df)), random_state=42)
