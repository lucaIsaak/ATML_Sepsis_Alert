"""
NarrativeAgent — reasoning agent for clinical narrative generation.

Replaces the fixed prompt template → LLM pipeline with an agent that:
  1. Collects context from multiple sources via explicit tools
  2. Reasons about what is most clinically relevant for THIS patient
  3. Adjusts tone based on trajectory (stable HIGH vs deteriorating MODERATE)
  4. Generates a narrative grounded in the assembled context

The agent has four tools it calls before generating:
  _tool_assess_vital_trajectories  → what direction are key vitals moving?
  _tool_check_thresholds           → which Sepsis-3 / clinical limits are violated?
  _tool_assess_tone                → stable / worsening / rapidly deteriorating?
  _tool_build_prompt               → assemble final context string for LLM

This means two patients with the same risk score (e.g. 0.65) get
meaningfully different narratives:
  Patient A — stable for 12h:
      "Risk is HIGH but vitals have been stable. Continue close monitoring."
  Patient B — MAP falling for 4h, lactate rising:
      "MAP has crossed the Sepsis-3 threshold and continues to fall.
       Lactate rising. Immediate bedside assessment required."
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.narrative.ollama_client import OllamaClient
from src.narrative.prompts import AGENT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from src.explainability.shap_explainer import SHAPExplanation


# ------------------------------------------------------------------ #
# Threshold definitions (mirror VITAL_THRESHOLDS in patient_buffer)   #
# Using feature column names that exist in the predictions DataFrame   #
# ------------------------------------------------------------------ #

_THRESHOLD_CHECKS = [
    # (feature_col,          label,        threshold, direction,  severity,   message_template)
    ("map_last",      "MAP",         65.0,  "below", "critical",
     "MAP {val:.0f} mmHg — below Sepsis-3 threshold (65 mmHg)"),
    ("map_last",      "MAP",         70.0,  "below", "warning",
     "MAP {val:.0f} mmHg — approaching hypotension"),
    ("heart_rate_last", "Heart rate", 130.0, "above", "critical",
     "HR {val:.0f} bpm — severe tachycardia"),
    ("heart_rate_last", "Heart rate", 100.0, "above", "warning",
     "HR {val:.0f} bpm — tachycardia"),
    ("spo2_min",      "SpO2",        90.0,  "below", "critical",
     "SpO2 {val:.0f}% — severe hypoxia"),
    ("spo2_min",      "SpO2",        94.0,  "below", "warning",
     "SpO2 {val:.0f}% — below normal range"),
    ("resp_rate_last", "Resp rate",  30.0,  "above", "critical",
     "RR {val:.0f}/min — severe tachypnea"),
    ("resp_rate_last", "Resp rate",  22.0,  "above", "warning",
     "RR {val:.0f}/min — tachypnea (Sepsis-3 criterion)"),
    ("lactate_last",  "Lactate",     4.0,   "above", "critical",
     "Lactate {val:.1f} mmol/L — critically elevated (Sepsis-3 criterion)"),
    ("lactate_last",  "Lactate",     2.0,   "above", "warning",
     "Lactate {val:.1f} mmol/L — elevated (normal < 2.0 mmol/L)"),
]


def _safe_float(val) -> float | None:
    """Return float or None if value is NaN / None."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


class NarrativeAgent:
    """
    Reasoning agent for clinical narrative generation.

    Usage:
        agent = NarrativeAgent(client=OllamaClient(cfg))
        narrative = agent.generate(explanation, features)

        # or streaming:
        for chunk in agent.stream_generate(explanation, features):
            yield chunk
    """

    def __init__(self, client: OllamaClient):
        self.client = client

    # ---------------------------------------------------------------- #
    # Tool 1 — Vital trajectories                                        #
    # ---------------------------------------------------------------- #

    def _tool_assess_vital_trajectories(
        self, features: dict
    ) -> dict[str, str]:
        """
        Determine directional trajectory for each vital using trend/delta features.

        Returns a dict of {vital_label: description} for vitals with a
        meaningful trend, e.g.:
          {"MAP": "falling (-2.1 mmHg/hr)", "Lactate": "rising (+0.8 mmol/L)"}

        Only vitals with a clinically significant trend are included —
        stable vitals are omitted to keep the narrative focused.
        """
        trajectories: dict[str, str] = {}

        # MAP — falling is the most dangerous pattern in sepsis
        map_trend = _safe_float(features.get("map_trend"))
        if map_trend is not None:
            if map_trend < -1.0:
                trajectories["MAP"] = f"falling ({map_trend:.1f} mmHg/hr)"
            elif map_trend > 1.0:
                trajectories["MAP"] = f"rising ({map_trend:+.1f} mmHg/hr)"

        # Heart rate — rising tachycardia + falling MAP = distributive shock pattern
        hr_trend = _safe_float(features.get("heart_rate_trend"))
        if hr_trend is not None:
            if hr_trend > 2.0:
                trajectories["Heart rate"] = f"rising ({hr_trend:+.1f} bpm/hr)"
            elif hr_trend < -2.0:
                trajectories["Heart rate"] = f"falling ({hr_trend:.1f} bpm/hr)"

        # SpO2 — falling is urgent
        spo2_trend = _safe_float(features.get("spo2_trend"))
        if spo2_trend is not None and spo2_trend < -0.5:
            trajectories["SpO2"] = f"falling ({spo2_trend:.1f}%/hr)"

        # Respiratory rate — rising suggests compensation or worsening
        rr_trend = _safe_float(features.get("resp_rate_trend"))
        if rr_trend is not None and rr_trend > 1.0:
            trajectories["Resp rate"] = f"rising ({rr_trend:+.1f}/min per hr)"

        # Lactate — use delta (batch lab, not continuous)
        lactate_delta = _safe_float(features.get("lactate_delta"))
        if lactate_delta is not None:
            if lactate_delta > 0.5:
                trajectories["Lactate"] = f"rising (+{lactate_delta:.1f} mmol/L since first draw)"
            elif lactate_delta < -0.5:
                trajectories["Lactate"] = f"falling ({lactate_delta:.1f} mmol/L since first draw)"

        # Creatinine — rising delta suggests acute kidney injury
        creatinine_delta = _safe_float(features.get("creatinine_delta"))
        if creatinine_delta is not None and creatinine_delta > 0.3:
            trajectories["Creatinine"] = f"rising (+{creatinine_delta:.1f} mg/dL — possible AKI)"

        return trajectories

    # ---------------------------------------------------------------- #
    # Tool 2 — Threshold checks                                          #
    # ---------------------------------------------------------------- #

    def _tool_check_thresholds(
        self, features: dict
    ) -> list[dict]:
        """
        Check key clinical thresholds using feature column values.

        Returns a list of threshold violations sorted critical-first.
        Only the most severe violation per vital is reported.
        """
        alerts: list[dict] = []
        reported: set[str] = set()

        for feat_name, label, threshold, direction, severity, msg_tmpl in _THRESHOLD_CHECKS:
            if label in reported:
                continue  # already reported a more severe violation for this vital
            val = _safe_float(features.get(feat_name))
            if val is None:
                continue
            violated = (
                (direction == "below" and val < threshold)
                or (direction == "above" and val > threshold)
            )
            if violated:
                alerts.append({
                    "label": label,
                    "severity": severity,
                    "message": msg_tmpl.format(val=val),
                })
                reported.add(label)

        alerts.sort(key=lambda a: 0 if a["severity"] == "critical" else 1)
        return alerts

    # ---------------------------------------------------------------- #
    # Tool 3 — Tone assessment                                           #
    # ---------------------------------------------------------------- #

    def _tool_assess_tone(
        self,
        risk_score: float,
        trajectories: dict[str, str],
        threshold_alerts: list[dict],
    ) -> str:
        """
        Decide the clinical tone of the narrative.

        This is the key reasoning step — two patients with the same risk
        score receive different narrative tones based on their trajectory.

        Returns one of:
          "RAPIDLY DETERIORATING"
          "WORSENING"
          "HIGH — currently stable"
          "MODERATE — monitor closely"
        """
        critical_count = sum(
            1 for a in threshold_alerts if a["severity"] == "critical"
        )
        # Count vitals moving in a dangerous direction
        worsening_vitals = sum(
            1 for vital, desc in trajectories.items()
            if (vital in ("MAP", "SpO2") and "falling" in desc)
            or (vital in ("Heart rate", "Resp rate", "Lactate", "Creatinine")
                and "rising" in desc)
        )

        if critical_count >= 2 or (critical_count >= 1 and worsening_vitals >= 2):
            return "RAPIDLY DETERIORATING — immediate assessment required"
        if critical_count >= 1 or worsening_vitals >= 2:
            return "WORSENING — close monitoring required"
        if risk_score >= 0.8:
            return "CRITICALLY HIGH — currently stable"
        if risk_score >= 0.6:
            return "HIGH — currently stable"
        return "MODERATE — monitor closely"

    # ---------------------------------------------------------------- #
    # Tool 4 — Prompt assembly                                           #
    # ---------------------------------------------------------------- #

    def _tool_build_prompt(
        self,
        explanation: "SHAPExplanation",
        trajectories: dict[str, str],
        threshold_alerts: list[dict],
        tone: str,
        few_shot_context: str = "",
        alert_context: str = "",
    ) -> str:
        """
        Assemble the full structured context for the LLM.

        The prompt is ordered by clinical priority:
          1. Risk score + trajectory tone (overall picture)
          2. Threshold alerts (what is critically wrong right now)
          3. Trending vitals (what is changing)
          4. SHAP drivers (why the model flagged this patient)
          5. Alert context (has the doctor already been notified?)
          6. Few-shot examples (what good narratives look like)
        """
        lines: list[str] = []

        lines.append(
            f"RISK SCORE: {explanation.risk_score:.2f} ({explanation.risk_label})"
        )
        lines.append(f"CLINICAL TRAJECTORY: {tone}\n")

        if threshold_alerts:
            lines.append("VITAL THRESHOLD ALERTS (act on these first):")
            for alert in threshold_alerts:
                lines.append(f"  [{alert['severity'].upper()}] {alert['message']}")
            lines.append("")

        if trajectories:
            lines.append("TRENDING VITALS (direction over the observation window):")
            for vital, desc in trajectories.items():
                lines.append(f"  - {vital}: {desc}")
            lines.append("")

        lines.append("TOP SHAP DRIVERS (why the model flagged this patient):")
        for i, feat in enumerate(explanation.top_features[:5], 1):
            val = _safe_float(feat.get("value"))
            val_str = f"{val:.1f}" if val is not None else "N/A"
            label = feat.get("label") or feat.get("feature", "")
            direction = (feat.get("direction") or "").replace("_", " ")
            shap = feat.get("shap", 0) or 0
            lines.append(
                f"  {i}. {label} = {val_str} → {direction} (SHAP: {shap:+.3f})"
            )
        lines.append("")

        if alert_context:
            lines.append(f"ALERT CONTEXT: {alert_context}")
            lines.append("")

        if few_shot_context:
            lines.append("EXAMPLES OF WELL-RATED NARRATIVES FROM THIS UNIT:")
            lines.append(few_shot_context)
            lines.append("")

        lines.append(
            "Generate a concise SBAR nurse alert. "
            "Prioritise THRESHOLD ALERTS and TRAJECTORY over raw scores. "
            "Do NOT state a confirmed diagnosis. "
            "Do NOT prescribe specific treatments or dosages."
        )

        return "\n".join(lines)

    # ---------------------------------------------------------------- #
    # Public API                                                         #
    # ---------------------------------------------------------------- #

    def generate(
        self,
        explanation: "SHAPExplanation",
        features: dict,
        few_shot_context: str = "",
        alert_context: str = "",
    ) -> str:
        """
        Generate a clinical narrative (non-streaming).

        Args:
            explanation:      SHAP explanation for this patient
            features:         flat feature dict from predictions DataFrame
            few_shot_context: high-rated narrative examples (from RAG)
            alert_context:    e.g. "First alert" or "Doctor notified 2h ago"
        """
        trajectories = self._tool_assess_vital_trajectories(features)
        thresholds   = self._tool_check_thresholds(features)
        tone         = self._tool_assess_tone(
            explanation.risk_score, trajectories, thresholds
        )
        prompt = self._tool_build_prompt(
            explanation, trajectories, thresholds, tone,
            few_shot_context, alert_context,
        )
        return self.client._chat(AGENT_SYSTEM_PROMPT, prompt)

    def stream_generate(
        self,
        explanation: "SHAPExplanation",
        features: dict,
        few_shot_context: str = "",
        alert_context: str = "",
    ):
        """
        Stream a clinical narrative chunk by chunk.

        Collects all context synchronously (fast), then streams the LLM
        output incrementally for the typewriter effect in the React UI.

        Yields text chunks as they arrive from Ollama.
        """
        trajectories = self._tool_assess_vital_trajectories(features)
        thresholds   = self._tool_check_thresholds(features)
        tone         = self._tool_assess_tone(
            explanation.risk_score, trajectories, thresholds
        )
        prompt = self._tool_build_prompt(
            explanation, trajectories, thresholds, tone,
            few_shot_context, alert_context,
        )
        yield from self.client._stream_chat(AGENT_SYSTEM_PROMPT, prompt)
