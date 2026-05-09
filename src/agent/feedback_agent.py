"""
FeedbackLoopAgent — monitors clinician feedback and decides when to act.

Two feedback streams are monitored in parallel:
  1. Clinical feedback  (logs/feedback.jsonl)
     - "confirmed_sepsis" → true positive label
     - "flagged_wrong"    → false positive label

  2. Narrative ratings  (logs/narrative_feedback.jsonl)
     - 1–5 star rating from clinician after reading the LLM narrative
     - Optional free-text correction note

Decision logic
--------------
  WAIT     — not enough data yet (< 10 clinical records OR < 5 narrative ratings)
  FLAG     — quality issue detected:
               • Mean narrative rating < 2.5 (clinicians unhappy with LLM output)
               • Rating std > 1.5 (contradictory / inconsistent feedback)
               • False positive rate > 40% (model alerts too many non-sepsis patients)
  RETRAIN  — sufficient signal to improve:
               • ≥ 20 confirmed sepsis labels accumulated  AND
               • False positive rate > 30% (model has consistent systematic error)

The agent is stateless — call evaluate() at any time and it reads
the current log files fresh, so it reflects the latest clinician feedback.

This is deterministic rule-based reasoning (not a second LLM call).
The rules are intentionally conservative: RETRAIN is only triggered when
there is both enough positive labels AND evidence of false-positive overload.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_CLINICAL_LOG   = Path("logs/feedback.jsonl")
_NARRATIVE_LOG  = Path("logs/narrative_feedback.jsonl")

# Thresholds (tunable via config in a future version)
_MIN_RECORDS_TO_ACT     = 10    # need this many clinical records before FLAG/RETRAIN
_MIN_NARRATIVE_TO_ACT   = 5     # need this many narrative ratings before using them
_FLAG_RATING_MEAN       = 2.5   # flag if mean star rating falls below this
_FLAG_RATING_STD        = 1.5   # flag if rating std exceeds this (contradictory feedback)
_FLAG_FP_RATE           = 0.40  # flag if false positive rate exceeds 40 %
_RETRAIN_MIN_CONFIRMED  = 20    # need at least 20 confirmed sepsis labels for retraining
_RETRAIN_FP_RATE        = 0.30  # retrain if false positive rate exceeds 30 %


# ------------------------------------------------------------------ #
# Result dataclass                                                     #
# ------------------------------------------------------------------ #

@dataclass
class FeedbackDecision:
    """
    The agent's current decision based on accumulated clinician feedback.

    Fields
    ------
    decision        : "WAIT" | "FLAG" | "RETRAIN"
    reason          : Human-readable explanation (shown in the UI card)
    details         : Dict of computed metrics surfaced for transparency
    evaluated_at    : ISO timestamp of this evaluation
    clinical_total  : Total clinical feedback records
    confirmed_sepsis: Count of confirmed_sepsis records
    flagged_wrong   : Count of flagged_wrong records
    fp_rate         : False positive rate (flagged_wrong / total), or None
    narrative_total : Total narrative ratings
    mean_rating     : Mean star rating, or None if insufficient data
    std_rating      : Std of star ratings, or None if insufficient data
    low_rated_pct   : Fraction of ratings ≤ 2 stars (quality signal)
    correction_notes: Recent free-text corrections from clinicians (up to 5)
    """

    decision:         str
    reason:           str
    details:          dict = field(default_factory=dict)
    evaluated_at:     str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    clinical_total:   int  = 0
    confirmed_sepsis: int  = 0
    flagged_wrong:    int  = 0
    fp_rate:          float | None = None
    narrative_total:  int  = 0
    mean_rating:      float | None = None
    std_rating:       float | None = None
    low_rated_pct:    float | None = None
    correction_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict for the API response."""
        return {
            "decision":         self.decision,
            "reason":           self.reason,
            "details":          self.details,
            "evaluated_at":     self.evaluated_at,
            "clinical_total":   self.clinical_total,
            "confirmed_sepsis": self.confirmed_sepsis,
            "flagged_wrong":    self.flagged_wrong,
            "fp_rate":          round(self.fp_rate, 3) if self.fp_rate is not None else None,
            "narrative_total":  self.narrative_total,
            "mean_rating":      round(self.mean_rating, 2) if self.mean_rating is not None else None,
            "std_rating":       round(self.std_rating, 2) if self.std_rating is not None else None,
            "low_rated_pct":    round(self.low_rated_pct, 3) if self.low_rated_pct is not None else None,
            "correction_notes": self.correction_notes,
        }


# ------------------------------------------------------------------ #
# Agent                                                                #
# ------------------------------------------------------------------ #

class FeedbackLoopAgent:
    """
    Rule-based agent that monitors clinician feedback and decides when
    to flag issues or trigger model retraining.

    Usage
    -----
        agent = FeedbackLoopAgent()
        decision = agent.evaluate()
        print(decision.decision, decision.reason)
    """

    def evaluate(self) -> FeedbackDecision:
        """
        Read both feedback logs and return the current decision.

        This is the single public method — call it at any time.
        The result reflects the latest state of the log files.
        """
        clinical = self._load_clinical()
        narrative = self._load_narrative()

        # --- Compute clinical metrics ---
        clinical_total   = len(clinical)
        confirmed        = sum(1 for r in clinical if r.get("feedback_type") == "confirmed_sepsis")
        flagged          = sum(1 for r in clinical if r.get("feedback_type") == "flagged_wrong")
        fp_rate          = flagged / clinical_total if clinical_total > 0 else None

        # --- Compute narrative metrics ---
        narrative_total  = len(narrative)
        ratings          = [r["rating"] for r in narrative if isinstance(r.get("rating"), (int, float))]
        mean_rating      = (sum(ratings) / len(ratings)) if ratings else None
        std_rating       = self._std(ratings) if len(ratings) >= 2 else None
        low_rated        = sum(1 for r in ratings if r <= 2)
        low_rated_pct    = low_rated / len(ratings) if ratings else None

        # Recent correction notes (non-empty, most recent first)
        notes = [
            r["correction_note"]
            for r in reversed(narrative)
            if r.get("correction_note", "").strip()
        ][:5]

        details = {
            "clinical_records":      clinical_total,
            "confirmed_sepsis":      confirmed,
            "flagged_wrong":         flagged,
            "fp_rate":               round(fp_rate, 3) if fp_rate is not None else None,
            "narrative_ratings":     narrative_total,
            "mean_star_rating":      round(mean_rating, 2) if mean_rating is not None else None,
            "rating_std":            round(std_rating, 2) if std_rating is not None else None,
            "low_rated_pct":         round(low_rated_pct, 3) if low_rated_pct is not None else None,
            "thresholds": {
                "min_clinical_to_act":    _MIN_RECORDS_TO_ACT,
                "min_narrative_to_act":   _MIN_NARRATIVE_TO_ACT,
                "flag_rating_below":      _FLAG_RATING_MEAN,
                "flag_fp_rate_above":     _FLAG_FP_RATE,
                "retrain_min_confirmed":  _RETRAIN_MIN_CONFIRMED,
                "retrain_fp_rate_above":  _RETRAIN_FP_RATE,
            },
        }

        # --- Decision logic ---

        # WAIT — insufficient data
        if clinical_total < _MIN_RECORDS_TO_ACT:
            reason = (
                f"Collecting data: {clinical_total}/{_MIN_RECORDS_TO_ACT} clinical "
                f"feedback records and {narrative_total}/{_MIN_NARRATIVE_TO_ACT} "
                f"narrative ratings accumulated."
            )
            return FeedbackDecision(
                decision="WAIT", reason=reason, details=details,
                clinical_total=clinical_total, confirmed_sepsis=confirmed,
                flagged_wrong=flagged, fp_rate=fp_rate,
                narrative_total=narrative_total, mean_rating=mean_rating,
                std_rating=std_rating, low_rated_pct=low_rated_pct,
                correction_notes=notes,
            )

        # RETRAIN — strong signal for systematic improvement
        if confirmed >= _RETRAIN_MIN_CONFIRMED and fp_rate is not None and fp_rate > _RETRAIN_FP_RATE:
            reason = (
                f"Retraining recommended: {confirmed} confirmed sepsis labels collected "
                f"with a false positive rate of {fp_rate:.0%} "
                f"(threshold: >{_RETRAIN_FP_RATE:.0%}). "
                f"Sufficient labelled data to improve model decision boundary."
            )
            return FeedbackDecision(
                decision="RETRAIN", reason=reason, details=details,
                clinical_total=clinical_total, confirmed_sepsis=confirmed,
                flagged_wrong=flagged, fp_rate=fp_rate,
                narrative_total=narrative_total, mean_rating=mean_rating,
                std_rating=std_rating, low_rated_pct=low_rated_pct,
                correction_notes=notes,
            )

        # FLAG — quality issue detected; enumerate all active flag reasons
        flag_reasons: list[str] = []

        if (
            narrative_total >= _MIN_NARRATIVE_TO_ACT
            and mean_rating is not None
            and mean_rating < _FLAG_RATING_MEAN
        ):
            flag_reasons.append(
                f"mean narrative rating {mean_rating:.1f}/5 "
                f"(below threshold {_FLAG_RATING_MEAN}/5)"
            )

        if (
            narrative_total >= _MIN_NARRATIVE_TO_ACT
            and std_rating is not None
            and std_rating > _FLAG_RATING_STD
        ):
            flag_reasons.append(
                f"rating variance σ={std_rating:.1f} "
                f"(threshold σ>{_FLAG_RATING_STD}) — clinicians disagree on quality"
            )

        if fp_rate is not None and fp_rate > _FLAG_FP_RATE:
            flag_reasons.append(
                f"false positive rate {fp_rate:.0%} "
                f"(threshold >{_FLAG_FP_RATE:.0%}) — too many incorrect alerts"
            )

        if flag_reasons:
            reason = "Human review recommended: " + "; ".join(flag_reasons) + "."
            if notes:
                reason += f" Most recent clinician note: \"{notes[0]}\""
            return FeedbackDecision(
                decision="FLAG", reason=reason, details=details,
                clinical_total=clinical_total, confirmed_sepsis=confirmed,
                flagged_wrong=flagged, fp_rate=fp_rate,
                narrative_total=narrative_total, mean_rating=mean_rating,
                std_rating=std_rating, low_rated_pct=low_rated_pct,
                correction_notes=notes,
            )

        # All thresholds are fine — continue monitoring
        rating_str = f", mean narrative rating {mean_rating:.1f}/5" if mean_rating is not None else ""
        fp_str = f", FP rate {fp_rate:.0%}" if fp_rate is not None else ""
        reason = (
            f"All metrics within acceptable range: {clinical_total} clinical records "
            f"({confirmed} confirmed sepsis, {flagged} false positives)"
            f"{fp_str}{rating_str}. Continue monitoring."
        )
        return FeedbackDecision(
            decision="WAIT", reason=reason, details=details,
            clinical_total=clinical_total, confirmed_sepsis=confirmed,
            flagged_wrong=flagged, fp_rate=fp_rate,
            narrative_total=narrative_total, mean_rating=mean_rating,
            std_rating=std_rating, low_rated_pct=low_rated_pct,
            correction_notes=notes,
        )

    # ---------------------------------------------------------------- #
    # Private helpers                                                    #
    # ---------------------------------------------------------------- #

    @staticmethod
    def _load_clinical() -> list[dict]:
        """Load all clinical feedback records."""
        if not _CLINICAL_LOG.exists():
            return []
        records: list[dict] = []
        with _CLINICAL_LOG.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _load_narrative() -> list[dict]:
        """Load all narrative feedback records."""
        if not _NARRATIVE_LOG.exists():
            return []
        records: list[dict] = []
        with _NARRATIVE_LOG.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _std(values: list[float]) -> float:
        """Population standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)
