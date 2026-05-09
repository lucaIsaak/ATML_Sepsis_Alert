"""
Clinician feedback storage.

Persists alert-level feedback (confirmed sepsis / flagged wrong) to an
append-only JSONL file.  No external database required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path("logs/feedback.jsonl")


def save_feedback(stay_id: int, feedback_type: str, risk_score: float) -> None:
    """
    Append a clinician feedback record to the log.

    Parameters
    ----------
    stay_id       : ICU stay identifier
    feedback_type : "confirmed_sepsis" or "flagged_wrong"
    risk_score    : model score at the time of feedback
    """
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stay_id": int(stay_id),
        "feedback_type": feedback_type,
        "risk_score": float(risk_score),
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_feedback_for_patient(stay_id: int) -> dict | None:
    """
    Return the most recent feedback record for a stay, or None if none exists.
    """
    if not _LOG_PATH.exists():
        return None
    last = None
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("stay_id") == int(stay_id):
                    last = record
            except json.JSONDecodeError:
                continue
    return last


def load_training_labels():
    """
    Load all clinician feedback as a training-ready DataFrame.

    Returns a DataFrame with columns:
        stay_id        : int
        feedback_type  : "confirmed_sepsis" | "flagged_wrong"
        sepsis_label   : int   (1 = confirmed sepsis, 0 = flagged wrong)
        low_confidence : bool  (True for flagged_wrong — absence of alert
                                does not equal absence of disease)

    Only the most recent feedback per stay_id is kept.
    """
    import pandas as pd  # pylint: disable=import-outside-toplevel

    empty = pd.DataFrame(
        columns=["stay_id", "feedback_type", "sepsis_label", "low_confidence"]
    )
    if not _LOG_PATH.exists():
        return empty

    rows: list[dict] = []
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return empty

    # Keep most recent feedback per stay_id
    by_stay: dict[int, dict] = {}
    for row in rows:
        by_stay[int(row["stay_id"])] = row

    records = [
        {
            "stay_id":        sid,
            "feedback_type":  r["feedback_type"],
            "sepsis_label":   1 if r["feedback_type"] == "confirmed_sepsis" else 0,
            "low_confidence": r["feedback_type"] == "flagged_wrong",
        }
        for sid, r in by_stay.items()
    ]
    return pd.DataFrame(records)
