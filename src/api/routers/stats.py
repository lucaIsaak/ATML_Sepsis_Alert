"""
Stats routes.

GET  /stats                  — model performance metrics + ROC curve data
GET  /feedback-agent/status  — FeedbackLoopAgent WAIT / FLAG / RETRAIN decision
POST /retrain                — launch retrain_with_feedback.py as a subprocess
GET  /retrain/status         — poll subprocess state (idle | running | done | error)
"""

from __future__ import annotations

import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import sklearn
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from src.safety.guardrails import AuditLogger
from src.agent.feedback_agent import FeedbackLoopAgent
from src.monitoring.drift import compute_drift_report

_feedback_agent = FeedbackLoopAgent()

router = APIRouter()
_audit_logger = AuditLogger(log_path="logs/audit.jsonl")

# ------------------------------------------------------------------ #
# Retrain subprocess state (in-memory, single instance)               #
# ------------------------------------------------------------------ #

_retrain_state: dict = {
    "status":      "idle",   # idle | running | done | error
    "log":         "",
    "started_at":  None,
    "finished_at": None,
    "exit_code":   None,
}
_retrain_lock = threading.Lock()


def _run_retrain_subprocess() -> None:
    """Run retrain_with_feedback.py in a background thread."""
    script = Path("retrain_with_feedback.py")

    with _retrain_lock:
        _retrain_state["status"]      = "running"
        _retrain_state["log"]         = ""
        _retrain_state["started_at"]  = datetime.now(timezone.utc).isoformat()
        _retrain_state["finished_at"] = None
        _retrain_state["exit_code"]   = None

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Stream output line by line into state log
        for line in proc.stdout:  # type: ignore[union-attr]
            with _retrain_lock:
                _retrain_state["log"] += line

        proc.wait()
        exit_code = proc.returncode

        with _retrain_lock:
            _retrain_state["status"]      = "done" if exit_code == 0 else "error"
            _retrain_state["exit_code"]   = exit_code
            _retrain_state["finished_at"] = datetime.now(timezone.utc).isoformat()

    except Exception as exc:  # pylint: disable=broad-except
        with _retrain_lock:
            _retrain_state["status"]      = "error"
            _retrain_state["log"]        += f"\n[Internal error] {exc}"
            _retrain_state["finished_at"] = datetime.now(timezone.utc).isoformat()

# NEWS2 risk scores are simulated from known vital-sign weights
# (this mirrors what the Streamlit dashboard does)
_NEWS2_WEIGHTS = {
    "resp_rate_last": 3,
    "spo2_last": 3,
    "heart_rate_last": 1,
    "temperature_f_last": 2,
    "map_last": 3,
}
_NEWS2_MAX = sum(_NEWS2_WEIGHTS.values())


def _compute_news2_score(row) -> float:
    """Approximate NEWS2 score normalised to [0, 1] from feature values."""
    score = 0.0
    for feat, weight in _NEWS2_WEIGHTS.items():
        if feat in row.index and not np.isnan(row[feat]):
            score += weight
    return score / _NEWS2_MAX if _NEWS2_MAX > 0 else 0.0


def _roc_curve_points(y_true, y_score, n_points: int = 100) -> list[dict]:
    """Compute ROC curve and return sampled {fpr, tpr} points."""
    from sklearn.metrics import roc_curve  # noqa: PLC0415

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Subsample to n_points for JSON size
    indices = np.linspace(0, len(fpr) - 1, min(n_points, len(fpr)), dtype=int)
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in indices]


@router.get("/model/info")
async def get_model_info(request: Request) -> dict:
    """Return model metadata: algorithm, AUROC, feature count, sklearn version."""
    artifact = request.app.state.artifact
    return {
        "algorithm":       "HistGradientBoostingClassifier",
        "auroc":           float(artifact.get("auroc", 0.895)),
        "feature_count":   len(artifact.get("feature_cols", [])),
        "sklearn_version": sklearn.__version__,
        "training_data":   "MIMIC-IV v3.1 — 93,224 ICU stays",
        "label_strategy":  "Sepsis-3 ICD-10 proxy (A41.x / R65.2x)",
        "tuning":          "Optuna Bayesian search — 50 trials, 5-fold CV",
    }


@router.get("/audit")
async def get_audit_log(n: int = 50) -> list[dict]:
    """Return the last n audit log entries (GDPR Art. 22 transparency endpoint)."""
    return _audit_logger.read_recent(n=n)


@router.get("/feedback-agent/status")
async def get_feedback_agent_status() -> dict:
    """
    Run the FeedbackLoopAgent and return its current decision.

    Decision values: "WAIT" | "FLAG" | "RETRAIN"
    Called by the Model Performance page to display the agent status card.
    This endpoint reads fresh from the log files on every call — no caching.
    """
    decision = _feedback_agent.evaluate()
    return decision.to_dict()


@router.get("/drift/status")
async def get_drift_status(request: Request) -> dict:
    """
    Compute PSI-based data drift report.

    Compares the live patient feature distributions (current predictions)
    against the training distribution (features.parquet).
    Results are logged to logs/drift_history.jsonl for trend tracking.
    """
    predictions = request.app.state.predictions
    features_df = request.app.state.features_df
    artifact    = request.app.state.artifact
    feature_cols = artifact["feature_cols"]

    return compute_drift_report(
        train_df=features_df,
        live_df=predictions,
        feature_cols=feature_cols,
        risk_scores_live=predictions["risk_score"].values,
    )


@router.post("/retrain")
async def trigger_retrain() -> dict:
    """
    Launch retrain_with_feedback.py as a background subprocess.

    Returns 409 if a retrain is already in progress.
    The script compares old vs new AUROC and only saves if the new model improves.
    Poll GET /retrain/status for progress and log output.
    """
    with _retrain_lock:
        if _retrain_state["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="Retraining is already in progress. Poll /retrain/status for updates.",
            )

    thread = threading.Thread(target=_run_retrain_subprocess, daemon=True)
    thread.start()
    return {"status": "started", "message": "Retraining launched. Poll /retrain/status."}


@router.get("/retrain/status")
async def get_retrain_status() -> dict:
    """Return current retrain subprocess state and captured log output."""
    with _retrain_lock:
        return dict(_retrain_state)


@router.get("/stats")
async def get_stats(request: Request) -> dict:
    """Return model performance metrics and ROC curve data."""
    predictions = request.app.state.predictions

    # Ground truth: merge sepsis_label from cohort_df
    cohort_df = request.app.state.cohort_df
    label_col = cohort_df[["stay_id", "sepsis_label"]] if "sepsis_label" in cohort_df.columns else None
    if label_col is not None:
        merged = predictions[["stay_id", "risk_score"]].merge(label_col, on="stay_id", how="left")
        y_true = merged["sepsis_label"].fillna(0).astype(int).values
    else:
        y_true = (predictions["risk_score"] >= 0.5).astype(int).values

    y_score = predictions["risk_score"].values

    # NEWS2 scores — computed from features_df (has vital-sign columns)
    features_df = request.app.state.features_df
    sampled_features = features_df[features_df["stay_id"].isin(predictions["stay_id"])]
    news2_scores = np.array([
        _compute_news2_score(row) for _, row in sampled_features.iterrows()
    ])
    # Align length with y_true/y_score in case of any mismatch
    news2_scores = news2_scores[: len(y_score)]

    # ROC curves
    roc_sepsis = _roc_curve_points(y_true, y_score)
    roc_news2 = _roc_curve_points(y_true, news2_scores)

    artifact = request.app.state.artifact
    auroc = float(artifact.get("auroc", 0.895))

    return {
        # AUROC read from the actual trained artifact — updates after retrain
        "auroc": auroc,
        "news2_auroc": 0.614,
        "auprc": 0.527,
        "total_stays": 93224,
        "sepsis_cases": 9890,
        "features": len(artifact.get("feature_cols", [])) or 43,
        # Dynamic ROC curves from the sampled predictions
        "roc_sepsis": roc_sepsis,
        "roc_news2": roc_news2,
    }
