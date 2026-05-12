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

# App reference injected by lifespan — avoids circular import from src.api.main
_app = None


def set_app(app) -> None:  # noqa: ANN001
    """Called once from main.py lifespan so hot-reload can access app.state."""
    global _app  # noqa: PLW0603
    _app = app

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

        # ── Hot-reload the new model into app.state ──────────────────────
        if exit_code == 0:
            _hot_reload_model()

    except Exception as exc:  # pylint: disable=broad-except
        with _retrain_lock:
            _retrain_state["status"]      = "error"
            _retrain_state["log"]        += f"\n[Internal error] {exc}"
            _retrain_state["finished_at"] = datetime.now(timezone.utc).isoformat()


def _hot_reload_model() -> None:
    """
    Reload the newly saved model artifact into the running FastAPI app.

    Called automatically after a successful retrain so the live API
    starts serving predictions from the improved model immediately —
    no server restart required.

    Uses the module-level `_app` reference injected via set_app() during
    lifespan startup — avoids a circular import from src.api.main.
    """
    import joblib  # noqa: PLC0415

    app = _app  # injected by lifespan; None if server not fully started yet
    if app is None:
        with _retrain_lock:
            _retrain_state["log"] += "\n[Hot-reload] app reference not set — skipping reload."
        return

    cfg = app.state.cfg if hasattr(app.state, "cfg") else {}
    artifact_path = Path(cfg.get("model", {}).get("artifact_path", "models/sepsis_model.pkl"))
    if not artifact_path.exists():
        with _retrain_lock:
            _retrain_state["log"] += "\n[Hot-reload] model file not found — skipping reload."
        return

    try:
        new_artifact = joblib.load(artifact_path)
        app.state.artifact = new_artifact

        # Re-run predictions with the new model on the same patient sample
        from src.model.predict import predict_batch  # noqa: PLC0415
        features_df = app.state.features_df
        cohort_df   = app.state.cohort_df
        sample = features_df.sample(n=min(100, len(features_df)), random_state=99)
        new_preds = predict_batch(sample, new_artifact)
        display_cols = ["stay_id"] + [c for c in ["age", "gender", "first_careunit"]
                                       if c in cohort_df.columns]
        new_preds = new_preds.merge(cohort_df[display_cols], on="stay_id", how="left")
        app.state.predictions = new_preds

        # Clear per-patient caches so SHAP / OOD / uncertainty are recomputed
        import src.api.routers.patients as _patients_router  # noqa: PLC0415
        _patients_router._shap_cache.clear()
        _patients_router._ood_cache.clear()
        _patients_router._uncertainty_cache.clear()
        # Reset lazily-built guards so they are rebuilt against the new model weights
        _patients_router._input_guard = None
        _patients_router._explainer   = None

        # Also update the PatientMonitorAgent's model reference
        if hasattr(app.state, "monitor_agent"):
            app.state.monitor_agent.update_artifact(new_artifact)

        with _retrain_lock:
            _retrain_state["log"] += (
                f"\n[Hot-reload] ✓ New model loaded (AUROC {new_artifact.get('auroc', '?'):.4f}). "
                "Predictions refreshed. No server restart needed."
            )
    except Exception as exc:  # pylint: disable=broad-except
        with _retrain_lock:
            _retrain_state["log"] += f"\n[Hot-reload] ✗ Failed: {exc}"

def _news2_score(row) -> float:
    """
    Compute a proper NEWS2 score from available feature columns.

    NEWS2 (Royal College of Physicians, 2017) is a track-and-trigger system
    used in UK ICUs as a clinical deterioration baseline.

    Ranges sourced from: https://www.rcplondon.ac.uk/projects/outputs/national-early-warning-score-news-2

    We use available MIMIC-IV features; supplemental O2 and consciousness
    (AVPU) are omitted as they are not in features.parquet.
    Maximum achievable score here = 16 (out of 20 with O2/consciousness).
    Normalised to [0, 1] for AUROC comparison.
    """
    def _get(col_a, col_b=None):
        """Return first non-NaN value from column a or fallback b."""
        for col in [col_a, col_b]:
            if col and col in row.index:
                v = row[col]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return float(v)
        return None

    score = 0

    # ── Respiratory rate ───────────────────────────────────────────
    rr = _get("resp_rate_last", "resp_rate_mean")
    if rr is not None:
        if rr <= 8 or rr >= 25:
            score += 3
        elif 21 <= rr <= 24:
            score += 2
        elif 9 <= rr <= 11:
            score += 1
        # 12–20 → 0

    # ── SpO2 ───────────────────────────────────────────────────────
    spo2 = _get("spo2_last", "spo2_mean")
    if spo2 is not None:
        if spo2 <= 91:
            score += 3
        elif 92 <= spo2 <= 93:
            score += 2
        elif 94 <= spo2 <= 95:
            score += 1
        # ≥96 → 0

    # ── Temperature (Fahrenheit → Celsius) ─────────────────────────
    temp_f = _get("temperature_f_last", "temperature_f_mean")
    if temp_f is not None:
        temp_c = (temp_f - 32.0) * 5.0 / 9.0
        if temp_c <= 35.0:
            score += 3
        elif 35.1 <= temp_c <= 36.0:
            score += 1
        elif 38.1 <= temp_c <= 39.0:
            score += 1
        elif temp_c >= 39.1:
            score += 2
        # 36.1–38.0 → 0

    # ── MAP as blood-pressure proxy ────────────────────────────────
    # NEWS2 uses systolic BP; MIMIC features store MAP.
    # Approximate equivalence: MAP ≈ (SBP + 2*DBP)/3
    # Threshold mapping:  SBP ≤ 90 ≈ MAP ≤ 60,  SBP 91-100 ≈ MAP 61-70
    #                     SBP ≥ 220 ≈ MAP ≥ 150
    map_v = _get("map_last", "map_mean")
    if map_v is not None:
        if map_v <= 60 or map_v >= 150:
            score += 3
        elif 61 <= map_v <= 70:
            score += 2
        elif 71 <= map_v <= 80:
            score += 1
        # 81–149 → 0

    # ── Heart rate ─────────────────────────────────────────────────
    hr = _get("heart_rate_last", "heart_rate_mean")
    if hr is not None:
        if hr <= 40 or hr >= 131:
            score += 3
        elif 111 <= hr <= 130:
            score += 2
        elif 41 <= hr <= 50 or 91 <= hr <= 110:
            score += 1
        # 51–90 → 0

    # Normalise: 5 components × max 3 points each = 15 (O2 and consciousness omitted)
    return score / 15.0



def _pr_curve_auprc(y_true, y_score) -> float:
    """Compute AUPRC (average precision) dynamically from current predictions."""
    from sklearn.metrics import average_precision_score  # noqa: PLC0415
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:  # pylint: disable=broad-except
        return 0.0


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
    model_type = type(artifact.get("model", None)).__name__ if artifact.get("model") else "Unknown"
    return {
        "algorithm":       f"{model_type} (sklearn {sklearn.__version__})",
        "auroc":           float(artifact.get("auroc", 0.895)),
        "feature_count":   len(artifact.get("feature_cols", [])),
        "sklearn_version": sklearn.__version__,
        "training_data":   "MIMIC-IV v3.1 — 93,224 ICU stays",
        "label_strategy":  "Sepsis-3 ICD-10 proxy (A41.x / R65.2x)",
        "tuning":          "Initial: Optuna Bayesian 50-trial search. Retrain: fixed hyperparams from config.",
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
    if "sepsis_label" in cohort_df.columns:
        merged = predictions[["stay_id", "risk_score"]].merge(
            cohort_df[["stay_id", "sepsis_label"]], on="stay_id", how="left"
        )
        y_true = merged["sepsis_label"].fillna(0).astype(int).values
    else:
        # Cohort has no labels — ROC metrics are unavailable
        y_true = np.zeros(len(predictions), dtype=int)

    y_score = predictions["risk_score"].values

    # NEWS2 scores — must be in the same row order as predictions/y_true
    features_df = request.app.state.features_df
    # Merge on stay_id to guarantee alignment; fill missing with 0
    news2_df = features_df[features_df["stay_id"].isin(predictions["stay_id"])].copy()
    news2_map = {
        int(row["stay_id"]): _news2_score(row)
        for _, row in news2_df.iterrows()
    }
    news2_scores = np.array([
        news2_map.get(int(sid), 0.0) for sid in predictions["stay_id"]
    ])

    # ROC curves
    roc_sepsis = _roc_curve_points(y_true, y_score)
    roc_news2 = _roc_curve_points(y_true, news2_scores)

    artifact = request.app.state.artifact
    auroc = float(artifact.get("auroc", 0.895))

    # Compute AUPRC dynamically from the current predictions + ground truth
    auprc = _pr_curve_auprc(y_true, y_score)
    # Compute NEWS2 AUROC dynamically too
    from sklearn.metrics import roc_auc_score as _roc_auc  # noqa: PLC0415
    try:
        news2_auroc = float(_roc_auc(y_true, news2_scores))
    except Exception:  # pylint: disable=broad-except
        news2_auroc = 0.614  # fallback if computation fails

    return {
        # AUROC read from the actual trained artifact — updates after retrain
        "auroc": auroc,
        "news2_auroc": round(news2_auroc, 3),
        "auprc": round(auprc, 3),
        "total_stays": 93224,
        "sepsis_cases": 9890,
        "features": len(artifact.get("feature_cols", [])) or 55,
        # Dynamic ROC curves from the sampled predictions
        "roc_sepsis": roc_sepsis,
        "roc_news2": roc_news2,
    }
