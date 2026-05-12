"""
SepsisAlert FastAPI Backend

Start the server:
    uvicorn src.api.main:app --reload --port 8000

Start the frontend:
    cd frontend && npm install && npm run dev
"""

import asyncio
import logging
import traceback
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
import os
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

import sklearn  # noqa: F401 — must be imported before joblib deserialises sklearn models
import numpy as np
import pandas as pd
import joblib
import yaml

# Project root — always resolve to ATML_Sepsis_Alert/ regardless of cwd
ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)  # ensure all relative paths (parquet, config) resolve correctly
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import patients, narrative, feedback, stats
from src.api.routers.stats import set_app as _set_stats_app


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_predictions(features_df: pd.DataFrame, artifact: dict, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Sample patients, run predictions, merge display columns."""
    from src.model.predict import predict_batch  # noqa: PLC0415
    sampled = features_df.sample(n=min(250, len(features_df)), random_state=99)
    preds = predict_batch(sampled, artifact)
    # age is already in preds (from features.parquet); only pull gender/first_careunit from cohort
    # to avoid pandas renaming age → age_x / age_y on merge
    display_cols = ["stay_id"] + [c for c in ["gender", "first_careunit"]
                                   if c in cohort_df.columns]
    return preds.merge(cohort_df[display_cols], on="stay_id", how="left")


async def _periodic_monitoring(
    app: FastAPI,
    interval_seconds: int = 3600,
    run_immediately: bool = False,
) -> None:
    """
    Background task: run PatientMonitorAgent over the current patient sample
    every `interval_seconds` (default: 1 hour).

    This keeps the audit log current and updates per-patient memory
    (risk history, alert suppression) without manual intervention.

    When `run_immediately` is True the first pass executes right away
    (used by lifespan startup instead of a blocking synchronous call).
    Otherwise a 30 s startup grace-period is applied.
    """
    if not run_immediately:
        await asyncio.sleep(30)

    while True:
        try:
            loop = asyncio.get_running_loop()
            monitor: "PatientMonitorAgent" = app.state.monitor_agent  # type: ignore[name-defined]
            predictions: pd.DataFrame = app.state.predictions
            if not predictions.empty:
                features_df: pd.DataFrame = app.state.features_df
                live_ids = set(predictions["stay_id"].tolist())
                live_features = features_df[features_df["stay_id"].isin(live_ids)]
                ts = datetime.now()
                # Run the blocking monitoring pass in a thread-pool executor
                # so the event loop stays responsive during Ollama / SHAP calls.
                alerts = await loop.run_in_executor(
                    None,
                    monitor.process_from_dataframe,
                    live_features,
                    ts,
                )
                if alerts:
                    print(f"[Monitor] Cycle complete — {len(alerts)} alert(s) dispatched.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("[Monitor] Background cycle error: %s\n%s", exc, traceback.format_exc())
        await asyncio.sleep(interval_seconds)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, data and predictions at startup; start background monitor."""
    cfg = load_config()

    # GDPR hard gate — validate at startup, not at first request.
    # Prevents a misconfigured provider from constructing a patient-data-laden
    # prompt and failing only after the SHAP summary has been assembled.
    _provider = cfg.get("narrative", {}).get("provider", "ollama")
    if _provider != "ollama":
        raise RuntimeError(
            f"\n\n  [SepsisAlert] narrative.provider is set to '{_provider}'.\n"
            "  Only 'ollama' is permitted — patient feature data is included in\n"
            "  the LLM prompt and must never leave the hospital network (GDPR Art. 9).\n"
            "  Set narrative.provider: 'ollama' in config.yaml.\n"
        )

    # Load model artifact (relative paths now safe — os.chdir(ROOT) at module level)
    artifact = joblib.load(cfg["model"]["artifact_path"])

    # Load feature + cohort data
    _features_path = Path("data/processed/features.parquet")
    _cohort_path   = Path("data/processed/cohort.parquet")
    if not _features_path.exists() or not _cohort_path.exists():
        raise RuntimeError(
            "\n\n"
            "  [SepsisAlert] Required data files are missing.\n"
            "  Run the demo setup first:\n\n"
            "      python setup_demo.py\n\n"
            "  Then restart the server."
        )
    features_df = pd.read_parquet(_features_path)
    cohort_df   = pd.read_parquet(_cohort_path)

    # ── Multivariate OOD statistics ────────────────────────────────────
    # Prefer stats saved by train.py (computed from training split only).
    # Fall back to computing from features_df only for legacy artifacts that
    # pre-date the training_stats save — avoids test-set contamination.
    if artifact.get("training_cov_inv") is not None:
        print("[Startup] Multivariate OOD stats loaded from artifact (training-split).")
    else:
        try:
            _fea_cols = artifact.get("feature_cols", [])
            _X_train  = features_df[_fea_cols].dropna()
            if len(_X_train) > len(_fea_cols) + 1:
                _mean    = _X_train.mean().values.astype(float)
                _cov     = np.cov(_X_train.values.T)
                _cov_reg = _cov + 1e-6 * np.eye(len(_fea_cols))
                try:
                    _cov_inv = np.linalg.inv(_cov_reg)
                except np.linalg.LinAlgError:
                    _cov_inv = np.linalg.pinv(_cov_reg)
                artifact["training_mean"]    = _mean
                artifact["training_cov_inv"] = _cov_inv
                print(
                    f"[Startup] Multivariate OOD ready (legacy fallback) — "
                    f"{len(_fea_cols)}-feature covariance from {len(_X_train):,} patients."
                )
        except Exception as _exc:  # pylint: disable=broad-except
            print(f"[Startup] Multivariate OOD stats skipped: {_exc}")

    # Build initial predictions
    predictions = _build_predictions(features_df, artifact, cohort_df)

    # ── Instantiate PatientMonitorAgent (reuses already-loaded artifact) ──
    # Import here to avoid circular imports at module level
    from src.agent.monitor_agent import PatientMonitorAgent  # noqa: PLC0415
    monitor_agent = PatientMonitorAgent.from_artifact(artifact, cfg)

    # Store in app state
    app.state.cfg           = cfg
    app.state.artifact      = artifact
    app.state.predictions   = predictions
    app.state.features_df   = features_df
    app.state.cohort_df     = cohort_df
    app.state.monitor_agent = monitor_agent

    # Inject app reference into the stats router (avoids circular import in hot-reload)
    _set_stats_app(app)

    # ── Pseudonymization sanity check ─────────────────────────────────────
    # Hashed IDs (HMAC-SHA256 → uint64 or hex string) are large and non-sequential.
    # If stay_ids look like small sequential integers the hospital may not have
    # applied pseudonymization before sending data. This is a soft warning only —
    # demo synthetic data intentionally uses sequential IDs.
    _stay_ids = features_df["stay_id"].dropna().astype(int).tolist()[:20]
    if _stay_ids and max(_stay_ids) < 1_000_000:
        _deltas = [abs(_stay_ids[i + 1] - _stay_ids[i]) for i in range(min(9, len(_stay_ids) - 1))]
        if _deltas and sum(_deltas) / len(_deltas) < 100:
            warnings.warn(
                "\n\n  [SepsisAlert] stay_ids appear to be small sequential integers.\n"
                "  In production, the hospital must pseudonymize patient IDs with\n"
                "  HMAC-SHA256 before sending data to this system (see README GDPR section).\n"
                "  This warning is expected for synthetic demo data.\n",
                stacklevel=1,
            )

    # ── Ollama health check ────────────────────────────────────────────────
    _ollama_url = cfg.get("narrative", {}).get("ollama_base_url", "http://localhost:11434")
    try:
        requests.get(f"{_ollama_url}/api/tags", timeout=3)
        print("[Startup] Ollama is running — narrative endpoints ready.")
    except Exception:
        warnings.warn(
            "\n\n  [SepsisAlert] Ollama is not running.\n"
            "  Narrative endpoints will fail until you run:\n\n"
            "      ollama serve\n\n"
            "  Pull the model first if needed:  ollama pull mistral:7b\n",
            stacklevel=1,
        )

    # Start periodic background monitoring (1-hour cycle).
    # run_immediately=True: first pass executes in the thread pool right away,
    # keeping the event loop free (no blocking Ollama / SHAP calls on startup).
    _monitor_task = asyncio.create_task(
        _periodic_monitoring(app, run_immediately=True)
    )

    yield

    # Graceful shutdown
    _monitor_task.cancel()
    try:
        await _monitor_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="SepsisAlert API",
    version="1.0.0",
    description="ICU sepsis early-warning system — REST API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(patients.router, prefix="/api")
app.include_router(narrative.router, prefix="/api")
app.include_router(feedback.router, prefix="/api")
app.include_router(stats.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
