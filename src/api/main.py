"""
SepsisAlert FastAPI Backend

Start the server:
    uvicorn src.api.main:app --reload --port 8000

Start the frontend:
    cd frontend && npm install && npm run dev
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path

import sklearn  # noqa: F401 — must be imported before joblib deserialises sklearn models
import pandas as pd
import joblib
import yaml

# Project root — always resolve to ATML_Sepsis_Alert/ regardless of cwd
ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(ROOT)  # ensure all relative paths (parquet, config) resolve correctly
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import patients, narrative, feedback, stats


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, data and predictions at startup."""
    cfg = load_config()

    # Load model artifact (relative paths now safe — os.chdir(ROOT) at module level)
    artifact = joblib.load(cfg["model"]["artifact_path"])

    # Load feature + cohort data
    features_df = pd.read_parquet("data/processed/features.parquet")
    cohort_df = pd.read_parquet("data/processed/cohort.parquet")

    # Sample 100 patients, run predictions on features only
    from src.model.predict import predict_batch  # noqa: PLC0415
    sampled_features = features_df.sample(n=min(100, len(features_df)), random_state=99)
    predictions = predict_batch(sampled_features, artifact)

    # Merge display columns from cohort (age, gender, first_careunit)
    display_cols = ["stay_id"] + [c for c in ["age", "gender", "first_careunit"] if c in cohort_df.columns]
    predictions = predictions.merge(cohort_df[display_cols], on="stay_id", how="left")

    # Store in app state
    app.state.cfg = cfg
    app.state.artifact = artifact
    app.state.predictions = predictions
    app.state.features_df = features_df
    app.state.cohort_df = cohort_df

    yield


app = FastAPI(
    title="SepsisAlert API",
    version="1.0.0",
    description="ICU sepsis early-warning system — REST API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
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
