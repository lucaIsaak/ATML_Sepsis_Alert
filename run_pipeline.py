"""
SepsisAlert — Full Pipeline Runner

Runs all steps in order:
    1. Extract cohort
    2. Engineer features
    3. Train model
    4. Evaluate model

Usage:
    python run_pipeline.py              # run all steps
    python run_pipeline.py --from train # skip data steps, start from training
"""

import argparse
import sys
from pathlib import Path


STEPS = ["cohort", "features", "train", "evaluate"]


def run_cohort():
    print("\n" + "="*50)
    print("STEP 1/4 — Cohort Extraction")
    print("="*50)
    from src.data.cohort import load_config, extract_cohort, save_cohort
    cfg = load_config()
    df = extract_cohort(cfg)
    save_cohort(df, cfg)


def run_features():
    print("\n" + "="*50)
    print("STEP 2/4 — Feature Engineering")
    print("="*50)
    import pandas as pd
    from src.data.features import load_config, extract_features, save_features
    cfg = load_config()
    cohort = pd.read_parquet(Path(cfg["data"]["processed_path"]) / "cohort.parquet")
    features = extract_features(cohort, cfg)
    save_features(features, cfg)


def run_train():
    print("\n" + "="*50)
    print("STEP 3/4 — Model Training")
    print("="*50)
    from src.model.train import train
    train()


def run_evaluate():
    print("\n" + "="*50)
    print("STEP 4/4 — Model Evaluation")
    print("="*50)
    from src.model.evaluate import evaluate
    evaluate()


STEP_FNS = {
    "cohort": run_cohort,
    "features": run_features,
    "train": run_train,
    "evaluate": run_evaluate,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=STEPS,
        default="cohort",
        help="Start pipeline from this step (skips earlier steps)",
    )
    parser.add_argument(
        "--only",
        choices=STEPS,
        help="Run only this single step",
    )
    args = parser.parse_args()

    if args.only:
        steps_to_run = [args.only]
    else:
        start_idx = STEPS.index(args.start_from)
        steps_to_run = STEPS[start_idx:]

    print(f"Running steps: {' → '.join(steps_to_run)}")

    for step in steps_to_run:
        try:
            STEP_FNS[step]()
        except Exception as e:
            print(f"\n❌ Step '{step}' failed: {e}")
            sys.exit(1)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
