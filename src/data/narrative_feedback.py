"""
Narrative quality feedback storage with few-shot and RAG retrieval.

Each record stores the clinician's star rating, optional correction note,
the full narrative text, the SHAP summary, and the SHAP feature vector.

The SHAP vector enables cosine-similarity search so the most clinically
similar past patient can be surfaced as a RAG context example.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path("logs/narrative_feedback.jsonl")


# ------------------------------------------------------------------ #
# Write                                                                #
# ------------------------------------------------------------------ #

def save_narrative_feedback(
    stay_id: int,
    rating: int,
    correction_note: str,
    narrative_text: str,
    shap_summary: str,
    model_used: str,
    shap_vector: dict[str, float],
) -> None:
    """
    Append a narrative feedback record to the log.

    Parameters
    ----------
    stay_id         : ICU stay identifier
    rating          : 1–5 star rating from clinician
    correction_note : free-text correction (may be empty)
    narrative_text  : full LLM narrative that was rated
    shap_summary    : formatted SHAP string sent to the LLM
    model_used      : Ollama model name (e.g. "mistral:7b")
    shap_vector     : dict mapping feature label → SHAP value
    """
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stay_id": int(stay_id),
        "rating": int(rating),
        "correction_note": correction_note,
        "narrative_text": narrative_text,
        "shap_summary": shap_summary,
        "model_used": model_used,
        "shap_vector": {k: float(v) for k, v in shap_vector.items()} if shap_vector else {},
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ #
# Read helpers                                                         #
# ------------------------------------------------------------------ #

def _load_all() -> list[dict]:
    """Load all records from the feedback log."""
    if not _LOG_PATH.exists():
        return []
    records = []
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_few_shot_examples(
    min_rating: int = 4,
    max_examples: int = 2,
    model_used: str | None = None,
) -> list[dict]:
    """
    Return up to `max_examples` high-rated narratives for few-shot prompting.

    Filtered by `min_rating` and optionally by `model_used`.
    Returned in descending rating order.
    """
    records = _load_all()
    filtered = [
        r for r in records
        if r.get("rating", 0) >= min_rating
        and (model_used is None or r.get("model_used") == model_used)
        and r.get("narrative_text", "").strip()
    ]
    filtered.sort(key=lambda r: r["rating"], reverse=True)
    return filtered[:max_examples]


def find_similar_narratives(
    current_shap_vector: dict[str, float],
    top_n: int = 1,
    min_rating: int = 4,
    model_used: str | None = None,
) -> list[dict]:
    """
    Return the `top_n` most clinically similar past narratives by cosine
    similarity of SHAP vectors.

    Parameters
    ----------
    current_shap_vector : feature label → SHAP value for the current patient
    top_n               : number of results to return
    min_rating          : only consider records at or above this rating
    model_used          : if set, only match records from this model

    Returns
    -------
    List of record dicts, each augmented with a "similarity" float (0–1).
    """
    records = _load_all()
    candidates = [
        r for r in records
        if r.get("rating", 0) >= min_rating
        and r.get("shap_vector")
        and r.get("narrative_text", "").strip()
        and (model_used is None or r.get("model_used") == model_used)
    ]

    if not candidates:
        return []

    scored = []
    for record in candidates:
        sim = _cosine_similarity(current_shap_vector, record["shap_vector"])
        scored.append({**record, "similarity": sim})

    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:top_n]


def export_finetune_data(min_rating: int = 4) -> Path:
    """
    Export high-rated narrative feedback as Alpaca-format JSONL for LoRA fine-tuning.

    Each record becomes one instruction/input/output training pair.
    Raises ValueError if fewer than 5 records meet the threshold.

    Returns the path to the exported JSONL file.
    """
    records = _load_all()
    eligible = [
        r for r in records
        if r.get("rating", 0) >= min_rating
        and r.get("narrative_text", "").strip()
        and r.get("shap_summary", "").strip()
    ]

    if len(eligible) < 5:
        raise ValueError(
            f"Only {len(eligible)} records meet min_rating={min_rating}. "
            "Collect more clinician ratings in the dashboard before fine-tuning."
        )

    out_path = Path("data/feedback/finetune_pairs.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instruction = (
        "You are a clinical decision support assistant for ICU sepsis monitoring. "
        "Generate a concise SBAR-structured alert for bedside nursing staff based on "
        "the SHAP feature importance values provided. Never make definitive diagnoses "
        "or prescribe specific treatments."
    )

    with out_path.open("w", encoding="utf-8") as f:
        for r in eligible:
            pair = {
                "instruction": instruction,
                "input":       r["shap_summary"],
                "output":      r["narrative_text"],
            }
            f.write(json.dumps(pair) + "\n")

    return out_path


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse feature vectors (dicts).

    Only features present in both vectors contribute to the dot product.
    Returns 0.0 if either vector has zero magnitude.
    """
    common_keys = set(a) & set(b)
    if not common_keys:
        return 0.0

    dot = sum(a[k] * b[k] for k in common_keys)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
