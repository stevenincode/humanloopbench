"""Detector validation against RAGTruth expert hallucination labels.

Loads ~150 QA items from wandb/RAGTruth-processed, runs the grounding
detector, and compares predicted failures vs. gold span-level labels to
produce precision, recall, and F1.

Do NOT redistribute RAGTruth data — use locally for validation only.
The dataset is loaded from HF at runtime and never written to data/.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from humanloopbench.detectors import detect_grounding
from humanloopbench.schemas import ModelOutput, Sample, ValidationResult

REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "validation_report.json"

# Score threshold for calling a prediction a "failure".
# Kept at 0.5 for validation (more conservative than the 0.3 queue filter).
PREDICT_THRESHOLD = 0.5

GenerateFn = Callable[..., str]


# --------------------------------------------------------------------------- #
# RAGTruth loader
# --------------------------------------------------------------------------- #


def _get_field(row: dict, *candidates: str, default: str = "") -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            return str(row[key])
    return default


def load_ragtruth(limit: int = 150) -> list[tuple[Sample, ModelOutput, bool]]:
    """Load RAGTruth summarization slice from HF (wandb/RAGTruth-processed).

    Returns list of (Sample, ModelOutput, gold_failure) triples.
    gold_failure=True when hallucination_labels is non-empty (expert span labels).
    Dataset is summarization-based; 'question' holds the summarization prompt.
    """
    import json as _json

    from datasets import load_dataset

    ds = load_dataset("wandb/RAGTruth-processed", split="train")

    items: list[tuple[Sample, ModelOutput, bool]] = []
    for i, row in enumerate(ds):
        if len(items) >= limit:
            break

        row = dict(row)

        question = _get_field(row, "query", "question", "input")
        context = _get_field(row, "context", "passage", "source", "document")
        response = _get_field(row, "output", "response", "answer", "generated_text")

        if not question or not context or not response:
            continue

        # hallucination_labels is a JSON string of span dicts; non-empty = hallucination
        raw = row.get("hallucination_labels", "[]")
        try:
            labels = _json.loads(raw) if isinstance(raw, str) else (raw or [])
        except (_json.JSONDecodeError, TypeError):
            labels = []
        gold_failure = len(labels) > 0

        sample_id = f"ragtruth_{i}"
        sample = Sample(
            id=sample_id,
            question=question,
            context=context,
            reference="",
            answerable=True,
        )
        output = ModelOutput(sample_id=sample_id, model="ragtruth", text=response)
        items.append((sample, output, gold_failure))

    return items


# --------------------------------------------------------------------------- #
# Run validation
# --------------------------------------------------------------------------- #


def run_validation(
    items: list[tuple[Sample, ModelOutput, bool]],
    generate_fn: GenerateFn,
    *,
    predict_threshold: float = PREDICT_THRESHOLD,
    workers: int = 3,
) -> list[ValidationResult]:
    """Run grounding detector on each item and return ValidationResult list."""
    import concurrent.futures

    from tqdm import tqdm

    from humanloopbench.schemas import candidate_id as make_cid
    from humanloopbench.schemas import output_id as make_oid

    def _validate_one(item: tuple[Sample, ModelOutput, bool]) -> ValidationResult:
        sample, output, gold_failure = item
        candidate = detect_grounding(
            sample, output, generate_fn, two_step=False, min_score=0.0,
        )
        oid = make_oid(sample.id, output.model)
        cid = make_cid(oid, "hallucination")
        predicted = candidate is not None and candidate.score >= predict_threshold
        return ValidationResult(
            candidate_id=cid, predicted_failure=predicted, gold_failure=gold_failure
        )

    results: list[ValidationResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(
            pool.map(_validate_one, items), total=len(items), desc="validating", unit="item"
        ):
            results.append(result)

    return results


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_validation_metrics(results: list[ValidationResult]) -> dict:
    """Compute precision, recall, F1, and accuracy from ValidationResult list."""
    tp = sum(1 for r in results if r.predicted_failure and r.gold_failure)
    fp = sum(1 for r in results if r.predicted_failure and not r.gold_failure)
    fn = sum(1 for r in results if not r.predicted_failure and r.gold_failure)
    tn = sum(1 for r in results if not r.predicted_failure and not r.gold_failure)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(results) if results else 0.0

    return {
        "total": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "predict_threshold": PREDICT_THRESHOLD,
        "gold_positive_rate": round(sum(1 for r in results if r.gold_failure) / len(results), 4)
        if results
        else 0.0,
    }


# --------------------------------------------------------------------------- #
# Save / load report
# --------------------------------------------------------------------------- #


def save_report(metrics: dict, results: list[ValidationResult]) -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": metrics,
        "results": [r.model_dump() for r in results],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return REPORT_PATH


def load_report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
