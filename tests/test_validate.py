"""Unit tests for validate.py — no network or live judge required."""

from __future__ import annotations

import json

import pytest

from humanloopbench.schemas import ModelOutput, Sample
from humanloopbench.validate import compute_validation_metrics, run_validation

# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #


def _stub_judge(score: float):
    body = json.dumps({"score": score, "explanation": "stubbed"})

    def _fn(model: str, prompt: str, *, system: str | None = None) -> str:  # noqa: ARG001
        return body

    return _fn


def _item(sample_id: str, answer: str, gold_failure: bool) -> tuple:
    sample = Sample(
        id=sample_id,
        question="What is X?",
        context="X is described in this passage.",
        reference="",
        answerable=True,
    )
    output = ModelOutput(sample_id=sample_id, model="ragtruth", text=answer)
    return sample, output, gold_failure


# --------------------------------------------------------------------------- #
# compute_validation_metrics
# --------------------------------------------------------------------------- #


def test_metrics_perfect_detector():
    from humanloopbench.schemas import ValidationResult

    results = [
        ValidationResult(candidate_id="a", predicted_failure=True, gold_failure=True),
        ValidationResult(candidate_id="b", predicted_failure=False, gold_failure=False),
    ]
    m = compute_validation_metrics(results)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_metrics_all_false_positives():
    from humanloopbench.schemas import ValidationResult

    results = [
        ValidationResult(candidate_id="a", predicted_failure=True, gold_failure=False),
        ValidationResult(candidate_id="b", predicted_failure=True, gold_failure=False),
    ]
    m = compute_validation_metrics(results)
    assert m["precision"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)  # no TP, no FN → recall=0
    assert m["f1"] == pytest.approx(0.0)


def test_metrics_all_false_negatives():
    from humanloopbench.schemas import ValidationResult

    results = [
        ValidationResult(candidate_id="a", predicted_failure=False, gold_failure=True),
        ValidationResult(candidate_id="b", predicted_failure=False, gold_failure=True),
    ]
    m = compute_validation_metrics(results)
    assert m["precision"] == pytest.approx(0.0)  # no TP → precision=0
    assert m["recall"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)


def test_metrics_mixed():
    from humanloopbench.schemas import ValidationResult

    # TP=2, FP=1, FN=1, TN=1  →  P=2/3, R=2/3, F1=2/3
    results = [
        ValidationResult(candidate_id="a", predicted_failure=True, gold_failure=True),
        ValidationResult(candidate_id="b", predicted_failure=True, gold_failure=True),
        ValidationResult(candidate_id="c", predicted_failure=True, gold_failure=False),  # FP
        ValidationResult(candidate_id="d", predicted_failure=False, gold_failure=True),  # FN
        ValidationResult(candidate_id="e", predicted_failure=False, gold_failure=False),
    ]
    m = compute_validation_metrics(results)
    assert m["tp"] == 2
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_metrics_empty():
    m = compute_validation_metrics([])
    assert m["total"] == 0
    assert m["f1"] == 0.0


# --------------------------------------------------------------------------- #
# run_validation
# --------------------------------------------------------------------------- #


def test_run_validation_high_score_predicts_failure():
    items = [_item("q1", "Some hallucinated answer.", gold_failure=True)]
    results = run_validation(items, _stub_judge(0.9), predict_threshold=0.5)
    assert len(results) == 1
    assert results[0].predicted_failure is True
    assert results[0].gold_failure is True


def test_run_validation_low_score_predicts_non_failure():
    items = [_item("q1", "Correct grounded answer.", gold_failure=False)]
    results = run_validation(items, _stub_judge(0.1), predict_threshold=0.5)
    assert len(results) == 1
    assert results[0].predicted_failure is False
    assert results[0].gold_failure is False


def test_run_validation_threshold_respected():
    items = [_item("q1", "Borderline answer.", gold_failure=True)]
    # score=0.5, threshold=0.5 → predicted True (>= threshold)
    r_above = run_validation(items, _stub_judge(0.5), predict_threshold=0.5)
    assert r_above[0].predicted_failure is True

    # score=0.49, threshold=0.5 → predicted False
    r_below = run_validation(items, _stub_judge(0.49), predict_threshold=0.5)
    assert r_below[0].predicted_failure is False


def test_run_validation_candidate_id_format():
    items = [_item("ragtruth_0", "Some answer.", gold_failure=True)]
    results = run_validation(items, _stub_judge(0.8))
    assert "ragtruth_0" in results[0].candidate_id
    assert "hallucination" in results[0].candidate_id
