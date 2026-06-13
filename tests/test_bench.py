"""Unit tests for bench.py — no filesystem or network required."""

from __future__ import annotations

import pytest

from humanloopbench.bench import compute_metrics
from humanloopbench.schemas import Annotation, FailureCandidate


def _candidate(sample_id: str, model: str, category: str, score: float) -> FailureCandidate:
    return FailureCandidate(
        output_id=f"{sample_id}::{model}",
        category=category,
        score=score,
        explanation="test explanation",
    )


def _annotation(sample_id: str, model: str, category: str, label: str) -> Annotation:
    from humanloopbench.schemas import candidate_id as make_cid

    cid = make_cid(f"{sample_id}::{model}", category)
    return Annotation(candidate_id=cid, label=label)


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #


def test_metrics_empty_candidates():
    df = compute_metrics([], [])
    assert list(df.columns) == ["model", "category", "total", "confirmed", "failure_rate"]
    assert len(df) == 0


def test_metrics_no_confirmed():
    candidates = [_candidate("q1", "model-a", "hallucination", 0.8)]
    annotations = [_annotation("q1", "model-a", "hallucination", "rejected")]
    df = compute_metrics(candidates, annotations)
    assert df.iloc[0]["confirmed"] == 0
    assert df.iloc[0]["failure_rate"] == 0.0


def test_metrics_all_confirmed():
    candidates = [
        _candidate("q1", "model-a", "hallucination", 0.9),
        _candidate("q2", "model-a", "hallucination", 0.8),
    ]
    annotations = [
        _annotation("q1", "model-a", "hallucination", "confirmed"),
        _annotation("q2", "model-a", "hallucination", "confirmed"),
    ]
    df = compute_metrics(candidates, annotations)
    row = df[df["model"] == "model-a"].iloc[0]
    assert row["total"] == 2
    assert row["confirmed"] == 2
    assert row["failure_rate"] == pytest.approx(1.0)


def test_metrics_multiple_models_and_categories():
    candidates = [
        _candidate("q1", "model-a", "hallucination", 0.9),
        _candidate("q1", "model-a", "factual_error", 0.7),
        _candidate("q2", "model-b", "hallucination", 0.6),
        _candidate("q3", "model-b", "hallucination", 0.5),
    ]
    annotations = [
        _annotation("q1", "model-a", "hallucination", "confirmed"),
        _annotation("q1", "model-a", "factual_error", "rejected"),
        _annotation("q2", "model-b", "hallucination", "confirmed"),
        _annotation("q3", "model-b", "hallucination", "rejected"),
    ]
    df = compute_metrics(candidates, annotations)

    # model-a hallucination: 1/1 = 1.0
    row = df[(df["model"] == "model-a") & (df["category"] == "hallucination")].iloc[0]
    assert row["failure_rate"] == pytest.approx(1.0)

    # model-a factual_error: 0/1 = 0.0
    row = df[(df["model"] == "model-a") & (df["category"] == "factual_error")].iloc[0]
    assert row["failure_rate"] == pytest.approx(0.0)

    # model-b hallucination: 1/2 = 0.5
    row = df[(df["model"] == "model-b") & (df["category"] == "hallucination")].iloc[0]
    assert row["failure_rate"] == pytest.approx(0.5)


def test_metrics_unannotated_candidates_not_counted_as_confirmed():
    candidates = [_candidate("q1", "model-a", "hallucination", 0.9)]
    df = compute_metrics(candidates, [])  # no annotations
    assert df.iloc[0]["confirmed"] == 0
    assert df.iloc[0]["failure_rate"] == pytest.approx(0.0)
