"""Unit tests for detectors.py — all judge calls are stubbed (no network)."""

from __future__ import annotations

import json

import pytest

from humanloopbench.detectors import (
    _parse_judge,
    detect_grounding,
    detect_reasoning,
    run_detectors,
)
from humanloopbench.schemas import ModelOutput, Sample

# --------------------------------------------------------------------------- #
# Stub helpers
# --------------------------------------------------------------------------- #


def _stub_judge(score: float, explanation: str):
    """Return a generate_fn that always responds with fixed score + explanation."""
    body = json.dumps({"score": score, "explanation": explanation})

    def _fn(model: str, prompt: str, *, system: str | None = None) -> str:  # noqa: ARG001
        return body

    return _fn


_SAMPLE_ANSWERABLE = Sample(
    id="q1",
    question="What is the capital of France?",
    context="France is a country in Western Europe. Its capital is Paris.",
    reference="Paris",
    answerable=True,
)

_SAMPLE_UNANSWERABLE = Sample(
    id="q2",
    question="What is the population of Mars?",
    context="Mars is the fourth planet from the Sun.",
    reference="",
    answerable=False,
)

_OUTPUT_CORRECT = ModelOutput(sample_id="q1", model="test-model", text="Paris")
_OUTPUT_WRONG = ModelOutput(sample_id="q1", model="test-model", text="London")
_OUTPUT_REFUSAL = ModelOutput(sample_id="q2", model="test-model", text="NOT IN CONTEXT")
_OUTPUT_HALLUCINATED = ModelOutput(
    sample_id="q2", model="test-model", text="About 5 billion people."
)


# --------------------------------------------------------------------------- #
# _parse_judge
# --------------------------------------------------------------------------- #


def test_parse_judge_valid_json():
    raw = '{"score": 0.8, "explanation": "Answer unsupported."}'
    score, expl = _parse_judge(raw)
    assert score == pytest.approx(0.8)
    assert "unsupported" in expl.lower()


def test_parse_judge_json_in_prose():
    raw = 'Here is my assessment: {"score": 0.2, "explanation": "Grounded."} done.'
    score, expl = _parse_judge(raw)
    assert score == pytest.approx(0.2)


def test_parse_judge_clamps_out_of_range():
    raw = '{"score": 1.5, "explanation": "Way off."}'
    score, _ = _parse_judge(raw)
    assert score == pytest.approx(1.0)


def test_parse_judge_fallback_on_garbage():
    score, expl = _parse_judge("I have no idea")
    assert score == pytest.approx(0.5)
    assert expl == "I have no idea"


# --------------------------------------------------------------------------- #
# detect_grounding
# --------------------------------------------------------------------------- #


def test_grounding_correct_answer_below_threshold():
    # Judge says score=0.1 → below MIN_SCORE → None
    c = detect_grounding(_SAMPLE_ANSWERABLE, _OUTPUT_CORRECT, _stub_judge(0.1, "Grounded."))
    assert c is None


def test_grounding_hallucination_detected():
    # Answerable sample, judge returns high score
    stub = _stub_judge(0.85, "London not in context.")
    c = detect_grounding(_SAMPLE_ANSWERABLE, _OUTPUT_WRONG, stub)
    assert c is not None
    assert c.category == "hallucination"
    assert c.score == pytest.approx(0.85)
    assert c.explanation == "London not in context."
    assert c.output_id == "q1::test-model"


def test_grounding_unanswerable_correct_refusal_skipped():
    # Model says NOT IN CONTEXT for unanswerable sample → None (correct behaviour)
    c = detect_grounding(_SAMPLE_UNANSWERABLE, _OUTPUT_REFUSAL, _stub_judge(0.9, "irrelevant"))
    assert c is None


def test_grounding_unanswerable_hallucination_boosted():
    # Model gives an answer for an unanswerable question → score boosted by 0.3
    stub = _stub_judge(0.5, "Ungrounded.")
    c = detect_grounding(_SAMPLE_UNANSWERABLE, _OUTPUT_HALLUCINATED, stub)
    assert c is not None
    assert c.score == pytest.approx(0.8)  # 0.5 + 0.3


# --------------------------------------------------------------------------- #
# detect_reasoning
# --------------------------------------------------------------------------- #


def test_reasoning_skips_unanswerable_sample():
    c = detect_reasoning(_SAMPLE_UNANSWERABLE, _OUTPUT_HALLUCINATED, _stub_judge(0.9, "Wrong."))
    assert c is None


def test_reasoning_correct_answer_below_threshold():
    c = detect_reasoning(_SAMPLE_ANSWERABLE, _OUTPUT_CORRECT, _stub_judge(0.05, "Correct."))
    assert c is None


def test_reasoning_wrong_answer_detected():
    c = detect_reasoning(_SAMPLE_ANSWERABLE, _OUTPUT_WRONG, _stub_judge(0.9, "London ≠ Paris."))
    assert c is not None
    assert c.category == "factual_error"
    assert c.score == pytest.approx(0.9)
    assert "London" in c.explanation


def test_reasoning_refusal_on_answerable_boosted():
    # two_step=False isolates the refusal boost logic without the verification call
    output = ModelOutput(sample_id="q1", model="test-model", text="NOT IN CONTEXT")
    c = detect_reasoning(
        _SAMPLE_ANSWERABLE, output, _stub_judge(0.2, "Should have answered."), two_step=False
    )
    assert c is not None
    assert c.score == pytest.approx(0.4)  # 0.2 + 0.2 refusal boost
    assert "NOT IN CONTEXT" in c.explanation


# --------------------------------------------------------------------------- #
# run_detectors
# --------------------------------------------------------------------------- #


def test_run_detectors_unmatched_output_skipped():
    orphan = ModelOutput(sample_id="no-such-id", model="test-model", text="whatever")
    result = run_detectors(
        [_SAMPLE_ANSWERABLE],
        [orphan],
        _stub_judge(0.9, "bad"),
    )
    assert result == []


def test_run_detectors_answerable_uses_reasoning():
    # Answerable sample → reasoning detector → factual_error (MECE routing)
    result = run_detectors(
        [_SAMPLE_ANSWERABLE],
        [_OUTPUT_WRONG],
        _stub_judge(0.9, "Failure."),
    )
    assert len(result) == 1
    assert result[0].category == "factual_error"


def test_run_detectors_unanswerable_uses_grounding():
    # Unanswerable sample + model answered → grounding detector → hallucination
    result = run_detectors(
        [_SAMPLE_UNANSWERABLE],
        [_OUTPUT_HALLUCINATED],
        _stub_judge(0.9, "Failure."),
    )
    assert len(result) == 1
    assert result[0].category == "hallucination"


def test_run_detectors_output_id_format():
    result = run_detectors(
        [_SAMPLE_ANSWERABLE],
        [_OUTPUT_WRONG],
        _stub_judge(0.9, "Failure."),
    )
    for c in result:
        assert c.output_id == "q1::test-model"
        assert c.explanation  # every candidate must carry an explanation
