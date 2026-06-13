from humanloopbench.schemas import (
    Annotation,
    FailureCandidate,
    ModelOutput,
    Sample,
    ValidationResult,
    candidate_id,
    output_id,
)


def test_sample_roundtrip():
    s = Sample(
        id="q1",
        question="Capital of France?",
        context="France's capital is Paris.",
        reference="Paris",
    )
    assert s.question.startswith("Capital")
    assert Sample(**s.model_dump()) == s


def test_output_and_candidate_ids():
    oid = output_id("q1", "gpt-oss")
    assert oid == "q1::gpt-oss"
    cid = candidate_id(oid, "hallucination")
    assert cid == "q1::gpt-oss::hallucination"


def test_failure_candidate_sample_id_and_bounds():
    c = FailureCandidate(
        output_id="q1::gpt-oss",
        category="hallucination",
        score=0.9,
        explanation="Answer not supported by context.",
    )
    assert c.sample_id == "q1"
    assert 0.0 <= c.score <= 1.0


def test_model_output_and_annotation():
    o = ModelOutput(sample_id="q1", model="gpt-oss", text="Paris")
    assert o.model == "gpt-oss"
    a = Annotation(candidate_id="q1::gpt-oss::hallucination", label="confirmed")
    assert a.label == "confirmed"
    assert a.annotator == "human"


def test_validation_result():
    vr = ValidationResult(
        candidate_id="q1::gpt-oss::hallucination",
        predicted_failure=True,
        gold_failure=True,
    )
    assert vr.predicted_failure is True
    assert vr.gold_failure is True
