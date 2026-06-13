"""Core data model for HumanLoopBench.

Everything flows through these types:
detectors emit FailureCandidate, humans produce Annotation,
the benchmark = confirmed candidates.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FailureCategory = Literal["hallucination", "refusal", "factual_error"]
AnnotationLabel = Literal["confirmed", "rejected", "unsure"]


class Sample(BaseModel):
    """One grounded-QA item: a question to be answered from `context`."""

    id: str
    question: str
    context: str
    reference: str = ""  # gold answer; empty when unanswerable
    answerable: bool = True  # False = context does NOT support any answer (SQuAD v2)


class ModelOutput(BaseModel):
    """A model's answer to a Sample."""

    sample_id: str
    model: str
    text: str


class FailureCandidate(BaseModel):
    """A suspected failure flagged by a detector. Always carries an explanation."""

    output_id: str  # f"{sample_id}::{model}"
    category: FailureCategory
    score: float = Field(ge=0.0, le=1.0)  # higher = more likely a failure
    explanation: str

    @property
    def sample_id(self) -> str:
        return self.output_id.split("::", 1)[0]


class Annotation(BaseModel):
    """A human judgment on a FailureCandidate."""

    candidate_id: str  # f"{output_id}::{category}"
    label: AnnotationLabel
    notes: str = ""
    annotator: str = "human"


def output_id(sample_id: str, model: str) -> str:
    return f"{sample_id}::{model}"


def candidate_id(output_id_: str, category: FailureCategory) -> str:
    return f"{output_id_}::{category}"


class ValidationResult(BaseModel):
    """Detector prediction vs. expert gold label (for precision/recall on RAGTruth)."""

    candidate_id: str
    predicted_failure: bool  # detector said this is a failure
    gold_failure: bool  # expert label says it is a failure
