"""LLM-as-Judge failure detectors.

Two detectors:
  grounding  — answer not supported by context (strong prior when answerable=False).
  reasoning  — answer factually wrong vs. reference (skipped when no reference).

Both return FailureCandidate | None. Only candidates with score >= MIN_SCORE are emitted.
Every candidate carries an explanation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from humanloopbench.schemas import FailureCandidate, ModelOutput, Sample, output_id

JUDGE_MODEL = "gpt-oss:120b"
MIN_SCORE = 0.3
BORDERLINE_MAX = 0.7  # scores in [MIN_SCORE, BORDERLINE_MAX] trigger a second verification call

CANDIDATES_DIR = Path(__file__).resolve().parents[2] / "data" / "candidates"

# Type alias: compatible with models.generate(model, prompt, *, system=...) -> str
GenerateFn = Callable[..., str]

# --------------------------------------------------------------------------- #
# Judge prompts
# --------------------------------------------------------------------------- #

_GROUNDING_SYSTEM = (
    "You are a hallucination detector. Determine whether the model answer is fully "
    "supported by the context.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"score": <float 0.0-1.0>, "explanation": "<one sentence>"}\n\n'
    "score = probability this is a GROUNDING FAILURE (not supported by context). "
    "1.0 = definite failure, 0.0 = fully grounded."
)

_GROUNDING_PROMPT = """\
Context:
{context}

Question: {question}

Model answer: {answer}

Context contains the answer: {answerable}

Respond with JSON only."""

_REASONING_SYSTEM = (
    "You are a factual accuracy judge. Compare the model answer to the reference "
    "answer and decide if the model answer is factually wrong or misleading.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"score": <float 0.0-1.0>, "explanation": "<one sentence>"}\n\n'
    "score = probability this is a REASONING FAILURE (factually wrong). "
    "1.0 = clearly wrong, 0.0 = correct."
)

_REASONING_PROMPT = """\
Question: {question}

Reference answer: {reference}

Model answer: {answer}

Respond with JSON only."""

# Agent B: verification prompt for borderline scores
_VERIFY_SYSTEM = (
    "You are a careful second-opinion reviewer. A previous judge estimated a failure "
    "probability for a model answer. Re-examine it skeptically: is there any reasonable "
    "interpretation under which the answer is acceptable?\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"score": <float 0.0-1.0>, "explanation": "<one sentence>"}\n\n'
    "score = your revised probability this is a genuine failure."
)

_VERIFY_PROMPT = """\
Previous failure probability: {prev_score:.2f}
Previous reasoning: {prev_explanation}

--- Original assessment context ---
{context}

Reconsider and respond with JSON only."""


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _parse_judge(text: str) -> tuple[float, str]:
    """Parse JSON from judge response -> (score, explanation). Clamps score to [0,1]."""
    match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            explanation = str(data.get("explanation", text)).strip()
            return score, explanation
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return 0.5, text.strip()


def _is_refusal(text: str) -> bool:
    return "NOT IN CONTEXT" in text.upper()


def _verify_score(
    prev_score: float,
    prev_explanation: str,
    context_summary: str,
    generate_fn: GenerateFn,
    judge_model: str,
) -> tuple[float, str]:
    """Second verification call for borderline scores. Returns averaged (score, explanation)."""
    prompt = _VERIFY_PROMPT.format(
        prev_score=prev_score,
        prev_explanation=prev_explanation,
        context=context_summary,
    )
    raw = generate_fn(judge_model, prompt, system=_VERIFY_SYSTEM)
    score2, explanation2 = _parse_judge(raw)
    avg_score = round((prev_score + score2) / 2, 4)
    return avg_score, explanation2


# --------------------------------------------------------------------------- #
# Public detector functions
# --------------------------------------------------------------------------- #


def detect_grounding(
    sample: Sample,
    output: ModelOutput,
    generate_fn: GenerateFn,
    *,
    judge_model: str = JUDGE_MODEL,
    min_score: float = MIN_SCORE,
    two_step: bool = True,
) -> FailureCandidate | None:
    """Flag answers not supported by context.

    If answerable=False and the model did NOT say NOT IN CONTEXT, score is
    boosted by 0.3 (strong prior: confident answer to unanswerable question).
    If answerable=False and the model correctly refused, returns None immediately.
    With two_step=True (default), borderline scores trigger a second verification call.
    """
    answer = output.text.strip()
    refusal = _is_refusal(answer)

    if not sample.answerable and refusal:
        return None  # correct behaviour — not a failure

    prompt = _GROUNDING_PROMPT.format(
        context=sample.context,
        question=sample.question,
        answer=answer,
        answerable="Yes" if sample.answerable else "No",
    )
    raw = generate_fn(judge_model, prompt, system=_GROUNDING_SYSTEM)
    score, explanation = _parse_judge(raw)

    if not sample.answerable and not refusal:
        score = min(1.0, score + 0.3)

    if two_step and min_score <= score <= BORDERLINE_MAX:
        ctx = f"Q: {sample.question}\nAnswer: {answer}\nContext snippet: {sample.context[:300]}"
        score, explanation = _verify_score(score, explanation, ctx, generate_fn, judge_model)

    if score < min_score:
        return None

    return FailureCandidate(
        output_id=output_id(sample.id, output.model),
        category="hallucination",
        score=round(score, 4),
        explanation=explanation,
    )


def detect_reasoning(
    sample: Sample,
    output: ModelOutput,
    generate_fn: GenerateFn,
    *,
    judge_model: str = JUDGE_MODEL,
    min_score: float = MIN_SCORE,
    two_step: bool = True,
) -> FailureCandidate | None:
    """Flag answers that are factually wrong vs. the reference.

    Skips unanswerable samples (no reference to compare against).
    If the model refused (NOT IN CONTEXT) when the context does contain the answer,
    the score is boosted by 0.2.
    With two_step=True (default), borderline scores trigger a second verification call.
    """
    if not sample.answerable or not sample.reference:
        return None

    answer = output.text.strip()
    refusal = _is_refusal(answer)

    prompt = _REASONING_PROMPT.format(
        question=sample.question,
        reference=sample.reference,
        answer=answer,
    )
    raw = generate_fn(judge_model, prompt, system=_REASONING_SYSTEM)
    score, explanation = _parse_judge(raw)

    if refusal:
        score = min(1.0, score + 0.2)
        explanation = f"Model said NOT IN CONTEXT but a reference answer exists. {explanation}"

    if two_step and min_score <= score <= BORDERLINE_MAX:
        ctx = f"Q: {sample.question}\nReference: {sample.reference}\nAnswer: {answer}"
        score, explanation = _verify_score(score, explanation, ctx, generate_fn, judge_model)

    if score < min_score:
        return None

    category = "refusal" if refusal else "factual_error"
    return FailureCandidate(
        output_id=output_id(sample.id, output.model),
        category=category,
        score=round(score, 4),
        explanation=explanation,
    )


def run_detectors(
    samples: list[Sample],
    outputs: list[ModelOutput],
    generate_fn: GenerateFn,
    *,
    judge_model: str = JUDGE_MODEL,
) -> list[FailureCandidate]:
    """Run both detectors over all (sample, output) pairs. Skips unmatched outputs."""
    sample_map = {s.id: s for s in samples}
    candidates: list[FailureCandidate] = []

    for out in outputs:
        sample = sample_map.get(out.sample_id)
        if sample is None:
            continue
        detector = detect_grounding if not sample.answerable else detect_reasoning
        c = detector(sample, out, generate_fn, judge_model=judge_model)
        if c is not None:
            candidates.append(c)

    return candidates


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def save_candidates(
    candidates: list[FailureCandidate], model: str, source: str = "demo"
) -> Path:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{model.replace('/', '_').replace(':', '_')}__{source}"
    path = CANDIDATES_DIR / f"{slug}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.model_dump_json() + "\n")
    return path


def load_candidates(model: str, source: str = "squad_v2") -> list[FailureCandidate]:
    slug = f"{model.replace('/', '_').replace(':', '_')}__{source}"
    path = CANDIDATES_DIR / f"{slug}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [FailureCandidate(**r) for r in rows]
