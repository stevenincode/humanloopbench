"""Grounded-QA task: load samples, build prompts, run a model over them.

Two sources:
  - "demo"     : small bundled fixture (data/samples_demo.jsonl) — works offline, no deps.
  - "squad_v2" : rajpurkar/squad_v2 from Hugging Face (CC BY-SA 4.0). Needs `datasets`
                 and network. Mixes answerable + unanswerable items; the unanswerable
                 ones are the cleanest grounding-failure targets.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from humanloopbench.schemas import ModelOutput, Sample, output_id

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RUNS_DIR = DATA_DIR / "runs"
SAMPLES_DIR = DATA_DIR / "samples"
DEMO_FILE = DATA_DIR / "samples_demo.jsonl"

# The model is told it MAY refuse when the context lacks the answer. A confident
# answer to an unanswerable question is then a clean grounding failure.
SYSTEM = (
    "You answer strictly from the provided context. "
    "If the context does not contain the answer, reply exactly: NOT IN CONTEXT."
)
PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def build_prompt(sample: Sample) -> str:
    return PROMPT_TEMPLATE.format(context=sample.context, question=sample.question)


def load_demo(limit: int | None = None) -> list[Sample]:
    lines = DEMO_FILE.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    samples = [Sample(**row) for row in rows]
    return samples[:limit] if limit else samples


def load_squad_v2(limit: int = 150, *, split: str = "validation") -> list[Sample]:
    """Real path. Balances answerable/unanswerable roughly 50/50 up to `limit`."""
    from datasets import load_dataset  # imported lazily so demo path needs no install

    ds = load_dataset("rajpurkar/squad_v2", split=split)
    answerable, unanswerable = [], []
    for row in ds:
        ans = row["answers"]["text"]
        s = Sample(
            id=str(row["id"]),
            question=row["question"],
            context=row["context"],
            reference=ans[0] if ans else "",
            answerable=bool(ans),
        )
        (answerable if s.answerable else unanswerable).append(s)
        if len(answerable) >= limit // 2 and len(unanswerable) >= limit // 2:
            break
    half = limit // 2
    mixed = answerable[:half] + unanswerable[: limit - half]
    return mixed


def load_samples(source: str = "demo", limit: int = 150) -> list[Sample]:
    if source == "demo":
        return load_demo(limit)
    if source == "squad_v2":
        return load_squad_v2(limit)
    raise ValueError(f"unknown source: {source!r} (use 'demo' or 'squad_v2')")


def run_samples(
    samples: list[Sample],
    model: str,
    generate_fn: Callable[[str, str], str],
) -> list[ModelOutput]:
    """Generate an answer per sample. `generate_fn(model, prompt) -> text`.

    Injecting generate_fn keeps this unit-testable without network or a live model.
    """
    from tqdm import tqdm

    outputs: list[ModelOutput] = []
    for s in tqdm(samples, desc=f"querying {model}", unit="sample"):
        text = generate_fn(model, build_prompt(s))
        outputs.append(ModelOutput(sample_id=s.id, model=model, text=text))
    return outputs


def _model_slug(model: str, source: str) -> str:
    return f"{model.replace('/', '_').replace(':', '_')}__{source}"


def save_outputs(outputs: list[ModelOutput], model: str, source: str = "demo") -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{_model_slug(model, source)}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for o in outputs:
            f.write(o.model_dump_json() + "\n")
    return path


def load_outputs(model: str, source: str = "squad_v2") -> list[ModelOutput]:
    path = RUNS_DIR / f"{_model_slug(model, source)}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [ModelOutput(**r) for r in rows]


def save_samples_snapshot(samples: list[Sample], source: str) -> Path:
    """Persist sample metadata so the review UI can display question + context.

    Merges into any existing snapshot (keyed by sample id) instead of replacing it,
    so a small `run --limit N` never drops samples that earlier runs depend on.
    Existing order is kept; new ids are appended; re-saved ids are updated in place.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"{source}.jsonl"
    merged: dict[str, Sample] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing = Sample(**json.loads(line))
                merged[existing.id] = existing
    for s in samples:
        merged[s.id] = s
    with path.open("w", encoding="utf-8") as f:
        for s in merged.values():
            f.write(s.model_dump_json() + "\n")
    return path


def load_samples_snapshot(source: str, limit: int | None = None) -> list[Sample] | None:
    """Load a previously saved snapshot; returns None if not found (caller falls back to HF)."""
    path = SAMPLES_DIR / f"{source}.jsonl"
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    samples = [Sample(**r) for r in rows]
    return samples[:limit] if limit else samples


def load_all_samples() -> dict[str, Sample]:
    """Load all saved sample snapshots and return a {sample_id: Sample} map."""
    samples: dict[str, Sample] = {}
    if not SAMPLES_DIR.exists():
        return samples
    for path in SAMPLES_DIR.glob("*.jsonl"):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        for r in rows:
            s = Sample(**r)
            samples[s.id] = s
    return samples


# id helper re-exported for convenience
__all__ = [
    "SYSTEM",
    "build_prompt",
    "load_samples",
    "load_demo",
    "load_squad_v2",
    "run_samples",
    "save_outputs",
    "load_outputs",
    "save_samples_snapshot",
    "load_all_samples",
    "output_id",
]
