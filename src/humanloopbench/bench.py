"""Benchmark export + per-model/per-category failure rate metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from humanloopbench.detectors import CANDIDATES_DIR
from humanloopbench.schemas import Annotation, FailureCandidate
from humanloopbench.schemas import candidate_id as make_cid

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark"


def load_all_candidates() -> list[FailureCandidate]:
    """Load candidates from every model file in data/candidates/."""
    candidates: list[FailureCandidate] = []
    if not CANDIDATES_DIR.exists():
        return candidates
    for path in sorted(CANDIDATES_DIR.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        candidates.extend(FailureCandidate(**r) for r in rows)
    return candidates


def compute_metrics(
    candidates: list[FailureCandidate],
    annotations: list[Annotation],
) -> pd.DataFrame:
    """Return per-model × per-category failure rates for confirmed annotations."""
    confirmed_ids = {a.candidate_id for a in annotations if a.label == "confirmed"}

    rows = []
    for c in candidates:
        cid = make_cid(c.output_id, c.category)
        # output_id = "sample_id::model_name"
        model = c.output_id.split("::", 1)[1] if "::" in c.output_id else c.output_id
        rows.append(
            {
                "model": model,
                "category": c.category,
                "candidate_id": cid,
                "confirmed": cid in confirmed_ids,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["model", "category", "total", "confirmed", "failure_rate"])

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["model", "category"])
        .agg(total=("candidate_id", "count"), confirmed=("confirmed", "sum"))
        .reset_index()
    )
    summary["failure_rate"] = (summary["confirmed"] / summary["total"]).round(3)
    return summary


def export_hf(
    candidates: list[FailureCandidate],
    annotations: list[Annotation],
    min_score: float = 0.5,
) -> Path:
    """Export confirmed failures as a HF Dataset to data/benchmark/."""
    from datasets import Dataset

    confirmed_ids = {a.candidate_id for a in annotations if a.label == "confirmed"}
    ann_map = {a.candidate_id: a for a in annotations}

    records = []
    for c in candidates:
        cid = make_cid(c.output_id, c.category)
        if cid not in confirmed_ids or c.score < min_score:
            continue
        ann = ann_map[cid]
        records.append(
            {
                "candidate_id": cid,
                "output_id": c.output_id,
                "category": c.category,
                "score": c.score,
                "explanation": c.explanation,
                "label": ann.label,
                "notes": ann.notes,
                "source": "squad_v2",
                "license": "CC BY-SA 4.0",
            }
        )

    ds = Dataset.from_list(records)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(BENCHMARK_DIR))
    return BENCHMARK_DIR


def save_metrics(df: pd.DataFrame) -> Path:
    """Save per-model × per-category failure rates to data/metrics.json."""
    path = BENCHMARK_DIR.parent / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return path
