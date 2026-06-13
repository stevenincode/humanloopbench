"""Annotation store: append-only JSONL, shared by Streamlit, MCP server, and agents."""

from __future__ import annotations

import json
from pathlib import Path

from humanloopbench.schemas import Annotation

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ANNOTATIONS_FILE = DATA_DIR / "annotations.jsonl"


def load_annotations() -> list[Annotation]:
    if not ANNOTATIONS_FILE.exists():
        return []
    lines = ANNOTATIONS_FILE.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [Annotation(**r) for r in rows]


def save_annotation(annotation: Annotation) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ANNOTATIONS_FILE.open("a", encoding="utf-8") as f:
        f.write(annotation.model_dump_json() + "\n")


def get_annotated_ids() -> set[str]:
    return {a.candidate_id for a in load_annotations()}
