"""MCP server for HumanLoopBench annotation queue.

Tools:
  get_next_candidate  — next unreviewed FailureCandidate (highest score first)
  submit_annotation   — save a confirmed/rejected/unsure label
  get_queue_stats     — total / annotated / remaining counts

Usage:
  Start the server:  python -m humanloopbench.mcp_server
  Default SSE URL:   http://127.0.0.1:8765/sse

Register in .claude/settings.json:
  {"mcpServers": {"humanloopbench": {"url": "http://127.0.0.1:8765/sse"}}}
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from humanloopbench.detectors import CANDIDATES_DIR
from humanloopbench.schemas import Annotation, FailureCandidate
from humanloopbench.schemas import candidate_id as make_cid
from humanloopbench.store import get_annotated_ids, save_annotation
from humanloopbench.task import RUNS_DIR, load_all_samples

MCP_PORT = 8765
mcp = FastMCP("humanloopbench-review", port=MCP_PORT)


def _load_outputs() -> dict[str, str]:
    """Return {output_id: answer_text} from all run files."""
    outputs: dict[str, str] = {}
    if not RUNS_DIR.exists():
        return outputs
    for path in RUNS_DIR.glob("*.jsonl"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                r = json.loads(line)
                oid = f"{r['sample_id']}::{r['model']}"
                outputs[oid] = r["text"]
    return outputs


def _all_candidates() -> list[FailureCandidate]:
    candidates: list[FailureCandidate] = []
    if not CANDIDATES_DIR.exists():
        return candidates
    for path in sorted(CANDIDATES_DIR.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        candidates.extend(FailureCandidate(**r) for r in rows)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


@mcp.tool()
def get_next_candidate() -> str:
    """Get the next unreviewed failure candidate from the queue (highest score first).

    Returns a JSON object with candidate details including question, context,
    model answer, score, category, and detector explanation. Returns a message
    if the queue is empty.
    """
    annotated = get_annotated_ids()
    sample_map = load_all_samples()
    output_map = _load_outputs()
    all_candidates = _all_candidates()
    remaining_after = sum(
        1 for c2 in all_candidates if make_cid(c2.output_id, c2.category) not in annotated
    )

    for i, c in enumerate(all_candidates):
        cid = make_cid(c.output_id, c.category)
        if cid in annotated:
            continue

        model = c.output_id.split("::", 1)[1] if "::" in c.output_id else c.output_id
        sample = sample_map.get(c.sample_id)
        position = i + 1
        remaining_after_this = remaining_after - 1

        result: dict = {
            "candidate_id": cid,
            "position": f"{position} of {len(all_candidates)}",
            "remaining_after_this": remaining_after_this,
            "category": c.category,
            "score": c.score,
            "model": model,
            "model_answer": output_map.get(c.output_id, "(not found)"),
            "explanation": c.explanation,
        }
        if sample:
            result["question"] = sample.question
            result["context"] = sample.context
            result["ground_truth"] = sample.reference
            result["answerable"] = sample.answerable

        return json.dumps(result, indent=2, ensure_ascii=False)

    return json.dumps({"status": "queue_empty", "message": "All candidates have been reviewed."})


@mcp.tool()
def submit_annotation(candidate_id: str, label: str, notes: str = "") -> str:
    """Save a human annotation for a failure candidate.

    Args:
        candidate_id: The candidate_id returned by get_next_candidate.
        label: One of 'confirmed', 'rejected', or 'unsure'.
        notes: Optional explanation for the decision.

    Returns a confirmation message.
    """
    valid_labels = {"confirmed", "rejected", "unsure"}
    if label not in valid_labels:
        return json.dumps({"error": f"label must be one of {sorted(valid_labels)}"})

    ann = Annotation(candidate_id=candidate_id, label=label, notes=notes, annotator="agent")
    save_annotation(ann)
    return json.dumps({"status": "saved", "candidate_id": candidate_id, "label": label})


@mcp.tool()
def get_queue_stats() -> str:
    """Return current annotation queue statistics: total, annotated, and remaining counts."""
    all_candidates = _all_candidates()
    annotated = get_annotated_ids()
    total = len(all_candidates)
    done = sum(1 for c in all_candidates if make_cid(c.output_id, c.category) in annotated)
    return json.dumps({"total": total, "annotated": done, "remaining": total - done})


if __name__ == "__main__":
    mcp.run(transport="sse")
