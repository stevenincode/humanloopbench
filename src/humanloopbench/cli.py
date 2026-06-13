"""HumanLoopBench CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from humanloopbench import detectors, models, task
from humanloopbench.schemas import Annotation
from humanloopbench.schemas import candidate_id as make_cid

app = typer.Typer(
    help="HumanLoopBench — human-in-the-loop LLM failure detection.",
    no_args_is_help=True,
)


@app.command()
def smoke(model: str = "gpt-oss:120b") -> None:
    """Verify model access."""
    typer.echo(f"calling {model} ...")
    typer.echo(models.smoke_test(model))


@app.command()
def run(
    model: str = typer.Option(..., help="Model to query"),
    source: str = "demo",
    limit: int = 150,
) -> None:
    """Run a model over the grounded-QA task and save outputs.

    source: 'demo' (offline fixture) or 'squad_v2' (Hugging Face).
    """
    samples = task.load_samples(source=source, limit=limit)
    typer.echo(f"loaded {len(samples)} samples from '{source}'; querying {model} ...")

    def gen(m: str, prompt: str) -> str:
        return models.generate(m, prompt, system=task.SYSTEM)

    outputs = task.run_samples(samples, model, gen)
    path = task.save_outputs(outputs, model, source)
    task.save_samples_snapshot(samples, source)
    typer.echo(f"saved {len(outputs)} outputs -> {path}")


@app.command()
def detect(
    model: str = typer.Option(..., help="Model whose outputs to evaluate"),
    source: str = "demo",
    limit: int = 150,
    judge: str = detectors.JUDGE_MODEL,
    no_two_step: bool = typer.Option(False, "--no-two-step", help="Disable verification pass"),
    workers: int = typer.Option(3, help="Parallel judge call workers (keep ≤5 to avoid 429s)"),
) -> None:
    """Run detectors over saved outputs; write FailureCandidates.

    source: same value used in 'run' ('demo' or 'squad_v2').
    """
    import concurrent.futures

    outputs = task.load_outputs(model, source)
    typer.echo(f"loaded {len(outputs)} outputs for {model}")

    samples = task.load_samples_snapshot(source, limit)
    if samples is not None:
        typer.echo(f"loaded {len(samples)} from snapshot; detectors (workers={workers}) ...")
    else:
        samples = task.load_samples(source=source, limit=limit)
        typer.echo(f"loaded {len(samples)} from HF '{source}'; detectors (workers={workers}) ...")

    two_step = not no_two_step

    def judge_fn(m: str, prompt: str, *, system: str | None = None) -> str:
        return models.generate(m, prompt, system=system)

    sample_map = {s.id: s for s in samples}

    def _detect_one(out_detector):
        out, detector = out_detector
        sample = sample_map.get(out.sample_id)
        if sample is None:
            return None
        return detector(sample, out, judge_fn, judge_model=judge, two_step=two_step)

    tasks = [
        (
            out,
            detectors.detect_grounding
            if not sample_map[out.sample_id].answerable
            else detectors.detect_reasoning,
        )
        for out in outputs
        if out.sample_id in sample_map
    ]
    from tqdm import tqdm

    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(
            pool.map(_detect_one, tasks), total=len(tasks), desc="detecting", unit="output"
        ):
            if result is not None:
                candidates.append(result)

    path = detectors.save_candidates(candidates, model, source)
    typer.echo(f"found {len(candidates)} failure candidates -> {path}")


@app.command()
def review() -> None:
    """Launch the Streamlit human-in-the-loop review UI."""
    app_path = Path(__file__).parent / "review_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


@app.command()
def pre_annotate(
    model: str = detectors.JUDGE_MODEL,
    limit: int = typer.Option(50, help="Max candidates for agent to label in one run"),
) -> None:
    """Autonomously pre-annotate the review queue via LLM.

    Loops through unreviewed candidates, reasons about each, and
    submits confirmed/rejected/unsure labels. Humans spot-check afterwards.
    """
    from humanloopbench.detectors import CANDIDATES_DIR
    from humanloopbench.store import get_annotated_ids, save_annotation
    from humanloopbench.task import load_all_samples

    _AGENT_SYSTEM = (
        "You are a pre-annotation agent for an LLM failure benchmark. "
        "For each candidate, decide whether it is a genuine failure.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"label": "confirmed|rejected|unsure", "notes": "<reason>", "stop": false}\n\n'
        'Set "stop": true to end the session early (e.g. if too many edge cases appear). '
        "Use 'unsure' sparingly — only for genuinely ambiguous cases."
    )

    def _agent_prompt(candidate: dict) -> str:
        lines = [
            f"Category: {candidate.get('category', '?')}",
            f"Score: {candidate.get('score', '?')}",
            f"Detector explanation: {candidate.get('explanation', '?')}",
        ]
        if "question" in candidate:
            lines.append(f"Question: {candidate['question']}")
        if "reference" in candidate:
            lines.append(f"Reference answer: {candidate['reference']}")
        lines.append(f"Model answer: {candidate.get('answer', candidate.get('output_id', '?'))}")
        return "\n".join(lines)

    def _parse_decision(text: str) -> dict:
        import re

        m = re.search(r"\{[^{}]*\"label\"[^{}]*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                pass
        return {"label": "unsure", "notes": "could not parse agent response", "stop": False}

    all_candidates_raw: list[dict] = []
    if CANDIDATES_DIR.exists():
        for path in sorted(CANDIDATES_DIR.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():  # noqa: E501
                if line.strip():
                    all_candidates_raw.append(json.loads(line))

    all_candidates_raw.sort(key=lambda c: c.get("score", 0), reverse=True)

    annotated = get_annotated_ids()
    sample_map = load_all_samples()

    from tqdm import tqdm

    processed = 0
    pbar = tqdm(total=min(limit, len(all_candidates_raw)), desc="pre-annotating", unit="candidate")
    for raw in all_candidates_raw:
        if processed >= limit:
            break

        from humanloopbench.schemas import FailureCandidate

        c = FailureCandidate(**raw)
        cid = make_cid(c.output_id, c.category)

        if cid in annotated:
            continue

        sample = sample_map.get(c.sample_id)
        payload: dict = {
            "category": c.category,
            "score": c.score,
            "explanation": c.explanation,
            "output_id": c.output_id,
        }
        if sample:
            payload["question"] = sample.question
            payload["reference"] = sample.reference

        prompt = _agent_prompt(payload)
        raw_resp = models.generate(model, prompt, system=_AGENT_SYSTEM)
        decision = _parse_decision(raw_resp)

        label = decision.get("label", "unsure")
        if label not in {"confirmed", "rejected", "unsure"}:
            label = "unsure"
        notes = decision.get("notes", "")

        save_annotation(Annotation(candidate_id=cid, label=label, notes=notes, annotator="agent"))
        annotated.add(cid)
        processed += 1
        pbar.update(1)
        pbar.set_postfix(label=label)

        if decision.get("stop"):
            typer.echo("agent requested stop.")
            break

    pbar.close()
    typer.echo(f"pre-annotation done: {processed} candidates labeled.")


@app.command()
def demo_reset(
    force: bool = typer.Option(False, "--force", help="Skip confirmation prompt"),
) -> None:
    """Reset to a clean state for a human-in-the-loop demo run.

    Deletes: data/annotations.jsonl, data/candidates/*.jsonl, data/benchmark/.
    Keeps:   data/runs/ (model outputs — expensive to regenerate).
    """
    import shutil

    from humanloopbench.detectors import CANDIDATES_DIR
    from humanloopbench.store import ANNOTATIONS_FILE

    DATA_DIR = ANNOTATIONS_FILE.parent
    BENCHMARK_DIR = DATA_DIR / "benchmark"

    targets = []
    if ANNOTATIONS_FILE.exists():
        targets.append(ANNOTATIONS_FILE)
    if CANDIDATES_DIR.exists():
        targets.extend(sorted(CANDIDATES_DIR.glob("*.jsonl")))
    if BENCHMARK_DIR.exists():
        targets.append(BENCHMARK_DIR)

    if not targets:
        typer.echo("Nothing to reset.")
        return

    typer.echo("Will delete:")
    for t in targets:
        typer.echo(f"  {t.relative_to(DATA_DIR.parent)}")

    if not force:
        typer.confirm("Proceed?", abort=True)

    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
        else:
            t.unlink()
        typer.echo(f"  deleted {t.name}")

    typer.echo("Reset done. Run 'humanloopbench detect' then 'review' to start fresh.")


@app.command()
def bench(
    min_score: float = typer.Option(0.5, help="Only export confirmed failures with score >= this"),
) -> None:
    """Export confirmed failures + print model comparison table."""
    from humanloopbench import bench as bench_mod
    from humanloopbench.store import load_annotations

    annotations = load_annotations()
    candidates = bench_mod.load_all_candidates()

    if not candidates:
        typer.echo("No candidates found. Run `detect` first.")
        raise typer.Exit(1)

    confirmed = sum(1 for a in annotations if a.label == "confirmed")
    typer.echo(f"{len(candidates)} candidates, {confirmed} confirmed failures.")

    metrics = bench_mod.compute_metrics(candidates, annotations)
    typer.echo("\n--- Failure rates ---")
    typer.echo(metrics.to_string(index=False))

    metrics_path = bench_mod.save_metrics(metrics)
    typer.echo(f"metrics saved -> {metrics_path}")

    if confirmed > 0:
        export_path = bench_mod.export_hf(candidates, annotations, min_score=min_score)
        typer.echo(f"exported HF dataset (score>={min_score}) -> {export_path}")
    else:
        typer.echo("skipping HF export (no confirmed failures yet).")


@app.command()
def validate(
    limit: int = 150,
    judge: str = detectors.JUDGE_MODEL,
    threshold: float = 0.5,
    workers: int = typer.Option(3, help="Parallel judge call workers"),
) -> None:
    """Validate grounding detector against RAGTruth expert labels.

    Loads ~150 QA items from wandb/RAGTruth-processed, runs the grounding
    detector, and reports precision/recall/F1 vs. gold hallucination labels.
    Saves full report to data/validation_report.json.
    """
    from humanloopbench import validate as val_mod

    typer.echo(f"loading RAGTruth QA slice (limit={limit}) ...")
    items = val_mod.load_ragtruth(limit=limit)
    typer.echo(
        f"loaded {len(items)} items "
        f"({sum(1 for *_, g in items if g)} gold failures, "
        f"{sum(1 for *_, g in items if not g)} gold non-failures)"
    )

    def judge_fn(m: str, prompt: str, *, system: str | None = None) -> str:
        return models.generate(m, prompt, system=system)

    typer.echo(
        f"running grounding detector (judge={judge}, threshold={threshold}, workers={workers}) ..."
    )
    results = val_mod.run_validation(items, judge_fn, predict_threshold=threshold, workers=workers)

    metrics = val_mod.compute_validation_metrics(results)
    path = val_mod.save_report(metrics, results)

    typer.echo("\n--- Detector validation report ---")
    typer.echo(f"  Items evaluated : {metrics['total']}")
    typer.echo(f"  Gold +ve rate   : {metrics['gold_positive_rate']:.1%}")
    typer.echo(f"  Precision       : {metrics['precision']:.3f}")
    typer.echo(f"  Recall          : {metrics['recall']:.3f}")
    typer.echo(f"  F1              : {metrics['f1']:.3f}")
    typer.echo(f"  Accuracy        : {metrics['accuracy']:.3f}")
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    typer.echo(f"  TP/FP/FN/TN     : {tp}/{fp}/{fn}/{tn}")
    typer.echo(f"\nfull report saved -> {path}")


if __name__ == "__main__":
    app()
