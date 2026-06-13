"""Streamlit HITL review UI — the demo centerpiece.

Launch via: humanloopbench review
"""

from __future__ import annotations

import html as _html
import json

import pandas as pd
import streamlit as st

from humanloopbench.bench import BENCHMARK_DIR
from humanloopbench.detectors import CANDIDATES_DIR, JUDGE_MODEL
from humanloopbench.schemas import Annotation, FailureCandidate
from humanloopbench.schemas import candidate_id as make_cid
from humanloopbench.store import get_annotated_ids, load_annotations, save_annotation
from humanloopbench.task import RUNS_DIR, load_all_samples
from humanloopbench.validate import REPORT_PATH

st.set_page_config(
    page_title="HumanLoopBench — Failure Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Data loading (cached per session)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=5)
def _load_candidates() -> list[FailureCandidate]:
    candidates: list[FailureCandidate] = []
    if not CANDIDATES_DIR.exists():
        return candidates
    for path in sorted(CANDIDATES_DIR.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        candidates.extend(FailureCandidate(**r) for r in rows)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


@st.cache_data(ttl=5)
def _load_outputs() -> dict[str, str]:
    """Return {output_id: answer_text} from all run files."""
    outputs: dict[str, str] = {}
    if not RUNS_DIR.exists():
        return outputs
    for path in RUNS_DIR.glob("*.jsonl"):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        for r in rows:
            oid = f"{r['sample_id']}::{r['model']}"
            outputs[oid] = r["text"]
    return outputs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _queue(candidates: list[FailureCandidate], annotated_ids: set[str]) -> list[FailureCandidate]:
    return [c for c in candidates if make_cid(c.output_id, c.category) not in annotated_ids]


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "#e74c3c"
    if score >= 0.5:
        return "#e67e22"
    return "#f39c12"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------- #
# Browse reviewed items
# --------------------------------------------------------------------------- #

_LABEL_COLOR = {"confirmed": "#27ae60", "rejected": "#e74c3c", "unsure": "#f39c12"}


def _render_browse(
    candidates: list[FailureCandidate],
    ann_list: list,
    sample_map: dict,
    output_map: dict,
) -> None:
    ann_map = {a.candidate_id: a for a in ann_list}

    _fc, _fs = st.columns([1, 2])
    with _fc:
        filter_label = st.selectbox(
            "Filter by label", ["all", "confirmed", "rejected", "unsure", "has notes"], index=0
        )
    with _fs:
        filter_score = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)

    shown = 0
    for c in candidates:
        cid = make_cid(c.output_id, c.category)
        ann = ann_map.get(cid)
        if ann is None:
            continue
        if filter_label == "has notes":
            if not ann.notes:
                continue
        elif filter_label != "all" and ann.label != filter_label:
            continue
        if c.score < filter_score:
            continue

        label_color = _LABEL_COLOR.get(ann.label, "#888")
        _CAT_COLORS = {"hallucination": "#3498db", "refusal": "#e67e22", "factual_error": "#9b59b6"}
        badge_cat = _CAT_COLORS.get(c.category, "#888")
        model_name = c.output_id.split("::", 1)[1] if "::" in c.output_id else c.output_id
        sample = sample_map.get(c.sample_id)
        answer = output_map.get(c.output_id, "_(not found)_")

        header = (
            f'<span style="background:{badge_cat};color:white;padding:2px 8px;'
            f'border-radius:3px;font-size:0.8em" title="{c.explanation}">'
            f"{c.category.upper()}</span> "
            f'<span style="color:{label_color};font-weight:bold">{ann.label.upper()}</span> '
            f"&nbsp;conf={c.score:.2f}&nbsp; <code>{model_name}</code>"
        )
        with st.expander(
            sample.question[:90] + "…" if sample else cid[:60], expanded=False
        ):
            st.markdown(header, unsafe_allow_html=True)
            if sample:
                st.caption(f"**Q:** {sample.question}")
                st.caption(f"**Ref:** {sample.reference or '(unanswerable)'}")
                with st.expander("Context", expanded=False):
                    st.write(sample.context)
            st.info(answer[:300])
            st.warning(c.explanation)
            if ann.notes:
                st.caption(f"Notes: {ann.notes}")
        shown += 1
        if shown >= 100:
            st.caption("(showing first 100 — adjust filters to narrow down)")
            break


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #


def main() -> None:  # noqa: PLR0912, PLR0915
    st.markdown("""
<style>
.block-container{padding-top:.4rem!important;padding-bottom:0!important}
section[data-testid="stMain"]{overflow:hidden!important}
div[data-testid="stAppViewBlockContainer"]{overflow:hidden!important}
</style>""", unsafe_allow_html=True)

    all_candidates = _load_candidates()
    sample_map = load_all_samples()
    output_map = _load_outputs()

    if not all_candidates:
        st.warning("No candidates found. Run `humanloopbench detect --model <name>` first.")
        return

    annotated_ids = get_annotated_ids()
    queue = _queue(all_candidates, annotated_ids)
    total = len(all_candidates)
    done = total - len(queue)
    ann_list = load_annotations()
    confirmed = sum(1 for a in ann_list if a.label == "confirmed")
    rejected  = sum(1 for a in ann_list if a.label == "rejected")
    unsure    = sum(1 for a in ann_list if a.label == "unsure")

    # ── Sidebar ── theme-safe: #888 for labels, inherit for numbers ───────────
    with st.sidebar:
        pct_int = int(done / total * 100) if total else 0
        st.markdown(
            f'<div style="padding:6px 0 14px 0">'
            f'<div style="color:#888;font-size:0.68em;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px">Progress</div>'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:baseline;margin-bottom:5px">'
            f'<span style="font-size:1.8em;font-weight:700">{done}</span>'
            f'<span style="color:#888;font-size:0.85em">/ {total} reviewed</span>'
            f'</div>'
            f'<div style="height:4px;background:rgba(128,128,128,.2);'
            f'border-radius:2px;margin-bottom:5px">'
            f'<div style="width:{pct_int}%;height:4px;background:#4a80d4;'
            f'border-radius:2px"></div></div>'
            f'<div style="color:#888;font-size:0.78em">{len(queue)} remaining</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            f'<div style="display:grid;grid-template-columns:1fr 1fr;'
            f'gap:10px 6px;padding:4px 0 10px 0">'
            f'<div><div style="color:#888;font-size:0.65em;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px">Confirmed</div>'
            f'<div style="color:#2ecc71;font-size:1.5em;font-weight:700">{confirmed}</div></div>'
            f'<div><div style="color:#888;font-size:0.65em;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px">Rejected</div>'
            f'<div style="color:#e74c3c;font-size:1.5em;font-weight:700">{rejected}</div></div>'
            f'<div><div style="color:#888;font-size:0.65em;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px">Unsure</div>'
            f'<div style="color:#f39c12;font-size:1.5em;font-weight:700">{unsure}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        with st.expander("Confidence guide", expanded=True):
            st.markdown(
                '<div style="font-size:0.8em;line-height:1.6">'
                '<b>Judge confidence</b> = estimated failure probability<br><br>'
                '🔴 <b>≥ 0.7</b> high<br>'
                '🟠 <b>0.5–0.7</b> borderline †<br>'
                '🟡 <b>0.3–0.5</b> weak †<br>'
                '⬛ <b>&lt; 0.3</b> filtered out<br><br>'
                '<span style="color:#888">† 2 verification calls, averaged</span><br><br>'
                '<b>Boosters:</b><br>'
                '<code>+0.3</code> unanswerable Q, model answered<br>'
                '<code>+0.2</code> answerable Q, model refused'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── Empty-queue: tabs ─────────────────────────────────────────────────────
    if not queue:
        st.caption("✓ All candidates reviewed — run `humanloopbench bench` to export.")
        tab_browse, tab_metrics, tab_validate = st.tabs(["Browse", "Metrics", "Validation"])
        with tab_browse:
            _render_browse(all_candidates, ann_list, sample_map, output_map)
        with tab_metrics:
            metrics_path = BENCHMARK_DIR.parent / "metrics.json"
            if metrics_path.exists():
                rows = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                st.caption(f"Source: `{metrics_path}` — refresh to reload after `bench`")
            else:
                st.info("No metrics yet. Run `humanloopbench bench` first.")
        with tab_validate:
            if REPORT_PATH.exists():
                report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
                m = report.get("metrics", {})
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Precision", f"{m.get('precision', 0):.3f}")
                mc2.metric("Recall",    f"{m.get('recall', 0):.3f}")
                mc3.metric("F1",        f"{m.get('f1', 0):.3f}")
                mc4.metric("Accuracy",  f"{m.get('accuracy', 0):.3f}")
                st.caption(
                    f"TP {m.get('tp')} / FP {m.get('fp')} / "
                    f"FN {m.get('fn')} / TN {m.get('tn')}  "
                    f"·  threshold {m.get('predict_threshold')}  "
                    f"·  {m.get('total')} items"
                )
            else:
                st.info("No validation report yet. Run `humanloopbench validate` first.")
        return

    # ── Review card ───────────────────────────────────────────────────────────
    c = queue[0]
    cid = make_cid(c.output_id, c.category)
    sample = sample_map.get(c.sample_id)
    answer_text = output_map.get(c.output_id, "(answer not found — rerun `detect`)")
    model_name = c.output_id.split("::", 1)[1] if "::" in c.output_id else c.output_id

    _CAT_META = {
        "hallucination": ("#3498db", "Model answered when context has no answer"),
        "refusal":       ("#e67e22", "Model said NOT IN CONTEXT when answer exists"),
        "factual_error": ("#9b59b6", "Model used context but got the answer wrong"),
    }
    badge_color, badge_tip = _CAT_META.get(c.category, ("#888", c.category))
    conf_color = _score_color(c.score)
    tier = "high" if c.score >= 0.7 else "borderline" if c.score >= 0.5 else "weak"
    pct = int(done / total * 100) if total else 0

    # ── Scrollable content zone (height controlled by CSS flex:1) ───────────
    with st.container(border=False):
        # Progress ribbon
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">'
            f'<div style="flex:1;height:3px;background:rgba(128,128,128,.2);border-radius:2px">'
            f'<div style="width:{pct}%;height:3px;background:{badge_color};'
            f'border-radius:2px"></div></div>'
            f'<span style="color:#888;font-size:0.72em;white-space:nowrap">'
            f'{done + 1} / {total}&nbsp;·&nbsp;{len(queue) - 1} remaining</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Two-column: Question | Ground truth + Model answer
        col_left, col_right = st.columns(2)

        with col_left:
            if sample:
                q = _html.escape(sample.question)
                st.markdown(
                    f'<div style="color:#888;font-size:0.7em;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px">'
                    f'Question</div>'
                    f'<div style="font-size:1.02em;line-height:1.7;font-weight:500;'
                    f'margin-bottom:14px">{q}</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Context", expanded=True):
                    st.write(sample.context)
            else:
                st.caption(f"Sample `{c.sample_id}` — run `humanloopbench run`.")

        with col_right:
            gt_text = (
                sample.reference if (sample and sample.reference)
                else "(unanswerable — no answer in context)"
            )
            gt  = _html.escape(gt_text)
            ans = _html.escape(answer_text).replace("\n", "<br>")
            st.markdown(
                f'<div style="color:#888;font-size:0.7em;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px">'
                f'Ground truth</div>'
                f'<div style="background:{_rgba("#27ae60", 0.08)};'
                f'border:1px solid {_rgba("#27ae60", 0.3)};'
                f'border-left:3px solid #27ae60;border-radius:8px;'
                f'padding:10px 14px;margin-bottom:14px;font-size:0.9em;line-height:1.55">'
                f'<span style="color:#27ae60;margin-right:6px">✓</span>{gt}'
                f'</div>'
                f'<div style="color:#888;font-size:0.7em;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">'
                f'Model answer '
                f'<code style="font-size:0.88em;font-weight:400;color:#888">'
                f'{model_name}</code></div>'
                f'<div style="background:{_rgba("#e74c3c", 0.07)};'
                f'border:1px solid {_rgba("#e74c3c", 0.3)};'
                f'border-left:3px solid #e74c3c;border-radius:8px;'
                f'padding:10px 14px;font-size:0.9em;line-height:1.55">'
                f'<span style="color:#e74c3c;margin-right:6px">✗</span>{ans}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Judge analysis card
        bar_w = max(4, int(c.score * 80))
        score_svg = (
            f'<svg width="80" height="6" style="vertical-align:middle;margin:0 4px 0 6px">'
            f'<rect width="80" height="6" rx="3" fill="rgba(128,128,128,.2)"/>'
            f'<rect width="{bar_w}" height="6" rx="3" fill="{conf_color}"/>'
            f'</svg>'
        )
        expl = _html.escape(c.explanation)
        st.markdown(
            f'<div style="background:{_rgba(badge_color, 0.06)};'
            f'border:1px solid {_rgba(badge_color, 0.3)};'
            f'border-left:4px solid {badge_color};border-radius:10px;'
            f'padding:12px 18px;margin-top:14px">'
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'flex-wrap:wrap;margin-bottom:10px">'
            f'<span style="background:{badge_color};color:#fff;padding:3px 13px;'
            f'border-radius:20px;font-size:0.78em;font-weight:700;letter-spacing:.05em;'
            f'cursor:default" title="{badge_tip}">{c.category.upper()}</span>'
            f'{score_svg}'
            f'<span style="color:{conf_color};font-weight:700;font-size:1.05em">'
            f'{c.score:.2f}</span>'
            f'<span style="color:{conf_color};background:{_rgba(conf_color, 0.12)};'
            f'font-size:0.7em;padding:2px 9px;border-radius:10px;font-weight:700;'
            f'letter-spacing:.05em">{tier.upper()}</span>'
            f'<span style="margin-left:auto;color:#888;font-size:0.75em">'
            f'Judge: <code style="background:transparent">{JUDGE_MODEL}</code>'
            f'</span></div>'
            f'<div style="border-top:1px solid rgba(128,128,128,.2);padding-top:10px;'
            f'font-size:0.88em;line-height:1.6">'
            f'<span style="color:#888;font-size:0.76em;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.08em;margin-right:8px">Analysis ›</span>'
            f'{expl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Annotation bar — fixed outside scroll zone ────────────────────────────
    st.markdown(
        '<div style="border-top:1px solid rgba(128,128,128,.25);margin:6px 0 6px 0"></div>',
        unsafe_allow_html=True,
    )
    notes = st.text_input(
        "", key=f"notes_{cid}",
        placeholder="Note (optional) — saved with annotation",
        label_visibility="collapsed",
    )
    col_conf, col_unsure, col_rej = st.columns(3)
    submitted = False
    label = None
    with col_conf:
        if st.button("✓  Confirmed failure", use_container_width=True, type="primary"):
            label, submitted = "confirmed", True
    with col_unsure:
        if st.button("?  Unsure", use_container_width=True):
            label, submitted = "unsure", True
    with col_rej:
        if st.button("✗  Rejected (not a failure)", use_container_width=True):
            label, submitted = "rejected", True
    if submitted and label:
        save_annotation(Annotation(candidate_id=cid, label=label, notes=notes))
        _load_candidates.clear()
        st.rerun()


if __name__ == "__main__":
    main()
