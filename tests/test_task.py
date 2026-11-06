from humanloopbench import task
from humanloopbench.schemas import ModelOutput, Sample


def test_load_demo_has_answerable_and_unanswerable():
    samples = task.load_demo()
    assert len(samples) >= 6
    assert any(s.answerable for s in samples)
    assert any(not s.answerable for s in samples)
    # unanswerable items should carry no reference
    assert all(s.reference == "" for s in samples if not s.answerable)


def test_load_demo_limit():
    assert len(task.load_demo(limit=3)) == 3


def test_build_prompt_contains_context_and_question():
    s = Sample(id="x", question="Q?", context="Some context.", reference="A")
    p = task.build_prompt(s)
    assert "Some context." in p
    assert "Q?" in p
    assert p.strip().endswith("Answer:")


def test_run_samples_with_stub_generator():
    samples = task.load_demo(limit=4)
    calls = []

    def fake_gen(model: str, prompt: str) -> str:
        calls.append((model, prompt))
        return "stub answer"

    outputs = task.run_samples(samples, "test-model", fake_gen)
    assert len(outputs) == 4
    assert all(isinstance(o, ModelOutput) for o in outputs)
    assert all(o.model == "test-model" for o in outputs)
    assert len(calls) == 4


def test_save_and_load_outputs_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(task, "RUNS_DIR", tmp_path)
    outs = [ModelOutput(sample_id="a", model="m:1", text="hi")]
    path = task.save_outputs(outs, "m:1", "demo")
    assert path.exists()
    loaded = task.load_outputs("m:1", "demo")
    assert loaded == outs


def test_load_samples_unknown_source():
    try:
        task.load_samples(source="nope")
    except ValueError as e:
        assert "unknown source" in str(e)
    else:
        raise AssertionError("expected ValueError")


def _s(i: str, q: str = "q") -> Sample:
    return Sample(id=i, question=q, context="c", reference="r", answerable=True)


def test_samples_snapshot_merges_instead_of_overwriting(tmp_path, monkeypatch):
    monkeypatch.setattr(task, "SAMPLES_DIR", tmp_path)
    task.save_samples_snapshot([_s("a"), _s("b"), _s("c")], "squad_v2")
    # A smaller later run (e.g. --limit 1) must not drop a/c; new ids are appended.
    task.save_samples_snapshot([_s("b", q="updated"), _s("d")], "squad_v2")
    loaded = task.load_samples_snapshot("squad_v2")
    assert [s.id for s in loaded] == ["a", "b", "c", "d"]
    assert loaded[1].question == "updated"
