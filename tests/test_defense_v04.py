import json
from pathlib import Path

import httpx
import pytest

from unified_multi_agent_coordination.corpus_v04 import write_corpus
from unified_multi_agent_coordination import defense_study_v04
from unified_multi_agent_coordination.defense_study_v04 import (
    _completion,
    _prompt,
    collect,
    validate_prerequisites,
)
from unified_multi_agent_coordination.study_analysis_v04 import analyze, write_analysis


def test_v04_prompt_uses_public_authoritative_input_but_never_reference_labels(tmp_path):
    root = tmp_path / "v0.4"
    write_corpus(root)
    public = json.loads((root / "public/cases.json").read_text())
    hidden = json.loads((root / "hidden/reference-labels.json").read_text())

    rendered = json.dumps(_prompt(public["cases"][0]))

    assert "admitted_request" in rendered
    assert "registry_snapshot" in rendered
    assert hidden["labels"][0]["minimum_justification"] not in rendered
    assert "minimum_justification" not in rendered
    assert validate_prerequisites(root, check_runtime=False)["case_count"] == 36


def test_v04_scoring_is_impossible_before_collection_completion(tmp_path):
    corpus = tmp_path / "v0.4"
    write_corpus(corpus)
    run = tmp_path / "run"
    run.mkdir()

    with pytest.raises(RuntimeError, match="blocked until collection is complete"):
        analyze(run, corpus)


def test_v04_frozen_analysis_uses_case_units_and_configuration_specific_latency():
    result = analyze(
        Path("demo_runs/v0.4/20260713T121506Z-a0ecd2bf0e"),
        Path("corpus/v0.4"),
    )

    assert result["analysis_schema_version"] == "0.4.1"
    assert result["observation_counts"]["natural_language_rule_only"] == 36
    assert result["observation_counts"]["structured_oracle_upper_bound"] == 36
    assert result["observation_counts"]["hybrid_bridge"] == 540
    assert result["observation_counts"]["llm_only_no_symbolic_gate"] == 540
    assert (
        result["metrics"]["natural_language_rule_only"]["latency_ms"]["median"]
        < result["metrics"]["llm_only_no_symbolic_gate"]["latency_ms"]["median"]
        < result["metrics"]["hybrid_bridge"]["latency_ms"]["median"]
    )
    assert result["preregistered_acceptance"]["every_model_recall_at_least_0_60"] is False
    assert all(
        item["seed_honored_by_backend"] == "not_verifiable_from_response"
        for item in result["cross_seed_stability"].values()
    )


def test_v04_analysis_writer_refuses_to_overwrite(tmp_path):
    target = tmp_path / "analysis.json"
    write_analysis({"timings": {}, "result": "first"}, target)

    with pytest.raises(FileExistsError):
        write_analysis({"timings": {}, "result": "second"}, target)


def test_v04_runtime_prerequisites_reject_wrong_corpus_and_missing_models(
    tmp_path, monkeypatch
):
    root = tmp_path / "v0.4"
    write_corpus(root)
    monkeypatch.setattr(
        defense_study_v04,
        "validate_frozen_labels",
        lambda _root: {"annotation_type": "author_labeled"},
    )
    public_path = root / "public/cases.json"
    public = json.loads(public_path.read_text())
    public["version"] = "wrong"
    public_path.write_text(json.dumps(public))
    with pytest.raises(RuntimeError, match="requires corpus v0.4"):
        validate_prerequisites(root, check_runtime=False)

    public["version"] = "0.4"
    public_path.write_text(json.dumps(public))
    monkeypatch.setattr(defense_study_v04, "_installed_models", lambda: set())
    with pytest.raises(RuntimeError, match="Missing pinned models"):
        validate_prerequisites(root)


def test_v04_completion_preserves_success_and_reports_http_failure():
    def success(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "model", "choices": []}, request=request)

    with httpx.Client(transport=httpx.MockTransport(success)) as client:
        assert _completion(client, "model", 11, [])["model"] == "model"

    def failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(failure)) as client:
        with pytest.raises(RuntimeError, match=r"completion failed \(503\): offline"):
            _completion(client, "model", 11, [])


def test_v04_collection_is_immutable_and_resumable_only_while_incomplete(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    output = tmp_path / "runs"
    monkeypatch.setattr(
        defense_study_v04,
        "validate_prerequisites",
        lambda _root: {"corpus_hash": "a" * 64, "case_count": 36},
    )
    monkeypatch.setattr(defense_study_v04, "MODELS", ("local/model",))
    monkeypatch.setattr(defense_study_v04, "SEEDS", (11,))
    monkeypatch.setattr(defense_study_v04, "_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        defense_study_v04,
        "_provenance",
        lambda corpus_hash: {"corpus_hash": corpus_hash, "dirty_state": False},
    )
    calls = 0

    def completion(_client, model, _seed, _messages):
        nonlocal calls
        calls += 1
        return {
            "model": model,
            "choices": [{"message": {"content": '{"draft":{"selections":[]}}'}}],
        }

    monkeypatch.setattr(defense_study_v04, "_completion", completion)
    run = collect(corpus, output)
    assert calls == 72  # one initial and one repair response for every case
    assert len(list(run.glob("*/seed-*/*.json"))) == 36
    complete = json.loads((run / "collection-complete.json").read_text())
    assert complete == {
        "complete": True,
        "expected_outputs": 36,
        "labels_loaded_during_collection": False,
    }
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        collect(corpus, output, run)


def test_v04_resume_rejects_corpus_hash_mismatch(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    run = tmp_path / "resume"
    run.mkdir()
    (run / "provenance.json").write_text(json.dumps({"corpus_hash": "old"}))
    monkeypatch.setattr(
        defense_study_v04,
        "validate_prerequisites",
        lambda _root: {"corpus_hash": "new", "case_count": 36},
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        collect(corpus, tmp_path / "output", run)
