import json
from pathlib import Path

import pytest

from unified_multi_agent_coordination.corpus import audit_corpus, write_corpus
from unified_multi_agent_coordination.defense_study import _prompt, validate_frozen_labels
from unified_multi_agent_coordination.evidence_preflight import run_preflight
from unified_multi_agent_coordination.study_analysis import _decision, analyze
from unified_multi_agent_coordination.defense_study import LinguisticBatchOutput


def test_author_provenance_freezes_hash_and_passes_audit(tmp_path):
    manifest = write_corpus(tmp_path)
    provenance = validate_frozen_labels(tmp_path)
    audit = json.loads((tmp_path / "review/internal-consistency-audit.json").read_text())

    assert provenance["annotation_type"] == "author_labeled"
    assert provenance["independent_adjudication"] is False
    assert provenance["frozen"] is True
    assert provenance["corpus_hash"] == manifest["corpus_hash"]
    assert audit["passed"] is True


@pytest.mark.parametrize("field,value", [("frozen", False), ("corpus_hash", "stale")])
def test_collection_gate_rejects_unfrozen_or_mismatched_provenance(tmp_path, field, value):
    write_corpus(tmp_path)
    path = tmp_path / "label-provenance.json"
    provenance = json.loads(path.read_text())
    provenance[field] = value
    path.write_text(json.dumps(provenance))

    with pytest.raises(RuntimeError):
        validate_frozen_labels(tmp_path)


def test_labels_are_not_present_in_model_prompt(tmp_path):
    write_corpus(tmp_path)
    public = json.loads((tmp_path / "public/cases.json").read_text())
    labels = json.loads((tmp_path / "hidden/reference-labels.json").read_text())
    prompt_text = json.dumps([_prompt(case) for case in public["cases"]])

    assert "minimum_justification" not in prompt_text
    assert not any(label["minimum_justification"] in prompt_text for label in labels["labels"])


def test_frozen_corpus_rejects_on_disk_label_mutation(tmp_path):
    write_corpus(tmp_path)
    path = tmp_path / "hidden/reference-labels.json"
    labels = json.loads(path.read_text())
    labels["labels"][0]["feasible"] = not labels["labels"][0]["feasible"]
    path.write_text(json.dumps(labels))

    with pytest.raises(RuntimeError, match="Frozen corpus cannot be changed"):
        write_corpus(tmp_path)


def test_consistency_audit_detects_duplicate_and_contradictory_label(tmp_path):
    write_corpus(tmp_path)
    public = json.loads((tmp_path / "public/cases.json").read_text())
    labels = json.loads((tmp_path / "hidden/reference-labels.json").read_text())
    public["cases"].append(public["cases"][0])
    labels["labels"][0]["feasible"] = False
    labels["labels"][0]["status"] = "completed"

    audit = audit_corpus(public, labels)

    assert audit["passed"] is False
    assert any("duplicate" in error for error in audit["errors"])
    assert any("contradictory" in error for error in audit["errors"])


def test_repository_preflight_accepts_author_label_methodology():
    result = run_preflight(Path("corpus/v0.3"))

    assert result["passed"] is True
    assert result["independent_adjudication"] is False


def test_configuration_ablation_does_not_mutate_shared_linguistic_output():
    public = json.loads(Path("corpus/v0.3/public/cases.json").read_text())
    case = public["cases"][0]
    linguistic = LinguisticBatchOutput.model_validate(
        {
            "interpreted_request": case["request"],
            "candidate_plans": [case["shared_raw_proposal"]],
        }
    )
    before = linguistic.model_dump(mode="json")

    _decision("ablation_contract_validation", case, linguistic)

    assert linguistic.model_dump(mode="json") == before


def test_scoring_is_blocked_before_collection_completion(tmp_path):
    with pytest.raises(RuntimeError, match="collection is complete"):
        analyze(tmp_path, Path("corpus/v0.3"))
