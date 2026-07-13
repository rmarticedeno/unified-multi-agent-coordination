import json

from unified_multi_agent_coordination.corpus_v04 import write_corpus


def test_v04_corpus_is_frozen_domain_specific_and_has_preregistered_thresholds(tmp_path):
    manifest = write_corpus(tmp_path / "v0.4")
    public = json.loads((tmp_path / "v0.4/public/cases.json").read_text())
    provenance = json.loads((tmp_path / "v0.4/label-provenance.json").read_text())

    assert manifest["case_count"] == 36
    assert manifest["base_case_count"] == 24
    assert manifest["variation_count"] == 12
    assert "defense corpus task" not in " ".join(c["request_text"] for c in public["cases"])
    assert provenance["frozen"] is True
    assert provenance["independent_adjudication"] is False
    assert manifest["success_criteria"]["minimum_per_model_feasible_recall"] == 0.60
