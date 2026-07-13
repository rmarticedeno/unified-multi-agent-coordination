import json

import pytest

from unified_multi_agent_coordination.corpus_v04 import write_corpus
from unified_multi_agent_coordination.defense_study_v04 import _prompt, validate_prerequisites
from unified_multi_agent_coordination.study_analysis_v04 import analyze


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
