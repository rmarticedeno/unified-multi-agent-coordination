import json
from collections import Counter

import pytest

from unified_multi_agent_coordination import defense_study_v05
from unified_multi_agent_coordination.corpus_v05 import (
    DOMAINS,
    INVALIDITY_CATEGORIES,
    build_corpus,
    write_corpus,
)
from unified_multi_agent_coordination.defense_study_v05 import (
    DirectCoordinatorOutput,
    _direct_issues,
    _direct_prompt,
    _hybrid_prompt,
    _public_corpus,
    validate_collection,
    validate_prerequisites,
)
from unified_multi_agent_coordination.models import SolutionProposal
from unified_multi_agent_coordination.study_analysis_v05 import (
    _baseline_decision,
    _clustered_difference_interval,
)


def test_v05_corpus_has_fresh_balanced_matched_pairs_and_hidden_oracle(tmp_path):
    root = tmp_path / "v0.5"
    manifest = write_corpus(root)
    public = json.loads((root / "public/cases.json").read_text())
    hidden = json.loads((root / "hidden/reference-labels.json").read_text())
    cases = public["cases"]

    assert manifest["case_count"] == 48
    assert manifest["matched_pair_count"] == 24
    assert manifest["primary_output_count"] == 1440
    assert len({case["pair_id"] for case in cases}) == 24
    assert Counter(case["domain"] for case in cases) == {domain: 6 for domain in DOMAINS}
    assert Counter(case["invalidity_category"] for case in cases) == {
        category: 6 for category in INVALIDITY_CATEGORIES
    }
    assert not any("oracle_proposal" in case for case in cases)
    assert all("oracle_proposal" in label for label in hidden["labels"])
    assert len(json.loads((root / "development/cases.json").read_text())["cases"]) == 8
    assert len(json.loads((root / "runtime/cases.json").read_text())["cases"]) == 12


def test_v05_public_loader_never_reads_hidden_labels(tmp_path):
    root = tmp_path / "v0.5"
    write_corpus(root)
    (root / "hidden/reference-labels.json").write_text("not valid JSON")

    assert len(_public_corpus(root)["cases"]) == 48
    result = validate_prerequisites(root, check_runtime=False, require_clean=False)
    assert result["labels_loaded"] is False
    assert result["expected_outputs"] == 1440


def test_v05_prompts_have_equal_public_inputs_without_labels(tmp_path):
    root = tmp_path / "v0.5"
    write_corpus(root)
    case = _public_corpus(root)["cases"][0]
    hybrid = json.dumps(_hybrid_prompt(case))
    direct = json.dumps(_direct_prompt(case))

    for field in ("request_text", "admitted_request", "registry_snapshot", "payload"):
        assert field in hybrid
        assert field in direct
    assert "minimum_justification" not in hybrid + direct
    assert "oracle_proposal" not in hybrid + direct


def test_v05_oracle_and_non_oracle_greedy_baseline_match_frozen_design():
    for case in build_corpus():
        label = {**case["reference"], "oracle_proposal": case["oracle_proposal"]}
        public = {
            key: value
            for key, value in case.items()
            if key not in {"reference", "oracle_proposal"}
        }
        expected = label["feasible"]
        assert _baseline_decision("structured_oracle_upper_bound", public, label)[0] is expected
        assert _baseline_decision("greedy_symbolic", public, label)[0] is expected


def test_direct_repair_validator_reports_only_public_referential_errors():
    case = build_corpus()[0]
    public = {
        key: value
        for key, value in case.items()
        if key not in {"reference", "oracle_proposal"}
    }
    proposal = SolutionProposal.model_validate(case["oracle_proposal"])
    proposal.tasks[0].assigned_to = "unknown"
    output = DirectCoordinatorOutput(decision="accept", rationale="", proposal=proposal)

    assert _direct_issues(public, output) == [
        {"code": "unknown_agent", "requirement_id": proposal.tasks[0].requirement_id}
    ]


def test_v05_collection_identity_validation_rejects_missing_and_excess_calls(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(defense_study_v05, "ARMS", ("hybrid", "direct"))
    monkeypatch.setattr(defense_study_v05, "MODELS", ("model",))
    monkeypatch.setattr(defense_study_v05, "SEEDS", (11,))
    cases = [{"case_id": "a"}, {"case_id": "b"}]
    root = tmp_path / "run"
    for arm in defense_study_v05.ARMS:
        for case in cases:
            path = root / "outputs" / arm / "model" / "seed-11" / f"{case['case_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "identity": {
                            "arm": arm,
                            "case_id": case["case_id"],
                            "model_id": "model",
                            "seed": 11,
                        },
                        "call_count": 2,
                    }
                )
            )
    assert validate_collection(root, cases)["observed_outputs"] == 4

    first = next(root.glob("outputs/*/*/seed-*/*.json"))
    record = json.loads(first.read_text())
    record["call_count"] = 3
    first.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="bad_call_budget=1"):
        validate_collection(root, cases)


def test_matched_pair_bootstrap_is_deterministic_and_preserves_pair_clusters():
    left = []
    right = []
    for pair in ("p1", "p2", "p3"):
        for feasible in (True, False):
            case = f"{pair}-{feasible}"
            common = {
                "case_id": case,
                "pair_id": pair,
                "model_id": "m",
                "seed": 1,
            }
            left.append({**common, "correct": True})
            right.append({**common, "correct": pair == "p3"})

    first = _clustered_difference_interval(left, right, samples=1000)
    second = _clustered_difference_interval(left, right, samples=1000)
    assert first == second
    assert first[0] >= 0
