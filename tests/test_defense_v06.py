import csv
import io
import json
from collections import Counter
from pathlib import Path

import pytest

from unified_multi_agent_coordination import defense_study_v06
from unified_multi_agent_coordination.corpus_v06 import (
    CATEGORIES,
    build_corpus,
    write_corpus,
)
from unified_multi_agent_coordination.defense_study_v06 import (
    LLM_ARMS,
    _canonical_hash,
    _load_public,
    _messages,
    _public_payload,
    freeze_protocol,
    lexical_selection,
    output_path,
    selection_schema,
    symbolic_authorization,
    validate_collection,
    validate_prerequisites,
)
from unified_multi_agent_coordination.study_analysis_v06 import (
    _bootstrap,
    _metrics,
    _one_sided_upper_zero,
    _semantic_match,
)


def test_v06_corpus_has_40_balanced_pairs_and_hidden_semantic_references(tmp_path):
    root = tmp_path / "v0.6"
    manifest = write_corpus(root)
    public = json.loads((root / "public/cases.json").read_text())
    hidden = json.loads((root / "hidden/reference-labels.json").read_text())

    assert manifest["case_count"] == 80
    assert manifest["matched_pair_count"] == 40
    assert manifest["expected_llm_outputs"] == 2400
    assert len({case["pair_id"] for case in public["cases"]}) == 40
    assert sum(case["case_type"] == "matched_feasible" for case in public["cases"]) == 40
    counts = Counter(case["category"] for case in public["cases"])
    assert set(counts) == set(CATEGORIES)
    assert min(counts.values()) >= 12
    assert not any("reference" in case for case in public["cases"])
    assert all("typed_reference_request" in label for label in hidden["labels"])


def test_v06_blind_review_package_contains_no_author_labels(tmp_path):
    root = tmp_path / "v0.6"
    write_corpus(root)
    content = (root / "review/blind-label-review.csv").read_text()
    rows = list(csv.DictReader(io.StringIO(content)))

    assert len(rows) == 80
    assert all(row["reviewer_feasible"] == "" for row in rows)
    assert "minimum_justification" not in content
    assert "typed_reference_request" not in content
    instructions = json.loads((root / "review/reviewer-instructions.json").read_text())
    assert instructions["author_labels_included"] is False


def test_v06_arms_receive_identical_public_payload_and_dynamic_public_enums():
    case = {key: value for key, value in build_corpus()[0].items() if key != "reference"}
    hybrid = _messages(case, LLM_ARMS[0])
    direct = _messages(case, LLM_ARMS[1])
    assert hybrid[1] == direct[1]
    assert "typed_reference_request" not in json.dumps(hybrid + direct)

    hybrid_schema = selection_schema(case, LLM_ARMS[0])
    direct_schema = selection_schema(case, LLM_ARMS[1])
    capability_ids = {item["capability_id"] for item in case["capability_catalog"]}
    assert (
        set(hybrid_schema["properties"]["goal_capability_ids"]["items"]["enum"]) == capability_ids
    )
    assert hybrid_schema["properties"]["goal_capability_ids"]["minItems"] == 1
    assert direct_schema["properties"]["selected_agent_ids"]["minItems"] == 1
    assert "selected_agent_ids" not in hybrid_schema["properties"]


def test_v06_author_semantic_oracle_matches_real_symbolic_boundary():
    for case in build_corpus():
        public = {key: value for key, value in case.items() if key != "reference"}
        accepted, evidence = symbolic_authorization(
            public, case["reference"]["typed_reference_request"]
        )
        assert accepted is case["reference"]["feasible"], (case["case_id"], evidence)


def test_v06_lexical_control_uses_public_values_and_produces_nonempty_selection():
    for case in build_corpus():
        public = {key: value for key, value in case.items() if key != "reference"}
        selection = lexical_selection(public)
        assert selection["goal_capability_ids"]
        assert selection["required_policy_ids"]
        assert selection["required_artifact_contract_ids"]
        assert _semantic_match(selection, case["reference"]["typed_reference_request"]) in {
            True,
            False,
        }


def test_v06_protocol_lock_covers_corpus_prompts_schemas_models_and_code(tmp_path):
    root = tmp_path / "v0.6"
    write_corpus(root)
    lock = freeze_protocol(root)
    result = validate_prerequisites(root, check_runtime=False, require_clean=False)

    assert result["expected_outputs"] == 2400
    assert result["labels_loaded"] is False
    assert lock["dynamic_schemas_sha256"]
    assert len(lock["source_sha256"]) == 3
    lock["prompt_version"] = "changed"
    (root / "protocol-lock.json").write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match="protocol lock differs"):
        validate_prerequisites(root, check_runtime=False, require_clean=False)


def test_v06_collection_matrix_rejects_bad_budget_and_public_input(tmp_path, monkeypatch):
    monkeypatch.setattr(defense_study_v06, "LLM_ARMS", ("hybrid", "direct"))
    monkeypatch.setattr(defense_study_v06, "MODELS", ("model",))
    monkeypatch.setattr(defense_study_v06, "SEEDS", (11,))
    cases = [
        {
            "case_id": "a",
            "request_text": "request",
            "capability_catalog": [],
            "registry_snapshot": [],
            "policy": {},
            "payload": {},
        }
    ]
    root = tmp_path / "run"
    for arm in defense_study_v06.LLM_ARMS:
        path = output_path(root, arm=arm, model="model", seed=11, case_index=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "identity": {
                        "arm": arm,
                        "case_id": "a",
                        "model_id": "model",
                        "seed": 11,
                    },
                    "call_count": 2,
                    "public_input_hash": _canonical_hash(_public_payload(cases[0])),
                }
            )
        )
    assert validate_collection(root, cases)["observed_outputs"] == 2
    first = next(root.glob("o/*/*/s*/*.json"))
    record = json.loads(first.read_text())
    record["call_count"] = 3
    first.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="bad_call_budget=1"):
        validate_collection(root, cases)


def test_v06_compact_output_paths_fit_legacy_windows_limits(monkeypatch):
    monkeypatch.setattr(defense_study_v06, "MODELS", ("model",))
    path = output_path(
        Path("run"),
        arm=LLM_ARMS[0],
        model="model",
        seed=11,
        case_index=79,
    )
    assert len(str(path)) < 30


def test_v06_metrics_bootstrap_and_zero_event_bound_are_deterministic():
    rows = []
    for pair in range(4):
        rows.extend(
            (
                {
                    "case_id": f"p{pair}-f",
                    "pair_id": f"p{pair}",
                    "category": "c",
                    "accepted": True,
                    "system_accept": True,
                    "semantic_match": True,
                    "reference_feasible": True,
                    "correct": True,
                    "unsafe_acceptance": False,
                },
                {
                    "case_id": f"p{pair}-i",
                    "pair_id": f"p{pair}",
                    "category": "c",
                    "accepted": False,
                    "system_accept": False,
                    "semantic_match": True,
                    "reference_feasible": False,
                    "correct": True,
                    "unsafe_acceptance": False,
                },
            )
        )
    metrics = _metrics(rows)
    assert metrics["balanced_accuracy"] == 1
    assert metrics["mcc"] == 1
    views = {
        "hybrid_repaired": rows,
        "direct_repaired": rows,
        defense_study_v06.RULE_ARM: rows,
    }
    assert _bootstrap(views, samples=100) == _bootstrap(views, samples=100)
    assert 0 < _one_sided_upper_zero(40) < 0.1


def test_v06_public_loader_never_opens_hidden_labels(tmp_path):
    root = tmp_path / "v0.6"
    write_corpus(root)
    (root / "hidden/reference-labels.json").write_text("not json")
    assert len(_load_public(root)["cases"]) == 80
