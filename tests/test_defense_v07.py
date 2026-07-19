import asyncio
import json

from unified_multi_agent_coordination.corpus_v07 import CATEGORIES, build_phase, write_corpus
from unified_multi_agent_coordination.defense_study_v07 import (
    ARMS,
    MODELS,
    collect_observation,
    direct_schema,
    protocol_payload,
)
from unified_multi_agent_coordination.study_analysis_v07 import (
    _canonical_intent,
    exact_alias_baseline,
)
from unified_multi_agent_coordination.symbolic_benchmark_v07 import run_benchmark


def test_v07_corpus_has_qwen_development_and_balanced_confirmatory_pairs(tmp_path):
    development = build_phase("development")
    confirmatory = build_phase("confirmatory")

    assert len(development) == 48
    assert len(confirmatory) == 64
    assert {
        category: sum(item["category"] == category for item in development)
        for category in CATEGORIES
    } == {category: 6 for category in CATEGORIES}
    assert {
        category: sum(item["category"] == category for item in confirmatory)
        for category in CATEGORIES
    } == {category: 8 for category in CATEGORIES}
    assert sum(item["reference"]["feasible"] for item in confirmatory) == 32

    root = tmp_path / "v0.7"
    manifest = write_corpus(root, "Test Author")
    public = json.loads((root / "public/cases.json").read_text(encoding="utf-8"))
    hidden = json.loads(
        (root / "hidden/reference-labels.json").read_text(encoding="utf-8")
    )

    assert manifest["development_case_count"] == 48
    assert manifest["confirmatory_case_count"] == 64
    assert public["corpus_hash"] == hidden["corpus_hash"]
    assert all("reference" not in case for case in public["cases"])


def test_v07_protocol_hashes_the_production_path(tmp_path):
    root = tmp_path / "v0.7"
    write_corpus(root, "Test Author")

    payload = protocol_payload(root)

    assert payload["expected_outputs"] == 768
    assert payload["models"] == list(MODELS)
    assert payload["seeds"] == [11, 29]
    assert (
        "src/unified_multi_agent_coordination/semantic_admission.py"
        in payload["source_sha256"]
    )
    assert (
        "src/unified_multi_agent_coordination/symbolic_plan_compiler.py"
        in payload["source_sha256"]
    )


def test_v07_direct_schema_extends_but_does_not_replace_production_vocabulary():
    case = build_phase("development")[0]
    from unified_multi_agent_coordination.defense_study_v07 import _environment

    catalog, registry = _environment(case)
    schema = direct_schema(catalog, registry)

    assert "goals" in schema["properties"]
    assert "decision" in schema["properties"]
    assert "assignments" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_v07_collection_records_runtime_failures_without_losing_identity(monkeypatch):
    case = build_phase("development")[0]

    async def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic model failure")

    monkeypatch.setattr(
        "unified_multi_agent_coordination.defense_study_v07._hybrid_record",
        fail,
    )
    record = asyncio.run(
        collect_observation(
            case,
            phase="development",
            arm=ARMS[0],
            model=MODELS[0],
            seed=11,
        )
    )

    assert record["identity"]["case_id"] == case["case_id"]
    assert record["result"] is None
    assert record["runtime_error"] == "RuntimeError: synthetic model failure"


def test_v07_direct_semantics_ignore_direct_only_fields():
    case = build_phase("development")[0]
    observed = {
        **case["reference"]["intent"],
        "decision": "accept",
        "assignments": [],
        "execution_order": [],
    }

    assert _canonical_intent(observed, case) == _canonical_intent(
        case["reference"]["intent"],
        case,
    )


def test_exact_alias_baseline_is_deterministic():
    case = build_phase("development")[0]

    assert exact_alias_baseline(case) == exact_alias_baseline(case)


def test_symbolic_benchmark_enforces_recovery_permutation_and_search_bounds():
    result = run_benchmark(repetitions=1)

    assert result["all_invariants_passed"]
    assert result["invariants"] == {
        "oracle_compilation": True,
        "registry_permutation_invariance": True,
        "alternative_provider_recovery": True,
        "bounded_search_refusal": True,
    }
