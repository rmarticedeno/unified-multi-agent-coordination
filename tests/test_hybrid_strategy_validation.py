import json

from unified_multi_agent_coordination.feasibility import FeasibilityAnalyzer
from unified_multi_agent_coordination.hybrid_strategy_validation import (
    ACCEPTED_ANALYSIS_PATH,
    EXPECTED_PATH,
    FROZEN_CORPUS_ROOT,
    FROZEN_LABELS_PATH,
    SENTINEL_CASES,
    _canonical_intent,
    _public_cases,
    analyze_typed_corpus_replay,
    collect_typed_corpus_replay,
    strategy_catalog,
    strategy_registry,
)
from unified_multi_agent_coordination.semantic_admission import (
    SemanticIntentOutput,
    SemanticRequestAdmitter,
)
from unified_multi_agent_coordination.symbolic_plan_compiler import SymbolicPlanCompiler


def test_strategy_subset_is_fixed_at_eight_cases_and_four_sentinels():
    _, cases = _public_cases("all")
    _, sentinels = _public_cases("sentinel")

    assert len(cases) == 8
    assert len(sentinels) == 4
    assert {item["case_id"] for item in sentinels} == SENTINEL_CASES
    assert len({item["case_id"] for item in cases}) == 8


def test_strategy_reference_intents_match_the_real_admission_and_compiler():
    _, cases = _public_cases("all")
    expectations = json.loads(EXPECTED_PATH.read_text())["expectations"]
    catalog = strategy_catalog()

    for case in cases:
        expected = expectations[case["case_id"]]
        intent = SemanticIntentOutput.model_validate(expected["intent"])
        admission = SemanticRequestAdmitter().admit(
            case["request_text"],
            catalog,
            intent,
            strategy_registry(case["variant"], catalog),
        )
        accepted = False
        if admission.request is not None:
            compilation = SymbolicPlanCompiler(FeasibilityAnalyzer()).compile(
                admission.request,
                strategy_registry(case["variant"], catalog),
            )
            accepted = compilation.report.feasible
        assert accepted is expected["feasible"], case["case_id"]


def test_semantic_comparison_treats_explicit_and_default_policy_as_equivalent():
    catalog = strategy_catalog()
    explicit = {
        "interpretation_status": "resolved",
        "goals": [{
            "capability_id": "deliver",
            "trust_policy_id": "ordinary",
            "artifact_contract_id": "json",
        }],
        "forbidden_capability_ids": [],
        "forbidden_agent_ids": [],
        "unresolved_terms": [],
    }
    implicit = {
        **explicit,
        "goals": [{
            "capability_id": "deliver",
            "trust_policy_id": None,
            "artifact_contract_id": None,
        }],
    }

    assert _canonical_intent(explicit, catalog) == _canonical_intent(
        implicit, catalog
    )


def test_frozen_typed_corpus_replay_uses_the_shared_compiler_and_scores_separately(
    tmp_path,
):
    run = collect_typed_corpus_replay(FROZEN_CORPUS_ROOT, tmp_path)
    provenance = json.loads((run / "provenance.json").read_text())
    outputs = sorted((run / "typed").glob("*.json"))

    assert len(outputs) == 48
    assert provenance["expected_data_loaded_during_collection"] is False
    assert provenance["typed_request_llm_bypass"] is True
    assert provenance["model_calls"] == 0
    assert not (run / "analysis.json").exists()

    result = analyze_typed_corpus_replay(
        run, FROZEN_LABELS_PATH, ACCEPTED_ANALYSIS_PATH
    )

    assert result["metrics"]["correct"] == 48
    assert result["metrics"]["feasible_recall"] == 1.0
    assert result["metrics"]["false_acceptances"] == 0
    assert result["metrics"]["false_refusals"] == 0
    assert result["accepted_historical_hybrid_repaired"]["feasible_recall"] == (
        0.013888888888888888
    )
