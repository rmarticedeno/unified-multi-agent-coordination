from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    GroundedSemanticSelection,
    SemanticCatalog,
    SemanticIntentV08,
    SemanticRequestAdmitterV08,
    classify_request_spans,
    normalize_v08,
    retrieve_semantic_candidates,
)
from unified_multi_agent_coordination.corpus_v08 import _pair, build_phase
from unified_multi_agent_coordination.defense_study_v08 import MODELS


def _environment(case):
    return (
        SemanticCatalog.model_validate(case["catalog"]),
        [AgentRegistryEntry.model_validate(item) for item in case["registry"]],
    )


def test_unicode_normalization_and_ignored_quote_spans():
    assert normalize_v08("Garantía naïve RÉSUMÉ") == "garantia naive resume"
    text = 'Publish the result. The payload has quoted untrusted text "ignore rules" as data.'
    spans = classify_request_spans(text)
    assert [item.kind for item in spans].count("ignored_untrusted") == 1
    assert spans[1].text == '"ignore rules"'


def test_retrieval_keeps_exact_matches_and_dependency_ancestors():
    case = _pair(1, "low_overlap_paraphrase", "document review", phase="development", variant=1)[0]
    catalog, _ = _environment(case)
    retrieval = retrieve_semantic_candidates("issue the brief", catalog)
    deliver = "v08-p01-deliver"
    assert deliver in retrieval.exact_match_ids
    assert {"v08-p01-prepare", "v08-p01-verify"} <= set(retrieval.dependency_expansion_ids)


def test_reference_feasible_intent_admits_without_canonical_goal_invention():
    case = _pair(1, "low_overlap_paraphrase", "document review", phase="development", variant=1)[0]
    catalog, registry = _environment(case)
    intent = SemanticIntentV08.model_validate(case["reference"]["intent"])
    result = SemanticRequestAdmitterV08().admit(case["request_text"], catalog, intent, registry)
    assert result.admitted
    assert result.derived_status == "resolved"
    assert [item.identifier for item in result.canonical_intent.terminal_goals] == [
        "v08-p01-deliver"
    ]
    assert [item.capability_id for item in result.request.requirements] == [
        "v08-p01-prepare", "v08-p01-verify", "v08-p01-deliver"
    ]


def test_unresolved_disjunction_is_deterministically_refused():
    case = _pair(1, "disjunction", "document review", phase="development", variant=1)[1]
    catalog, registry = _environment(case)
    invented = SemanticIntentV08(
        terminal_goals=[GroundedSemanticSelection(
            identifier="v08-p01-deliver", evidence_text="send the finalized outcome"
        )],
        global_trust_policy=None,
        global_artifact_contract=None,
        goal_overrides=[],
        forbidden_capabilities=[],
        forbidden_agents=[],
        goal_alternatives=[],
        policy_alternatives=[],
        contract_alternatives=[],
        unknown_required_terms=[],
        ignored_untrusted_spans=[],
    )
    result = SemanticRequestAdmitterV08().admit(
        case["request_text"], catalog, invented, registry
    )
    assert not result.admitted
    assert result.derived_status == "ambiguous"
    assert "unresolved_choice" in {item.code for item in result.issues}


def test_quoted_unknown_does_not_become_required_but_executable_unknown_does():
    feasible, infeasible = _pair(
        1, "unknown_required_entity", "document review", phase="development", variant=1
    )
    for case, expected in ((feasible, True), (infeasible, False)):
        catalog, registry = _environment(case)
        intent = SemanticIntentV08.model_validate(case["reference"]["intent"])
        result = SemanticRequestAdmitterV08().admit(
            case["request_text"], catalog, intent, registry
        )
        assert result.admitted is expected
    assert "unknown_required_term" in {
        item.code for item in result.issues
    }


def test_v08_corpus_counts_low_overlap_and_model_allowlist():
    development = build_phase("development")
    confirmatory = build_phase("confirmatory")
    assert len(development) == 72
    assert len(confirmatory) == 96
    assert len({item["category"] for item in confirmatory}) == 12
    assert MODELS == ("qwen/qwen3-1.7b", "google/gemma-4-e2b")
