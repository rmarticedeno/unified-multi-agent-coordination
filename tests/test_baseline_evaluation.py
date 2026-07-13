import pytest

from unified_multi_agent_coordination.baseline_evaluation import (
    _summarize_rows,
    build_rule_only_proposal,
    llm_output_to_proposal,
    resolve_capability_identity,
)
from unified_multi_agent_coordination.end_to_end_scenarios import scenario_definitions
from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    CapabilityRequirement,
    CoordinationSdk,
    FeasibilityAnalyzer,
)


def _agent(agent_id, *skills):
    return AgentRegistryEntry(
        agent_id=agent_id,
        name=agent_id,
        service_endpoint=f"local://{agent_id}",
        skills=list(skills),
    )


def test_resolves_agent_id_with_single_advertised_skill():
    skill = CapabilityRequirement(name="extract key points")
    resolution = resolve_capability_identity(
        raw_name="note-extractor",
        assigned_agent_id="note-extractor",
        registry=[_agent("note-extractor", skill)],
    )

    assert resolution.resolved
    assert resolution.resolved_name == "extract key points"
    assert resolution.method == "agent_id_single_skill"


def test_resolves_exact_normalized_skill_name_when_agent_has_multiple_skills():
    summarize = CapabilityRequirement(name="summarize brief")
    classify = CapabilityRequirement(name="classify text")
    resolution = resolve_capability_identity(
        raw_name="Summarize Brief",
        assigned_agent_id="multi-agent",
        registry=[_agent("multi-agent", summarize, classify)],
    )

    assert resolution.resolved
    assert resolution.resolved_name == "summarize brief"
    assert resolution.method == "exact_normalized_skill_name"


def test_unresolved_llm_requirement_is_explicit():
    resolution = resolve_capability_identity(
        raw_name="invent missing skill",
        assigned_agent_id="unknown-agent",
        registry=[_agent("known-agent", CapabilityRequirement(name="summarize"))],
    )

    assert not resolution.resolved
    assert resolution.method == "unresolved"


@pytest.mark.asyncio
async def test_rule_only_plan_assigns_available_agents_and_bounded_auxiliary():
    scenario = next(
        item
        for item in scenario_definitions()
        if item.scenario_id == "data_privacy_redaction_pipeline"
    )
    sdk = CoordinationSdk()
    await scenario.setup(sdk)
    registry = await sdk.registry_snapshot(refresh=False)

    proposal = build_rule_only_proposal(scenario.request, registry)
    report = FeasibilityAnalyzer().check(scenario.request, registry, proposal)

    assert proposal.tasks[0].auxiliary_spec_id == "aux-1"
    assert [task.assigned_to for task in proposal.tasks[1:]] == [
        "transcript-redactor",
        "redaction-validator",
        "privacy-summary-generator",
    ]
    assert report.feasible
    await sdk.http_client.aclose()


@pytest.mark.asyncio
async def test_llm_output_to_proposal_resolves_agent_ids_and_dependencies():
    scenario = next(
        item
        for item in scenario_definitions()
        if item.scenario_id == "research_brief_production"
    )
    sdk = CoordinationSdk()
    await scenario.setup(sdk)
    registry = await sdk.registry_snapshot(refresh=False)
    model_output = {
        "required_artifacts": ["research_brief"],
        "candidate_tasks": [
            {
                "requirement_name": "note-extractor",
                "assigned_agent_id": "note-extractor",
                "depends_on": [],
            },
            {
                "requirement_name": "brief-summarizer",
                "assigned_agent_id": "brief-summarizer",
                "depends_on": ["note-extractor"],
            },
        ],
    }

    proposal, resolutions, unresolved_dependencies = llm_output_to_proposal(
        model_output,
        registry,
    )

    assert [item.resolved_name for item in resolutions] == [
        "extract key points",
        "summarize brief",
    ]
    assert proposal.tasks[1].depends_on == ["t1"]
    assert unresolved_dependencies == 0
    await sdk.http_client.aclose()


def test_baseline_summary_counts_false_accepts_and_resolution():
    rows = [
        {
            "status_matches_reference": True,
            "decision_matches_reference": False,
            "false_accept": True,
            "false_refuse": False,
            "accepted": True,
            "observed_status": "completed",
            "dispatch_attempts": 2,
            "dispatch_without_symbolic_authorization": True,
            "latency_ms": 10,
            "llm_tokens": 100,
            "requirement_resolution": {
                "resolved": 2,
                "unresolved": 1,
                "exact_name_matches": 1,
                "resolved_expected_names": 2,
            },
        }
    ]

    summary = _summarize_rows(rows)

    assert summary["false_accepts"] == 1
    assert summary["resolved_requirements"] == 2
    assert summary["unresolved_requirements"] == 1
    assert summary["exact_requirement_name_matches"] == 1
    assert summary["resolved_expected_names"] == 2
