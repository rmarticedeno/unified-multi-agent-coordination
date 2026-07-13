from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    CapabilityRequirement,
    DraftRequirementSelection,
    LinguisticPlanDraft,
    PlanHydrator,
    ProblemRequest,
)
from unified_multi_agent_coordination.runtime_policies import (
    SingleUseAuxiliaryLifecyclePolicy,
    UnsafeReusableAuxiliaryLifecyclePolicy,
)


def _agent(agent_id: str, *skills: CapabilityRequirement) -> AgentRegistryEntry:
    return AgentRegistryEntry(
        agent_id=agent_id,
        name=agent_id,
        service_endpoint=f"local://{agent_id}",
        skills=list(skills),
    )


def test_hydrator_constructs_executable_fields_from_authoritative_inputs():
    extract = CapabilityRequirement(
        name="extract", capability_id="cap-extract",
        validation_contract={"required_artifacts": ["facts"]},
    )
    assess = CapabilityRequirement(
        name="assess", capability_id="cap-assess",
        validation_contract={"required_artifacts": ["decision"]},
    )
    request = ProblemRequest(
        user_goal="Extract and assess", requirements=[extract, assess],
        required_artifacts=["facts", "decision"],
    )
    draft = LinguisticPlanDraft(selections=[
        DraftRequirementSelection(
            requirement_id=extract.requirement_id, capability_id="cap-extract"
        ),
        DraftRequirementSelection(
            requirement_id=assess.requirement_id,
            capability_id="cap-assess",
            depends_on_requirement_ids=[extract.requirement_id],
        ),
    ])

    result = PlanHydrator().hydrate(
        request, [_agent("z-agent", extract), _agent("a-agent", extract), _agent("judge", assess)], draft
    )

    assert result.complete
    assert result.proposal is not None
    assert result.proposal.selected_agents == {"t1": "a-agent", "t2": "judge"}
    assert result.proposal.tasks[1].depends_on == ["t1"]
    assert result.proposal.tasks[0].validation_contract == extract.validation_contract
    assert result.proposal.completion_contract.required_artifacts == ["facts", "decision"]


def test_hydrator_fails_closed_on_invention_omission_and_unresolved_language():
    requirement = CapabilityRequirement(name="extract", capability_id="cap-extract")
    request = ProblemRequest(user_goal="Extract", requirements=[requirement])
    draft = LinguisticPlanDraft(
        selections=[DraftRequirementSelection(
            requirement_id="invented", capability_id="admin-everything"
        )],
        unresolved_terms=["the safest provider"],
    )

    result = PlanHydrator().hydrate(request, [_agent("worker", requirement)], draft)

    assert result.proposal is None
    assert {issue.code for issue in result.issues} == {
        "unknown_requirement", "missing_requirement", "unresolved_term"
    }


def test_hydrator_rejects_cycles_before_task_creation():
    first = CapabilityRequirement(name="first")
    second = CapabilityRequirement(name="second")
    request = ProblemRequest(user_goal="Both", requirements=[first, second])
    draft = LinguisticPlanDraft(selections=[
        DraftRequirementSelection(
            requirement_id=first.requirement_id,
            capability_id=first.capability_id,
            depends_on_requirement_ids=[second.requirement_id],
        ),
        DraftRequirementSelection(
            requirement_id=second.requirement_id,
            capability_id=second.capability_id,
            depends_on_requirement_ids=[first.requirement_id],
        ),
    ])

    result = PlanHydrator().hydrate(
        request, [_agent("one", first), _agent("two", second)], draft
    )

    assert result.proposal is None
    assert [issue.code for issue in result.issues] == ["dependency_cycle"]


def test_auxiliary_lifecycle_defaults_to_single_use_and_unsafe_control_is_explicit():
    consumed: set[str] = set()
    strict = SingleUseAuxiliaryLifecyclePolicy()

    assert strict.claim("plan-1", consumed)
    assert not strict.claim("plan-1", consumed)
    assert not strict.claim("", consumed)
    assert UnsafeReusableAuxiliaryLifecyclePolicy().claim("plan-1", consumed)
