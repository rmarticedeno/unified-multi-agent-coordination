import asyncio

import pytest

from unified_multi_agent_coordination import (
    AgentAdmissionError,
    AgentAdmissionPolicy,
    AgentRegistryEntry,
    CapabilityRequirement,
    ConstraintSpec,
    CoordinationAgent,
    CoordinationPlanResult,
    CoordinationSdk,
    FeasibilityAnalyzer,
    ProblemRequest,
    SolutionProposal,
    StaticCredentialProvider,
    FeasibilityReport,
    TaskSpec,
    validate_trace,
)


def _entry(agent_id: str, capability: CapabilityRequirement) -> AgentRegistryEntry:
    return AgentRegistryEntry(
        agent_id=agent_id,
        name=agent_id,
        agent_kind="local_python",
        service_endpoint="local://handler",
        skills=[capability],
        input_modes=capability.input_modes,
        output_modes=capability.output_modes,
    )


def _single_plan(
    requirement: CapabilityRequirement,
    *,
    task_contract=None,
) -> tuple[ProblemRequest, SolutionProposal, AgentRegistryEntry]:
    request = ProblemRequest(
        user_goal="Execute one checked task.",
        requirements=[requirement],
        required_artifacts=["result"],
    )
    proposal = SolutionProposal(
        tasks=[
            TaskSpec(
                task_id="t1",
                requirement_name=requirement.name,
                capability_id=requirement.capability_id,
                assigned_to="agent",
                expected_artifacts=["result"],
                validation_contract=task_contract or {},
            )
        ],
        execution_order=["t1"],
        expected_artifacts=["result"],
    )
    return request, proposal, _entry("agent", requirement)


def test_validatorless_task_is_refused_even_when_artifact_is_promised():
    request, proposal, entry = _single_plan(CapabilityRequirement(name="work"))

    report = FeasibilityAnalyzer().check(request, [entry], proposal)

    assert not report.feasible
    assert report.validation_gaps == [
        {
            "task_id": "t1",
            "requirement_id": "work",
            "reason": "no enforceable validation contract",
        }
    ]


@pytest.mark.parametrize(
    ("constraint", "feasible"),
    [
        (
            ConstraintSpec(
                constraint_id="region-us",
                source="request_context",
                path="/region",
                operator="eq",
                expected="US",
            ),
            True,
        ),
        (
            ConstraintSpec(
                constraint_id="budget",
                source="request_context",
                path="/budget",
                operator="lte",
                expected=5,
            ),
            False,
        ),
        ("must be safe", False),
    ],
)
def test_typed_constraints_are_enforced_and_legacy_text_fails_closed(constraint, feasible):
    requirement = CapabilityRequirement(
        name="work",
        validation_contract={"json_schema": {"type": "object"}},
    )
    request, proposal, entry = _single_plan(requirement)
    request_data = request.model_dump(mode="json")
    request_data["context"] = {"region": "US", "budget": 10}
    request_data["constraints"] = [
        constraint.model_dump(mode="json")
        if isinstance(constraint, ConstraintSpec)
        else constraint
    ]
    request = ProblemRequest.model_validate(request_data)

    report = FeasibilityAnalyzer().check(request, [entry], proposal)

    assert report.feasible is feasible
    assert bool(report.constraint_violations) is (not feasible)


def test_capability_id_and_all_modes_must_match():
    requirement = CapabilityRequirement(
        name="analyze",
        capability_id="analysis-v2",
        input_modes=["text", "table"],
        output_modes=["json"],
        validation_contract={"json_schema": {"type": "object"}},
    )
    request, proposal, _ = _single_plan(requirement)
    advertised = requirement.model_copy(
        update={"capability_id": "analysis-v1", "input_modes": ["text"]}
    )

    report = FeasibilityAnalyzer().check(request, [_entry("agent", advertised)], proposal)

    assert not report.feasible
    assert report.capability_resolution[0]["resolved"] is False


def test_remote_card_cannot_self_assign_trust_and_http_is_fail_closed(monkeypatch):
    monkeypatch.delenv("COORDINATION_ALLOW_INSECURE_A2A", raising=False)
    strict = AgentAdmissionPolicy()
    sdk = CoordinationSdk(admission_policy=strict)
    card = {
        "name": "remote",
        "url": "http://remote.example/a2a",
        "trustLevel": "admin",
        "skills": [{"id": "work"}],
    }
    with pytest.raises(AgentAdmissionError):
        sdk.a2a_adapter.normalize_card(card)

    local_policy = AgentAdmissionPolicy(
        allow_insecure_development=True,
        default_trust_level="standard",
    )
    sdk = CoordinationSdk(admission_policy=local_policy)
    entry = sdk.a2a_adapter.normalize_card(card)
    assert entry.trust_level == "standard"


def test_required_card_credentials_must_be_locally_available():
    policy = AgentAdmissionPolicy(allow_insecure_development=True)
    card = {
        "name": "remote",
        "url": "http://remote.example/a2a",
        "skills": [{"id": "work"}],
        "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
        "security": [{"bearer": []}],
    }
    with pytest.raises(AgentAdmissionError):
        CoordinationSdk(admission_policy=policy).a2a_adapter.normalize_card(card)

    provider = StaticCredentialProvider(
        headers={"Authorization": "Bearer secret"},
        supported_schemes={"bearer"},
    )
    entry = CoordinationSdk(
        admission_policy=policy, credential_provider=provider
    ).a2a_adapter.normalize_card(card)
    assert entry.required_security_schemes == ["bearer"]
    assert "secret" not in entry.model_dump_json()


class _FixedPlanAgent(CoordinationAgent):
    def __init__(self, sdk, plan):
        super().__init__(sdk=sdk)
        self.plan = plan

    async def build_solution_plan(self, user_request, context=None, *, session_id="", plan_id=""):
        del user_request, context
        return self.plan.model_copy(update={"session_id": session_id, "plan_id": plan_id})


@pytest.mark.asyncio
async def test_failed_prerequisite_skips_only_descendants_and_independent_branch_continues():
    sdk = CoordinationSdk()
    contract = {"json_schema": {"type": "object"}}
    requirements = [
        CapabilityRequirement(name=name, validation_contract=contract)
        for name in ("root", "dependent", "independent")
    ]
    called = []

    def fail(_payload):
        called.append("root")
        raise RuntimeError("boom")

    def should_not_run(_payload):
        called.append("dependent")
        return {"artifacts": [{"name": "dependent", "kind": "data"}]}

    def succeed(_payload):
        called.append("independent")
        return {"artifacts": [{"name": "independent", "kind": "data"}]}

    for requirement, handler in zip(requirements, (fail, should_not_run, succeed)):
        sdk.register_local_agent(requirement.name, [requirement], handler)
    tasks = [
        TaskSpec(task_id="root", requirement_name="root", assigned_to="root", validation_contract=contract),
        TaskSpec(task_id="child", requirement_name="dependent", assigned_to="dependent", depends_on=["root"], validation_contract=contract),
        TaskSpec(task_id="other", requirement_name="independent", assigned_to="independent", validation_contract=contract),
    ]
    request = ProblemRequest(user_goal="branches", requirements=requirements)
    proposal = SolutionProposal(tasks=tasks, execution_order=["root", "child", "other"])
    report = FeasibilityAnalyzer().check(request, list(await sdk.registry_snapshot()), proposal)
    assert report.feasible
    plan = CoordinationPlanResult(request=request, proposal=proposal, feasibility_report=report)

    result = await _FixedPlanAgent(sdk, plan).coordinate(request, session_id="branch-session")

    assert called == ["root", "independent"]
    assert [item.status for item in result.task_results] == ["failed", "skipped", "completed"]
    assert result.task_results[1].metadata["blocked_by"] == ["root"]


@pytest.mark.asyncio
async def test_concurrent_sessions_return_only_their_persisted_trace_events():
    sdk = CoordinationSdk()
    requirement = CapabilityRequirement(
        name="work", validation_contract={"required_fields": ["value"]}
    )

    async def handler(payload):
        await asyncio.sleep(0)
        return {"artifacts": [{"name": "result", "kind": "data", "data": {"value": payload["value"]}}]}

    sdk.register_local_agent("worker", [requirement], handler)
    agent = CoordinationAgent(sdk=sdk)
    request = ProblemRequest(user_goal="work", requirements=[requirement], required_artifacts=["result"])
    first, second = await asyncio.gather(
        agent.coordinate(request, payload={"value": 1}, session_id="session-a"),
        agent.coordinate(request, payload={"value": 2}, session_id="session-b"),
    )

    assert first.status == second.status == "completed"
    assert {event.session_id for event in first.trace} == {"session-a"}
    assert {event.session_id for event in second.trace} == {"session-b"}
    assert validate_trace(first.trace, "session-a").complete
    assert validate_trace(second.trace, "session-b").complete


@pytest.mark.asyncio
async def test_nested_json_schema_types_are_enforced_at_runtime():
    sdk = CoordinationSdk()
    requirement = CapabilityRequirement(
        name="profile",
        validation_contract={
            "json_schema": {
                "type": "object",
                "properties": {
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "required": ["age"],
                                    "properties": {"age": {"type": "integer", "minimum": 18}},
                                }
                            },
                            "required": ["data"],
                        },
                    }
                },
                "required": ["artifacts"],
            }
        },
    )
    entry = sdk.register_local_agent(
        "profile-agent",
        [requirement],
        lambda _payload: {"artifacts": [{"kind": "data", "data": {"age": "18"}}]},
    )
    task = TaskSpec(
        task_id="t1",
        requirement_name="profile",
        assigned_to=entry.agent_id,
    )
    report = FeasibilityReport(feasible=True, matched_agents={"t1": entry.agent_id})

    result = await sdk.send_task(report, task, {})

    assert result.status == "failed"
    assert "integer" in result.error


def test_auxiliary_extraction_derives_pii_and_lifecycle_expires():
    sdk = CoordinationSdk()
    requirement = CapabilityRequirement(
        name="extract pii",
        description="extract email and phone",
        output_schema={"required": ["email", "phone"]},
        auxiliary_eligible=True,
    )
    from unified_multi_agent_coordination import BoundedAuxiliaryCapabilityFactory

    spec = BoundedAuxiliaryCapabilityFactory().specify(
        requirement, lifecycle="plan:p1:task:t1"
    )
    assert spec is not None
    task = TaskSpec(
        task_id="t1",
        requirement_name=requirement.name,
        auxiliary_spec_id=spec.spec_id,
        expected_artifacts=["pii"],
    )
    first = sdk._invoke_auxiliary(
        spec,
        task,
        {"text": "Email ana@example.com or call 555-0100."},
    )
    second = sdk._invoke_auxiliary(spec, task, {"text": "another@example.com"})

    assert first.status == "completed"
    assert first.artifacts[0]["data"] == {
        "email": "ana@example.com",
        "phone": "555-0100",
    }
    assert second.status == "refused"
    assert second.metadata["expired"] is True
