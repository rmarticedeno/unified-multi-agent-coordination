import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    CapabilityRequirement,
    CoordinationAgent,
    CoordinationPlanResult,
    CoordinationSdk,
    InMemoryCoordinationLedger,
    JsonlCoordinationLedger,
    LedgerEvent,
    ProblemRequest,
    RetryPolicy,
    SolutionProposal,
    TaskExecutionResult,
    TaskSpec,
    FeasibilityReport,
    ValidationContract,
    LeaseRecord,
    RemoteRegistryError,
)
from unified_multi_agent_coordination.hybrid_strategy_validation import strategy_catalog


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://registry.example")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("registry failed", request=request, response=response)

    def json(self):
        return self.payload


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


class SequenceHttpClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


def _card(name, skill="summarize"):
    return {
        "name": name,
        "url": f"http://{name}.example",
        "skills": [{"id": skill}],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
    }


@pytest.mark.asyncio
async def test_build_solution_plan_refreshes_registry_filters_self_and_authorizes_direct_plan():
    request = ProblemRequest(
        user_goal="Summarize the report.",
        requirements=[CapabilityRequirement(name="summarize", validation_contract={"json_schema": {"type": "object"}})],
        required_artifacts=["summary"],
    )
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        self_agent_id="coordinator",
        http_client=FakeHttpClient(
            FakeResponse({"agents": [_card("coordinator"), _card("summarizer")]})
        ),
    )
    agent = CoordinationAgent(sdk=sdk)

    result = await agent.build_solution_plan(request)

    assert [entry.agent_id for entry in result.registry_snapshot] == ["summarizer"]
    assert result.proposal.tasks[0].assigned_to == "summarizer"
    assert result.feasibility_report.feasible
    assert result.feasibility_report.matched_agents == {"t1": "summarizer"}


@pytest.mark.asyncio
async def test_build_solution_plan_records_remote_registry_failure_as_infeasible():
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        http_client=FakeHttpClient(FakeResponse(status_code=503)),
    )
    agent = CoordinationAgent(sdk=sdk)

    result = await agent.build_solution_plan(
        ProblemRequest(
            user_goal="Summarize.",
            requirements=[CapabilityRequirement(name="summarize", validation_contract={"json_schema": {"type": "object"}})],
        )
    )

    assert not result.feasibility_report.feasible
    assert result.feasibility_report.evidence[0].name == "registry_available"
    assert "Remote registry HTTP error" in result.feasibility_report.risks[0]


@pytest.mark.asyncio
async def test_build_solution_plan_retries_transient_registry_failure():
    request = ProblemRequest(
        user_goal="Summarize.",
        requirements=[CapabilityRequirement(name="summarize", validation_contract={"json_schema": {"type": "object"}})],
        required_artifacts=["summary"],
    )
    http_client = SequenceHttpClient(
        [
            FakeResponse(status_code=503),
            FakeResponse({"agents": [_card("summarizer")]}),
        ]
    )
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        http_client=http_client,
    )
    agent = CoordinationAgent(
        sdk=sdk,
        retry_policy=RetryPolicy(registry_retries=1, task_retries=0, backoff_s=0),
    )

    result = await agent.build_solution_plan(request)

    assert result.feasibility_report.feasible
    assert len(http_client.requests) == 2


@pytest.mark.asyncio
async def test_build_solution_plan_fails_closed_when_raw_request_has_no_semantic_catalog():
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        http_client=FakeHttpClient(FakeResponse({"agents": [_card("other", skill="classify")]})),
    )
    agent = CoordinationAgent(sdk=sdk)

    result = await agent.build_solution_plan("Summarize this report.")

    assert result.proposal.tasks == []
    assert not result.feasibility_report.feasible
    assert result.admission_issues[0]["code"] == "missing_catalog"


@pytest.mark.asyncio
async def test_raw_semantic_admission_fails_closed_without_or_during_interpreter():
    catalog = strategy_catalog()
    missing = CoordinationAgent(
        sdk=CoordinationSdk(),
        semantic_catalog=catalog,
    )

    missing_result = await missing.build_solution_plan("Deliver the result.")

    assert missing_result.admission_issues[0]["code"] == "missing_interpreter"

    class BrokenInterpreter:
        async def interpret(self, *_args, **_kwargs):
            raise RuntimeError("model transport failed")

    broken = CoordinationAgent(
        sdk=CoordinationSdk(),
        semantic_catalog=catalog,
        semantic_interpreter=BrokenInterpreter(),  # type: ignore[arg-type]
    )

    broken_result = await broken.build_solution_plan("Deliver the result.")

    assert broken_result.admission_issues[0]["code"] == "schema_invalid"
    assert "model transport failed" in broken_result.admission_issues[0]["message"]


@pytest.mark.asyncio
async def test_coordinate_dispatches_authorized_tasks_in_execution_order_and_aggregates():
    calls = []
    summarize = CapabilityRequirement(name="summarize", output_modes=["text"], validation_contract={"json_schema": {"type": "object"}})
    classify = CapabilityRequirement(name="classify", input_modes=["text"], output_modes=["json"], validation_contract={"json_schema": {"type": "object"}})
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))

    def summarize_handler(payload):
        calls.append("summarize")
        return {"artifacts": [{"name": "summary", "kind": "text", "text": payload["document"][:7]}]}

    def classify_handler(payload):
        calls.append("classify")
        return {"artifacts": [{"name": "label", "kind": "data", "data": {"label": "brief"}}]}

    summarizer = sdk.register_local_agent("Summarizer", [summarize], summarize_handler)
    classifier = sdk.register_local_agent("Classifier", [classify], classify_handler)
    agent = CoordinationAgent(sdk=sdk)
    request = ProblemRequest(
        user_goal="Summarize and classify.",
        requirements=[summarize, classify],
        required_artifacts=["summary", "label"],
    )

    result = await agent.coordinate(request, payload={"document": "hello world"})

    assert result.status == "completed"
    assert calls == ["summarize", "classify"]
    assert [task.agent_id for task in result.task_results] == [
        summarizer.agent_id,
        classifier.agent_id,
    ]
    assert result.artifacts == [
        {"name": "summary", "kind": "text", "text": "hello w"},
        {"name": "label", "kind": "data", "data": {"label": "brief"}},
    ]
    assert any(event.event_type == "sdk_task_completed" for event in result.trace)


@pytest.mark.asyncio
async def test_coordinate_returns_structured_infeasibility_without_dispatch():
    calls = []
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    sdk.register_local_agent(
        "Classifier",
        [CapabilityRequirement(name="classify")],
        lambda payload: calls.append("classify"),
    )
    agent = CoordinationAgent(sdk=sdk)
    request = ProblemRequest(
        user_goal="Summarize.",
        requirements=[CapabilityRequirement(name="summarize")],
        required_artifacts=["summary"],
    )

    result = await agent.coordinate(request)

    assert result.status == "infeasible"
    assert result.task_results == []
    assert calls == []
    assert not result.plan_result.feasibility_report.feasible


@pytest.mark.asyncio
async def test_build_solution_plan_authorizes_bounded_auxiliary_gap():
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    agent = CoordinationAgent(sdk=sdk)
    request = ProblemRequest(
        user_goal="Extract Q1 and Q2 revenue cells.",
        requirements=[
            CapabilityRequirement(
                name="extract revenue cells",
                description="extract Q1 and Q2 revenue values",
                output_schema={"required": ["q1_revenue", "q2_revenue"]},
                auxiliary_eligible=True,
            )
        ],
        required_artifacts=["revenue_pair"],
    )

    result = await agent.build_solution_plan(request)

    assert result.feasibility_report.feasible
    assert result.proposal.generated_nlp_agents
    assert result.proposal.tasks[0].auxiliary_spec_id == "aux-1"


@pytest.mark.asyncio
async def test_coordinate_executes_bounded_auxiliary_task():
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    agent = CoordinationAgent(sdk=sdk)
    request = ProblemRequest(
        user_goal="Extract Q1 and Q2 revenue cells.",
        requirements=[
            CapabilityRequirement(
                name="extract revenue cells",
                description="extract Q1 and Q2 revenue values",
                output_schema={"required": ["q1_revenue", "q2_revenue"]},
                auxiliary_eligible=True,
            )
        ],
        required_artifacts=["revenue_pair"],
    )

    result = await agent.coordinate(
        request,
        payload={"q1_revenue": 10, "q2_revenue": 15},
    )

    assert result.status == "completed"
    assert result.task_results[0].agent_kind == "auxiliary"
    assert result.artifacts[0]["data"] == {"q1_revenue": 10, "q2_revenue": 15}
    assert any(event.event_type == "sdk_auxiliary_task_completed" for event in result.trace)


@pytest.mark.asyncio
async def test_coordinate_retries_failed_task_and_records_attempts():
    calls = 0
    ledger = InMemoryCoordinationLedger()
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    requirement = CapabilityRequirement(name="summarize", side_effect_class="idempotent", validation_contract={"json_schema": {"type": "object"}})

    def handler(payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return {"artifacts": [{"kind": "text", "text": payload["text"]}]}

    sdk.register_local_agent("Summarizer", [requirement], handler)
    agent = CoordinationAgent(
        sdk=sdk,
        ledger=ledger,
        retry_policy=RetryPolicy(registry_retries=0, task_retries=1, backoff_s=0),
    )

    result = await agent.coordinate(
        ProblemRequest(user_goal="Summarize.", requirements=[requirement]),
        payload={"text": "ok"},
        session_id="session-retry",
    )

    events = ledger.events("session-retry")
    assert result.status == "completed"
    assert calls == 2
    assert [event.event_type for event in events].count("task_attempt_started") == 2
    assert any(event.event_type == "task_attempt_failed" for event in events)
    assert any(event.event_type == "task_attempt_completed" for event in events)


@pytest.mark.asyncio
async def test_coordinate_with_terminal_session_id_returns_recorded_result_without_redispatch():
    calls = 0
    ledger = InMemoryCoordinationLedger()
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    requirement = CapabilityRequirement(name="summarize", validation_contract={"json_schema": {"type": "object"}})

    def handler(payload):
        nonlocal calls
        calls += 1
        return {"artifacts": [{"kind": "text", "text": payload["text"]}]}

    sdk.register_local_agent("Summarizer", [requirement], handler)
    agent = CoordinationAgent(sdk=sdk, ledger=ledger)
    request = ProblemRequest(user_goal="Summarize.", requirements=[requirement])

    first = await agent.coordinate(request, payload={"text": "once"}, session_id="same-session")
    second = await agent.coordinate(request, payload={"text": "twice"}, session_id="same-session")

    assert first.status == "completed"
    assert second.artifacts == first.artifacts
    assert calls == 1


@pytest.mark.asyncio
async def test_resume_with_fresh_agent_skips_completed_tasks(tmp_path):
    calls = []
    ledger = JsonlCoordinationLedger(tmp_path / "ledger.jsonl")
    summarize = CapabilityRequirement(name="summarize", validation_contract={"json_schema": {"type": "object"}})
    classify = CapabilityRequirement(name="classify", validation_contract={"json_schema": {"type": "object"}})
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))

    summarizer = sdk.register_local_agent(
        "Summarizer",
        [summarize],
        lambda payload: calls.append("summarize") or {"artifacts": [{"kind": "text"}]},
    )
    classifier = sdk.register_local_agent(
        "Classifier",
        [classify],
        lambda payload: calls.append("classify") or {"artifacts": [{"kind": "data"}]},
    )
    request = ProblemRequest(
        user_goal="Summarize and classify.",
        requirements=[summarize, classify],
    )
    proposal = SolutionProposal(
        tasks=[
            TaskSpec(
                task_id="t1",
                requirement_name="summarize",
                assigned_to=summarizer.agent_id,
            ),
            TaskSpec(
                task_id="t2",
                requirement_name="classify",
                assigned_to=classifier.agent_id,
            ),
        ],
        execution_order=["t1", "t2"],
        completion_criteria=["all task validators pass"],
    )
    agent_report = CoordinationAgent(sdk=sdk).feasibility_analyzer.check(
        request,
        [summarizer, classifier],
        proposal,
    )
    plan_result = CoordinationPlanResult(
        session_id="recover-session",
        plan_id="recover-plan",
        request=request,
        proposal=proposal,
        feasibility_report=agent_report,
        registry_snapshot=[summarizer, classifier],
    )
    assert agent_report.feasible
    completed = TaskExecutionResult(
        session_id="recover-session",
        plan_id="recover-plan",
        attempt_id="t1-attempt-1",
        task_id="t1",
        agent_id=summarizer.agent_id,
        agent_kind="local_python",
        status="completed",
        artifacts=[{"kind": "text"}],
    )
    ledger.append(
        LedgerEvent(
            event_type="session_started",
            session_id="recover-session",
            plan_id="recover-plan",
            payload={"payload": {"text": "payload"}},
        )
    )
    ledger.append(
        LedgerEvent(
            event_type="plan_authorized",
            session_id="recover-session",
            plan_id="recover-plan",
            payload={"plan_result": plan_result.model_dump(mode="json")},
        )
    )
    ledger.append(
        LedgerEvent(
            event_type="task_attempt_completed",
            session_id="recover-session",
            plan_id="recover-plan",
            task_id="t1",
            attempt_id="t1-attempt-1",
            payload={"task_result": completed.model_dump(mode="json")},
        )
    )

    standby = CoordinationAgent(sdk=sdk, ledger=JsonlCoordinationLedger(ledger.path))
    result = await standby.resume_session("recover-session")

    assert result.status == "completed"
    assert calls == ["classify"]
    assert [task.task_id for task in result.task_results] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_resume_marks_unknown_in_flight_unsafe_attempt_without_redispatch(tmp_path):
    calls = []
    ledger = JsonlCoordinationLedger(tmp_path / "ledger.jsonl")
    requirement = CapabilityRequirement(name="control actuator", side_effect_class="unsafe")
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    actuator = sdk.register_local_agent(
        "Actuator",
        [requirement],
        lambda payload: calls.append("control") or {"artifacts": [{"kind": "data"}]},
    )
    request = ProblemRequest(
        user_goal="Control actuator.",
        requirements=[requirement],
    )
    proposal = SolutionProposal(
        tasks=[
            TaskSpec(
                task_id="t1",
                requirement_name=requirement.name,
                assigned_to=actuator.agent_id,
            )
        ],
        execution_order=["t1"],
    )
    report = CoordinationAgent(sdk=sdk).feasibility_analyzer.check(
        request,
        [actuator],
        proposal,
    )
    plan_result = CoordinationPlanResult(
        session_id="unsafe-recover-session",
        plan_id="unsafe-recover-plan",
        request=request,
        proposal=proposal,
        feasibility_report=report,
        registry_snapshot=[actuator],
    )

    ledger.append(
        LedgerEvent(
            event_type="session_started",
            session_id=plan_result.session_id,
            plan_id=plan_result.plan_id,
            payload={"payload": {"command": "open"}},
        )
    )
    ledger.append(
        LedgerEvent(
            event_type="plan_authorized",
            session_id=plan_result.session_id,
            plan_id=plan_result.plan_id,
            payload={"plan_result": plan_result.model_dump(mode="json")},
        )
    )
    ledger.append(
        LedgerEvent(
            event_type="task_attempt_started",
            session_id=plan_result.session_id,
            plan_id=plan_result.plan_id,
            task_id="t1",
            attempt_id="t1-attempt-1",
            payload={
                "idempotency_key": "unsafe-recover-session:unsafe-recover-plan:t1:t1-attempt-1"
            },
        )
    )

    standby = CoordinationAgent(sdk=sdk, ledger=JsonlCoordinationLedger(ledger.path))
    result = await standby.resume_session(plan_result.session_id)

    assert result.status == "failed"
    assert result.task_results[0].status == "unknown"
    assert "refusing blind duplicate dispatch" in result.task_results[0].error
    assert calls == []


def test_agent_constructor_and_execution_helper_invariants():
    sdk = CoordinationSdk()
    with pytest.raises(ValueError, match="must be positive"):
        CoordinationAgent(sdk=sdk, max_concurrent_dispatches=0)
    agent = CoordinationAgent(sdk=sdk, self_agent_id="self")
    for status, expected in (
        ("unknown", "task_attempt_unknown"),
        ("completed", "task_attempt_completed"),
        ("timeout", "task_attempt_timeout"),
        ("failed", "task_attempt_failed"),
    ):
        assert agent._task_event_type(
            TaskExecutionResult(task_id="t", agent_id="a", status=status)
        ) == expected
    assert agent._idempotency_key("s", "p", "t", "a") == "s:p:t:a"
    assert agent._operation_key("s", 2, "t") == "s:2:t"
    assert agent._jsonable(ValidationContract()).get("json_schema") == {}
    assert agent._jsonable("plain") == "plain"


def test_coordination_metadata_dependency_artifacts_and_completion_contract():
    payload = CoordinationAgent._with_coordination_metadata(
        {"_coordination": {"existing": "kept"}},
        session_id="s",
        plan_id="p",
        plan_generation=2,
        task_id="t",
        attempt_id="a",
        coordinator_id="c",
        fencing_token=7,
        idempotency_key="attempt-key",
        operation_key="operation-key",
        registry_revision=9,
    )
    assert payload["_coordination"]["existing"] == "kept"
    assert payload["_coordination"]["fencing_token"] == 7
    task = TaskSpec(task_id="t2", requirement_name="work", depends_on=["t1", "missing"])
    enriched = CoordinationAgent._with_dependency_artifacts(
        payload,
        task,
        {"t1": [{"name": "summary", "kind": "text"}]},
    )
    assert enriched["_coordination"]["inputs_by_task"]["missing"] == []
    assert enriched["_coordination"]["previous_artifacts"][0]["name"] == "summary"

    proposal = SolutionProposal(
        tasks=[task],
        execution_order=["t2"],
        expected_artifacts=["summary"],
        completion_criteria=["summary exists"],
    )
    completed = TaskExecutionResult(task_id="t2", agent_id="a", status="completed")
    failed = completed.model_copy(update={"status": "failed"})
    assert CoordinationAgent._completion_contract_satisfied(
        proposal, [completed], [{"data": {"summary": "x"}}]
    ) is True
    assert CoordinationAgent._completion_contract_satisfied(proposal, [], []) is False
    assert CoordinationAgent._completion_contract_satisfied(
        proposal, [failed], [{"name": "summary"}]
    ) is False
    assert CoordinationAgent._artifact_named([{"name": "summary"}], "summary") is True
    assert CoordinationAgent._artifact_named([{"name": "other"}], "summary") is False


def test_self_filter_registry_failure_and_auxiliary_noop_paths():
    requirement = CapabilityRequirement(
        name="summarize",
        input_modes=["text"],
        output_modes=["text"],
        validation_contract=ValidationContract(json_schema={"type": "object"}),
    )
    available = AgentRegistryEntry(
        agent_id="worker",
        name="Worker",
        service_endpoint="http://worker",
        skills=[requirement],
    )
    unavailable = available.model_copy(update={"agent_id": "down", "status": "unavailable"})
    agent = CoordinationAgent(sdk=CoordinationSdk(), self_agent_id="worker")
    assert agent._without_self([available, unavailable]) == [unavailable]

    request = ProblemRequest(
        user_goal="Summarize",
        requirements=[requirement],
        required_artifacts=["summary"],
    )
    unassigned = SolutionProposal(
        tasks=[
            TaskSpec(
                task_id="t1",
                requirement_name="summarize",
                requirement_id=requirement.requirement_id,
                capability_id=requirement.capability_id,
                validation_contract=requirement.validation_contract,
            )
        ],
        execution_order=["t1"],
        expected_artifacts=["summary"],
    )
    unchanged = agent._with_auxiliary_specs(
        request,
        unassigned,
        FeasibilityReport(feasible=False, missing_capabilities=[]),
    )
    assert unchanged == unassigned

    failure = agent._registry_failure_result(
        request,
        {"extra": True},
        Exception("registry offline"),  # type: ignore[arg-type]
    )
    assert failure.feasibility_report.feasible is False
    assert failure.request.context["extra"] is True
    string_failure = agent._registry_failure_result(
        "goal", None, Exception("offline")  # type: ignore[arg-type]
    )
    assert string_failure.request.user_goal == "goal"


@pytest.mark.asyncio
async def test_registry_retry_and_lease_renewal_dispatch_paths(monkeypatch):
    sdk = CoordinationSdk()
    agent = CoordinationAgent(
        sdk=sdk,
        retry_policy=RetryPolicy(registry_retries=2, backoff_s=0),
        lease_renew_interval_s=0.001,
    )
    calls = 0

    async def flaky_snapshot(*, refresh=False):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RemoteRegistryError("temporary")
        return []

    monkeypatch.setattr(sdk, "registry_snapshot", flaky_snapshot)
    assert await agent._registry_snapshot_with_retry() == []
    assert calls == 3

    now = datetime.now(timezone.utc)
    lease = LeaseRecord(
        session_id="s",
        holder_id="c",
        fencing_token=1,
        expires_at=now + timedelta(seconds=30),
        heartbeat_at=now,
    )
    task = TaskSpec(task_id="t", requirement_name="work")
    report = FeasibilityReport(feasible=True, matched_agents={"t": "worker"})

    async def completed(*args, **kwargs):
        del args, kwargs
        return TaskExecutionResult(task_id="t", agent_id="worker", status="completed")

    monkeypatch.setattr(agent, "_bounded_send_task", completed)
    result = await agent._send_task_with_lease_renewal(
        report, task, {}, timeout_s=1, lease=lease
    )
    assert result.status == "completed"

    async def slow(*args, **kwargs):
        del args, kwargs
        await asyncio.sleep(10)

    async def renewal_failure(_lease):
        raise RuntimeError("quorum lost during renewal")

    monkeypatch.setattr(agent, "_bounded_send_task", slow)
    monkeypatch.setattr(agent, "_lease_renewal_loop", renewal_failure)
    with pytest.raises(RuntimeError, match="quorum lost"):
        await agent._send_task_with_lease_renewal(
            report, task, {}, timeout_s=1, lease=lease
        )


def test_fault_injection_order_payload_and_completion_helpers(monkeypatch):
    agent = CoordinationAgent(sdk=CoordinationSdk())
    agent._inject_fault("not-enabled")
    monkeypatch.setenv("COORDINATION_FAULT_AT", "point")
    monkeypatch.setenv("COORDINATION_FAULT_MODE", "abandon_lease")
    with pytest.raises(RuntimeError, match="Injected"):
        agent._inject_fault("point")
    assert agent._abandon_current_lease is True
    monkeypatch.setenv("COORDINATION_FAULT_MODE", "raise")
    with pytest.raises(RuntimeError, match="Injected"):
        agent._inject_fault("point")

    first = TaskSpec(task_id="first", requirement_name="one")
    second = TaskSpec(task_id="second", requirement_name="two")
    ordered = SolutionProposal(tasks=[first, second], execution_order=["second", "missing"])
    assert agent._ordered_tasks(ordered) == [second]
    unordered = ordered.model_copy(update={"execution_order": []})
    assert agent._ordered_tasks(unordered) == [first, second]
    shared = {"_coordination": {"session_id": "s"}, "first": {"value": 1}}
    assert agent._payload_for_task(shared, first) == {
        "value": 1,
        "_coordination": {"session_id": "s"},
    }
    assert agent._payload_for_task({}, first) == {}
    assert agent._payload_for_task({"value": 2}, first) == {"value": 2}
