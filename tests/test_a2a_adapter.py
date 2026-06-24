import asyncio

import pytest
from a2a import types as a2a_types

from unified_multi_agent_coordination import (
    A2AAdapter,
    AuthorizationError,
    CapabilityRequirement,
    FeasibilityAnalyzer,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


@pytest.mark.asyncio
async def test_agent_card_ingestion_and_authorized_dispatch():
    async def fetcher(url: str):
        return {
            "name": "summarizer",
            "description": "Summarizes text",
            "url": url,
            "skills": [{"id": "summarize", "description": "Create summaries"}],
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
        }

    sent = []

    async def sender(agent_id: str, payload: dict):
        sent.append((agent_id, payload))
        return {"status": "completed", "parts": [{"kind": "text", "text": "done"}]}

    adapter = A2AAdapter(fetcher, sender)
    agent = await adapter.register_from_card_url("http://summarizer.example")
    req = CapabilityRequirement(name="summarize")
    request = ProblemRequest(
        user_goal="Summarize.",
        requirements=[req],
        required_artifacts=["summary"],
    )
    proposal = SolutionProposal(
        tasks=[TaskSpec(task_id="t1", requirement_name="summarize", assigned_to="summarizer")],
        execution_order=["t1"],
        expected_artifacts=["summary"],
        completion_criteria=["summary artifact exists"],
    )
    report = FeasibilityAnalyzer().check(request, [agent], proposal)

    result = await adapter.send_task_after_authorization(
        report, proposal.tasks[0], {"input": "hello"}
    )

    assert result["status"] == "completed"
    assert sent == [("summarizer", {"input": "hello"})]
    assert adapter.trace[-1].event_type == "delegation_completed"


@pytest.mark.asyncio
async def test_dispatch_without_authorization_is_blocked():
    async def fetcher(url: str):
        return {}

    async def sender(agent_id: str, payload: dict):
        return {}

    adapter = A2AAdapter(fetcher, sender)
    task = TaskSpec(task_id="t1", requirement_name="summarize", assigned_to="summarizer")
    report = FeasibilityAnalyzer().check(
        ProblemRequest(user_goal="x", requirements=[CapabilityRequirement(name="summarize")]),
        [],
        SolutionProposal(tasks=[task]),
    )

    with pytest.raises(AuthorizationError):
        await adapter.send_task_after_authorization(report, task, {})

    assert adapter.trace[-1].event_type == "delegation_refused"


@pytest.mark.asyncio
async def test_timeout_is_recorded():
    async def fetcher(url: str):
        return {}

    async def sender(agent_id: str, payload: dict):
        await asyncio.sleep(0.05)
        return {}

    adapter = A2AAdapter(fetcher, sender)
    task = TaskSpec(task_id="t1", requirement_name="summarize", assigned_to="summarizer")
    report = FeasibilityAnalyzer().check(
        ProblemRequest(
            user_goal="x",
            requirements=[CapabilityRequirement(name="summarize")],
            required_artifacts=["summary"],
        ),
        [],
        SolutionProposal(
            tasks=[task],
            execution_order=["t1"],
            expected_artifacts=["summary"],
            completion_criteria=["done"],
        ),
    )
    report.feasible = True

    with pytest.raises(TimeoutError):
        await adapter.send_task_after_authorization(report, task, {}, timeout_s=0.001)

    assert adapter.trace[-1].event_type == "delegation_timeout"


def test_artifact_part_conversion_handles_text_data_and_file():
    async def fetcher(url: str):
        return {}

    async def sender(agent_id: str, payload: dict):
        return {}

    adapter = A2AAdapter(fetcher, sender)

    converted = adapter.convert_artifact_parts(
        [
            {"kind": "text", "text": "hello"},
            {"kind": "data", "data": {"value": 42}},
            {
                "kind": "file",
                "file": {
                    "name": "report.txt",
                    "mime_type": "text/plain",
                    "bytes": "aGVsbG8=",
                },
            },
        ]
    )

    assert converted[0] == {"kind": "text", "text": "hello"}
    assert converted[1] == {"kind": "data", "data": {"value": 42}}
    assert converted[2]["name"] == "report.txt"


def test_normalize_card_supports_protobuf_agent_card_interfaces():
    async def fetcher(url: str):
        return {}

    async def sender(agent_id: str, payload: dict):
        return {}

    adapter = A2AAdapter(fetcher, sender)
    card = a2a_types.AgentCard(
        name="summarizer",
        description="Summarizes text",
        version="1.0.0",
        supported_interfaces=[
            a2a_types.AgentInterface(
                url="http://summarizer.example/a2a",
                protocol_binding="JSONRPC",
            )
        ],
        skills=[
            a2a_types.AgentSkill(
                id="summarize",
                name="summarize",
                description="Create summaries",
                input_modes=["text"],
                output_modes=["text"],
            )
        ],
        default_input_modes=["text"],
        default_output_modes=["text"],
    )

    entry = adapter.normalize_card(card)

    assert entry.agent_id == "summarizer"
    assert entry.service_endpoint == "http://summarizer.example/a2a"
    assert entry.input_modes == ["text"]
    assert entry.output_modes == ["text"]
    assert entry.skills[0].name == "summarize"
