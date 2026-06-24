import pytest

from lingo.mock import MockLLM

from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    CapabilityRequirement,
    LingoLinguisticCoordinator,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


def _agent() -> AgentRegistryEntry:
    return AgentRegistryEntry(
        agent_id="summarizer",
        name="summarizer",
        service_endpoint="http://agent.example",
        skills=[
            CapabilityRequirement(
                name="summarize",
                input_modes=["text"],
                output_modes=["text"],
            )
        ],
        input_modes=["text"],
        output_modes=["text"],
    )


@pytest.mark.asyncio
async def test_structured_interpretation_valid_request():
    typed_request = ProblemRequest(
        user_goal="Summarize the report.",
        requirements=[
            CapabilityRequirement(
                name="summarize",
                input_modes=["text"],
                output_modes=["text"],
            )
        ],
        required_artifacts=["summary"],
    )
    coordinator = LingoLinguisticCoordinator(llm=MockLLM([typed_request]))

    result = await coordinator.interpret_request("Summarize this report.", [_agent()])

    assert result.user_goal == "Summarize the report."
    assert coordinator.state.interpreted_request["required_artifacts"] == ["summary"]
    assert coordinator.state.trace[-1]["event_type"] == "request_interpreted"


@pytest.mark.asyncio
async def test_structured_interpretation_records_ambiguity_as_context():
    ambiguous = ProblemRequest(
        user_goal="Analyze the document.",
        requirements=[],
        context={"ambiguities": ["analysis type is unspecified"]},
    )
    coordinator = LingoLinguisticCoordinator(llm=MockLLM([ambiguous]))

    result = await coordinator.interpret_request("Analyze it.", [_agent()])

    assert result.requirements == []
    assert result.context["ambiguities"] == ["analysis type is unspecified"]


@pytest.mark.asyncio
async def test_structured_interpretation_rejects_malformed_mock_output():
    coordinator = LingoLinguisticCoordinator(llm=MockLLM(["not a pydantic model"]))

    with pytest.raises(TypeError):
        await coordinator.interpret_request("Summarize this report.", [_agent()])


@pytest.mark.asyncio
async def test_candidate_solution_is_non_authoritative_until_checked():
    request = ProblemRequest(
        user_goal="Summarize the report.",
        requirements=[CapabilityRequirement(name="summarize")],
        required_artifacts=["summary"],
    )
    proposal = SolutionProposal(
        tasks=[
            TaskSpec(
                task_id="t1",
                requirement_name="summarize",
                assigned_to="summarizer",
                expected_artifacts=["summary"],
            )
        ],
        selected_agents={"t1": "summarizer"},
        execution_order=["t1"],
        expected_artifacts=["summary"],
        completion_criteria=["summary artifact exists"],
    )
    coordinator = LingoLinguisticCoordinator(llm=MockLLM([proposal]))

    result = await coordinator.propose_solution(request, [_agent()])

    assert result.tasks[0].assigned_to == "summarizer"
    assert coordinator.state.candidate_plan["tasks"][0]["task_id"] == "t1"
    assert coordinator.state.trace[-1]["event_type"] == "solution_proposed"
