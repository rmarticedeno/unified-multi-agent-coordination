import httpx
import pytest
from lingo.mock import MockLLM

from unified_multi_agent_coordination import (
    CapabilityRequirement,
    CoordinationAgent,
    CoordinationSdk,
    LingoLinguisticCoordinator,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


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
        requirements=[CapabilityRequirement(name="summarize")],
        required_artifacts=["summary"],
    )
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        self_agent_id="coordinator",
        http_client=FakeHttpClient(
            FakeResponse({"agents": [_card("coordinator"), _card("summarizer")]})
        ),
    )
    linguistic = LingoLinguisticCoordinator(llm=MockLLM([request]))
    agent = CoordinationAgent(sdk=sdk, linguistic_coordinator=linguistic)

    result = await agent.build_solution_plan("Summarize this report.")

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
            requirements=[CapabilityRequirement(name="summarize")],
        )
    )

    assert not result.feasibility_report.feasible
    assert result.feasibility_report.evidence[0].name == "registry_available"
    assert "Remote registry HTTP error" in result.feasibility_report.risks[0]


@pytest.mark.asyncio
async def test_build_solution_plan_uses_linguistic_planner_when_direct_match_is_missing():
    request = ProblemRequest(
        user_goal="Summarize the report.",
        requirements=[CapabilityRequirement(name="summarize")],
        required_artifacts=["summary"],
    )
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        http_client=FakeHttpClient(FakeResponse({"agents": [_card("other", skill="classify")]})),
    )
    linguistic = LingoLinguisticCoordinator(
        llm=MockLLM(
            [
                request,
                SolutionProposal(
                    tasks=[
                        TaskSpec(
                            task_id="t1",
                            requirement_name="summarize",
                            assigned_to="other",
                        )
                    ],
                    execution_order=["t1"],
                    expected_artifacts=["summary"],
                    completion_criteria=["summary exists"],
                ),
            ]
        )
    )
    agent = CoordinationAgent(sdk=sdk, linguistic_coordinator=linguistic)

    result = await agent.build_solution_plan("Summarize this report.")

    assert result.proposal.tasks[0].assigned_to == "other"
    assert not result.feasibility_report.feasible
