from fastapi.testclient import TestClient

from unified_multi_agent_coordination import (
    CapabilityRequirement,
    CoordinationSdk,
)
from unified_multi_agent_coordination.service import create_app


def _client_with_local_summarizer() -> TestClient:
    sdk = CoordinationSdk()
    sdk.register_local_agent(
        "Summarizer",
        [CapabilityRequirement(name="summarize", output_modes=["text"])],
        lambda payload: {
            "artifacts": [
                {
                    "kind": "text",
                    "text": payload.get("text", ""),
                }
            ]
        },
    )
    return TestClient(create_app(sdk=sdk))


def test_health_endpoint():
    client = TestClient(create_app(sdk=CoordinationSdk()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_registry_endpoint_returns_registered_agents():
    client = _client_with_local_summarizer()

    response = client.get("/registry")

    assert response.status_code == 200
    assert response.json()["agents"][0]["agent_id"] == "summarizer"


def test_plan_endpoint_returns_authorized_direct_plan():
    client = _client_with_local_summarizer()

    response = client.post(
        "/plan",
        json={
            "problem": {
                "user_goal": "Summarize.",
                "requirements": [{"name": "summarize"}],
                "required_artifacts": ["summary"],
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["feasibility_report"]["feasible"] is True


def test_coordinate_endpoint_returns_local_artifact():
    client = _client_with_local_summarizer()

    response = client.post(
        "/coordinate",
        json={
            "problem": {
                "user_goal": "Summarize.",
                "requirements": [{"name": "summarize"}],
                "required_artifacts": ["summary"],
            },
            "payload": {"text": "short summary"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["artifacts"] == [{"kind": "text", "text": "short summary"}]


def test_coordinate_endpoint_returns_infeasible_without_agent():
    client = TestClient(create_app(sdk=CoordinationSdk()))

    response = client.post(
        "/coordinate",
        json={
            "problem": {
                "user_goal": "Summarize.",
                "requirements": [{"name": "summarize"}],
                "required_artifacts": ["summary"],
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "infeasible"


def test_feasibility_endpoint_checks_explicit_plan():
    client = _client_with_local_summarizer()

    response = client.post(
        "/feasibility",
        json={
            "request": {
                "user_goal": "Summarize.",
                "requirements": [{"name": "summarize"}],
                "required_artifacts": ["summary"],
            },
            "proposal": {
                "tasks": [
                    {
                        "task_id": "t1",
                        "requirement_name": "summarize",
                        "assigned_to": "summarizer",
                    }
                ],
                "execution_order": ["t1"],
                "expected_artifacts": ["summary"],
                "completion_criteria": ["summary exists"],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["feasible"] is True
