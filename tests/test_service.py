from fastapi.testclient import TestClient

from unified_multi_agent_coordination import (
    CapabilityRequirement,
    CoordinationAgent,
    CoordinationSdk,
    InMemoryCoordinationLedger,
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
    assert body["session_id"]
    assert body["artifacts"] == [{"kind": "text", "text": "short summary"}]


def test_coordinate_endpoint_accepts_session_id_and_resume_returns_terminal_result():
    ledger = InMemoryCoordinationLedger()
    sdk = CoordinationSdk()
    calls = []
    sdk.register_local_agent(
        "Summarizer",
        [CapabilityRequirement(name="summarize", output_modes=["text"])],
        lambda payload: calls.append(payload.get("text", "")) or {
            "artifacts": [{"kind": "text", "text": payload.get("text", "")}]
        },
    )
    client = TestClient(create_app(sdk=sdk, agent=CoordinationAgent(sdk=sdk, ledger=ledger)))

    response = client.post(
        "/coordinate",
        json={
            "session_id": "api-session",
            "problem": {
                "user_goal": "Summarize.",
                "requirements": [{"name": "summarize"}],
            },
            "payload": {"text": "first"},
        },
    )
    resumed = client.post("/sessions/api-session/resume", json={})

    assert response.status_code == 200
    assert resumed.status_code == 200
    assert resumed.json()["session_id"] == "api-session"
    assert resumed.json()["artifacts"] == [{"kind": "text", "text": "first"}]
    assert calls == ["first"]


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
