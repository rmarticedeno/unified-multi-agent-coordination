"""HTTP-only Docker system tests for the coordination service."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: str
    reference_status: str
    observed_status: str
    passed: bool
    latency_ms: float
    response: JsonObject
    error: str = ""


async def run_system_tests() -> JsonObject:
    """Run the Docker system matrix through HTTP endpoints only."""
    base_url = os.getenv("COORDINATION_BASE_URL", "http://coordination-service:8000")
    bad_registry_url = os.getenv(
        "BAD_REGISTRY_BASE_URL", "http://coordination-bad-registry:8000"
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        await _wait_for_health(client, base_url)
        await _wait_for_health(client, bad_registry_url)

        registry_response = await client.get(f"{base_url}/registry")
        registry_response.raise_for_status()
        registry = registry_response.json()["agents"]

        outcomes = [
            await _health(client, base_url),
            _registry_discovery(registry),
            await _direct_remote_success(client, base_url),
            await _multi_remote_success(client, base_url),
            await _missing_capability_refusal(client, base_url),
            await _authority_refusal(client, base_url),
            await _mode_dependency_rejection(client, base_url, registry),
            await _bad_artifact_runtime_failure(client, base_url),
            await _timeout_runtime_failure(client, base_url),
            await _registry_outage(client, bad_registry_url),
            await _resume_session(client, base_url),
        ]

    return {
        "generated_at_unix": time.time(),
        "base_url": base_url,
        "bad_registry_base_url": bad_registry_url,
        "scenario_count": len(outcomes),
        "passed": all(outcome.passed for outcome in outcomes),
        "scenarios": [_scenario_record(outcome) for outcome in outcomes],
    }


async def _health(client: httpx.AsyncClient, base_url: str) -> ScenarioOutcome:
    started = time.perf_counter()
    try:
        response = await client.get(f"{base_url}/health")
        response.raise_for_status()
        body = response.json()
        passed = body.get("status") == "ok"
        return _outcome("health", "ok", body.get("status", ""), passed, started, body)
    except Exception as exc:
        return _outcome("health", "ok", "error", False, started, {}, str(exc))


def _registry_discovery(registry: list[JsonObject]) -> ScenarioOutcome:
    started = time.perf_counter()
    expected = {
        "summarizer",
        "extractor",
        "calculator",
        "formatter",
        "bad-artifact",
        "slow-agent",
        "control-agent",
    }
    observed = {str(agent.get("agent_id")) for agent in registry}
    passed = expected <= observed
    return _outcome(
        "registry_discovery",
        "all_agents_visible",
        "all_agents_visible" if passed else "missing_agents",
        passed,
        started,
        {"agents": registry, "missing": sorted(expected - observed)},
    )


async def _direct_remote_success(
    client: httpx.AsyncClient, base_url: str
) -> ScenarioOutcome:
    return await _coordinate_scenario(
        client,
        base_url,
        "remote_a2a_direct_success",
        "completed",
        {
            "problem": {
                "user_goal": "Summarize the document.",
                "requirements": [
                    {
                        "name": "summarize",
                        "input_modes": ["text"],
                        "output_modes": ["data"],
                        "validation_contract": {"required_fields": ["summary"]},
                    }
                ],
                "required_artifacts": ["summary"],
            },
            "payload": {"text": "short report"},
            "session_id": "docker-direct-success",
        },
        lambda body: (
            body.get("status") == "completed"
            and _trace_has_authorized_dispatch(body)
            and _artifact_field_exists(body, "summary")
        ),
    )


async def _multi_remote_success(
    client: httpx.AsyncClient, base_url: str
) -> ScenarioOutcome:
    return await _coordinate_scenario(
        client,
        base_url,
        "remote_a2a_multi_agent_success",
        "completed",
        {
            "problem": {
                "user_goal": "Extract facts, calculate a value, and format a report.",
                "requirements": [
                    {
                        "name": "extract facts",
                        "input_modes": ["text"],
                        "output_modes": ["facts"],
                        "validation_contract": {"required_fields": ["facts"]},
                    },
                    {
                        "name": "calculate metric",
                        "input_modes": ["facts"],
                        "output_modes": ["metric"],
                        "validation_contract": {"required_fields": ["metric"]},
                    },
                    {
                        "name": "format report",
                        "input_modes": ["metric"],
                        "output_modes": ["report"],
                        "validation_contract": {"required_artifacts": ["final_report"]},
                    },
                ],
                "required_artifacts": ["final_report"],
            },
            "payload": {"document": "alpha beta gamma"},
            "session_id": "docker-multi-success",
        },
        lambda body: (
            body.get("status") == "completed"
            and len(body.get("task_results", [])) == 3
            and _artifact_field_exists(body, "final_report")
            and _received_previous_count(body) == 1
            and _trace_has_authorized_dispatch(body)
        ),
    )


async def _missing_capability_refusal(
    client: httpx.AsyncClient, base_url: str
) -> ScenarioOutcome:
    return await _coordinate_scenario(
        client,
        base_url,
        "missing_capability_refusal",
        "infeasible",
        {
            "problem": {
                "user_goal": "Forecast next quarter.",
                "requirements": [{"name": "forecast revenue"}],
                "required_artifacts": ["forecast"],
            }
        },
        lambda body: (
            body.get("status") == "infeasible"
            and not body.get("task_results")
            and "forecast revenue"
            in body["plan_result"]["feasibility_report"]["missing_capabilities"]
        ),
    )


async def _authority_refusal(
    client: httpx.AsyncClient, base_url: str
) -> ScenarioOutcome:
    return await _coordinate_scenario(
        client,
        base_url,
        "authority_refusal",
        "infeasible",
        {
            "problem": {
                "user_goal": "Operate a protected actuator.",
                "requirements": [
                    {
                        "name": "control actuator",
                        "required_trust_level": "admin",
                    }
                ],
                "required_artifacts": ["control_receipt"],
            }
        },
        lambda body: (
            body.get("status") == "infeasible"
            and _predicate_failed(body, "authorized")
            and not body.get("task_results")
        ),
    )


async def _mode_dependency_rejection(
    client: httpx.AsyncClient,
    base_url: str,
    registry: list[JsonObject],
) -> ScenarioOutcome:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url}/feasibility",
            json={
                "request": {
                    "user_goal": "Summarize and then calculate.",
                    "requirements": [
                        {
                            "name": "summarize",
                            "input_modes": ["text"],
                            "output_modes": ["text"],
                        },
                        {
                            "name": "calculate metric",
                            "input_modes": ["number"],
                            "output_modes": ["metric"],
                        },
                    ],
                    "required_artifacts": ["metric"],
                },
                "proposal": {
                    "tasks": [
                        {
                            "task_id": "t1",
                            "requirement_name": "summarize",
                            "assigned_to": "summarizer",
                        },
                        {
                            "task_id": "t2",
                            "requirement_name": "calculate metric",
                            "assigned_to": "calculator",
                            "depends_on": ["t1"],
                        },
                    ],
                    "execution_order": ["t1", "t2"],
                    "expected_artifacts": ["metric"],
                    "completion_criteria": ["metric artifact exists"],
                },
                "registry_snapshot": registry,
            },
        )
        response.raise_for_status()
        body = response.json()
        passed = body.get("feasible") is False and _report_predicate_failed(
            body, "compatible"
        )
        return _outcome(
            "mode_dependency_rejection",
            "infeasible",
            "infeasible" if body.get("feasible") is False else "feasible",
            passed,
            started,
            {
                "plan_result": {
                    "feasibility_report": body,
                    "registry_snapshot": registry,
                },
                "feasible": body.get("feasible"),
            },
        )
    except Exception as exc:
        return _outcome(
            "mode_dependency_rejection",
            "infeasible",
            "error",
            False,
            started,
            {},
            str(exc),
        )


async def _bad_artifact_runtime_failure(
    client: httpx.AsyncClient, base_url: str
) -> ScenarioOutcome:
    return await _coordinate_scenario(
        client,
        base_url,
        "bad_artifact_runtime_failure",
        "failed",
        {
            "problem": {
                "user_goal": "Produce a validated artifact.",
                "requirements": [
                    {
                        "name": "produce validated artifact",
                        "validation_contract": {"required_fields": ["approved"]},
                    }
                ],
                "required_artifacts": ["approved"],
            },
            "session_id": "docker-bad-artifact",
        },
        lambda body: (
            body.get("status") == "failed"
            and body.get("plan_result", {})
            .get("feasibility_report", {})
            .get("feasible")
            is True
            and "approved" in body["task_results"][0].get("error", "")
        ),
    )


async def _timeout_runtime_failure(
    client: httpx.AsyncClient, base_url: str
) -> ScenarioOutcome:
    return await _coordinate_scenario(
        client,
        base_url,
        "timeout_runtime_failure",
        "failed",
        {
            "problem": {
                "user_goal": "Run the slow summarizer.",
                "requirements": [
                    {
                        "name": "slow summarize",
                        "validation_contract": {
                            "required_artifacts": ["slow_summary"]
                        },
                    }
                ],
                "required_artifacts": ["slow_summary"],
            },
            "timeout_s": 0.25,
            "session_id": "docker-timeout",
        },
        lambda body: (
            body.get("status") == "failed"
            and any(
                task.get("status") in {"timeout", "failed"}
                for task in body.get("task_results", [])
            )
        ),
    )


async def _registry_outage(
    client: httpx.AsyncClient, bad_registry_url: str
) -> ScenarioOutcome:
    return await _coordinate_scenario(
        client,
        bad_registry_url,
        "registry_outage",
        "infeasible",
        {
            "problem": {
                "user_goal": "Summarize despite missing registry.",
                "requirements": [{"name": "summarize"}],
            }
        },
        lambda body: (
            body.get("status") == "infeasible"
            and body["plan_result"]["feasibility_report"]["evidence"][0]["name"]
            == "registry_available"
        ),
    )


async def _resume_session(client: httpx.AsyncClient, base_url: str) -> ScenarioOutcome:
    started = time.perf_counter()
    try:
        first = await client.post(
            f"{base_url}/coordinate",
            json={
                "problem": {
                    "user_goal": "Summarize once.",
                    "requirements": [
                        {
                            "name": "summarize",
                            "validation_contract": {"required_fields": ["summary"]},
                        }
                    ],
                },
                "session_id": "docker-resume-session",
            },
        )
        first.raise_for_status()
        resumed = await client.post(
            f"{base_url}/sessions/docker-resume-session/resume",
            json={},
        )
        resumed.raise_for_status()
        first_body = first.json()
        body = resumed.json()
        passed = (
            first_body.get("status") == "completed"
            and body.get("status") == "completed"
            and body.get("artifacts") == first_body.get("artifacts")
        )
        return _outcome(
            "resume_session",
            "completed",
            body.get("status", ""),
            passed,
            started,
            {"first": first_body, "resumed": body},
        )
    except Exception as exc:
        return _outcome(
            "resume_session",
            "completed",
            "error",
            False,
            started,
            {},
            str(exc),
        )


async def _coordinate_scenario(
    client: httpx.AsyncClient,
    base_url: str,
    scenario_id: str,
    reference_status: str,
    request: JsonObject,
    validator,
) -> ScenarioOutcome:
    started = time.perf_counter()
    try:
        response = await client.post(f"{base_url}/coordinate", json=request)
        response.raise_for_status()
        body = response.json()
        observed = str(body.get("status", ""))
        return _outcome(
            scenario_id,
            reference_status,
            observed,
            observed == reference_status and validator(body),
            started,
            body,
        )
    except Exception as exc:
        return _outcome(
            scenario_id,
            reference_status,
            "error",
            False,
            started,
            {},
            str(exc),
        )


async def _wait_for_health(
    client: httpx.AsyncClient,
    base_url: str,
    attempts: int = 60,
) -> None:
    for _ in range(attempts):
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200 and response.json().get("status") == "ok":
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(f"{base_url} did not become healthy.")


def _trace_event_types(body: JsonObject) -> list[str]:
    return [str(event.get("event_type")) for event in body.get("trace", [])]


def _trace_has_authorized_dispatch(body: JsonObject) -> bool:
    events = _trace_event_types(body)
    if "plan_authorized" not in events or "task_attempt_started" not in events:
        return False
    return events.index("plan_authorized") < events.index("task_attempt_started")


def _artifact_field_exists(body: JsonObject, field: str) -> bool:
    wanted = field.lower()
    for artifact in body.get("artifacts", []):
        if wanted in {str(key).lower() for key in artifact}:
            return True
        data = artifact.get("data")
        if isinstance(data, dict) and wanted in {str(key).lower() for key in data}:
            return True
        if str(artifact.get("name", "")).lower() == wanted:
            return True
    return False


def _received_previous_count(body: JsonObject) -> int:
    for artifact in body.get("artifacts", []):
        data = artifact.get("data")
        if isinstance(data, dict) and "received_previous_artifact_count" in data:
            return int(data["received_previous_artifact_count"])
    return 0


def _predicate_failed(body: JsonObject, name: str) -> bool:
    report = body["plan_result"]["feasibility_report"]
    return _report_predicate_failed(report, name)


def _report_predicate_failed(report: JsonObject, name: str) -> bool:
    return any(
        item.get("name") == name and item.get("passed") is False
        for item in report.get("evidence", [])
    )


def _outcome(
    scenario_id: str,
    reference_status: str,
    observed_status: str,
    passed: bool,
    started: float,
    response: JsonObject,
    error: str = "",
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=scenario_id,
        reference_status=reference_status,
        observed_status=observed_status,
        passed=passed,
        latency_ms=(time.perf_counter() - started) * 1000,
        response=response,
        error=error,
    )


def _scenario_record(outcome: ScenarioOutcome) -> JsonObject:
    response = outcome.response
    evidence_response = _primary_response(response)
    trace_response = evidence_response if evidence_response else response
    plan_result = response.get("plan_result", {})
    if not plan_result and isinstance(evidence_response, dict):
        plan_result = evidence_response.get("plan_result", {})
    feasibility = plan_result.get("feasibility_report", {})
    registry_snapshot = plan_result.get("registry_snapshot", [])
    if not registry_snapshot and isinstance(response.get("agents"), list):
        registry_snapshot = response["agents"]
    return {
        "id": outcome.scenario_id,
        "reference_status": outcome.reference_status,
        "observed_status": outcome.observed_status,
        "passed": outcome.passed,
        "latency_ms": round(outcome.latency_ms, 2),
        "error": outcome.error,
        "registry_snapshot": registry_snapshot,
        "predicate_evidence": feasibility.get("evidence", []),
        "task_results": evidence_response.get("task_results", []),
        "artifacts": evidence_response.get("artifacts", []),
        "artifact_validation": _artifact_validation(evidence_response),
        "trace_event_types": _trace_event_types(trace_response),
        "no_dispatch_before_authorization": _trace_has_authorized_dispatch(
            trace_response
        ),
        "raw_response": response,
    }


def _primary_response(response: JsonObject) -> JsonObject:
    if isinstance(response.get("resumed"), dict):
        return response["resumed"]
    if isinstance(response.get("first"), dict):
        return response["first"]
    return response


def _artifact_validation(response: JsonObject) -> list[JsonObject]:
    validations: list[JsonObject] = []
    for task in response.get("task_results", []):
        metadata = task.get("metadata")
        validation_errors = []
        if isinstance(metadata, dict) and isinstance(
            metadata.get("validation_errors"), list
        ):
            validation_errors = [str(error) for error in metadata["validation_errors"]]

        status = str(task.get("status", ""))
        if validation_errors:
            validation_status = "failed"
        elif status == "completed":
            validation_status = "passed"
        elif status in {"failed", "timeout"}:
            validation_status = "not_applicable"
        else:
            validation_status = "unknown"

        validations.append(
            {
                "task_id": task.get("task_id", ""),
                "agent_id": task.get("agent_id", ""),
                "status": validation_status,
                "artifact_count": len(task.get("artifacts", [])),
                "errors": validation_errors,
            }
        )
    return validations


def main() -> None:
    """Run the system tests and write a JSON report."""
    report = asyncio.run(run_system_tests())
    output_path = Path(
        os.getenv("SYSTEM_REPORT_PATH", "demo_runs/docker_system_report.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for scenario in report["scenarios"]:
        print(
            f"{scenario['id']}: {scenario['observed_status']} "
            f"(reference={scenario['reference_status']}, passed={scenario['passed']})"
        )
    print(f"Wrote system report to {output_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
