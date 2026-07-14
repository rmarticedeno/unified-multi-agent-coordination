"""HTTP and store-level distributed coordinator checks for Docker Compose."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .coordination_ledger import LedgerEvent
from .coordination_store import (
    PostgresCoordinationStore,
    StaleFenceError,
    StoreInvariantError,
)


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class DistributedOutcome:
    scenario_id: str
    passed: bool
    latency_ms: float
    response: JsonObject
    error: str = ""


async def run_distributed_tests() -> JsonObject:
    """Run distributed checks across multiple coordinator replicas."""
    coordinator_a = os.getenv("COORDINATION_A_URL", "http://coordination-a:8000")
    coordinator_b = os.getenv("COORDINATION_B_URL", "http://coordination-b:8000")
    coordinator_c = os.getenv("COORDINATION_C_URL", "http://coordination-c:8000")
    slow_agent = os.getenv("SLOW_AGENT_URL", "http://slow-agent:8000")
    registry_url = os.getenv("COORDINATION_REGISTRY_URL", "http://registry:8000/agents")
    store_url = os.getenv("COORDINATION_STORE_URL", "")
    run_id = uuid4().hex[:8]
    async with httpx.AsyncClient(timeout=10.0) as client:
        await asyncio.gather(
            _wait_for_health(client, coordinator_a),
            _wait_for_health(client, coordinator_b),
            _wait_for_health(client, coordinator_c),
            _wait_for_health(client, slow_agent),
        )
        outcomes = [
            await _lease_conflict(client, coordinator_b, coordinator_c, run_id),
            await _recover_after_abandoned_attempt(
                client,
                coordinator_a,
                coordinator_b,
                slow_agent,
                run_id,
            ),
            await _process_exit_after_external_dispatch_unknown(
                client,
                coordinator_b,
                slow_agent,
                registry_url,
                store_url,
                run_id,
            ),
            await _stale_fence_rejection(store_url, run_id),
            await _postgres_invariant_probe(store_url, run_id),
        ]
        fixture_stats = await _fixture_stats(client, slow_agent)

    report = {
        "generated_at_unix": time.time(),
        "coordinators": {
            "a": coordinator_a,
            "b": coordinator_b,
            "c": coordinator_c,
        },
        "store_url_present": bool(store_url),
        "scenario_count": len(outcomes),
        "passed": all(outcome.passed for outcome in outcomes),
        "scenarios": [_outcome_record(outcome) for outcome in outcomes],
        "fixture_stats": fixture_stats,
        "metrics": _metrics(outcomes, fixture_stats),
    }
    return report


async def _lease_conflict(
    client: httpx.AsyncClient,
    owner_url: str,
    contender_url: str,
    run_id: str,
) -> DistributedOutcome:
    started = time.perf_counter()
    session_id = f"distributed-live-lease-conflict-{run_id}"
    payload = {
        "session_id": session_id,
        "problem": {
            "user_goal": "Slow summarize.",
            "requirements": [
                {
                    "name": "slow summarize",
                    "side_effect_class": "read_only",
                    "validation_contract": {
                        "required_artifacts": ["slow_summary"]
                    },
                }
            ],
            "required_artifacts": ["slow_summary"],
        },
        "payload": {"text": "lease conflict check"},
        "timeout_s": 5.0,
    }
    owner_task = asyncio.create_task(
        client.post(f"{owner_url}/coordinate", json=payload)
    )
    await asyncio.sleep(0.25)
    contender_response = await client.post(f"{contender_url}/coordinate", json=payload)
    owner_response = await owner_task
    owner_body = _safe_json(owner_response)
    passed = contender_response.status_code == 409 and owner_body.get("status") in {
        "completed",
        "failed",
    }
    return DistributedOutcome(
        scenario_id="live_lease_conflict",
        passed=passed,
        latency_ms=_elapsed_ms(started),
        response={
            "owner_status_code": owner_response.status_code,
            "owner_status": owner_body.get("status"),
            "contender_status_code": contender_response.status_code,
            "contender_body": _safe_json(contender_response),
        },
    )


async def _recover_after_abandoned_attempt(
    client: httpx.AsyncClient,
    faulting_url: str,
    standby_url: str,
    agent_url: str,
    run_id: str,
) -> DistributedOutcome:
    started = time.perf_counter()
    session_id = f"distributed-abandoned-attempt-{run_id}"
    payload = {
        "session_id": session_id,
        "problem": {
            "user_goal": "Slow summarize with coordinator failover.",
            "requirements": [
                {
                    "name": "slow summarize",
                    "side_effect_class": "read_only",
                    "validation_contract": {
                        "required_artifacts": ["slow_summary"]
                    },
                }
            ],
            "required_artifacts": ["slow_summary"],
        },
        "payload": {"text": "recover this"},
        "timeout_s": 5.0,
    }
    fault_status = 0
    fault_body: JsonObject = {}
    try:
        fault_response = await client.post(f"{faulting_url}/coordinate", json=payload)
        fault_status = fault_response.status_code
        fault_body = _safe_json(fault_response)
    except httpx.HTTPError as exc:
        fault_body = {"error": str(exc)}

    await asyncio.sleep(float(os.getenv("DISTRIBUTED_RECOVERY_WAIT_S", "0.8")))
    recovered = await client.post(
        f"{standby_url}/sessions/{session_id}/resume",
        json={"timeout_s": 5.0},
    )
    recovered_body = _safe_json(recovered)
    attempts = recovered_body.get("task_results") or []
    stats = await _fixture_stats(client, agent_url)
    session_task_key = f"{session_id}:t1"
    effectful_executions = int(
        (stats.get("effectful_executions_by_session_task") or {}).get(
            session_task_key,
            0,
        )
    )
    passed = (
        fault_status >= 500
        and recovered.status_code == 200
        and recovered_body.get("status") == "completed"
        and len(attempts) == 1
        and attempts[0].get("attempt_id") == "t1-attempt-2"
        and effectful_executions == 1
    )
    return DistributedOutcome(
        scenario_id="recover_after_abandoned_attempt",
        passed=passed,
        latency_ms=_elapsed_ms(started),
        response={
            "fault_status_code": fault_status,
            "fault_body": fault_body,
            "recovered_status_code": recovered.status_code,
            "recovered_body": recovered_body,
            "agent_session_task_key": session_task_key,
            "agent_effectful_executions_for_session_task": effectful_executions,
        },
    )


async def _process_exit_after_external_dispatch_unknown(
    client: httpx.AsyncClient,
    standby_url: str,
    agent_url: str,
    registry_url: str,
    store_url: str,
    run_id: str,
) -> DistributedOutcome:
    started = time.perf_counter()
    if not store_url:
        return DistributedOutcome(
            scenario_id="process_exit_after_external_dispatch_unknown",
            passed=False,
            latency_ms=_elapsed_ms(started),
            response={},
            error="COORDINATION_STORE_URL is not configured.",
        )
    port = int(os.getenv("DISTRIBUTED_CRASH_COORDINATOR_PORT", "8011"))
    session_id = f"distributed-exit-after-dispatch-{run_id}"
    child_url = f"http://127.0.0.1:{port}"
    env = {
        **os.environ,
        "COORDINATION_REGISTRY_URL": registry_url,
        "COORDINATION_STORE_URL": store_url,
        "COORDINATION_COORDINATOR_ID": "coordinator-exit-after-dispatch",
        "COORDINATION_FAULT_AT": "after_external_dispatch",
        "COORDINATION_FAULT_MODE": "exit",
        "COORDINATION_FAULT_EXIT_CODE": "91",
        "COORDINATION_LEASE_TTL_S": "2.0",
        "COORDINATION_LEASE_RENEW_INTERVAL_S": "0.5",
        "COORDINATION_SERVICE_HOST": "127.0.0.1",
        "COORDINATION_SERVICE_PORT": str(port),
        "COORDINATION_TASK_RETRIES": "0",
        "COORDINATION_ALLOW_INSECURE_A2A": "true",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "unified_multi_agent_coordination.service"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_exit_code: int | None = None
    child_stderr = ""
    fault_status = 0
    fault_body: JsonObject = {}
    try:
        await _wait_for_health(client, child_url, timeout_s=15.0)
        payload = {
            "session_id": session_id,
            "problem": {
                "user_goal": "Slow summarize with unknown side effects.",
                "requirements": [
                    {
                        "name": "slow summarize",
                        "side_effect_class": "unknown",
                        "validation_contract": {
                            "required_artifacts": ["slow_summary"]
                        },
                    }
                ],
                "required_artifacts": ["slow_summary"],
            },
            "payload": {"text": "do not blindly repeat this"},
            "timeout_s": 5.0,
        }
        try:
            fault_response = await client.post(f"{child_url}/coordinate", json=payload)
            fault_status = fault_response.status_code
            fault_body = _safe_json(fault_response)
        except httpx.HTTPError as exc:
            fault_body = {"error": str(exc)}
        try:
            child_exit_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            child_exit_code = process.wait(timeout=5)
        _, child_stderr = process.communicate(timeout=5)
        await asyncio.sleep(float(os.getenv("DISTRIBUTED_EXIT_RECOVERY_WAIT_S", "2.5")))
        recovered = await client.post(
            f"{standby_url}/sessions/{session_id}/resume",
            json={"timeout_s": 5.0},
        )
        recovered_body = _safe_json(recovered)
        attempts = recovered_body.get("task_results") or []
        task_statuses = [str(item.get("status") or "") for item in attempts]
        stats = await _fixture_stats(client, agent_url)
        session_task_key = f"{session_id}:t1"
        effectful_executions = int(
            (stats.get("effectful_executions_by_session_task") or {}).get(
                session_task_key,
                0,
            )
        )
        passed = (
            child_exit_code == 91
            and recovered.status_code == 200
            and recovered_body.get("status") == "failed"
            and task_statuses == ["unknown"]
            and effectful_executions == 1
        )
        return DistributedOutcome(
            scenario_id="process_exit_after_external_dispatch_unknown",
            passed=passed,
            latency_ms=_elapsed_ms(started),
            response={
                "fault_status_code": fault_status,
                "fault_body": fault_body,
                "child_exit_code": child_exit_code,
                "child_stderr_tail": child_stderr[-1200:],
                "recovered_status_code": recovered.status_code,
                "recovered_status": recovered_body.get("status"),
                "recovered_task_statuses": task_statuses,
                "agent_session_task_key": session_task_key,
                "agent_effectful_executions_for_session_task": effectful_executions,
            },
        )
    except Exception as exc:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        return DistributedOutcome(
            scenario_id="process_exit_after_external_dispatch_unknown",
            passed=False,
            latency_ms=_elapsed_ms(started),
            response={
                "child_exit_code": child_exit_code,
                "child_stderr_tail": child_stderr[-1200:],
            },
            error=str(exc),
        )


async def _stale_fence_rejection(store_url: str, run_id: str) -> DistributedOutcome:
    started = time.perf_counter()
    if not store_url:
        return DistributedOutcome(
            scenario_id="stale_fence_rejection",
            passed=False,
            latency_ms=_elapsed_ms(started),
            response={},
            error="COORDINATION_STORE_URL is not configured.",
        )
    store = PostgresCoordinationStore(store_url)
    try:
        session_id = f"distributed-stale-fence-{run_id}"
        lease1 = await store.acquire_lease(
            session_id,
            "stale-probe",
            ttl_s=0.1,
        )
        await asyncio.sleep(0.2)
        lease2 = await store.acquire_lease(
            session_id,
            "stale-probe",
            ttl_s=5.0,
        )
        rejected = False
        try:
            await store.append_event(
                LedgerEvent(
                    event_type="session_started",
                    session_id=session_id,
                    payload={"payload": {"probe": True}},
                ),
                lease=lease1,
            )
        except StaleFenceError:
            rejected = True
        return DistributedOutcome(
            scenario_id="stale_fence_rejection",
            passed=rejected and lease2.fencing_token == lease1.fencing_token + 1,
            latency_ms=_elapsed_ms(started),
            response={
                "first_token": lease1.fencing_token,
                "second_token": lease2.fencing_token,
                "stale_write_rejected": rejected,
            },
        )
    finally:
        await store.close()


async def _postgres_invariant_probe(store_url: str, run_id: str) -> DistributedOutcome:
    started = time.perf_counter()
    if not store_url:
        return DistributedOutcome(
            scenario_id="postgres_invariant_probe",
            passed=False,
            latency_ms=_elapsed_ms(started),
            response={},
            error="COORDINATION_STORE_URL is not configured.",
        )
    store = PostgresCoordinationStore(store_url)
    try:
        missing_attempt_rejected = False
        duplicate_terminal_rejected = False
        idempotency_lookup_ok = False
        release_token_incremented = False

        missing_session = f"distributed-pg-missing-attempt-{run_id}"
        missing_lease = await store.acquire_lease(
            missing_session,
            "pg-invariant-probe",
            ttl_s=5.0,
        )
        try:
            await store.append_event(
                LedgerEvent(
                    event_type="task_attempt_completed",
                    session_id=missing_session,
                    plan_id="plan",
                    task_id="t1",
                    attempt_id="t1-attempt-1",
                    payload={"task_result": {"status": "completed"}},
                ),
                lease=missing_lease,
            )
        except StoreInvariantError:
            missing_attempt_rejected = True

        terminal_session = f"distributed-pg-terminal-{run_id}"
        terminal_lease = await store.acquire_lease(
            terminal_session,
            "pg-invariant-probe",
            ttl_s=5.0,
        )
        await store.append_event(
            LedgerEvent(
                event_type="run_failed",
                session_id=terminal_session,
                plan_id="plan",
                payload={"run_result": {"status": "failed"}},
            ),
            lease=terminal_lease,
        )
        try:
            await store.append_event(
                LedgerEvent(
                    event_type="run_completed",
                    session_id=terminal_session,
                    plan_id="plan",
                    payload={"run_result": {"status": "completed"}},
                ),
                lease=terminal_lease,
            )
        except Exception:
            duplicate_terminal_rejected = True

        idem_session = f"distributed-pg-idempotency-{run_id}"
        idem_lease = await store.acquire_lease(
            idem_session,
            "pg-invariant-probe",
            ttl_s=5.0,
        )
        idem_key = f"{idem_session}:plan:t1:t1-attempt-1"
        await store.append_event(
            LedgerEvent(
                event_type="task_attempt_started",
                session_id=idem_session,
                plan_id="plan",
                task_id="t1",
                attempt_id="t1-attempt-1",
                payload={
                    "idempotency_key": idem_key,
                    "coordinator_id": "pg-invariant-probe",
                    "fencing_token": idem_lease.fencing_token,
                },
            ),
            lease=idem_lease,
        )
        await store.append_event(
            LedgerEvent(
                event_type="task_attempt_completed",
                session_id=idem_session,
                plan_id="plan",
                task_id="t1",
                attempt_id="t1-attempt-1",
                payload={"task_result": {"status": "completed", "task_id": "t1"}},
            ),
            lease=idem_lease,
        )
        prior = await store.task_result_by_idempotency_key(idem_key)
        idempotency_lookup_ok = bool(prior and prior.get("status") == "completed")

        release_session = f"distributed-pg-release-{run_id}"
        release1 = await store.acquire_lease(
            release_session,
            "pg-invariant-probe",
            ttl_s=5.0,
        )
        await store.release_lease(release1)
        release2 = await store.acquire_lease(
            release_session,
            "pg-invariant-probe",
            ttl_s=5.0,
        )
        release_token_incremented = release2.fencing_token == release1.fencing_token + 1
        stale_after_release_rejected = False
        try:
            await store.append_event(
                LedgerEvent(
                    event_type="session_started",
                    session_id=release_session,
                    payload={"payload": {"stale": True}},
                ),
                lease=release1,
            )
        except StaleFenceError:
            stale_after_release_rejected = True

        checks = {
            "missing_attempt_rejected": missing_attempt_rejected,
            "duplicate_terminal_rejected": duplicate_terminal_rejected,
            "idempotency_lookup_ok": idempotency_lookup_ok,
            "release_token_incremented": release_token_incremented,
            "stale_after_release_rejected": stale_after_release_rejected,
        }
        return DistributedOutcome(
            scenario_id="postgres_invariant_probe",
            passed=all(checks.values()),
            latency_ms=_elapsed_ms(started),
            response=checks,
        )
    finally:
        await store.close()


async def _wait_for_health(
    client: httpx.AsyncClient,
    base_url: str,
    timeout_s: float = 30.0,
) -> None:
    deadline = time.perf_counter() + timeout_s
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            response = await client.get(f"{base_url}/health")
            response.raise_for_status()
            if response.json().get("status") == "ok":
                return
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(0.2)
    raise RuntimeError(f"{base_url} did not become healthy: {last_error}")


def _metrics(outcomes: list[DistributedOutcome], fixture_stats: JsonObject) -> JsonObject:
    recovery = next(
        (outcome for outcome in outcomes if outcome.scenario_id == "recover_after_abandoned_attempt"),
        None,
    )
    stale = next(
        (outcome for outcome in outcomes if outcome.scenario_id == "stale_fence_rejection"),
        None,
    )
    lease = next(
        (outcome for outcome in outcomes if outcome.scenario_id == "live_lease_conflict"),
        None,
    )
    return {
        "recovery_time_ms": recovery.latency_ms if recovery else None,
        "stale_write_rejections": (
            1
            if stale
            and stale.response.get("stale_write_rejected") is True
            else 0
        ),
        "lease_conflicts_observed": (
            1
            if lease
            and lease.response.get("contender_status_code") == 409
            else 0
        ),
        "duplicate_dispatch_count": int(
            fixture_stats.get("duplicate_idempotency_key_requests") or 0
        ),
        "agent_side_duplicate_idempotency_key_requests": int(
            fixture_stats.get("duplicate_idempotency_key_requests") or 0
        ),
        "agent_side_repeated_session_task_effectful_executions": int(
            fixture_stats.get("repeated_session_task_effectful_executions") or 0
        ),
        "terminal_correctness": all(outcome.passed for outcome in outcomes),
    }


def _outcome_record(outcome: DistributedOutcome) -> JsonObject:
    return {
        "scenario_id": outcome.scenario_id,
        "passed": outcome.passed,
        "latency_ms": round(outcome.latency_ms, 3),
        "response": outcome.response,
        "error": outcome.error,
    }


def _safe_json(response: httpx.Response) -> JsonObject:
    try:
        body = response.json()
    except json.JSONDecodeError:
        return {"text": response.text}
    return body if isinstance(body, dict) else {"value": body}


async def _fixture_stats(client: httpx.AsyncClient, agent_url: str) -> JsonObject:
    response = await client.get(f"{agent_url}/fixture-stats")
    response.raise_for_status()
    return _safe_json(response)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def main() -> None:
    """Run distributed system tests and write the report JSON."""
    report = asyncio.run(run_distributed_tests())
    report_path = Path(
        os.getenv("DISTRIBUTED_REPORT_PATH", "demo_runs/distributed_system_report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
