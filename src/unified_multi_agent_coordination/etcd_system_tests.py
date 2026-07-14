"""Black-box consensus smoke tests for the etcd Docker topology."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from .etcd_client import EtcdClient


JsonObject = dict[str, Any]


async def run_etcd_system_tests() -> JsonObject:
    coordinators = [
        os.getenv("COORDINATION_A_URL", "http://coordination-a:8000"),
        os.getenv("COORDINATION_B_URL", "http://coordination-b:8000"),
        os.getenv("COORDINATION_C_URL", "http://coordination-c:8000"),
    ]
    endpoints = [
        value.strip()
        for value in os.getenv("ETCD_ENDPOINTS", "http://coordination-a:2379").split(",")
        if value.strip()
    ]
    run_id = uuid4().hex[:8]
    async with httpx.AsyncClient(timeout=10.0) as client:
        statuses = await _wait_for_voters(client, coordinators, target=3)
        registry = await _wait_for_registry(client, coordinators[1], "summarizer")
        coordinate = await client.post(
            f"{coordinators[1]}/coordinate",
            json={
                "session_id": f"etcd-coordinate-{run_id}",
                "problem": {
                    "user_goal": "Summarize through consensus-backed coordination.",
                    "requirements": [
                        {
                            "name": "summarize",
                            "side_effect_class": "read_only",
                            "validation_contract": {"required_artifacts": ["summary"]},
                        }
                    ],
                    "required_artifacts": ["summary"],
                },
                "payload": {"text": "consensus smoke test"},
            },
        )
        coordinate_body = _json(coordinate)
        conflict = await _lease_conflict(client, coordinators, run_id)
        forged = await client.post(
            f"{coordinators[0]}/internal/cluster/join",
            json={
                "message_type": "cluster_join",
                "cluster_id": "thesis-coordination",
                "node_id": "forged-node",
                "payload": {"voter_target": 3},
                "signature": "forged",
            },
        )
        agent_stats = _json(
            await client.get(
                f"{os.getenv('SUMMARIZER_URL', 'http://summarizer:8000')}/fixture-stats"
            )
        )

    etcd = EtcdClient(endpoints)
    try:
        history = await etcd.range(
            b"/umac/thesis-coordination/membership/history/", prefix=True
        )
        backend_status = await etcd.status()
    finally:
        await etcd.close()

    checks = {
        "three_voters": all(item.get("active_voters") == 3 for item in statuses),
        "membership_steady_state": all(
            item.get("steady_state") is True
            and item.get("pending_membership_changes") == 0
            and item.get("role") == "voter"
            for item in statuses
        ),
        "registry_is_cluster_wide": any(
            item.get("agent_id") == "summarizer" for item in registry.get("agents", [])
        ),
        "terminal_correctness": coordinate.status_code == 200
        and coordinate_body.get("status") == "completed",
        "live_lease_conflict": conflict.get("contender_status_code") == 409,
        "forged_join_rejected": forged.status_code == 401,
        "membership_history_present": bool(history.values),
        "duplicate_effect_count_zero": int(
            agent_stats.get("repeated_session_task_effectful_executions") or 0
        )
        == 0,
    }
    return {
        "generated_at_unix": time.time(),
        "passed": all(checks.values()),
        "checks": checks,
        "voter_target": 3,
        "coordinator_statuses": statuses,
        "leader": int(backend_status.get("leader") or 0),
        "revision": int((backend_status.get("header") or {}).get("revision") or 0),
        "membership_history": [json.loads(item.value) for item in history.values],
        "quorum_events": [],
        "discovery_paths": sorted(
            {str(item.get("discovery_method") or "") for item in statuses}
        ),
        "duplicate_effect_count": int(
            agent_stats.get("repeated_session_task_effectful_executions") or 0
        ),
        "terminal_result": coordinate_body,
        "lease_conflict": conflict,
    }


async def _wait_for_voters(
    client: httpx.AsyncClient,
    coordinators: list[str],
    *,
    target: int,
) -> list[JsonObject]:
    deadline = time.monotonic() + 90
    last: list[JsonObject] = []
    while time.monotonic() < deadline:
        try:
            responses = await asyncio.gather(
                *(client.get(f"{url}/cluster/status") for url in coordinators)
            )
            last = [_json(response) for response in responses]
            if all(
                response.status_code == 200
                and body.get("active_voters") == target
                and body.get("steady_state") is True
                and body.get("pending_membership_changes") == 0
                for response, body in zip(responses, last, strict=True)
            ):
                return last
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(f"Voter membership did not converge: {last}")


async def _wait_for_registry(
    client: httpx.AsyncClient, coordinator: str, agent_id: str
) -> JsonObject:
    deadline = time.monotonic() + 60
    last: JsonObject = {}
    while time.monotonic() < deadline:
        try:
            response = await client.get(f"{coordinator}/registry")
            last = _json(response)
            if any(item.get("agent_id") == agent_id for item in last.get("agents", [])):
                return last
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    raise RuntimeError(f"Agent registry did not converge: {last}")


async def _lease_conflict(
    client: httpx.AsyncClient, coordinators: list[str], run_id: str
) -> JsonObject:
    payload = {
        "session_id": f"etcd-conflict-{run_id}",
        "problem": {
            "user_goal": "Slow summarize.",
            "requirements": [
                {
                    "name": "slow summarize",
                    "side_effect_class": "read_only",
                    "validation_contract": {"required_artifacts": ["slow_summary"]},
                }
            ],
            "required_artifacts": ["slow_summary"],
        },
        "payload": {"text": "lease conflict"},
    }
    owner = asyncio.create_task(client.post(f"{coordinators[0]}/coordinate", json=payload))
    await asyncio.sleep(0.2)
    contender = await client.post(f"{coordinators[2]}/coordinate", json=payload)
    owner_response = await owner
    return {
        "owner_status_code": owner_response.status_code,
        "contender_status_code": contender.status_code,
        "owner": _json(owner_response),
        "contender": _json(contender),
    }


def _json(response: httpx.Response) -> JsonObject:
    value = response.json()
    return value if isinstance(value, dict) else {"value": value}


def main() -> None:
    report = asyncio.run(run_etcd_system_tests())
    path = Path(os.getenv("ETCD_REPORT_PATH", "demo_runs/etcd-system-report.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
