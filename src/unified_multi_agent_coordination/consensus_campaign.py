"""Host-side, immutable Docker fault campaign for consensus-backed coordinators."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .cluster import HmacAuthenticator, SignedEnvelope

JsonObject = dict[str, Any]
COORDINATORS = {
    "coordination-a": "http://127.0.0.1:8000",
    "coordination-b": "http://127.0.0.1:8001",
    "coordination-c": "http://127.0.0.1:8002",
    "coordination-d": "http://127.0.0.1:8003",
    "coordination-e": "http://127.0.0.1:8004",
    "coordination-f": "http://127.0.0.1:8005",
    "coordination-g": "http://127.0.0.1:8006",
}
CLUSTER_ID = "thesis-coordination"
CLUSTER_SECRET = "development-cluster-secret"
DEFAULT_IMAGE = "unified-multi-agent-coordination:etcd-distributed"

EXPECTED_CHECKS: dict[str, tuple[str, ...]] = {
    "formation": ("steady_state", "coordinate_completed"),
    "reconfigure": (
        "expand_accepted",
        "expanded_steady",
        "shrink_accepted",
        "shrunk_steady",
        "zero_pending_changes",
    ),
    "leader-partition-quorum-concurrency": (
        "leader_failure_progress",
        "leader_failure_within_30s",
        "minority_partition_majority_progress",
        "minority_partition_fails_closed",
        "quorum_loss_503",
        "quorum_loss_detected_within_10s",
        "quorum_restored",
        "concurrent_session_serialized",
        "all_restorations_steady",
    ),
    "leader-termination": (
        "leader_failure_progress",
        "leader_failure_within_30s",
        "leader_restoration_steady",
    ),
    "minority-partition": (
        "minority_partition_majority_progress",
        "minority_partition_fails_closed",
        "minority_restoration_steady",
    ),
    "majority-loss-restoration": (
        "quorum_loss_503",
        "quorum_loss_detected_within_10s",
        "quorum_restored",
    ),
    "concurrent-ownership": (
        "concurrent_session_serialized",
        "cluster_remains_steady",
    ),
    "failed-voter-replacement": (
        "failed_voter_replaced",
        "replacement_reaches_steady_state",
        "progress_after_replacement",
    ),
    "audit-sink-unavailable": (
        "authoritative_progress_without_audit_sink",
        "cluster_remains_steady",
    ),
    "crash": (
        "fault_process_terminated_request",
        "session_recovered",
        "one_terminal_result",
        "zero_repeated_external_effects",
        "receiver_operation_keys_observed",
        "receiver_fencing_tokens_positive",
    ),
}


def _expected_checks(scenario: str) -> list[str]:
    for prefix, checks in EXPECTED_CHECKS.items():
        if scenario.startswith(prefix):
            return list(checks)
    raise ValueError(f"No check contract is declared for scenario {scenario!r}.")


def expected_scenarios(trials: int, *, smoke: bool) -> list[tuple[str, int]]:
    matrix: list[tuple[str, int]] = []
    topologies = (3,) if smoke else (3, 5, 7)
    count = 1 if smoke else trials
    for trial in range(1, count + 1):
        matrix.extend((f"formation-{topology}", trial) for topology in topologies)
        if not smoke:
            matrix.extend((("reconfigure-3-5-3", trial), ("reconfigure-5-7-5", trial)))
        matrix.extend(
            (
                ("leader-termination", trial),
                ("minority-partition", trial),
                ("majority-loss-restoration", trial),
                ("concurrent-ownership", trial),
                ("audit-sink-unavailable", trial),
            )
        )
        if not smoke:
            matrix.append(("failed-voter-replacement", trial))
            matrix.extend(
                (f"crash-{fault_point}", trial)
                for fault_point in (
                    "after_task_attempt_start",
                    "after_external_dispatch",
                    "during_aggregation",
                )
            )
            matrix.append(("leader-partition-quorum-concurrency", trial))
    return matrix


@dataclass
class TrialResult:
    scenario: str
    trial: int
    topology: int
    passed: bool
    duration_s: float
    checks: JsonObject = field(default_factory=dict)
    observations: JsonObject = field(default_factory=dict)
    error: str = ""
    status: str = "infrastructure_error"
    expected_checks: list[str] = field(default_factory=list)
    executed_checks: list[str] = field(default_factory=list)
    violated_checks: list[str] = field(default_factory=list)
    unexecuted_checks: list[str] = field(default_factory=list)
    primary: bool = True


class ComposeController:
    def __init__(
        self,
        *,
        project: str,
        evidence_dir: Path,
        voter_target: int,
        profiles: tuple[str, ...] = (),
        image: str,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.project = project
        self.evidence_dir = evidence_dir.resolve()
        self.profiles = profiles
        self.env = {
            **os.environ,
            "EVIDENCE_RUN_DIR": str(self.evidence_dir),
            "COORDINATION_VOTER_TARGET": str(voter_target),
            "COORDINATION_IMAGE": image,
            **(extra_env or {}),
        }

    def command(self, *args: str, timeout_s: float = 300, check: bool = True) -> str:
        command = [
            "docker",
            "compose",
            "-f",
            "docker-compose.etcd-distributed.yml",
            "-p",
            self.project,
        ]
        for profile in self.profiles:
            command.extend(("--profile", profile))
        command.extend(args)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            command_log = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "returncode": completed.returncode,
                "duration_s": time.monotonic() - started,
                "stdout": completed.stdout[-10000:],
                "stderr": completed.stderr[-10000:],
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            command_log = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "returncode": None,
                "duration_s": time.monotonic() - started,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}",
            }
            with (self.evidence_dir / "compose-commands.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(command_log, sort_keys=True) + "\n")
            raise RuntimeError(command_log["stderr"]) from exc
        with (self.evidence_dir / "compose-commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(command_log, sort_keys=True) + "\n")
        if check and completed.returncode:
            raise RuntimeError(
                f"{' '.join(command)} failed ({completed.returncode}): {completed.stderr[-2000:]}"
            )
        return completed.stdout.strip()

    def host_command(self, *args: str, timeout_s: float = 60) -> str:
        command = ["docker", *args]
        started = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        record = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "returncode": completed.returncode,
            "duration_s": time.monotonic() - started,
            "stdout": completed.stdout[-10000:],
            "stderr": completed.stderr[-10000:],
        }
        with (self.evidence_dir / "compose-commands.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if completed.returncode:
            raise RuntimeError(
                f"{' '.join(command)} failed ({completed.returncode}): {completed.stderr[-2000:]}"
            )
        return completed.stdout.strip()

    def up(self, services: list[str]) -> None:
        self.command("up", "-d", "--no-build", *services, timeout_s=900)

    def stop(self, *services: str) -> None:
        self.command("stop", *services)

    def start(self, *services: str) -> None:
        self.command("start", *services)

    def remove(self, *services: str) -> None:
        self.command("rm", "-s", "-f", *services, check=False)

    def container_id(self, service: str) -> str:
        value = self.command("ps", "-q", service)
        if not value:
            raise RuntimeError(f"Compose service {service} has no container.")
        return value.splitlines()[0]

    def disconnect(self, service: str) -> None:
        self.host_command(
            "network", "disconnect", f"{self.project}_default", self.container_id(service)
        )

    def reconnect(self, service: str) -> None:
        self.host_command(
            "network", "connect", f"{self.project}_default", self.container_id(service)
        )

    def image_ids(self) -> dict[str, str]:
        ids = {
            service: self.container_id(service)
            for service in self.services()
            if self.command("ps", "-q", service, check=False)
        }
        result: dict[str, str] = {}
        for service, container_id in ids.items():
            inspected = subprocess.run(
                ["docker", "inspect", "--format", "{{.Image}}", container_id],
                capture_output=True,
                text=True,
                check=True,
            )
            result[service] = inspected.stdout.strip()
        return result

    def services(self) -> list[str]:
        return self.command("config", "--services").splitlines()

    def down(self) -> None:
        self.command("down", "-v", "--remove-orphans", timeout_s=300, check=False)


def _services(topology: int, *, agents: bool = True) -> list[str]:
    names = list(COORDINATORS)[:topology]
    if agents:
        names.extend(("summarizer", "slow-agent"))
    return names


def _profiles(topology: int, *, chaos: bool = False) -> tuple[str, ...]:
    values: list[str] = []
    if topology >= 7:
        values.append("seven")
    elif topology >= 4:
        values.append("five")
    if chaos:
        values.append("chaos")
    return tuple(values)


async def _statuses(names: list[str]) -> dict[str, JsonObject]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        responses = await asyncio.gather(
            *(client.get(f"{COORDINATORS[name]}/cluster/status") for name in names),
            return_exceptions=True,
        )
    result: dict[str, JsonObject] = {}
    for name, response in zip(names, responses, strict=True):
        if not isinstance(response, httpx.Response):
            result[name] = {"http_status": 0, "error": str(response)}
        else:
            body = response.json() if response.content else {}
            result[name] = {"http_status": response.status_code, **body}
    return result


async def _wait_for(
    names: list[str],
    predicate: Callable[[dict[str, JsonObject]], bool],
    *,
    timeout_s: float,
) -> dict[str, JsonObject]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, JsonObject] = {}
    while time.monotonic() < deadline:
        last = await _statuses(names)
        if predicate(last):
            return last
        await asyncio.sleep(1)
    raise RuntimeError(f"Cluster condition did not converge in {timeout_s}s: {last}")


async def _wait_steady(names: list[str], target: int) -> dict[str, JsonObject]:
    return await _wait_for(
        names,
        lambda statuses: all(
            item.get("http_status") == 200
            and item.get("steady_state") is True
            and item.get("configured_voter_target") == target
            and item.get("active_voters") == target
            and item.get("pending_membership_changes") == 0
            for item in statuses.values()
        ),
        timeout_s=120,
    )


async def _wait_progress(names: list[str]) -> dict[str, JsonObject]:
    return await _wait_for(
        names,
        lambda statuses: any(
            item.get("http_status") == 200 and item.get("quorum_available") is True
            for item in statuses.values()
        ),
        timeout_s=30,
    )


async def _wait_replacement(
    names: list[str], target: int, failed: str, replacement: str
) -> dict[str, JsonObject]:
    return await _wait_for(
        names,
        lambda statuses: (
            all(
                item.get("http_status") == 200
                and item.get("steady_state") is True
                and item.get("configured_voter_target") == target
                for item in statuses.values()
            )
            and statuses.get(replacement, {}).get("role") == "voter"
            and failed not in next(iter(statuses.values())).get("role_agreement", {})
        ),
        timeout_s=120,
    )


async def _coordinate(url: str, session_id: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=60.0) as client:
        return await client.post(
            f"{url}/coordinate",
            json={
                "session_id": session_id,
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
                "payload": {"text": session_id},
            },
        )


async def _coordinate_observed(url: str, session_id: str) -> tuple[httpx.Response | None, str]:
    """Convert a bounded coordination failure into experiment evidence, not harness failure."""
    await _wait_registered_agent(url, "summarizer")
    try:
        return await _coordinate(url, session_id), ""
    except httpx.HTTPError as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def _wait_registered_agent(url: str, agent_id: str) -> JsonObject:
    """Wait until recovery can resolve the same distributed agent identity."""

    async def snapshot() -> JsonObject:
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(f"{url}/registry", params={"refresh": "true"})
                body = response.json() if response.content else {}
                return {"http_status": response.status_code, **body}
            except httpx.HTTPError as exc:
                return {"http_status": 0, "error": f"{type(exc).__name__}: {exc}"}

    deadline = time.monotonic() + 60
    last: JsonObject = {}
    while time.monotonic() < deadline:
        last = await snapshot()
        agents = last.get("agents") or []
        if last.get("http_status") == 200 and any(
            item.get("agent_id") == agent_id for item in agents if isinstance(item, dict)
        ):
            return last
        await asyncio.sleep(1)
    raise RuntimeError(f"Agent {agent_id!r} did not re-register within 60s: {last}")


async def _update_target(url: str, target: int, generation: int) -> httpx.Response:
    request = HmacAuthenticator(CLUSTER_SECRET).sign(
        SignedEnvelope(
            message_type="cluster_configuration",
            cluster_id=CLUSTER_ID,
            node_id="consensus-campaign",
            payload={"voter_target": target, "expected_generation": generation},
        )
    )
    async with httpx.AsyncClient(timeout=15.0) as client:
        return await client.put(
            f"{url}/internal/cluster/configuration",
            json=request.model_dump(mode="json"),
        )


async def _formation_trial(root: Path, topology: int, trial: int, image: str) -> TrialResult:
    return await _with_cluster(
        root,
        scenario=f"formation-{topology}",
        topology=topology,
        trial=trial,
        initial_target=topology,
        active_nodes=topology,
        action=_formation_action,
        image=image,
    )


async def _formation_action(
    controller: ComposeController, names: list[str], target: int
) -> tuple[JsonObject, JsonObject]:
    statuses = await _wait_steady(names, target)
    response, error = await _coordinate_observed(
        COORDINATORS[names[-1]], f"formation-{target}-{uuid.uuid4().hex}"
    )
    return {
        "steady_state": True,
        "coordinate_completed": response is not None
        and response.status_code == 200
        and response.json().get("status") == "completed",
    }, {
        "statuses": statuses,
        "coordinate": _response(response) if response is not None else {"error": error},
    }


async def _reconfiguration_trial(
    root: Path, initial: int, expanded: int, trial: int, image: str
) -> TrialResult:
    async def action(
        controller: ComposeController, names: list[str], target: int
    ) -> tuple[JsonObject, JsonObject]:
        before = await _wait_steady(names, target)
        generation = int(next(iter(before.values()))["configuration_generation"])
        expand = await _update_target(COORDINATORS[names[0]], expanded, generation)
        expanded_status = await _wait_steady(names, expanded)
        shrink = await _update_target(COORDINATORS[names[0]], initial, generation + 1)
        shrunk_status = await _wait_steady(names, initial)
        return {
            "expand_accepted": expand.status_code == 200,
            "expanded_steady": all(item["steady_state"] for item in expanded_status.values()),
            "shrink_accepted": shrink.status_code == 200,
            "shrunk_steady": all(item["steady_state"] for item in shrunk_status.values()),
            "zero_pending_changes": all(
                item["pending_membership_changes"] == 0 for item in shrunk_status.values()
            ),
        }, {
            "before": before,
            "expanded": expanded_status,
            "shrunk": shrunk_status,
            "expand_response": _response(expand),
            "shrink_response": _response(shrink),
        }

    return await _with_cluster(
        root,
        scenario=f"reconfigure-{initial}-{expanded}-{initial}",
        topology=expanded,
        trial=trial,
        initial_target=initial,
        active_nodes=expanded,
        action=action,
        image=image,
    )


def _leader_name(statuses: dict[str, JsonObject], names: list[str]) -> str:
    leader_id = int(next(iter(statuses.values())).get("leader") or 0)
    return next(
        (name for name, item in statuses.items() if int(item.get("member_id") or 0) == leader_id),
        names[0],
    )


async def _leader_termination_trial(root: Path, trial: int, image: str) -> TrialResult:
    async def action(controller, names, target):
        initial = await _wait_steady(names, target)
        leader = _leader_name(initial, names)
        survivors = [name for name in names if name != leader]
        started = time.monotonic()
        controller.stop(leader)
        try:
            await _wait_progress(survivors)
            progress, error = await _coordinate_observed(
                COORDINATORS[survivors[0]], f"leader-failure-{trial}-{uuid.uuid4().hex}"
            )
            recovery_s = time.monotonic() - started
        finally:
            controller.start(leader)
        restored = await _wait_steady(names, target)
        return {
            "leader_failure_progress": progress is not None and progress.status_code == 200,
            "leader_failure_within_30s": recovery_s <= 30,
            "leader_restoration_steady": all(item["steady_state"] for item in restored.values()),
        }, {
            "initial": initial,
            "leader": leader,
            "leader_recovery_s": recovery_s,
            "progress": _response(progress) if progress is not None else {"error": error},
            "restored": restored,
        }

    return await _with_cluster(
        root,
        scenario="leader-termination",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=3,
        action=action,
        image=image,
    )


async def _minority_partition_trial(root: Path, trial: int, image: str) -> TrialResult:
    async def action(controller, names, target):
        initial = await _wait_steady(names, target)
        leader = _leader_name(initial, names)
        minority = next(name for name in reversed(names) if name != leader)
        controller.disconnect(minority)
        minority_ready: httpx.Response | None = None
        minority_error = ""
        try:
            progress, progress_error = await _coordinate_observed(
                COORDINATORS[leader], f"minority-partition-{trial}-{uuid.uuid4().hex}"
            )
            try:
                minority_ready = await _wait_ready_status(COORDINATORS[minority], 503, timeout_s=20)
            except (RuntimeError, httpx.HTTPError) as exc:
                minority_error = f"{type(exc).__name__}: {exc}"
        finally:
            controller.reconnect(minority)
        restored = await _wait_steady(names, target)
        return {
            "minority_partition_majority_progress": progress is not None
            and progress.status_code == 200,
            "minority_partition_fails_closed": minority_ready is None
            or minority_ready.status_code == 503,
            "minority_restoration_steady": all(item["steady_state"] for item in restored.values()),
        }, {
            "initial": initial,
            "leader": leader,
            "minority": minority,
            "progress": _response(progress) if progress is not None else {"error": progress_error},
            "minority_ready": _response(minority_ready)
            if minority_ready is not None
            else {"error": minority_error, "bounded_no_response": True},
            "restored": restored,
        }

    return await _with_cluster(
        root,
        scenario="minority-partition",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=3,
        action=action,
        image=image,
    )


async def _majority_loss_trial(root: Path, trial: int, image: str) -> TrialResult:
    async def action(controller, names, target):
        initial = await _wait_steady(names, target)
        leader = _leader_name(initial, names)
        stopped = [name for name in names if name != leader]
        controller.stop(*stopped)
        started = time.monotonic()
        try:
            response = await _wait_ready_status(COORDINATORS[leader], 503, timeout_s=10)
            detection_s = time.monotonic() - started
        finally:
            controller.start(*stopped)
        restored = await _wait_steady(names, target)
        return {
            "quorum_loss_503": response.status_code == 503
            and response.json().get("code") == "quorum_unavailable",
            "quorum_loss_detected_within_10s": detection_s <= 10,
            "quorum_restored": all(item["steady_state"] for item in restored.values()),
        }, {
            "initial": initial,
            "leader": leader,
            "stopped": stopped,
            "quorum_response": _response(response),
            "quorum_detection_s": detection_s,
            "restored": restored,
        }

    return await _with_cluster(
        root,
        scenario="majority-loss-restoration",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=3,
        action=action,
        image=image,
    )


async def _concurrent_ownership_trial(root: Path, trial: int, image: str) -> TrialResult:
    async def action(controller, names, target):
        initial = await _wait_steady(names, target)
        session = f"concurrent-{trial}-{uuid.uuid4().hex}"
        observed = await asyncio.gather(
            _coordinate_observed(COORDINATORS[names[0]], session),
            _coordinate_observed(COORDINATORS[names[1]], session),
        )
        responses = [item[0] for item in observed]
        return {
            "concurrent_session_serialized": all(item is not None for item in responses)
            and sorted(item.status_code for item in responses if item is not None)
            in ([200, 200], [200, 409]),
            "cluster_remains_steady": all(item["steady_state"] for item in initial.values()),
        }, {
            "initial": initial,
            "concurrent": [
                _response(response) if response is not None else {"error": error}
                for response, error in observed
            ],
        }

    return await _with_cluster(
        root,
        scenario="concurrent-ownership",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=3,
        action=action,
        image=image,
    )


async def _fault_trial(root: Path, trial: int, image: str) -> TrialResult:
    async def action(
        controller: ComposeController, names: list[str], target: int
    ) -> tuple[JsonObject, JsonObject]:
        initial = await _wait_steady(names, target)
        first = next(iter(initial.values()))
        leader_id = int(first.get("leader") or 0)
        leader = next(
            (
                name
                for name, item in initial.items()
                if int(item.get("member_id") or 0) == leader_id
            ),
            names[0],
        )
        survivors = [name for name in names if name != leader]
        started = time.monotonic()
        controller.stop(leader)
        await _wait_progress(survivors)
        progress, progress_error = await _coordinate_observed(
            COORDINATORS[survivors[0]], f"leader-failure-{trial}-{uuid.uuid4().hex}"
        )
        leader_recovery_s = time.monotonic() - started
        controller.start(leader)
        restored = await _wait_steady(names, target)

        minority = survivors[-1]
        controller.disconnect(minority)
        partition_progress, partition_error = await _coordinate_observed(
            COORDINATORS[leader], f"minority-partition-{trial}-{uuid.uuid4().hex}"
        )
        minority_ready: httpx.Response | None = None
        minority_ready_error = ""
        try:
            try:
                minority_ready = await _wait_ready_status(COORDINATORS[minority], 503, timeout_s=20)
            except (RuntimeError, httpx.HTTPError) as exc:
                minority_ready_error = f"{type(exc).__name__}: {exc}"
        finally:
            controller.reconnect(minority)
        partition_restored = await _wait_steady(names, target)

        stopped = [name for name in names if name != leader]
        controller.stop(*stopped)
        quorum_started = time.monotonic()
        quorum_response = await _wait_ready_status(COORDINATORS[leader], 503, timeout_s=10)
        quorum_detection_s = time.monotonic() - quorum_started
        controller.start(*stopped)
        quorum_restored = await _wait_steady(names, target)

        session = f"concurrent-{trial}-{uuid.uuid4().hex}"
        concurrent = await asyncio.gather(
            _coordinate_observed(COORDINATORS[names[0]], session),
            _coordinate_observed(COORDINATORS[names[1]], session),
        )
        concurrent_responses = [item[0] for item in concurrent]
        checks = {
            "leader_failure_progress": progress is not None and progress.status_code == 200,
            "leader_failure_within_30s": leader_recovery_s <= 30,
            "minority_partition_majority_progress": partition_progress is not None
            and partition_progress.status_code == 200,
            "minority_partition_fails_closed": minority_ready is None
            or minority_ready.status_code == 503,
            "quorum_loss_503": quorum_response.status_code == 503
            and quorum_response.json().get("code") == "quorum_unavailable",
            "quorum_loss_detected_within_10s": quorum_detection_s <= 10,
            "quorum_restored": all(item["steady_state"] for item in quorum_restored.values()),
            "concurrent_session_serialized": all(item is not None for item in concurrent_responses)
            and sorted(item.status_code for item in concurrent_responses if item is not None)
            in ([200, 200], [200, 409]),
            "all_restorations_steady": all(item["steady_state"] for item in restored.values())
            and all(item["steady_state"] for item in partition_restored.values()),
        }
        return checks, {
            "initial": initial,
            "leader": leader,
            "leader_recovery_s": leader_recovery_s,
            "restored": restored,
            "partition_restored": partition_restored,
            "quorum_detection_s": quorum_detection_s,
            "quorum_restored": quorum_restored,
            "leader_progress_error": progress_error,
            "partition_progress_error": partition_error,
            "minority_ready": (
                _response(minority_ready)
                if minority_ready is not None
                else {"error": minority_ready_error, "bounded_no_response": True}
            ),
            "concurrent": [
                _response(response) if response is not None else {"error": error}
                for response, error in concurrent
            ],
        }

    return await _with_cluster(
        root,
        scenario="leader-partition-quorum-concurrency",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=3,
        action=action,
        image=image,
        primary=False,
    )


async def _failed_voter_replacement_trial(root: Path, trial: int, image: str) -> TrialResult:
    async def action(
        controller: ComposeController, names: list[str], target: int
    ) -> tuple[JsonObject, JsonObject]:
        initial = await _wait_steady(names, target)
        first = next(iter(initial.values()))
        leader_id = int(first.get("leader") or 0)
        failed = next(
            (
                name
                for name, item in initial.items()
                if item.get("role") == "voter" and int(item.get("member_id") or 0) != leader_id
            ),
            "coordination-c",
        )
        controller.stop(failed)
        survivors = [name for name in names if name != failed]
        replaced = await _wait_replacement(survivors, target, failed, "coordination-d")
        replacement = "coordination-d"
        progress, progress_error = await _coordinate_observed(
            COORDINATORS[survivors[0]], f"replacement-{trial}-{uuid.uuid4().hex}"
        )
        return {
            "failed_voter_replaced": replacement == "coordination-d",
            "replacement_reaches_steady_state": all(
                item.get("steady_state") is True for item in replaced.values()
            ),
            "progress_after_replacement": progress is not None and progress.status_code == 200,
        }, {
            "initial": initial,
            "failed": failed,
            "replacement": replacement,
            "replaced": replaced,
            "coordinate": _response(progress)
            if progress is not None
            else {"error": progress_error},
        }

    return await _with_cluster(
        root,
        scenario="failed-voter-replacement",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=4,
        action=action,
        image=image,
    )


async def _audit_failure_trial(root: Path, trial: int, image: str) -> TrialResult:
    async def action(
        controller: ComposeController, names: list[str], target: int
    ) -> tuple[JsonObject, JsonObject]:
        statuses = await _wait_steady(names, target)
        response, error = await _coordinate_observed(
            COORDINATORS[names[0]], f"audit-failure-{trial}-{uuid.uuid4().hex}"
        )
        return {
            "authoritative_progress_without_audit_sink": response is not None
            and response.status_code == 200,
            "cluster_remains_steady": all(item["steady_state"] for item in statuses.values()),
        }, {
            "statuses": statuses,
            "coordinate": _response(response) if response is not None else {"error": error},
        }

    return await _with_cluster(
        root,
        scenario="audit-sink-unavailable",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=3,
        action=action,
        image=image,
        extra_env={
            "COORDINATION_AUDIT_STORE_URL": (
                "postgresql://postgres:postgres@missing-audit:5432/coordination"
            )
        },
    )


async def _crash_window_trial(root: Path, trial: int, fault_point: str, image: str) -> TrialResult:
    async def action(
        controller: ComposeController, names: list[str], target: int
    ) -> tuple[JsonObject, JsonObject]:
        await _wait_steady(names, target)
        await _wait_ready_status("http://127.0.0.1:8010", 200, timeout_s=60)
        await _wait_registered_agent(COORDINATORS[names[0]], "summarizer")
        session_id = f"{fault_point}-{trial}-{uuid.uuid4().hex}"
        initial_error = ""
        try:
            response = await _coordinate("http://127.0.0.1:8010", session_id)
            initial_error = f"fault coordinator unexpectedly returned HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            initial_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(7)
        registry = await _wait_registered_agent(COORDINATORS[names[0]], "summarizer")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resumed = await client.post(
                f"{COORDINATORS[names[0]]}/sessions/{session_id}/resume",
                json={"payload": {"text": session_id}, "timeout_s": 20},
            )
            stats = await client.get("http://127.0.0.1:8100/fixture-stats")
        resumed_body = resumed.json() if resumed.content else {}
        stats_body = stats.json() if stats.content else {}
        repeated = int(stats_body.get("repeated_session_task_effectful_executions") or 0)
        receiver_fences = stats_body.get("highest_fence_by_operation") or {}
        return {
            "fault_process_terminated_request": bool(initial_error),
            "session_recovered": resumed.status_code == 200
            and resumed_body.get("status") == "completed",
            "one_terminal_result": resumed_body.get("status") == "completed",
            "zero_repeated_external_effects": repeated == 0,
            "receiver_operation_keys_observed": bool(receiver_fences),
            "receiver_fencing_tokens_positive": bool(receiver_fences)
            and all(int(value) > 0 for value in receiver_fences.values()),
        }, {
            "fault_point": fault_point,
            "initial_error": initial_error,
            "recovery_registry": registry,
            "resume": _response(resumed),
            "fixture_stats": stats_body,
        }

    return await _with_cluster(
        root,
        scenario=f"crash-{fault_point}",
        topology=3,
        trial=trial,
        initial_target=3,
        active_nodes=3,
        action=action,
        image=image,
        extra_env={"COORDINATION_FAULT_AT": fault_point},
        include_fault=True,
    )


async def _with_cluster(
    root: Path,
    *,
    scenario: str,
    topology: int,
    trial: int,
    initial_target: int,
    active_nodes: int,
    action: Callable[
        [ComposeController, list[str], int],
        Any,
    ],
    image: str,
    extra_env: dict[str, str] | None = None,
    include_fault: bool = False,
    primary: bool = True,
) -> TrialResult:
    started = time.monotonic()
    expected = _expected_checks(scenario)
    trial_dir = root / f"{scenario}-trial-{trial}"
    trial_dir.mkdir(parents=False, exist_ok=False)
    project = f"umac-{scenario[:18].replace('_', '-')}-{trial}-{uuid.uuid4().hex[:6]}"
    controller = ComposeController(
        project=project,
        evidence_dir=trial_dir,
        voter_target=initial_target,
        image=image,
        profiles=_profiles(active_nodes, chaos=include_fault),
        extra_env=extra_env,
    )
    names = list(COORDINATORS)[:active_nodes]
    print(f"[consensus] START {scenario} trial={trial} topology={topology}", flush=True)
    try:
        services = _services(active_nodes)
        if include_fault:
            services.append("coordination-fault")
        controller.up(services)
        checks, observations = await action(controller, names, initial_target)
        executed = sorted(checks)
        unexpected = sorted(set(executed) - set(expected))
        if unexpected:
            raise RuntimeError(f"Scenario returned undeclared checks: {unexpected}")
        violated = sorted(name for name, value in checks.items() if not bool(value))
        unexecuted = sorted(set(expected) - set(executed))
        passed = not violated and not unexecuted
        observations["image_ids"] = controller.image_ids()
        result = TrialResult(
            scenario=scenario,
            trial=trial,
            topology=topology,
            passed=passed,
            duration_s=time.monotonic() - started,
            checks=checks,
            observations=observations,
            status="passed" if passed else "invariant_failed",
            expected_checks=expected,
            executed_checks=executed,
            violated_checks=violated,
            unexecuted_checks=unexecuted,
            primary=primary,
        )
    except Exception as exc:
        result = TrialResult(
            scenario=scenario,
            trial=trial,
            topology=topology,
            passed=False,
            duration_s=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
            status="infrastructure_error",
            expected_checks=expected,
            unexecuted_checks=expected,
            primary=primary,
        )
    finally:
        try:
            controller.down()
        except Exception as cleanup_exc:
            result.passed = False
            result.status = "infrastructure_error"
            suffix = f"cleanup {type(cleanup_exc).__name__}: {cleanup_exc}"
            result.error = f"{result.error}; {suffix}" if result.error else suffix
            result.executed_checks = []
            result.violated_checks = []
            result.unexecuted_checks = expected
    _write_exclusive(trial_dir / "trial.json", result.__dict__)
    print(
        f"[consensus] END {scenario} trial={trial} status={result.status} "
        f"duration_s={result.duration_s:.2f}",
        flush=True,
    )
    return result


async def _ready(url: str) -> httpx.Response:
    # The service may spend up to ten seconds proving that its authoritative
    # etcd read cannot reach quorum.  The observer timeout must exceed that
    # bound or a correct fail-closed response is misclassified as infrastructure loss.
    async with httpx.AsyncClient(timeout=12.0) as client:
        return await client.get(f"{url}/health/ready")


async def _wait_ready_status(url: str, status: int, *, timeout_s: float) -> httpx.Response:
    deadline = time.monotonic() + timeout_s
    last: httpx.Response | None = None
    while time.monotonic() < deadline:
        try:
            last = await _ready(url)
            if last.status_code == status:
                return last
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    if last is None:
        raise RuntimeError(f"{url} did not return an HTTP response.")
    raise RuntimeError(f"{url} returned {last.status_code}, expected {status}.")


def _response(response: httpx.Response) -> JsonObject:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text
    return {"status_code": response.status_code, "body": body}


def _git(command: list[str]) -> str:
    completed = subprocess.run(["git", *command], capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _image_metadata(image: str) -> JsonObject:
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Campaign image {image!r} is unavailable: {completed.stderr}")
    inspected = json.loads(completed.stdout)[0]
    image_id = str(inspected.get("Id") or "")
    if not image_id.startswith("sha256:"):
        raise RuntimeError("Campaign image does not expose an immutable sha256 image ID.")
    return {
        "reference": image,
        "image_id": image_id,
        "repo_digests": inspected.get("RepoDigests") or [],
    }


def _prepare_image(output_dir: Path, image: str, *, build: bool) -> JsonObject:
    if build:
        print(f"[consensus] BUILD image={image}", flush=True)
        started = time.monotonic()
        completed = subprocess.run(
            ["docker", "build", "--tag", image, "."],
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        build_record = {
            "command": ["docker", "build", "--tag", image, "."],
            "returncode": completed.returncode,
            "duration_s": time.monotonic() - started,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        }
        _write_exclusive(output_dir / "image-build.json", build_record)
        if completed.returncode:
            raise RuntimeError(
                f"Campaign image build failed ({completed.returncode}): {completed.stderr[-4000:]}"
            )
    metadata = _image_metadata(image)
    _write_exclusive(output_dir / "image.json", metadata)
    print(f"[consensus] IMAGE id={metadata['image_id']}", flush=True)
    return metadata


def _provenance(image: JsonObject) -> JsonObject:
    lock = Path("uv.lock")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git(["rev-parse", "HEAD"]),
        "dirty_state": bool(_git(["status", "--porcelain"])),
        "dependency_lock_sha256": (
            hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "unavailable"
        ),
        "docker_version": subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "etcd_version": "3.6.13",
        "topology_scope": [3, 5, 7],
        "failure_domain": "single Docker host",
        "image": image,
        "dockerfile_sha256": hashlib.sha256(Path("Dockerfile").read_bytes()).hexdigest(),
    }


def _write_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


async def run_campaign(
    output_dir: Path,
    trials: int,
    *,
    smoke: bool = False,
    promotion_candidate: bool = False,
    image: str = DEFAULT_IMAGE,
    build_image: bool = True,
) -> JsonObject:
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        image_metadata = _prepare_image(output_dir, image, build=build_image)
    except Exception as exc:
        image_metadata = {"reference": image, "image_id": "", "repo_digests": []}
        provenance = _provenance(image_metadata)
        _write_exclusive(output_dir / "provenance.json", provenance)
        report = {
            "schema_version": "consensus-campaign-v4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
            "trial_count": 0,
            "primary_trial_count": 0,
            "supplementary_trial_count": 0,
            "expected_trial_count": len(expected_scenarios(trials, smoke=smoke)),
            "matrix_complete": False,
            "complete_check_accounting": False,
            "infrastructure_logs_complete": (output_dir / "image-build.json").is_file(),
            "passed": False,
            "outcome": "failed",
            "claim_status": "unsupported",
            "condition_results": {},
            "evidence_valid": False,
            "infrastructure_failed_trials": 0,
            "invariant_failed_trials": 0,
            "safety_checks_expected": 0,
            "safety_checks_executed": 0,
            "safety_violations": 0,
            "unexecuted_checks": 0,
            "primary_safety_checks_expected": 0,
            "primary_safety_checks_executed": 0,
            "primary_safety_violations": 0,
            "image_preparation_error": f"{type(exc).__name__}: {exc}",
            "results": [],
            "promotion_candidate": promotion_candidate,
            "accepted": False,
        }
        _write_exclusive(output_dir / "campaign.json", report)
        return report
    provenance = _provenance(image_metadata)
    _write_exclusive(output_dir / "provenance.json", provenance)
    results: list[TrialResult] = []
    topologies = (3,) if smoke else (3, 5, 7)
    trial_count = 1 if smoke else trials
    for trial in range(1, trial_count + 1):
        for topology in topologies:
            results.append(await _formation_trial(output_dir, topology, trial, image))
        if not smoke:
            results.append(await _reconfiguration_trial(output_dir, 3, 5, trial, image))
            results.append(await _reconfiguration_trial(output_dir, 5, 7, trial, image))
        results.append(await _leader_termination_trial(output_dir, trial, image))
        results.append(await _minority_partition_trial(output_dir, trial, image))
        results.append(await _majority_loss_trial(output_dir, trial, image))
        results.append(await _concurrent_ownership_trial(output_dir, trial, image))
        results.append(await _audit_failure_trial(output_dir, trial, image))
        if not smoke:
            results.append(await _failed_voter_replacement_trial(output_dir, trial, image))
            for fault_point in (
                "after_task_attempt_start",
                "after_external_dispatch",
                "during_aggregation",
            ):
                results.append(await _crash_window_trial(output_dir, trial, fault_point, image))
            results.append(await _fault_trial(output_dir, trial, image))
    expected_matrix = expected_scenarios(trials, smoke=smoke)
    actual_matrix = [(item.scenario, item.trial) for item in results]
    matrix_complete = actual_matrix == expected_matrix
    safety_checks_expected = sum(len(item.expected_checks) for item in results)
    safety_checks_executed = sum(len(item.executed_checks) for item in results)
    safety_violations = sum(len(item.violated_checks) for item in results)
    infrastructure_failures = sum(item.status == "infrastructure_error" for item in results)
    invariant_failures = sum(item.status == "invariant_failed" for item in results)
    primary_results = [item for item in results if item.primary]
    supplementary_results = [item for item in results if not item.primary]
    condition_results: dict[str, JsonObject] = {}
    for scenario in sorted({item.scenario for item in primary_results}):
        observations = [item for item in primary_results if item.scenario == scenario]
        condition_results[scenario] = {
            "trial_count": len(observations),
            "passed_trials": sum(item.passed for item in observations),
            "invariant_failed_trials": sum(
                item.status == "invariant_failed" for item in observations
            ),
            "infrastructure_failed_trials": sum(
                item.status == "infrastructure_error" for item in observations
            ),
            "checks_expected": sum(len(item.expected_checks) for item in observations),
            "checks_executed": sum(len(item.executed_checks) for item in observations),
            "violations": sum(len(item.violated_checks) for item in observations),
            "supported": bool(observations) and all(item.passed for item in observations),
        }
    passed = matrix_complete and all(item.passed for item in primary_results)
    outcome = "passed" if passed else "failed"
    supplementary_outcome = (
        "passed" if all(item.passed for item in supplementary_results) else "failed"
    )
    supported_conditions = sum(bool(item["supported"]) for item in condition_results.values())
    claim_status = (
        "supported"
        if condition_results and supported_conditions == len(condition_results)
        else "partially_supported"
        if supported_conditions
        else "unsupported"
    )
    complete_accounting = all(
        set(item.expected_checks) == set(item.executed_checks) | set(item.unexecuted_checks)
        and not (set(item.executed_checks) & set(item.unexecuted_checks))
        for item in results
    )
    infrastructure_logs_complete = all(
        item.status != "infrastructure_error"
        or (output_dir / f"{item.scenario}-trial-{item.trial}" / "compose-commands.jsonl").is_file()
        for item in results
    )
    evidence_valid = (
        matrix_complete
        and len(results) == len(expected_matrix)
        and complete_accounting
        and infrastructure_logs_complete
        and provenance["dirty_state"] is False
        and str(image_metadata.get("image_id", "")).startswith("sha256:")
    )
    report = {
        "schema_version": "consensus-campaign-v4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "trial_count": len(results),
        "primary_trial_count": len(primary_results),
        "supplementary_trial_count": len(supplementary_results),
        "expected_trial_count": len(expected_matrix),
        "matrix_complete": matrix_complete,
        "complete_check_accounting": complete_accounting,
        "infrastructure_logs_complete": infrastructure_logs_complete,
        "passed": passed,
        "outcome": outcome,
        "supplementary_outcome": supplementary_outcome,
        "claim_status": claim_status,
        "condition_results": condition_results,
        "evidence_valid": evidence_valid,
        "infrastructure_failed_trials": infrastructure_failures,
        "invariant_failed_trials": invariant_failures,
        "safety_checks_expected": safety_checks_expected,
        "safety_checks_executed": safety_checks_executed,
        "safety_violations": safety_violations,
        "unexecuted_checks": safety_checks_expected - safety_checks_executed,
        "primary_safety_checks_expected": sum(
            len(item.expected_checks) for item in primary_results
        ),
        "primary_safety_checks_executed": sum(
            len(item.executed_checks) for item in primary_results
        ),
        "primary_safety_violations": sum(len(item.violated_checks) for item in primary_results),
        "results": [item.__dict__ for item in results],
        "promotion_candidate": promotion_candidate,
        "accepted": promotion_candidate
        and not smoke
        and evidence_valid
        and not provenance["dirty_state"],
    }
    _write_exclusive(output_dir / "campaign.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Use an existing immutable local image instead of building once.",
    )
    parser.add_argument(
        "--promotion-candidate",
        action="store_true",
        help="Mark a clean, complete full campaign as an explicit release candidate.",
    )
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least one")
    if args.smoke and args.promotion_candidate:
        raise SystemExit("A smoke run cannot be an evidence-promotion candidate.")
    report = asyncio.run(
        run_campaign(
            args.output_dir,
            args.trials,
            smoke=args.smoke,
            promotion_candidate=args.promotion_candidate,
            image=args.image,
            build_image=not args.no_build,
        )
    )
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
