from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from unified_multi_agent_coordination import coordinator_node
from unified_multi_agent_coordination.coordinator_node import (
    CoordinatorNodeLauncher,
    _csv_env,
    _env_bool,
)
from unified_multi_agent_coordination.etcd_client import EtcdKeyValue, EtcdRange


class FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _environment(monkeypatch: pytest.MonkeyPatch, data_dir: Path, **values: str) -> None:
    defaults = {
        "COORDINATION_CLUSTER_ID": "alpha",
        "COORDINATION_NODE_ID": "node-a",
        "COORDINATION_CLUSTER_SECRET": "secret",
        "COORDINATION_ETCD_DATA_DIR": str(data_dir),
        "COORDINATION_VOTER_TARGET": "3",
        "COORDINATION_DISCOVERY_SEEDS": "seed-a:8000, http://seed-b:8000",
        # Register launcher-written variables with monkeypatch so each test
        # restores the process environment after exercising run().
        "COORDINATION_STORE_URL": "",
        "COORDINATION_NODE_ROLE": "",
        "COORDINATION_MEMBER_ID": "",
        "COORDINATION_ADVERTISE_API_URL": "http://node-a:8000",
        "COORDINATION_ADVERTISE_PEER_URL": "http://node-a:2380",
        "COORDINATION_ADVERTISE_CLIENT_URL": "http://node-a:2379",
        "COORDINATION_DISCOVERY_METHOD": "test",
    }
    defaults.update(values)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)


def _assignment(path: Path, **updates: object) -> None:
    value = {
        "cluster_id": "alpha",
        "role": "client",
        "member_id": 0,
        "etcd_endpoints": ["http://etcd-a:2379"],
        "voter_target": 3,
        "seed_url": "http://seed-a:8000",
        **updates,
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "coordinator-assignment.json").write_text(json.dumps(value))


def test_launcher_environment_parsing_and_persisted_assignment(
    tmp_path, monkeypatch
) -> None:
    data = tmp_path / "etcd"
    _environment(monkeypatch, data)
    _assignment(data)
    launcher = CoordinatorNodeLauncher()
    assert launcher.persisted_assignment["role"] == "client"
    assert launcher.voter_target == 3
    assert launcher.discovery.seeds == ["http://seed-a:8000", "http://seed-b:8000"]

    launcher._persist_assignment({"role": "voter", "member_id": 12})
    persisted = json.loads(launcher.assignment_path.read_text())
    assert persisted == {"cluster_id": "alpha", "member_id": 12, "role": "voter"}

    launcher.assignment_path.write_text("not json")
    assert launcher._load_persisted_assignment() == {}
    launcher.assignment_path.write_text(json.dumps({"cluster_id": "other", "role": "voter"}))
    assert launcher._load_persisted_assignment() == {}


def test_launcher_rejects_unsupported_target_and_parses_helpers(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path, COORDINATION_VOTER_TARGET="2")
    with pytest.raises(ValueError, match="1, 3, 5, or 7"):
        CoordinatorNodeLauncher()
    monkeypatch.delenv("OPTIONAL_FLAG", raising=False)
    assert _env_bool("OPTIONAL_FLAG", default=True) is True
    monkeypatch.setenv("OPTIONAL_FLAG", "YES")
    assert _env_bool("OPTIONAL_FLAG") is True
    monkeypatch.setenv("CSV", " a, ,b ")
    assert _csv_env("CSV") == ["a", "b"]


@pytest.mark.asyncio
async def test_run_restores_persisted_client_assignment(tmp_path, monkeypatch) -> None:
    data = tmp_path / "etcd"
    _environment(monkeypatch, data)
    _assignment(data)
    process = FakeProcess()
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    launcher = CoordinatorNodeLauncher()
    launcher._supervise = AsyncMock(return_value=0)  # type: ignore[method-assign]
    launcher._install_signal_handlers = lambda: None  # type: ignore[method-assign]

    assert await launcher.run() == 0
    assert launcher.etcd_process is None
    assert launcher.etcd_endpoints == ["http://etcd-a:2379"]
    assert "etcd-a:2379" in coordinator_node.os.environ["COORDINATION_STORE_URL"]
    assert spawn.await_args.args[1:3] == ("-m", "unified_multi_agent_coordination.service")
    await launcher.shutdown()


@pytest.mark.asyncio
async def test_run_restores_persisted_voter_and_local_endpoint(tmp_path, monkeypatch) -> None:
    data = tmp_path / "etcd"
    _environment(monkeypatch, data)
    _assignment(
        data,
        role="voter",
        member_id=9,
        initial_cluster="node-a=http://node-a:2380",
        etcd_endpoints=["http://etcd-b:2379"],
    )
    (data / "member").mkdir()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=FakeProcess()))
    launcher = CoordinatorNodeLauncher()
    launcher._start_joined_member = AsyncMock()  # type: ignore[method-assign]
    launcher._supervise = AsyncMock(return_value=0)  # type: ignore[method-assign]
    launcher._install_signal_handlers = lambda: None  # type: ignore[method-assign]

    assert await launcher.run() == 0
    launcher._start_joined_member.assert_awaited_once_with(  # type: ignore[attr-defined]
        "node-a=http://node-a:2380", wait_for_health=False
    )
    assert launcher.etcd_endpoints[0] == "http://127.0.0.1:2379"
    await launcher.shutdown()


@pytest.mark.asyncio
async def test_run_bootstraps_or_joins_as_learner(tmp_path, monkeypatch) -> None:
    data = tmp_path / "bootstrap"
    _environment(monkeypatch, data, COORDINATION_BOOTSTRAP="true")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=FakeProcess()))
    launcher = CoordinatorNodeLauncher()
    launcher.discovery.discover = AsyncMock(return_value=(None, "none"))  # type: ignore[method-assign]
    launcher._start_bootstrap_member = AsyncMock()  # type: ignore[method-assign]
    launcher._supervise = AsyncMock(return_value=0)  # type: ignore[method-assign]
    launcher._install_signal_handlers = lambda: None  # type: ignore[method-assign]
    assert await launcher.run() == 0
    launcher._start_bootstrap_member.assert_awaited_once()  # type: ignore[attr-defined]
    assert json.loads(launcher.assignment_path.read_text())["role"] == "voter"
    await launcher.shutdown()

    joined_data = tmp_path / "joined"
    _environment(monkeypatch, joined_data, COORDINATION_BOOTSTRAP="false")
    joined = CoordinatorNodeLauncher()
    joined.discovery.discover = AsyncMock(  # type: ignore[method-assign]
        return_value=("http://seed:8000", "dns")
    )
    joined._join = AsyncMock(return_value={  # type: ignore[method-assign]
        "role": "learner",
        "member_id": 15,
        "initial_cluster": "node-a=http://node-a:2380",
        "etcd_endpoints": ["http://etcd-a:2379"],
        "voter_target": 3,
    })
    joined._start_joined_member = AsyncMock()  # type: ignore[method-assign]
    joined._supervise = AsyncMock(return_value=0)  # type: ignore[method-assign]
    joined._install_signal_handlers = lambda: None  # type: ignore[method-assign]
    assert await joined.run() == 0
    joined._start_joined_member.assert_awaited_once()  # type: ignore[attr-defined]
    assert joined.etcd_endpoints[0] == "http://127.0.0.1:2379"
    await joined.shutdown()


@pytest.mark.asyncio
async def test_start_etcd_builds_safe_membership_arguments(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path)
    healthy = FakeProcess()
    spawn = AsyncMock(return_value=healthy)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    launcher = CoordinatorNodeLauncher()
    await launcher._start_bootstrap_member(wait_for_health=False)
    args = spawn.await_args.args
    assert "--strict-reconfig-check=true" in args
    assert args[args.index("--initial-cluster-state") + 1] == "new"

    exited = FakeProcess(returncode=4)
    spawn.return_value = exited
    with pytest.raises(RuntimeError, match="exited with code 4"):
        await launcher._start_joined_member("node-a=http://node-a:2380", wait_for_health=False)


@pytest.mark.asyncio
async def test_wait_for_cluster_and_shutdown_are_bounded(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path)
    launcher = CoordinatorNodeLauncher()
    launcher.discovery.discover = AsyncMock(  # type: ignore[method-assign]
        side_effect=[(None, "none"), ("http://seed:8000", "dns")]
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    assert await launcher._wait_for_cluster() == "http://seed:8000"
    assert coordinator_node.os.environ["COORDINATION_DISCOVERY_METHOD"] == "dns"

    service = FakeProcess()
    etcd = FakeProcess()
    launcher.service_process = service  # type: ignore[assignment]
    launcher.etcd_process = etcd  # type: ignore[assignment]
    launcher.seed_url = "http://seed:8000"
    launcher._leave = AsyncMock()  # type: ignore[method-assign]
    await launcher.shutdown()
    assert service.terminated and etcd.terminated
    launcher._leave.assert_awaited_once()  # type: ignore[attr-defined]
    await launcher.shutdown()  # idempotent


@pytest.mark.asyncio
async def test_assignment_monitor_persists_authoritative_role(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path)
    launcher = CoordinatorNodeLauncher()
    launcher.etcd_endpoints = ["http://etcd:2379"]
    record = {
        "node_id": "node-a",
        "api_url": "http://node-a:8000",
        "peer_url": "http://node-a:2380",
        "client_url": "http://node-a:2379",
        "role": "voter",
        "member_id": 8,
        "initial_cluster": "node-a=http://node-a:2380",
        "voter_target": 3,
    }

    class Client:
        async def sync_endpoints(self) -> None:
            return None

        async def range(self, key: bytes) -> EtcdRange:
            launcher._stopping.set()
            return EtcdRange(
                values=[EtcdKeyValue(
                    key=key,
                    value=json.dumps(record).encode(),
                    create_revision=1,
                    mod_revision=1,
                    version=1,
                    lease=0,
                )],
                revision=1,
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(coordinator_node, "EtcdClient", lambda *args, **kwargs: Client())
    launcher._start_joined_member = AsyncMock()  # type: ignore[method-assign]
    await launcher._assignment_monitor()
    persisted = json.loads(launcher.assignment_path.read_text())
    assert persisted["role"] == "voter"
    launcher._start_joined_member.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_supervise_requires_service_and_returns_service_exit(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path)
    launcher = CoordinatorNodeLauncher()
    with pytest.raises(RuntimeError, match="was not started"):
        await launcher._supervise()

    launcher.service_process = FakeProcess(returncode=7)  # type: ignore[assignment]
    launcher.shutdown = AsyncMock()  # type: ignore[method-assign]
    assert await launcher._supervise() == 7
    launcher.shutdown.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_start_etcd_waits_for_health_and_times_out(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path)
    process = FakeProcess()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    class HealthyClient:
        attempts = 0

        async def status(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("starting")
            return {"leader": "1"}

        async def close(self):
            return None

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(coordinator_node, "EtcdClient", lambda *args, **kwargs: HealthyClient())
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    launcher = CoordinatorNodeLauncher()
    await launcher._start_bootstrap_member()

    class UnhealthyClient:
        async def status(self):
            raise RuntimeError("still starting")

        async def close(self):
            return None

    monkeypatch.setattr(coordinator_node, "EtcdClient", lambda *args, **kwargs: UnhealthyClient())
    with pytest.raises(RuntimeError, match="did not become healthy"):
        await launcher._start_bootstrap_member()


class _JoinResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = coordinator_node.httpx.Request("POST", "http://seed")
            response = coordinator_node.httpx.Response(self.status_code, request=request)
            raise coordinator_node.httpx.HTTPStatusError(
                "join failed", request=request, response=response
            )

    def json(self):
        return self.payload


class _JoinClient:
    responses = []
    posts = []

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def post(self, url, json):
        self.posts.append((url, json))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_join_and_leave_use_signed_envelopes(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path)
    launcher = CoordinatorNodeLauncher()
    response = launcher.authenticator.sign(
        coordinator_node.SignedEnvelope(
            message_type="cluster_join_response",
            cluster_id="alpha",
            node_id="seed",
            payload={"role": "client", "etcd_endpoints": ["http://etcd:2379"]},
        )
    )
    _JoinClient.responses = [
        _JoinResponse(503, {}),
        _JoinResponse(200, response.model_dump(mode="json")),
        _JoinResponse(200, {}),
    ]
    _JoinClient.posts = []
    monkeypatch.setattr(coordinator_node.httpx, "AsyncClient", _JoinClient)

    assignment = await launcher._join("http://seed")
    assert assignment["role"] == "client"
    await launcher._leave("http://seed")
    assert _JoinClient.posts[-1][0].endswith("/internal/cluster/leave")


@pytest.mark.asyncio
async def test_run_rejects_persisted_assignment_without_endpoints(tmp_path, monkeypatch) -> None:
    data = tmp_path / "empty-endpoints"
    _environment(monkeypatch, data)
    _assignment(data, etcd_endpoints=[])
    launcher = CoordinatorNodeLauncher()
    with pytest.raises(RuntimeError, match="no usable etcd"):
        await launcher.run()


@pytest.mark.asyncio
async def test_wait_for_cluster_stops_and_monitor_bootstraps_missing_member(tmp_path, monkeypatch) -> None:
    _environment(monkeypatch, tmp_path, COORDINATION_BOOTSTRAP="true")
    launcher = CoordinatorNodeLauncher()
    launcher._stopping.set()
    with pytest.raises(RuntimeError, match="stopped before"):
        await launcher._wait_for_cluster()

    launcher._stopping.clear()
    launcher.etcd_endpoints = ["http://etcd:2379"]
    record = {
        "node_id": "node-a",
        "api_url": "http://node-a:8000",
        "peer_url": "http://node-a:2380",
        "client_url": "http://node-a:2379",
        "role": "voter",
        "member_id": 8,
        "initial_cluster": "",
        "voter_target": 3,
    }

    class Client:
        async def sync_endpoints(self):
            return None

        async def range(self, key):
            launcher._stopping.set()
            return EtcdRange(
                values=[
                    EtcdKeyValue(
                        key=key,
                        value=json.dumps(record).encode(),
                        create_revision=1,
                        mod_revision=1,
                        version=1,
                        lease=0,
                    )
                ],
                revision=1,
            )

        async def close(self):
            return None

    monkeypatch.setattr(coordinator_node, "EtcdClient", lambda *args, **kwargs: Client())
    launcher._start_bootstrap_member = AsyncMock()  # type: ignore[method-assign]
    await launcher._assignment_monitor()
    launcher._start_bootstrap_member.assert_awaited_once()  # type: ignore[attr-defined]
