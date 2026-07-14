import httpx
import pytest
from fastapi import FastAPI
import unified_multi_agent_coordination.service as service_module

from unified_multi_agent_coordination import (
    CapabilityRequirement,
    CoordinationAgent,
    CoordinationSdk,
    InMemoryCoordinationLedger,
    JsonlCoordinationStore,
)
from unified_multi_agent_coordination.service import create_app
from unified_multi_agent_coordination.etcd_client import EtcdQuorumUnavailableError
from unified_multi_agent_coordination.agent_registry import RegisteredAgent
from unified_multi_agent_coordination.cluster import (
    ClusterConfiguration,
    ConfigurationConflictError,
    CoordinatorNodeRecord,
    HmacAuthenticator,
    SignedEnvelope,
)
from unified_multi_agent_coordination.coordination_store import StaleFenceError
from unified_multi_agent_coordination.models import AgentRegistryEntry


def _app_with_local_summarizer() -> FastAPI:
    sdk = CoordinationSdk()
    sdk.register_local_agent(
        "Summarizer",
        [
            CapabilityRequirement(
                name="summarize",
                output_modes=["text"],
                validation_contract={"json_schema": {"type": "object"}},
            )
        ],
        lambda payload: {
            "artifacts": [
                {"name": "summary", "kind": "text", "text": payload.get("text", "")}
            ]
        },
    )
    return create_app(sdk=sdk)


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )


@pytest.mark.asyncio
async def test_health_endpoint():
    async with _client(create_app(sdk=CoordinationSdk())) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_registry_endpoint_returns_registered_agents():
    async with _client(_app_with_local_summarizer()) as client:
        response = await client.get("/registry")
    assert response.status_code == 200
    assert response.json()["agents"][0]["agent_id"] == "summarizer"


@pytest.mark.asyncio
async def test_plan_endpoint_returns_authorized_direct_plan():
    async with _client(_app_with_local_summarizer()) as client:
        response = await client.post(
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


@pytest.mark.asyncio
async def test_coordinate_endpoint_returns_local_artifact():
    async with _client(_app_with_local_summarizer()) as client:
        response = await client.post(
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
    assert body["artifacts"] == [
        {"name": "summary", "kind": "text", "text": "short summary"}
    ]


@pytest.mark.asyncio
async def test_coordinate_endpoint_accepts_session_id_and_resume_returns_terminal_result():
    ledger = InMemoryCoordinationLedger()
    sdk = CoordinationSdk()
    calls: list[str] = []
    sdk.register_local_agent(
        "Summarizer",
        [
            CapabilityRequirement(
                name="summarize",
                output_modes=["text"],
                validation_contract={"json_schema": {"type": "object"}},
            )
        ],
        lambda payload: calls.append(payload.get("text", ""))
        or {"artifacts": [{"kind": "text", "text": payload.get("text", "")}]},
    )
    app = create_app(sdk=sdk, agent=CoordinationAgent(sdk=sdk, ledger=ledger))
    async with _client(app) as client:
        response = await client.post(
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
        resumed = await client.post("/sessions/api-session/resume", json={})
    assert response.status_code == 200
    assert resumed.status_code == 200
    assert resumed.json()["session_id"] == "api-session"
    assert resumed.json()["artifacts"] == [{"kind": "text", "text": "first"}]
    assert calls == ["first"]


@pytest.mark.asyncio
async def test_coordinate_endpoint_returns_infeasible_without_agent():
    async with _client(create_app(sdk=CoordinationSdk())) as client:
        response = await client.post(
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


@pytest.mark.asyncio
async def test_coordinate_endpoint_returns_conflict_for_live_foreign_lease():
    store = JsonlCoordinationStore()
    await store.acquire_lease("locked-session", "other-coordinator", ttl_s=30)
    sdk = CoordinationSdk()
    agent = CoordinationAgent(sdk=sdk, store=store, coordinator_id="api-coordinator")
    async with _client(create_app(sdk=sdk, agent=agent)) as client:
        response = await client.post(
            "/coordinate",
            json={
                "session_id": "locked-session",
                "problem": {
                    "user_goal": "Summarize.",
                    "requirements": [{"name": "summarize"}],
                },
            },
        )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_feasibility_endpoint_checks_explicit_plan():
    async with _client(_app_with_local_summarizer()) as client:
        response = await client.post(
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


@pytest.mark.asyncio
async def test_coordinate_endpoint_returns_machine_readable_quorum_error(
    monkeypatch: pytest.MonkeyPatch,
):
    sdk = CoordinationSdk()
    agent = CoordinationAgent(sdk=sdk)

    async def unavailable(*args, **kwargs):
        del args, kwargs
        raise EtcdQuorumUnavailableError("etcdserver: no leader")

    monkeypatch.setattr(agent, "coordinate", unavailable)
    async with _client(create_app(sdk=sdk, agent=agent)) as client:
        response = await client.post(
            "/coordinate",
            json={
                "problem": {
                    "user_goal": "Summarize.",
                    "requirements": [{"name": "summarize"}],
                }
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "code": "quorum_unavailable",
        "retryable": True,
        "message": "etcdserver: no leader",
    }


@pytest.mark.asyncio
async def test_standalone_health_metrics_cluster_and_input_errors():
    app = create_app(sdk=CoordinationSdk())
    async with _client(app) as client:
        assert (await client.get("/health/live")).status_code == 200
        ready = await client.get("/health/ready")
        cluster = await client.get("/cluster/status")
        metrics = await client.get("/metrics")
        missing_input = await client.post("/plan", json={})
        unavailable_distributed = await client.post(
            "/internal/cluster/join",
            json={"message_type": "cluster_join", "cluster_id": "c", "node_id": "n"},
        )
        missing_session = await client.post("/sessions/missing/resume")

    assert ready.json() == {"status": "ready", "cluster": None}
    assert cluster.json()["role"] == "standalone"
    assert metrics.json()["joins"] == 0
    assert missing_input.status_code == 422
    assert unavailable_distributed.status_code == 503
    assert missing_session.status_code == 404


class _FakeMembership:
    def __init__(self):
        self.configuration = ClusterConfiguration(cluster_id="service-cluster", voter_target=3)
        self.current_node = CoordinatorNodeRecord(
            node_id="coordinator-a", api_url="http://a", role="voter"
        )
        self.removed = []
        self.conflict = False

    async def status(self):
        return {"state": "ready", "steady_state": True, "generation": self.configuration.generation}

    async def join(self, request):
        if request.node_id == "conflict":
            raise ValueError("duplicate node")
        return {"node_id": request.node_id, "role": "learner"}

    async def join_status(self, node_id):
        if node_id == "missing":
            raise KeyError(node_id)
        return {"node_id": node_id, "role": "voter"}

    async def leave(self, node_id):
        self.removed.append(node_id)

    async def update_voter_target(self, target, *, expected_generation, updated_by):
        if self.conflict:
            raise ConfigurationConflictError("stale generation")
        self.configuration = self.configuration.model_copy(
            update={
                "voter_target": target,
                "generation": expected_generation + 1,
                "updated_by": updated_by,
            }
        )
        return self.configuration


class _FakeRegistry:
    def __init__(self):
        self.records = {}

    async def register(self, record, *, ttl_s=30):
        self.records[record.entry.agent_id] = record
        return record

    async def heartbeat(self, agent_id):
        if agent_id not in self.records:
            raise KeyError(agent_id)
        return 99

    async def remove(self, agent_id):
        self.records.pop(agent_id, None)

    async def close(self):
        return None


def _distributed_app():
    sdk = CoordinationSdk()
    registry = _FakeRegistry()
    sdk.distributed_registry = registry
    app = create_app(sdk=sdk)
    membership = _FakeMembership()
    authenticator = HmacAuthenticator("cluster-secret")
    app.state.membership = membership
    app.state.authenticator = authenticator
    app.state.agent_authenticator = authenticator
    return app, membership, registry, authenticator


def _signed(authenticator, message_type, *, node_id="node-b", payload=None):
    return authenticator.sign(
        SignedEnvelope(
            message_type=message_type,
            cluster_id="service-cluster",
            node_id=node_id,
            payload=payload or {},
        )
    )


@pytest.mark.asyncio
async def test_signed_cluster_membership_routes_and_generation_cas():
    app, membership, _, authenticator = _distributed_app()
    async with _client(app) as client:
        status = await client.get("/cluster/status")
        joined = await client.post(
            "/internal/cluster/join",
            json=_signed(authenticator, "cluster_join").model_dump(mode="json"),
        )
        join_request = _signed(authenticator, "cluster_join_status", node_id="node-b")
        join_status = await client.get(
            "/internal/cluster/join/node-b",
            params={
                "timestamp": join_request.timestamp,
                "nonce": join_request.nonce,
                "signature": join_request.signature,
            },
        )
        left = await client.post(
            "/internal/cluster/leave",
            json=_signed(authenticator, "cluster_leave").model_dump(mode="json"),
        )
        updated = await client.put(
            "/internal/cluster/configuration",
            json=_signed(
                authenticator,
                "cluster_configuration",
                payload={"voter_target": 5, "expected_generation": 0},
            ).model_dump(mode="json"),
        )

    assert status.json()["state"] == "ready"
    assert joined.json()["payload"]["role"] == "learner"
    assert join_status.json()["payload"]["role"] == "voter"
    assert left.json() == {"status": "removed"}
    assert membership.removed == ["node-b"]
    assert updated.json()["voter_target"] == 5
    assert updated.json()["generation"] == 1


@pytest.mark.asyncio
async def test_signed_agent_registry_routes_and_failures():
    app, _, registry, authenticator = _distributed_app()
    record = RegisteredAgent(
        entry=AgentRegistryEntry(
            agent_id="remote", name="Remote", service_endpoint="http://remote"
        ),
        supports_fencing=True,
    )
    async with _client(app) as client:
        registered = await client.post(
            "/internal/agents/register",
            json=_signed(
                authenticator,
                "agent_register",
                payload={"record": record.model_dump(mode="json"), "ttl_s": 45},
            ).model_dump(mode="json"),
        )
        heartbeat = await client.post(
            "/internal/agents/remote/heartbeat",
            json=_signed(authenticator, "agent_heartbeat").model_dump(mode="json"),
        )
        missing = await client.post(
            "/internal/agents/missing/heartbeat",
            json=_signed(authenticator, "agent_heartbeat").model_dump(mode="json"),
        )
        removed = await client.request(
            "DELETE",
            "/internal/agents/remote",
            json=_signed(authenticator, "agent_remove").model_dump(mode="json"),
        )

    assert registered.status_code == 200
    assert heartbeat.json()["ttl"] == 99
    assert missing.status_code == 404
    assert removed.json() == {"status": "removed"}
    assert registry.records == {}


@pytest.mark.asyncio
async def test_signed_route_rejections_and_configuration_errors():
    app, membership, _, authenticator = _distributed_app()
    async with _client(app) as client:
        wrong_type = await client.post(
            "/internal/cluster/join",
            json=_signed(authenticator, "wrong").model_dump(mode="json"),
        )
        unsigned = SignedEnvelope(
            message_type="cluster_join", cluster_id="service-cluster", node_id="node-b"
        )
        bad_signature = await client.post(
            "/internal/cluster/join", json=unsigned.model_dump(mode="json")
        )
        conflict_join = await client.post(
            "/internal/cluster/join",
            json=_signed(authenticator, "cluster_join", node_id="conflict").model_dump(mode="json"),
        )
        missing_target = await client.put(
            "/internal/cluster/configuration",
            json=_signed(
                authenticator,
                "cluster_configuration",
                payload={"expected_generation": 0},
            ).model_dump(mode="json"),
        )
        missing_generation = await client.put(
            "/internal/cluster/configuration",
            json=_signed(
                authenticator,
                "cluster_configuration",
                payload={"voter_target": 3},
            ).model_dump(mode="json"),
        )
        membership.conflict = True
        conflict_generation = await client.put(
            "/internal/cluster/configuration",
            json=_signed(
                authenticator,
                "cluster_configuration",
                payload={"voter_target": 3, "expected_generation": 0},
            ).model_dump(mode="json"),
        )

    assert wrong_type.status_code == 422
    assert bad_signature.status_code == 401
    assert conflict_join.status_code == 409
    assert missing_target.status_code == 422
    assert missing_generation.status_code == 422
    assert conflict_generation.status_code == 409


@pytest.mark.asyncio
async def test_resume_maps_stale_fence_and_updates_metrics(monkeypatch):
    sdk = CoordinationSdk()
    agent = CoordinationAgent(sdk=sdk)

    async def stale(*args, **kwargs):
        raise StaleFenceError("stale")

    monkeypatch.setattr(agent, "resume_session", stale)
    app = create_app(sdk=sdk, agent=agent)
    async with _client(app) as client:
        response = await client.post("/sessions/stale/resume")
        metrics = await client.get("/metrics")
    assert response.status_code == 409
    assert metrics.json()["stale_fence_rejections"] == 1


def test_environment_factories_cover_standalone_distributed_and_audit_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    monkeypatch.setenv("COORDINATION_REQUEST_TIMEOUT_S", "4.5")
    monkeypatch.setenv("COORDINATION_LEDGER_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("COORDINATION_LEASE_TTL_S", "8")
    monkeypatch.setenv("COORDINATION_REGISTRY_RETRIES", "3")
    monkeypatch.setenv("COORDINATION_TASK_RETRIES", "2")
    monkeypatch.setenv("COORDINATION_RETRY_BACKOFF_S", "0.01")
    monkeypatch.setenv("COORDINATION_MAX_CONCURRENT_DISPATCHES", "7")
    sdk = service_module.sdk_from_env()
    agent = service_module.agent_from_env(sdk)
    assert sdk.request_timeout_s == 4.5
    assert agent.lease_ttl_s == 8
    assert agent._dispatch_semaphore._value == 7

    monkeypatch.setenv("COORDINATION_AUDIT_STORE_URL", "postgresql://audit/db")
    with pytest.raises(ValueError, match="requires etcd"):
        service_module.agent_from_env(sdk)
    monkeypatch.setenv("COORDINATION_STORE_URL", "etcd://one:2379")
    monkeypatch.setenv("COORDINATION_AUDIT_STORE_URL", "redis://audit")
    with pytest.raises(ValueError, match="PostgreSQL"):
        service_module.agent_from_env(service_module.sdk_from_env())

    monkeypatch.setenv("COORDINATION_AUDIT_STORE_URL", "postgresql://audit/db")
    distributed_sdk = service_module.sdk_from_env()
    distributed_agent = service_module.agent_from_env(distributed_sdk)
    assert distributed_sdk.distributed_registry is not None
    assert isinstance(distributed_agent.store, service_module.AuditProjectingCoordinationStore)


def test_distributed_environment_component_construction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COORDINATION_STORE_URL", "etcd://a:2379,b:2379")
    monkeypatch.setenv("COORDINATION_CLUSTER_ID", "env-cluster")
    monkeypatch.setenv("COORDINATION_COORDINATOR_ID", "env-node")
    monkeypatch.setenv("COORDINATION_CLUSTER_SECRET", "secret")
    monkeypatch.setenv("COORDINATION_VOTER_TARGET", "5")
    monkeypatch.setenv("COORDINATION_NODE_ROLE", "learner")
    monkeypatch.setenv("COORDINATION_MEMBER_ID", "42")
    monkeypatch.setenv("COORDINATION_ADVERTISE_API_URL", "http://node:8000")
    monkeypatch.setenv("COORDINATION_ADVERTISE_PEER_URL", "http://node:2380")
    monkeypatch.setenv("COORDINATION_ADVERTISE_CLIENT_URL", "http://node:2379")
    monkeypatch.setenv("COORDINATION_DISCOVERY_METHOD", "multicast")

    membership, authenticator, responder = service_module._distributed_components_from_env()

    assert membership is not None
    assert membership.configuration.cluster_id == "env-cluster"
    assert membership.configuration.voter_target == 5
    assert membership.current_node.role == "learner"
    assert membership.current_node.member_id == 42
    assert authenticator is not None
    assert responder is not None
    assert service_module._env_bool("COORDINATION_ALLOW_INSECURE_CLUSTER") is False


@pytest.mark.asyncio
async def test_readiness_join_status_and_registry_unavailable_error_branches():
    app, membership, _, authenticator = _distributed_app()

    async def broken_status():
        raise RuntimeError("membership unavailable")

    membership.status = broken_status
    app.state.sdk.distributed_registry = None
    async with _client(app) as client:
        ready = await client.get("/health/ready")
        missing_join = _signed(authenticator, "cluster_join_status", node_id="missing")
        join_status = await client.get(
            "/internal/cluster/join/missing",
            params={
                "timestamp": missing_join.timestamp,
                "nonce": missing_join.nonce,
                "signature": missing_join.signature,
            },
        )
        register = await client.post(
            "/internal/agents/register",
            json=_signed(authenticator, "agent_register", payload={}).model_dump(mode="json"),
        )
        heartbeat = await client.post(
            "/internal/agents/a/heartbeat",
            json=_signed(authenticator, "agent_heartbeat").model_dump(mode="json"),
        )
        remove = await client.request(
            "DELETE",
            "/internal/agents/a",
            json=_signed(authenticator, "agent_remove").model_dump(mode="json"),
        )

    assert ready.status_code == 503
    assert join_status.status_code == 404
    assert register.status_code == 503
    assert heartbeat.status_code == 503
    assert remove.status_code == 503


@pytest.mark.asyncio
async def test_distributed_lifespan_starts_and_closes_every_component(monkeypatch):
    calls = []

    class Client:
        async def close(self):
            calls.append("membership-client-close")

    class Membership:
        client = Client()

        async def start(self):
            calls.append("membership-start")

        async def stop(self):
            calls.append("membership-stop")

    class Responder:
        async def start(self):
            calls.append("responder-start")

        async def stop(self):
            calls.append("responder-stop")

    class Registry:
        async def close(self):
            calls.append("registry-close")

    membership = Membership()
    responder = Responder()
    monkeypatch.setenv("COORDINATION_AGENT_REGISTRATION_SECRET", "agent-secret")
    monkeypatch.setattr(
        service_module,
        "_distributed_components_from_env",
        lambda: (membership, HmacAuthenticator("cluster-secret"), responder),
    )
    sdk = CoordinationSdk()
    sdk.distributed_registry = Registry()
    app = create_app(sdk=sdk)
    original_close = app.state.agent.store.close

    async def store_close():
        calls.append("store-close")
        await original_close()

    app.state.agent.store.close = store_close
    async with app.router.lifespan_context(app):
        assert calls == ["membership-start", "responder-start"]

    assert calls == [
        "membership-start",
        "responder-start",
        "responder-stop",
        "membership-stop",
        "membership-client-close",
        "store-close",
        "registry-close",
    ]
