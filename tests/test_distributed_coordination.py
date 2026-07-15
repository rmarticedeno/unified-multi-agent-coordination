from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from unified_multi_agent_coordination.agent_registry import (
    EtcdAgentRegistry,
    InMemoryAgentRegistry,
    RegisteredAgent,
)
from unified_multi_agent_coordination.cluster import (
    ClusterConfiguration,
    ConfigurationConflictError,
    CoordinatorNodeRecord,
    HmacAuthenticator,
    MembershipManager,
    MembershipOperation,
    SignedEnvelope,
)
from unified_multi_agent_coordination.cluster_discovery import ClusterDiscovery
from unified_multi_agent_coordination.coordination_ledger import LedgerEvent
from unified_multi_agent_coordination.coordination_store import (
    CoordinationStoreError,
    LeaseConflictError,
    StaleFenceError,
    StoreInvariantError,
)
from unified_multi_agent_coordination.etcd_client import (
    EtcdClient,
    EtcdError,
    EtcdKeyValue,
    EtcdQuorumUnavailableError,
    EtcdRange,
)
from unified_multi_agent_coordination.etcd_store import (
    EtcdCoordinationStore,
    etcd_endpoints_from_url,
    etcd_store_from_url,
)
from unified_multi_agent_coordination.models import AgentRegistryEntry


@pytest.mark.parametrize("target", [1, 3, 5, 7])
def test_voter_targets_accept_supported_odd_values(target: int) -> None:
    assert ClusterConfiguration(cluster_id="cluster", voter_target=target).voter_target == target


@pytest.mark.parametrize("target", [0, -1, 2, 4, 6, 8, 9])
def test_voter_targets_reject_unsafe_values(target: int) -> None:
    with pytest.raises(ValueError, match="1, 3, 5, or 7"):
        ClusterConfiguration(cluster_id="cluster", voter_target=target)


def test_hmac_rejects_tampering_replay_expiry_and_wrong_cluster() -> None:
    authenticator = HmacAuthenticator("shared-secret")
    signed = authenticator.sign(
        SignedEnvelope(message_type="cluster_join", cluster_id="alpha", node_id="node-a")
    )
    authenticator.verify(signed, expected_cluster_id="alpha")
    with pytest.raises(ValueError, match="replayed"):
        authenticator.verify(signed, expected_cluster_id="alpha")

    tampered = signed.model_copy(update={"nonce": "different", "payload": {"peer_url": "forged"}})
    with pytest.raises(ValueError, match="signature"):
        authenticator.verify(tampered, expected_cluster_id="alpha")

    expired = authenticator.sign(
        SignedEnvelope(
            message_type="cluster_join",
            cluster_id="alpha",
            node_id="node-a",
            timestamp=time.time() - 31,
        )
    )
    with pytest.raises(ValueError, match="timestamp"):
        authenticator.verify(expired, expected_cluster_id="alpha")

    wrong_cluster = authenticator.sign(
        SignedEnvelope(message_type="cluster_join", cluster_id="beta", node_id="node-a")
    )
    with pytest.raises(ValueError, match="another cluster"):
        authenticator.verify(wrong_cluster, expected_cluster_id="alpha")


@pytest.mark.asyncio
async def test_discovery_prefers_dns_and_uses_multicast_only_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth = HmacAuthenticator("secret")
    discovery = ClusterDiscovery(
        cluster_id="alpha",
        node_id="node-b",
        authenticator=auth,
        seeds=["seed-a:8000", "seed-b:8000"],
    )

    async def reachable(seed: str) -> bool:
        return seed.endswith("seed-b:8000")

    monkeypatch.setattr(discovery, "_reachable", reachable)
    monkeypatch.setattr(
        discovery,
        "_multicast_discover",
        lambda: pytest.fail("multicast must not run when a DNS seed is healthy"),
    )
    assert await discovery.discover() == ("http://seed-b:8000", "dns")
    await discovery.close()

    fallback = ClusterDiscovery(
        cluster_id="alpha",
        node_id="node-b",
        authenticator=auth,
        seeds=["seed-a:8000"],
    )

    async def unreachable(_seed: str) -> bool:
        return False

    response = auth.sign(
        SignedEnvelope(
            message_type="coordinator_discovery_response",
            cluster_id="alpha",
            node_id="node-a",
            payload={"api_url": "http://node-a:8000"},
        )
    )
    monkeypatch.setattr(fallback, "_reachable", unreachable)
    monkeypatch.setattr(fallback, "_multicast_discover", lambda: response)
    assert await fallback.discover() == ("http://node-a:8000", "multicast")
    await fallback.close()


@pytest.mark.asyncio
async def test_etcd_client_rotates_endpoints_and_decodes_watch_events() -> None:
    calls: list[str] = []
    watch_line = json.dumps(
        {
            "result": {
                "header": {"revision": "7"},
                "events": [
                    {
                        "type": "PUT",
                        "kv": {
                            "key": _b64(b"/watched"),
                            "value": _b64(b"value"),
                            "create_revision": "7",
                            "mod_revision": "7",
                            "version": "1",
                            "lease": "0",
                        },
                    }
                ],
            }
        }
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "unavailable":
            raise httpx.ConnectError("offline", request=request)
        if request.url.path == "/v3/watch":
            return httpx.Response(200, text=watch_line + "\n")
        return httpx.Response(200, json={"header": {"revision": "6"}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = EtcdClient(
        ["http://unavailable:2379", "http://healthy:2379"],
        http_client=http_client,
    )
    assert await client.put(b"key", b"value") == 6
    assert calls[:2] == [
        "http://unavailable:2379/v3/kv/put",
        "http://healthy:2379/v3/kv/put",
    ]
    event = await anext(client.watch(b"/watched"))
    assert event.event_type == "PUT"
    assert event.value.key == b"/watched"
    assert event.value.value == b"value"
    assert event.revision == 7
    await http_client.aclose()


@pytest.mark.asyncio
async def test_etcd_client_distinguishes_semantic_errors_from_quorum_loss() -> None:
    async def semantic(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "member is not a learner"}, request=request)

    semantic_http = httpx.AsyncClient(transport=httpx.MockTransport(semantic))
    semantic_client = EtcdClient(["http://etcd"], http_client=semantic_http)
    with pytest.raises(EtcdError, match="not a learner"):
        await semantic_client.member_promote(10)
    await semantic_http.aclose()

    async def no_leader(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "etcdserver: no leader"}, request=request)

    quorum_http = httpx.AsyncClient(transport=httpx.MockTransport(no_leader))
    quorum_client = EtcdClient(["http://etcd"], http_client=quorum_http)
    with pytest.raises(EtcdQuorumUnavailableError, match="quorum"):
        await quorum_client.put(b"key", b"value")
    await quorum_http.aclose()


@pytest.mark.asyncio
async def test_etcd_store_contract_fencing_attempt_and_terminal_invariants() -> None:
    backend = FakeEtcdClient()
    store = EtcdCoordinationStore(["http://unused"], cluster_id="alpha", client=backend)
    first = await store.acquire_lease("session", "coordinator-a", 10)
    with pytest.raises(LeaseConflictError):
        await store.acquire_lease("session", "coordinator-b", 10)

    newer = await store.acquire_lease("session", "coordinator-a", 10)
    assert newer.fencing_token > first.fencing_token
    with pytest.raises(StaleFenceError):
        await store.append_event(
            LedgerEvent(event_type="session_started", session_id="session"),
            first,
        )

    with pytest.raises(StoreInvariantError, match="no started attempt"):
        await store.append_event(
            LedgerEvent(
                event_type="task_attempt_completed",
                session_id="session",
                task_id="task-1",
                attempt_id="attempt-1",
                payload={"task_result": {"status": "completed"}},
            ),
            newer,
        )

    await store.append_event(
        LedgerEvent(
            event_type="task_attempt_started",
            session_id="session",
            task_id="task-1",
            attempt_id="attempt-1",
            payload={"idempotency_key": "attempt-key", "task": {"task_id": "task-1"}},
        ),
        newer,
    )
    await store.append_event(
        LedgerEvent(
            event_type="task_attempt_completed",
            session_id="session",
            task_id="task-1",
            attempt_id="attempt-1",
            payload={"task_result": {"status": "completed", "task_id": "task-1"}},
        ),
        newer,
    )
    assert await store.task_result_by_idempotency_key("attempt-key") == {
        "status": "completed",
        "task_id": "task-1",
    }

    terminal = LedgerEvent(
        event_type="run_completed",
        session_id="session",
        payload={"run_result": {"status": "completed"}},
    )
    await store.append_event(terminal, newer)
    with pytest.raises(StoreInvariantError, match="terminal"):
        await store.append_event(terminal, newer)


def _registry_record(
    agent_id: str,
    *,
    status: str = "available",
    scope: str = "remote",
    owner: str = "",
    lease: int = 0,
) -> RegisteredAgent:
    return RegisteredAgent(
        entry=AgentRegistryEntry(
            agent_id=agent_id,
            name=agent_id,
            agent_kind="remote_a2a",
            service_endpoint=f"http://{agent_id}:8000",
            status=status,
            availability_scope=scope,
            owner_node_id=owner,
        ),
        availability_scope=scope,
        owner_node_id=owner,
        backend_lease_id=lease,
    )


@pytest.mark.asyncio
async def test_in_memory_registry_lifecycle_and_missing_heartbeat() -> None:
    registry = InMemoryAgentRegistry()
    record = _registry_record("agent-a")
    assert await registry.register(record) == record
    assert await registry.snapshot() == [record.entry]
    assert await registry.heartbeat("agent-a") == 30
    with pytest.raises(KeyError):
        await registry.heartbeat("missing")
    await registry.remove("agent-a")
    assert await registry.snapshot() == []
    assert registry.revision == 2
    await registry.close()


@pytest.mark.asyncio
async def test_etcd_registry_filters_records_renews_and_revokes_leases() -> None:
    backend = FakeEtcdClient()
    registry = EtcdAgentRegistry(backend, cluster_id="alpha")  # type: ignore[arg-type]

    registered = await registry.register(_registry_record("agent-a"), ttl_s=12)
    assert registered.backend_lease_id
    assert registered.registration_revision == registry.revision
    assert await registry.snapshot() == [registered.entry]
    records = await registry.records()
    assert records[0].registration_revision > 0
    assert await registry.heartbeat("agent-a") == 10

    unavailable = _registry_record("agent-b", status="unavailable")
    await registry.register(unavailable)
    malformed_local = _registry_record("agent-c", scope="node_local", owner="")
    await backend.put(
        registry._key("agent-c"), malformed_local.model_dump_json().encode(), lease=999
    )
    assert [item.agent_id for item in await registry.snapshot()] == ["agent-a"]

    with pytest.raises(ValueError, match="Node-local"):
        await registry.register(_registry_record("local", scope="node_local", owner="node-a"))
    with pytest.raises(KeyError, match="missing"):
        await registry.heartbeat("missing")
    await backend.put(
        registry._key("no-lease"), _registry_record("no-lease").model_dump_json().encode()
    )
    with pytest.raises(ValueError, match="no renewable"):
        await registry.heartbeat("no-lease")
    for invalid in ("", "bad/id"):
        with pytest.raises(ValueError, match="path segment"):
            registry._key(invalid)

    lease = registered.backend_lease_id
    await registry.remove("agent-a")
    assert not any(item.lease == lease for item in backend.values.values())
    await registry.remove("unknown")
    await registry.close()


@pytest.mark.asyncio
async def test_membership_configuration_rejects_even_update() -> None:
    backend = FakeEtcdClient()
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="alpha"),
        CoordinatorNodeRecord(node_id="node-a", api_url="http://node-a:8000"),
    )
    await manager.initialize()
    with pytest.raises(ValueError, match="1, 3, 5, or 7"):
        await manager.update_voter_target(4, expected_generation=0, updated_by="node-a")


@pytest.mark.asyncio
async def test_membership_configuration_uses_generation_compare_and_swap() -> None:
    backend = FakeEtcdClient()
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="alpha"),
        CoordinatorNodeRecord(node_id="node-a", api_url="http://node-a:8000"),
    )
    await manager.initialize()

    updated = await manager.update_voter_target(5, expected_generation=0, updated_by="operator")

    assert updated.voter_target == 5
    assert updated.generation == 1
    assert updated.updated_by == "operator"
    with pytest.raises(ConfigurationConflictError, match="stale"):
        await manager.update_voter_target(7, expected_generation=0, updated_by="stale-operator")


@pytest.mark.asyncio
async def test_membership_reconciliation_adds_promotes_and_reaches_steady_state() -> None:
    backend = FakeEtcdClient()
    backend.members = [
        {
            "ID": 1,
            "name": "node-a",
            "peerURLs": ["http://node-a:2380"],
            "clientURLs": ["http://node-a:2379"],
            "isLearner": False,
        }
    ]
    current = CoordinatorNodeRecord(
        node_id="node-a",
        api_url="http://node-a:8000",
        peer_url="http://node-a:2380",
        client_url="http://node-a:2379",
        role="voter",
        member_id=1,
    )
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="alpha", voter_target=3),
        current,
    )
    await manager.initialize()
    for suffix in ("b", "c"):
        record = CoordinatorNodeRecord(
            node_id=f"node-{suffix}",
            api_url=f"http://node-{suffix}:8000",
            peer_url=f"http://node-{suffix}:2380",
            client_url=f"http://node-{suffix}:2379",
        )
        await backend.put(
            manager._key(f"membership/intents/{record.node_id}"),
            record.model_dump_json().encode(),
        )
        lease = await backend.grant_lease(15)
        await backend.put(
            manager._key(f"nodes/{record.node_id}"),
            record.model_dump_json().encode(),
            lease=lease,
        )

    async def synchronize(node_id: str) -> None:
        intent = await manager._intent(node_id)
        assert intent is not None
        current_node = backend.values[manager._key(f"nodes/{node_id}")]
        await backend.put(
            manager._key(f"nodes/{node_id}"),
            intent.model_dump_json().encode(),
            lease=current_node.lease,
        )

    await manager._reconcile_once()
    await synchronize("node-b")
    await manager._reconcile_once()
    await synchronize("node-b")
    await manager._reconcile_once()
    await synchronize("node-c")
    await manager._reconcile_once()
    await synchronize("node-c")

    status = await manager.status()
    assert status["state"] == "ready"
    assert status["steady_state"] is True
    assert status["active_voters"] == 3
    assert status["pending_membership_changes"] == 0


@pytest.mark.asyncio
async def test_reconciliation_repairs_stale_learner_intent_after_backend_promotion() -> None:
    backend = FakeEtcdClient()
    backend.members = [
        {"ID": 1, "isLearner": False, "peerURLs": ["http://a:2380"]},
        {"ID": 2, "isLearner": False, "peerURLs": ["http://b:2380"]},
        {"ID": 3, "isLearner": False, "peerURLs": ["http://c:2380"]},
    ]
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="role-repair", voter_target=3),
        CoordinatorNodeRecord(
            node_id="node-a",
            api_url="http://a",
            peer_url="http://a:2380",
            role="voter",
            member_id=1,
        ),
    )
    await manager.initialize()
    stale = CoordinatorNodeRecord(
        node_id="node-b",
        api_url="http://b",
        peer_url="http://b:2380",
        role="learner",
        member_id=2,
        voter_target=3,
    )
    await backend.put(manager._key("membership/intents/node-b"), stale.model_dump_json().encode())

    assert await manager._repair_authoritative_roles(list(backend.members)) is True
    repaired = await manager._intent("node-b")
    assert repaired is not None
    assert repaired.role == "voter"
    assert await manager._repair_authoritative_roles(list(backend.members)) is False


@pytest.mark.asyncio
async def test_membership_join_status_leave_and_client_assignment_paths() -> None:
    backend = FakeEtcdClient()
    backend.members = [
        {
            "ID": 1,
            "name": "node-a",
            "peerURLs": ["http://node-a:2380"],
            "clientURLs": ["http://node-a:2379"],
            "isLearner": False,
        }
    ]
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="joins", voter_target=3),
        CoordinatorNodeRecord(
            node_id="node-a",
            api_url="http://node-a:8000",
            peer_url="http://node-a:2380",
            role="voter",
        ),
    )
    await manager.initialize()
    assert manager.current_node.member_id == 1

    learner_request = SignedEnvelope(
        message_type="cluster_join",
        cluster_id="joins",
        node_id="node-b",
        payload={
            "voter_target": 3,
            "peer_url": "http://node-b:2380",
            "api_url": "http://node-b:8000",
            "client_url": "http://node-b:2379",
        },
    )
    assignment = await manager.join(learner_request)
    assert assignment["role"] == "learner"
    assert (await manager.join(learner_request))["member_id"] == assignment["member_id"]
    assert (await manager.join_status("node-b"))["role"] == "learner"
    with pytest.raises(KeyError):
        await manager.join_status("missing")
    with pytest.raises(ValueError, match="conflicting voter target"):
        await manager.join(
            SignedEnvelope(
                message_type="cluster_join",
                cluster_id="joins",
                node_id="wrong",
                payload={"voter_target": 5},
            )
        )
    with pytest.raises(ValueError, match="peer URL"):
        await manager.join(
            SignedEnvelope(
                message_type="cluster_join",
                cluster_id="joins",
                node_id="no-peer",
                payload={"voter_target": 3},
            )
        )

    for member in backend.members:
        member["isLearner"] = False
    backend.members.append(
        {
            "ID": 3,
            "peerURLs": ["http://node-c:2380"],
            "clientURLs": ["http://node-c:2379"],
            "isLearner": False,
        }
    )
    client_assignment = await manager.join(
        SignedEnvelope(
            message_type="cluster_join",
            cluster_id="joins",
            node_id="node-d",
            payload={"voter_target": 3, "api_url": "http://node-d:8000"},
        )
    )
    assert client_assignment["role"] == "client"
    await manager.leave("node-b")
    await manager.leave("unknown")
    assert await manager._intent("node-b") is None


@pytest.mark.asyncio
async def test_membership_initialization_and_concurrent_configuration_failures() -> None:
    backend = FakeEtcdClient()
    first = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="configuration", voter_target=3),
        CoordinatorNodeRecord(node_id="a", api_url="http://a"),
    )
    await first.initialize()
    conflicting = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="configuration", voter_target=5),
        CoordinatorNodeRecord(node_id="b", api_url="http://b"),
    )
    with pytest.raises(ValueError, match="conflicts"):
        await conflicting.initialize()

    original_transaction = backend.transaction

    async def reject_transaction(**kwargs):
        del kwargs
        return {"succeeded": False, "header": {"revision": str(backend.revision)}}

    backend.transaction = reject_transaction
    with pytest.raises(ConfigurationConflictError, match="concurrently"):
        await first.update_voter_target(5, expected_generation=0, updated_by="operator")
    backend.transaction = original_transaction


@pytest.mark.asyncio
async def test_excess_and_failed_voters_are_removed_without_removing_leader() -> None:
    backend = FakeEtcdClient()
    backend.members = [
        {
            "ID": member_id,
            "peerURLs": [f"http://node-{member_id}:2380"],
            "clientURLs": [f"http://node-{member_id}:2379"],
            "isLearner": False,
        }
        for member_id in range(1, 6)
    ]
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="removal", voter_target=3),
        CoordinatorNodeRecord(node_id="node-1", api_url="http://node-1", member_id=1, role="voter"),
        failed_voter_grace_s=0,
    )
    await manager.initialize()
    for member_id in range(1, 6):
        record = CoordinatorNodeRecord(
            node_id=f"node-{member_id}",
            api_url=f"http://node-{member_id}",
            peer_url=f"http://node-{member_id}:2380",
            member_id=member_id,
            role="voter",
        )
        await backend.put(
            manager._key(f"membership/intents/{record.node_id}"),
            record.model_dump_json().encode(),
        )
        await backend.put(
            manager._key(f"nodes/{record.node_id}"),
            record.model_dump_json().encode(),
            lease=await backend.grant_lease(15),
        )
    await manager._remove_excess_voter(list(backend.members))
    assert len(backend.members) == 4
    assert any(member["ID"] == 1 for member in backend.members)

    failed_member = 4
    await backend.delete(manager._key(f"nodes/node-{failed_member}"))
    client = CoordinatorNodeRecord(
        node_id="replacement",
        api_url="http://replacement",
        peer_url="http://replacement:2380",
        role="client",
    )
    await backend.put(
        manager._key("membership/intents/replacement"), client.model_dump_json().encode()
    )
    await backend.put(
        manager._key("nodes/replacement"),
        client.model_dump_json().encode(),
        lease=await backend.grant_lease(15),
    )
    assert await manager._replace_failed_voter(list(backend.members), []) is False
    marker = manager._key(f"membership/failures/{failed_member}")
    await backend.put(marker, str(time.time() - 1).encode())
    assert await manager._replace_failed_voter(list(backend.members), []) is True
    assert not any(member["ID"] == failed_member for member in backend.members)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action", ["add_learner", "promote_learner", "remove_voter", "remove_member"]
)
async def test_membership_operation_recovers_idempotently_in_each_phase(action: str) -> None:
    backend = FakeEtcdClient()
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id=f"recover-{action}", voter_target=3),
        CoordinatorNodeRecord(node_id="node-a", api_url="http://a"),
    )
    await manager.initialize()
    intent = CoordinatorNodeRecord(
        node_id="node-b",
        api_url="http://b",
        peer_url="http://b:2380",
        role="client" if action == "add_learner" else "learner",
        member_id=0 if action == "add_learner" else 2,
    )
    await backend.put(manager._key("membership/intents/node-b"), intent.model_dump_json().encode())
    if action != "add_learner":
        backend.members = [
            {
                "ID": 2,
                "peerURLs": ["http://b:2380"],
                "clientURLs": ["http://b:2379"],
                "isLearner": action == "promote_learner",
            }
        ]
    operation = MembershipOperation(
        generation=0,
        action=action,
        node_id="node-b",
        member_id=intent.member_id,
        peer_url=intent.peer_url,
    )
    await backend.put(manager._key("membership/operation"), operation.model_dump_json().encode())

    assert await manager._recover_membership_operation(list(backend.members)) is True
    assert await manager._membership_operation() is None
    recovered = await manager._intent("node-b")
    assert recovered is not None
    if action == "promote_learner":
        assert recovered.role == "voter"
    elif action in {"remove_voter", "remove_member"}:
        assert recovered.role == "client"
        assert recovered.member_id == 0
    else:
        assert recovered.role == "learner"
        assert recovered.member_id > 0


@pytest.mark.asyncio
async def test_reconciler_removes_stranded_learner_once_voter_target_is_met() -> None:
    backend = FakeEtcdClient()
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="stranded-learner", voter_target=3),
        CoordinatorNodeRecord(node_id="node-a", api_url="http://a"),
    )
    await manager.initialize()
    backend.members = [
        {"ID": member_id, "peerURLs": [f"http://node-{member_id}:2380"], "isLearner": False}
        for member_id in (1, 2, 3)
    ] + [{"ID": 4, "peerURLs": ["http://node-4:2380"], "isLearner": True}]
    learner = CoordinatorNodeRecord(
        node_id="node-4",
        api_url="http://node-4",
        peer_url="http://node-4:2380",
        member_id=4,
        role="learner",
    )
    await backend.put(manager._key("membership/intents/node-4"), learner.model_dump_json().encode())

    await manager._remove_excess_learner([backend.members[-1]])

    assert not any(member["ID"] == 4 for member in backend.members)
    recovered = await manager._intent("node-4")
    assert recovered is not None
    assert recovered.role == "client"
    assert recovered.member_id == 0


@pytest.mark.asyncio
async def test_etcd_store_complete_authoritative_lifecycle_and_queries() -> None:
    backend = FakeEtcdClient()
    store = EtcdCoordinationStore([], cluster_id="store", client=backend)
    lease = await store.acquire_lease("session", "coordinator-a", 30)
    assert lease.fencing_token > 0
    assert (await store.renew_lease(lease, 30)).heartbeat_at >= lease.heartbeat_at

    events = [
        LedgerEvent(
            event_type="session_started",
            session_id="session",
            payload={"payload": {"goal": "test"}},
        ),
        LedgerEvent(
            event_type="plan_authorized",
            session_id="session",
            plan_id="plan",
            payload={"plan_result": {"plan_generation": 1}},
        ),
        LedgerEvent(
            event_type="task_attempt_started",
            session_id="session",
            plan_id="plan",
            task_id="task",
            attempt_id="attempt",
            payload={"idempotency_key": "operation-key", "task": {"value": 1}},
        ),
        LedgerEvent(
            event_type="task_attempt_completed",
            session_id="session",
            plan_id="plan",
            task_id="task",
            attempt_id="attempt",
            payload={"task_result": {"status": "completed", "artifacts": []}},
        ),
        LedgerEvent(
            event_type="run_completed",
            session_id="session",
            plan_id="plan",
            payload={"run_result": {"status": "completed"}},
        ),
    ]
    for event in events:
        assert await store.append_event(event, lease) == event

    assert len(await store.events("session")) >= len(events)
    assert (await store.session_state("session")).terminal_result is not None
    result = await store.task_result_by_idempotency_key("operation-key")
    assert result is not None and result["status"] == "completed"
    assert await store.task_result_by_idempotency_key("missing") is None
    assert await store.ready() is True
    await store.release_lease(lease)
    await store.release_lease(lease)
    await store.close()


@pytest.mark.asyncio
async def test_etcd_store_rejects_lease_and_transaction_invariant_violations() -> None:
    backend = FakeEtcdClient()
    store = EtcdCoordinationStore([], cluster_id="reject", client=backend)
    lease = await store.acquire_lease("session", "coordinator-a", 30)
    with pytest.raises(LeaseConflictError, match="leased by"):
        await store.acquire_lease("session", "coordinator-b", 30)
    with pytest.raises(StoreInvariantError, match="does not match"):
        await store.append_event(
            LedgerEvent(event_type="session_started", session_id="other"), lease
        )
    with pytest.raises(StoreInvariantError, match="no started attempt"):
        await store.append_event(
            LedgerEvent(
                event_type="task_attempt_failed",
                session_id="session",
                task_id="missing",
                attempt_id="missing",
            ),
            lease,
        )

    await store.append_event(LedgerEvent(event_type="run_failed", session_id="session"), lease)
    with pytest.raises(StoreInvariantError, match="terminal"):
        await store.append_event(
            LedgerEvent(event_type="run_completed", session_id="session"), lease
        )

    wrong_holder = lease.model_copy(update={"holder_id": "other"})
    with pytest.raises(LeaseConflictError, match="leased by"):
        await store._validate_lease(wrong_holder)
    stale = lease.model_copy(update={"fencing_token": lease.fencing_token + 100})
    with pytest.raises(StaleFenceError):
        await store._validate_lease(stale)
    await backend.delete(store._session_key("session", "owner"))
    with pytest.raises(LeaseConflictError, match="no active lease"):
        await store._validate_lease(lease)

    original_transaction = backend.transaction

    async def reject(**kwargs):
        del kwargs
        return {"succeeded": False}

    backend.transaction = reject
    with pytest.raises(LeaseConflictError, match="raced"):
        await store.acquire_lease("raced", "coordinator-a", 30)
    with pytest.raises(CoordinationStoreError, match="comparison failed"):
        await store.append_event(LedgerEvent(event_type="custom", session_id="free"))
    backend.transaction = original_transaction


@pytest.mark.asyncio
async def test_membership_background_lifecycle_and_non_owner_reconcile(monkeypatch) -> None:
    backend = FakeEtcdClient()
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="lifecycle", voter_target=1),
        CoordinatorNodeRecord(node_id="node-a", api_url="http://a"),
        reconcile_interval_s=0.01,
        registration_ttl_s=0.01,
    )
    await manager.start()
    assert len(manager._tasks) == 2
    await manager.stop()
    assert manager._stopping.is_set()

    manager._stopping.clear()
    calls = []

    async def one_sleep(_seconds):
        manager._stopping.set()

    async def reconcile_failure():
        calls.append("reconcile")
        raise RuntimeError("retry")

    monkeypatch.setattr(asyncio, "sleep", one_sleep)
    original_reconcile = manager._reconcile_once
    manager._reconcile_once = reconcile_failure
    await manager._reconcile_loop()
    assert calls == ["reconcile"]
    manager._reconcile_once = original_reconcile

    manager._stopping.clear()
    manager.current_node = manager.current_node.model_copy(update={"backend_lease_id": 123})
    manager._sync_assignment = AsyncMock()  # type: ignore[method-assign]
    await manager._heartbeat_loop()
    manager._sync_assignment.assert_awaited_once()  # type: ignore[attr-defined]

    manager._stopping.clear()
    manager._sync_assignment.reset_mock()  # type: ignore[attr-defined]
    manager._register_current_node = AsyncMock()  # type: ignore[method-assign]
    backend.keep_alive = AsyncMock(side_effect=RuntimeError("partition"))  # type: ignore[method-assign]
    await manager._heartbeat_once()
    manager._register_current_node.assert_awaited_once()  # type: ignore[attr-defined]
    manager._sync_assignment.assert_awaited_once()  # type: ignore[attr-defined]

    manager._stopping.clear()
    await backend.put(manager._key("membership/reconciler"), b"other", lease=44)
    await manager._reconcile_once()


@pytest.mark.asyncio
async def test_membership_noop_safety_guards_and_assignment_sync() -> None:
    backend = FakeEtcdClient()
    backend.members = [
        {
            "ID": 1,
            "peerURLs": ["http://a:2380"],
            "clientURLs": ["http://a:2379"],
            "isLearner": False,
        }
    ]
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="guards", voter_target=3),
        CoordinatorNodeRecord(
            node_id="node-a",
            api_url="http://a",
            peer_url="http://a:2380",
            role="voter",
            member_id=1,
        ),
    )
    await manager.initialize()
    await manager._assign_client_as_learner(list(backend.members))
    await manager._remove_excess_voter(list(backend.members))
    assert await manager._replace_failed_voter(list(backend.members), []) is False

    await backend.delete(manager._key("membership/intents/node-a"))
    await manager._sync_assignment()
    same = manager.current_node.model_copy(
        update={"backend_lease_id": manager.current_node.backend_lease_id}
    )
    await backend.put(manager._key("membership/intents/node-a"), same.model_dump_json().encode())
    await manager._sync_assignment()
    changed = same.model_copy(update={"role": "client", "member_id": 0})
    await backend.put(manager._key("membership/intents/node-a"), changed.model_dump_json().encode())
    await manager._sync_assignment()
    assert manager.current_node.role == "client"


@pytest.mark.asyncio
async def test_membership_operation_completed_and_invalid_recovery_paths() -> None:
    backend = FakeEtcdClient()
    manager = MembershipManager(
        backend,
        ClusterConfiguration(cluster_id="recovery-errors", voter_target=3),
        CoordinatorNodeRecord(node_id="node-a", api_url="http://a"),
    )
    await manager.initialize()
    completed = MembershipOperation(
        generation=0,
        action="remove_voter",
        node_id="node-b",
        status="completed",
    )
    await backend.put(manager._key("membership/operation"), completed.model_dump_json().encode())
    assert await manager._recover_membership_operation([]) is True
    assert await manager._recover_membership_operation([]) is False

    missing_assignment = MembershipOperation(
        generation=0,
        action="add_learner",
        node_id="missing",
        peer_url="http://missing:2380",
    )
    await backend.put(
        manager._key("membership/operation"), missing_assignment.model_dump_json().encode()
    )
    with pytest.raises(EtcdError, match="without member and intent"):
        await manager._recover_membership_operation([])
    await backend.delete(manager._key("membership/operation"))

    missing_promotion = MembershipOperation(
        generation=0,
        action="promote_learner",
        node_id="missing",
        member_id=999,
    )
    await backend.put(
        manager._key("membership/operation"), missing_promotion.model_dump_json().encode()
    )
    with pytest.raises(EtcdError, match="missing etcd member"):
        await manager._recover_membership_operation([])


def test_etcd_store_url_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert etcd_endpoints_from_url("etcd://a:2379,https://b:2379") == [
        "http://a:2379",
        "https://b:2379",
    ]
    monkeypatch.setenv("COORDINATION_CLUSTER_ID", "url-cluster")
    store = etcd_store_from_url("etcd://a:2379")
    assert store.cluster_id == "url-cluster"
    assert store.client.endpoints == ["http://a:2379"]


class FakeEtcdClient:
    def __init__(self) -> None:
        self.revision = 0
        self.next_lease = 100
        self.values: dict[bytes, EtcdKeyValue] = {}
        self.members: list[dict[str, Any]] = []
        self.next_member = 1000

    async def close(self) -> None:
        return None

    async def sync_endpoints(self) -> None:
        return None

    async def range(self, key: bytes, *, prefix: bool = False) -> EtcdRange:
        if prefix:
            values = [value for item_key, value in self.values.items() if item_key.startswith(key)]
        else:
            values = [self.values[key]] if key in self.values else []
        return EtcdRange(values=values, revision=self.revision)

    async def put(self, key: bytes, value: bytes, *, lease: int = 0) -> int:
        self.revision += 1
        existing = self.values.get(key)
        self.values[key] = EtcdKeyValue(
            key=key,
            value=value,
            create_revision=existing.create_revision if existing else self.revision,
            mod_revision=self.revision,
            version=existing.version + 1 if existing else 1,
            lease=lease,
        )
        return self.revision

    async def delete(self, key: bytes, *, prefix: bool = False) -> int:
        keys = [item for item in self.values if item.startswith(key)] if prefix else [key]
        deleted = 0
        for item in keys:
            deleted += int(self.values.pop(item, None) is not None)
        if deleted:
            self.revision += 1
        return deleted

    async def transaction(
        self,
        *,
        compare: list[dict[str, Any]],
        success: list[dict[str, Any]],
        failure: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del failure
        succeeded = all(self._compare(item) for item in compare)
        if not succeeded:
            return {"succeeded": False, "header": {"revision": str(self.revision)}}
        if success:
            self.revision += 1
        for operation in success:
            request = operation.get("request_put")
            if request is None:
                continue
            key = _decode(request["key"])
            value = _decode(request.get("value") or "")
            lease = int(request.get("lease") or 0)
            existing = self.values.get(key)
            self.values[key] = EtcdKeyValue(
                key=key,
                value=value,
                create_revision=existing.create_revision if existing else self.revision,
                mod_revision=self.revision,
                version=existing.version + 1 if existing else 1,
                lease=lease,
            )
        return {"succeeded": True, "header": {"revision": str(self.revision)}}

    async def grant_lease(self, ttl_s: float) -> int:
        del ttl_s
        self.next_lease += 1
        return self.next_lease

    async def keep_alive(self, lease_id: int) -> int:
        del lease_id
        return 10

    async def revoke_lease(self, lease_id: int) -> None:
        for key in [key for key, value in self.values.items() if value.lease == lease_id]:
            self.values.pop(key)
        self.revision += 1

    async def status(self) -> dict[str, Any]:
        return {"header": {"revision": str(self.revision)}, "leader": "1"}

    async def member_list(self) -> dict[str, Any]:
        return {"members": list(self.members)}

    async def member_add(self, peer_url: str, *, learner: bool = True) -> dict[str, Any]:
        self.next_member += 1
        member = {
            "ID": self.next_member,
            "peerURLs": [peer_url],
            "clientURLs": [],
            "isLearner": learner,
        }
        self.members.append(member)
        return {"member": member, "members": list(self.members)}

    async def member_promote(self, member_id: int) -> dict[str, Any]:
        member = next(item for item in self.members if item["ID"] == member_id)
        member["isLearner"] = False
        return {"members": list(self.members)}

    async def member_remove(self, member_id: int) -> dict[str, Any]:
        self.members = [item for item in self.members if item["ID"] != member_id]
        return {"members": list(self.members)}

    def _compare(self, compare: dict[str, Any]) -> bool:
        key = _decode(compare["key"])
        current = self.values.get(key)
        target = compare["target"]
        actual = current.version if target == "VERSION" and current else 0
        if target == "MOD":
            actual = current.mod_revision if current else 0
        expected = int(compare.get("version") or compare.get("mod_revision") or 0)
        if compare["result"] == "EQUAL":
            return actual == expected
        raise AssertionError(f"Unsupported comparison: {compare}")


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode()


def _decode(value: str) -> bytes:
    return base64.b64decode(value)
