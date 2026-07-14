from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
import pytest

from unified_multi_agent_coordination.cluster import (
    ClusterConfiguration,
    CoordinatorNodeRecord,
    HmacAuthenticator,
    MembershipManager,
    SignedEnvelope,
)
from unified_multi_agent_coordination.cluster_discovery import ClusterDiscovery
from unified_multi_agent_coordination.coordination_ledger import LedgerEvent
from unified_multi_agent_coordination.coordination_store import (
    LeaseConflictError,
    StaleFenceError,
    StoreInvariantError,
)
from unified_multi_agent_coordination.etcd_client import (
    EtcdClient,
    EtcdKeyValue,
    EtcdRange,
)
from unified_multi_agent_coordination.etcd_store import EtcdCoordinationStore


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

    tampered = signed.model_copy(
        update={"nonce": "different", "payload": {"peer_url": "forged"}}
    )
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


@pytest.mark.asyncio
async def test_membership_configuration_rejects_even_update() -> None:
    manager = MembershipManager(
        FakeEtcdClient(),
        ClusterConfiguration(cluster_id="alpha"),
        CoordinatorNodeRecord(node_id="node-a", api_url="http://node-a:8000"),
    )
    with pytest.raises(ValueError, match="1, 3, 5, or 7"):
        await manager.update_voter_target(4)


class FakeEtcdClient:
    def __init__(self) -> None:
        self.revision = 0
        self.next_lease = 100
        self.values: dict[bytes, EtcdKeyValue] = {}

    async def close(self) -> None:
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
        return {"members": []}

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
