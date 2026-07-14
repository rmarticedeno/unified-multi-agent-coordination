"""Authenticated coordinator membership and automatic etcd reconciliation."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .etcd_client import EtcdClient, EtcdError


JsonObject = dict[str, Any]
PROTOCOL_VERSION = 1
VALID_VOTER_TARGETS = {1, 3, 5, 7}


class ClusterConfiguration(BaseModel):
    cluster_id: str
    voter_target: int = 3
    protocol_version: int = PROTOCOL_VERSION

    @field_validator("voter_target")
    @classmethod
    def validate_voter_target(cls, value: int) -> int:
        if value not in VALID_VOTER_TARGETS:
            raise ValueError("Voter target must be one of 1, 3, 5, or 7.")
        return value


class CoordinatorNodeRecord(BaseModel):
    node_id: str
    api_url: str
    peer_url: str = ""
    client_url: str = ""
    role: Literal["voter", "learner", "client", "draining"] = "client"
    member_id: int = 0
    voter_target: int = 3
    initial_cluster: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    backend_lease_id: int = 0


class SignedEnvelope(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    message_type: str
    cluster_id: str
    node_id: str
    timestamp: float = Field(default_factory=time.time)
    nonce: str = Field(default_factory=lambda: uuid.uuid4().hex)
    payload: JsonObject = Field(default_factory=dict)
    signature: str = ""


class HmacAuthenticator:
    """Canonical HMAC signing with timestamp and nonce replay protection."""

    def __init__(
        self,
        secret: str,
        *,
        max_clock_skew_s: float = 30.0,
        nonce_ttl_s: float = 120.0,
        allow_insecure: bool = False,
    ) -> None:
        if not secret and not allow_insecure:
            raise ValueError("A cluster HMAC secret is required.")
        self.secret = secret.encode()
        self.max_clock_skew_s = max_clock_skew_s
        self.nonce_ttl_s = nonce_ttl_s
        self.allow_insecure = allow_insecure
        self._seen_nonces: dict[str, float] = {}

    def sign(self, envelope: SignedEnvelope) -> SignedEnvelope:
        if self.allow_insecure and not self.secret:
            return envelope.model_copy(update={"signature": "insecure"})
        signature = hmac.new(
            self.secret,
            _canonical_envelope(envelope),
            hashlib.sha256,
        ).hexdigest()
        return envelope.model_copy(update={"signature": signature})

    def verify(
        self,
        envelope: SignedEnvelope,
        *,
        expected_cluster_id: str,
        consume_nonce: bool = True,
    ) -> None:
        if envelope.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unsupported cluster protocol version.")
        if envelope.cluster_id != expected_cluster_id:
            raise ValueError("Discovery or join message belongs to another cluster.")
        now = time.time()
        if abs(now - envelope.timestamp) > self.max_clock_skew_s:
            raise ValueError("Signed cluster message timestamp is outside the allowed window.")
        self._prune_nonces(now)
        if consume_nonce and envelope.nonce in self._seen_nonces:
            raise ValueError("Signed cluster message nonce was replayed.")
        if not (self.allow_insecure and envelope.signature == "insecure"):
            expected = hmac.new(
                self.secret,
                _canonical_envelope(envelope),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected, envelope.signature):
                raise ValueError("Invalid cluster message signature.")
        if consume_nonce:
            self._seen_nonces[envelope.nonce] = now

    def _prune_nonces(self, now: float) -> None:
        self._seen_nonces = {
            nonce: seen
            for nonce, seen in self._seen_nonces.items()
            if now - seen <= self.nonce_ttl_s
        }


class MembershipManager:
    """etcd membership facade and lease-elected reconciliation loop."""

    def __init__(
        self,
        client: EtcdClient,
        configuration: ClusterConfiguration,
        current_node: CoordinatorNodeRecord,
        *,
        reconcile_interval_s: float = 2.0,
        registration_ttl_s: float = 15.0,
        failed_voter_grace_s: float = 60.0,
    ) -> None:
        self.client = client
        self.configuration = configuration
        self.current_node = current_node
        self.reconcile_interval_s = reconcile_interval_s
        self.registration_ttl_s = registration_ttl_s
        self.failed_voter_grace_s = failed_voter_grace_s
        self.root = f"/umac/{configuration.cluster_id}".encode()
        self._tasks: list[asyncio.Task[Any]] = []
        self._stopping = asyncio.Event()

    async def initialize(self) -> None:
        authoritative = await self._load_configuration()
        if authoritative is not None:
            if authoritative.voter_target != self.configuration.voter_target:
                raise ValueError(
                    "Configured voter target conflicts with authoritative cluster target."
                )
            self.configuration = authoritative
        else:
            await self.client.put(
                self._key("configuration"),
                self.configuration.model_dump_json().encode(),
            )
        if self.current_node.role in {"voter", "learner"} and not self.current_node.member_id:
            await self._resolve_current_member_id()
        if await self._intent(self.current_node.node_id) is None:
            await self.client.put(
                self._key(f"membership/intents/{self.current_node.node_id}"),
                self.current_node.model_dump_json().encode(),
            )
        await self._register_current_node()

    async def start(self) -> None:
        await self.initialize()
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._reconcile_loop()),
        ]

    async def stop(self) -> None:
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.current_node.backend_lease_id:
            try:
                await self.client.revoke_lease(self.current_node.backend_lease_id)
            except Exception:
                pass

    async def join(self, envelope: SignedEnvelope) -> JsonObject:
        payload = envelope.payload
        requested_target = int(payload.get("voter_target") or 3)
        if requested_target != self.configuration.voter_target:
            raise ValueError("Joining node has a conflicting voter target.")
        node_id = envelope.node_id
        peer_url = str(payload.get("peer_url") or "")
        api_url = str(payload.get("api_url") or "")
        client_url = str(payload.get("client_url") or "")
        members = await self.member_records()
        existing = await self._intent(node_id)
        if existing is not None:
            return self._assignment(existing, members)
        voter_count = sum(not member.get("isLearner", False) for member in members)
        already_added = next(
            (
                member
                for member in members
                if peer_url
                and peer_url
                in (member.get("peerURLs") or member.get("peer_urls") or [])
            ),
            None,
        )
        role: Literal["voter", "learner", "client", "draining"]
        if already_added is not None:
            member_id = int(already_added.get("ID") or already_added.get("id") or 0)
            role = "learner" if already_added.get("isLearner", False) else "voter"
        elif voter_count < self.configuration.voter_target:
            if not peer_url:
                raise ValueError("A peer URL is required for an etcd learner.")
            added = await self.client.member_add(peer_url, learner=True)
            member = added.get("member") or {}
            member_id = int(member.get("ID") or member.get("id") or 0)
            role = "learner"
            members = added.get("members") or members
        else:
            member_id = 0
            role = "client"
        record = CoordinatorNodeRecord(
            node_id=node_id,
            api_url=api_url,
            peer_url=peer_url,
            client_url=client_url,
            role=role,
            member_id=member_id,
            voter_target=requested_target,
        )
        initial_cluster = _initial_cluster(members, node_id=node_id, member_id=member_id)
        record = record.model_copy(update={"initial_cluster": initial_cluster})
        await self.client.put(
            self._key(f"membership/intents/{node_id}"),
            record.model_dump_json().encode(),
        )
        await self._record_history("node_joined", record)
        return self._assignment(record, members)

    async def join_status(self, node_id: str) -> JsonObject:
        record = await self._intent(node_id)
        if record is None:
            raise KeyError(node_id)
        return self._assignment(record, await self.member_records())

    async def leave(self, node_id: str) -> None:
        records = await self.node_records()
        record = next((item for item in records if item.node_id == node_id), None)
        record = record or await self._intent(node_id)
        if record and record.member_id:
            await self.client.member_remove(record.member_id)
        if record:
            await self._record_history("node_left", record)
        await self.client.delete(self._key(f"nodes/{node_id}"))
        await self.client.delete(self._key(f"membership/intents/{node_id}"))

    async def update_voter_target(self, voter_target: int) -> ClusterConfiguration:
        payload = self.configuration.model_dump()
        payload["voter_target"] = voter_target
        updated = ClusterConfiguration.model_validate(payload)
        await self.client.put(self._key("configuration"), updated.model_dump_json().encode())
        self.configuration = updated
        self.current_node = self.current_node.model_copy(
            update={"voter_target": voter_target}
        )
        for prefix in ("nodes/", "membership/intents/"):
            records = await self.client.range(self._key(prefix), prefix=True)
            for item in records.values:
                record = CoordinatorNodeRecord.model_validate_json(item.value)
                changed = record.model_copy(update={"voter_target": voter_target})
                await self.client.put(
                    item.key,
                    changed.model_dump_json().encode(),
                    lease=item.lease if prefix == "nodes/" else 0,
                )
        await self._record_history("voter_target_changed", self.current_node)
        return updated

    async def node_records(self) -> list[CoordinatorNodeRecord]:
        result = await self.client.range(self._key("nodes/"), prefix=True)
        return [CoordinatorNodeRecord.model_validate_json(item.value) for item in result.values]

    async def member_records(self) -> list[JsonObject]:
        result = await self.client.member_list()
        return list(result.get("members") or [])

    async def status(self) -> JsonObject:
        members = await self.member_records()
        nodes = await self.node_records()
        backend_status = await self.client.status()
        voter_ids = {
            int(member.get("ID") or member.get("id") or 0)
            for member in members
            if not member.get("isLearner", False)
        }
        active_member_ids = {node.member_id for node in nodes if node.member_id}
        voters = len(voter_ids & active_member_ids)
        learners = sum(bool(member.get("isLearner", False)) for member in members)
        quorum = len(voter_ids) // 2 + 1 if voter_ids else 0
        below_target = voters < self.configuration.voter_target
        development_target = self.configuration.voter_target < 3
        return {
            "cluster_id": self.configuration.cluster_id,
            "node_id": self.current_node.node_id,
            "role": self.current_node.role,
            "configured_voter_target": self.configuration.voter_target,
            "active_voters": voters,
            "learners": learners,
            "coordinator_nodes": len(nodes),
            "coordinator_only_nodes": sum(node.role == "client" for node in nodes),
            "quorum": quorum,
            "quorum_available": voters >= quorum,
            "leader": int(backend_status.get("leader") or 0),
            "revision": int((backend_status.get("header") or {}).get("revision") or 0),
            "discovery_method": os.getenv(
                "COORDINATION_DISCOVERY_METHOD", "configured"
            ),
            "pending_membership_changes": sum(
                node.role in {"learner", "draining"} for node in nodes
            ),
            "degraded": below_target or development_target,
            "degraded_reason": (
                "active voter count is below configured target"
                if below_target
                else (
                    "voter target below three is development/degraded mode"
                    if development_target
                    else ""
                )
            ),
        }

    async def _register_current_node(self) -> None:
        lease_id = await self.client.grant_lease(self.registration_ttl_s)
        self.current_node = self.current_node.model_copy(
            update={"backend_lease_id": lease_id}
        )
        await self.client.put(
            self._key(f"nodes/{self.current_node.node_id}"),
            self.current_node.model_dump_json().encode(),
            lease=lease_id,
        )

    async def _heartbeat_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(max(self.registration_ttl_s / 3, 1.0))
            await self.client.keep_alive(self.current_node.backend_lease_id)
            await self._sync_assignment()

    async def _reconcile_loop(self) -> None:
        while not self._stopping.is_set():
            await asyncio.sleep(self.reconcile_interval_s)
            try:
                await self._reconcile_once()
            except Exception:
                # Readiness/status surfaces the degraded cluster; the elected
                # loop retries instead of terminating the coordinator.
                continue

    async def _reconcile_once(self) -> None:
        lock_lease = await self.client.grant_lease(max(self.reconcile_interval_s * 3, 3))
        lock_key = self._key("membership/reconciler")
        from .etcd_client import compare_version, request_put

        acquired = await self.client.transaction(
            compare=[compare_version(lock_key, "EQUAL", 0)],
            success=[request_put(lock_key, self.current_node.node_id.encode(), lease=lock_lease)],
        )
        if not acquired.get("succeeded"):
            await self.client.revoke_lease(lock_lease)
            return
        try:
            authoritative = await self._load_configuration()
            if authoritative is not None:
                self.configuration = authoritative
            members = await self.member_records()
            voters = [member for member in members if not member.get("isLearner", False)]
            learners = [member for member in members if member.get("isLearner", False)]
            if await self._replace_failed_voter(voters, learners):
                return
            if len(voters) < self.configuration.voter_target and learners:
                member_id = int(learners[0].get("ID") or learners[0].get("id") or 0)
                if member_id:
                    try:
                        await self.client.member_promote(member_id)
                    except EtcdError:
                        return
                    await self._mark_promoted(member_id)
                    return
            if len(voters) < self.configuration.voter_target:
                await self._assign_client_as_learner(members)
                return
            if len(voters) > self.configuration.voter_target:
                await self._remove_excess_voter(voters)
        finally:
            await self.client.revoke_lease(lock_lease)

    async def _mark_promoted(self, member_id: int) -> None:
        intents = await self.client.range(self._key("membership/intents/"), prefix=True)
        for item in intents.values:
            record = CoordinatorNodeRecord.model_validate_json(item.value)
            if record.member_id != member_id:
                continue
            promoted = record.model_copy(update={"role": "voter"})
            await self.client.put(item.key, promoted.model_dump_json().encode())
            await self.client.put(
                self._key(f"membership/history/{uuid.uuid4().hex}"),
                json.dumps(
                    {
                        "event": "member_promoted",
                        "node_id": record.node_id,
                        "member_id": member_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    sort_keys=True,
                ).encode(),
            )

    async def _assign_client_as_learner(self, members: list[JsonObject]) -> None:
        active_nodes = await self.node_records()
        candidate = next(
            (node for node in active_nodes if node.role == "client" and node.peer_url),
            None,
        )
        if candidate is None:
            return
        added = await self.client.member_add(candidate.peer_url, learner=True)
        updated_members = list(added.get("members") or members)
        member = added.get("member") or {}
        member_id = int(member.get("ID") or member.get("id") or 0)
        if not member_id:
            raise EtcdError("etcd returned no learner member ID.")
        assigned = candidate.model_copy(
            update={
                "role": "learner",
                "member_id": member_id,
                "initial_cluster": _initial_cluster(
                    updated_members,
                    node_id=candidate.node_id,
                    member_id=member_id,
                ),
            }
        )
        await self.client.put(
            self._key(f"membership/intents/{candidate.node_id}"),
            assigned.model_dump_json().encode(),
        )
        await self._record_history("learner_assigned", assigned)

    async def _remove_excess_voter(self, voters: list[JsonObject]) -> None:
        backend_status = await self.client.status()
        leader_id = int(backend_status.get("leader") or 0)
        nodes = await self.node_records()
        by_member = {node.member_id: node for node in nodes if node.member_id}
        removable = [
            int(member.get("ID") or member.get("id") or 0)
            for member in voters
            if int(member.get("ID") or member.get("id") or 0) in by_member
            and int(member.get("ID") or member.get("id") or 0) != leader_id
        ]
        if not removable:
            return
        member_id = removable[-1]
        node = by_member[member_id]
        draining = node.model_copy(update={"role": "draining"})
        await self.client.put(
            self._key(f"membership/intents/{node.node_id}"),
            draining.model_dump_json().encode(),
        )
        await self.client.member_remove(member_id)
        await self._record_history("voter_removed", draining)
        client = draining.model_copy(
            update={"role": "client", "member_id": 0, "initial_cluster": ""}
        )
        await self.client.put(
            self._key(f"membership/intents/{node.node_id}"),
            client.model_dump_json().encode(),
        )

    async def _replace_failed_voter(
        self,
        voters: list[JsonObject],
        learners: list[JsonObject],
    ) -> bool:
        active_nodes = await self.node_records()
        active_member_ids = {node.member_id for node in active_nodes if node.member_id}
        voter_ids = {
            int(member.get("ID") or member.get("id") or 0) for member in voters
        }
        healthy_voters = voter_ids & active_member_ids
        quorum = len(voters) // 2 + 1
        if len(healthy_voters) < quorum:
            return False
        replacement_available = bool(learners) or any(
            node.role == "client" and node.peer_url for node in active_nodes
        )
        if not replacement_available:
            return False

        intents = await self.client.range(self._key("membership/intents/"), prefix=True)
        by_member: dict[int, tuple[bytes, CoordinatorNodeRecord]] = {}
        for item in intents.values:
            record = CoordinatorNodeRecord.model_validate_json(item.value)
            if record.member_id:
                by_member[record.member_id] = (item.key, record)

        now = time.time()
        for member_id in voter_ids:
            marker_key = self._key(f"membership/failures/{member_id}")
            if member_id in active_member_ids:
                await self.client.delete(marker_key)
                continue
            marker = await self.client.range(marker_key)
            if not marker.values:
                await self.client.put(marker_key, str(now).encode())
                continue
            missing_since = float(marker.values[0].value.decode())
            if now - missing_since < self.failed_voter_grace_s:
                continue
            assignment = by_member.get(member_id)
            if assignment is None:
                continue
            intent_key, failed = assignment
            await self.client.member_remove(member_id)
            await self._record_history("failed_voter_removed", failed)
            reassigned = failed.model_copy(
                update={"role": "client", "member_id": 0, "initial_cluster": ""}
            )
            await self.client.put(intent_key, reassigned.model_dump_json().encode())
            await self.client.delete(marker_key)
            return True
        return False

    async def _sync_assignment(self) -> None:
        assigned = await self._intent(self.current_node.node_id)
        if assigned is None:
            return
        synchronized = assigned.model_copy(
            update={"backend_lease_id": self.current_node.backend_lease_id}
        )
        if synchronized == self.current_node:
            return
        self.current_node = synchronized
        await self.client.put(
            self._key(f"nodes/{self.current_node.node_id}"),
            self.current_node.model_dump_json().encode(),
            lease=self.current_node.backend_lease_id,
        )

    async def _resolve_current_member_id(self) -> None:
        members = await self.member_records()
        for member in members:
            peer_urls = member.get("peerURLs") or member.get("peer_urls") or []
            if self.current_node.peer_url not in peer_urls:
                continue
            self.current_node = self.current_node.model_copy(
                update={
                    "member_id": int(member.get("ID") or member.get("id") or 0),
                    "initial_cluster": _initial_cluster(
                        members,
                        node_id=self.current_node.node_id,
                        member_id=int(member.get("ID") or member.get("id") or 0),
                    ),
                }
            )
            return

    async def _load_configuration(self) -> ClusterConfiguration | None:
        existing = await self.client.range(self._key("configuration"))
        if not existing.values:
            return None
        return ClusterConfiguration.model_validate_json(existing.values[0].value)

    async def _intent(self, node_id: str) -> CoordinatorNodeRecord | None:
        result = await self.client.range(self._key(f"membership/intents/{node_id}"))
        if not result.values:
            return None
        return CoordinatorNodeRecord.model_validate_json(result.values[0].value)

    def _assignment(
        self, record: CoordinatorNodeRecord, members: list[JsonObject]
    ) -> JsonObject:
        return {
            "role": record.role,
            "member_id": record.member_id,
            "initial_cluster": record.initial_cluster,
            "etcd_endpoints": [
                url
                for item in members
                for url in (item.get("clientURLs") or item.get("client_urls") or [])
            ],
            "voter_target": self.configuration.voter_target,
        }

    async def _record_history(
        self, event: str, record: CoordinatorNodeRecord
    ) -> None:
        await self.client.put(
            self._key(f"membership/history/{uuid.uuid4().hex}"),
            json.dumps(
                {
                    "event": event,
                    "node_id": record.node_id,
                    "member_id": record.member_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ).encode(),
        )

    def _key(self, suffix: str) -> bytes:
        return self.root + b"/" + suffix.encode()


def _canonical_envelope(envelope: SignedEnvelope) -> bytes:
    payload = envelope.model_dump(mode="json", exclude={"signature"})
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _initial_cluster(
    members: list[JsonObject],
    *,
    node_id: str,
    member_id: int,
) -> str:
    entries: list[str] = []
    for item in members:
        item_id = int(item.get("ID") or item.get("id") or 0)
        name = str(item.get("name") or (node_id if item_id == member_id else f"member-{item_id}"))
        peer_urls = item.get("peerURLs") or item.get("peer_urls") or []
        if peer_urls:
            entries.append(f"{name}={peer_urls[0]}")
    return ",".join(entries)
