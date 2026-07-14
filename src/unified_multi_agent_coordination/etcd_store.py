"""etcd-backed authoritative coordination state."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .coordination_ledger import CoordinationSessionState, LedgerEvent, _fold_session
from .coordination_store import (
    CoordinationStoreError,
    LeaseConflictError,
    StaleFenceError,
    StoreInvariantError,
)
from .etcd_client import (
    EtcdClient,
    EtcdKeyValue,
    compare_mod_revision,
    compare_version,
    request_put,
)
from .models import LeaseRecord


JsonObject = dict[str, Any]


class EtcdCoordinationStore:
    """Coordination store whose safety decisions are committed through Raft."""

    TERMINAL_EVENTS = {"run_completed", "run_failed", "plan_infeasible"}
    RESULT_EVENTS = {
        "task_attempt_completed",
        "task_attempt_failed",
        "task_attempt_timeout",
        "task_attempt_unknown",
    }

    def __init__(
        self,
        endpoints: list[str],
        *,
        cluster_id: str = "default",
        client: EtcdClient | None = None,
    ) -> None:
        self.client = client or EtcdClient(endpoints)
        self.cluster_id = cluster_id
        self.root = f"/umac/{cluster_id}".encode()

    async def acquire_lease(
        self,
        session_id: str,
        holder_id: str,
        ttl_s: float,
    ) -> LeaseRecord:
        owner_key = self._session_key(session_id, "owner")
        current = await self.client.range(owner_key)
        compare: list[JsonObject]
        if current.values:
            owner = _loads(current.values[0].value)
            if owner.get("holder_id") != holder_id:
                raise LeaseConflictError(
                    f"Session {session_id} is leased by {owner.get('holder_id', 'unknown')}."
                )
            compare = [
                compare_mod_revision(owner_key, "EQUAL", current.values[0].mod_revision)
            ]
        else:
            compare = [compare_version(owner_key, "EQUAL", 0)]

        backend_lease_id = await self.client.grant_lease(ttl_s)
        now = _utc_now()
        owner_value = _dumps({"holder_id": holder_id, "acquired_at": now.isoformat()})
        result = await self.client.transaction(
            compare=compare,
            success=[request_put(owner_key, owner_value, lease=backend_lease_id)],
        )
        if not result.get("succeeded"):
            await self.client.revoke_lease(backend_lease_id)
            raise LeaseConflictError(f"Session {session_id} lease acquisition raced.")
        fence = int((result.get("header") or {}).get("revision") or 0)
        lease = LeaseRecord(
            session_id=session_id,
            holder_id=holder_id,
            fencing_token=fence,
            expires_at=now + timedelta(seconds=ttl_s),
            heartbeat_at=now,
            backend_lease_id=backend_lease_id,
        )
        await self._append_unfenced(
            LedgerEvent(
                event_type="lease_acquired",
                session_id=session_id,
                payload={
                    "holder_id": holder_id,
                    "fencing_token": fence,
                    "expires_at": lease.expires_at.isoformat(),
                },
            ),
            lease,
        )
        return lease

    async def renew_lease(self, lease: LeaseRecord, ttl_s: float) -> LeaseRecord:
        await self._validate_lease(lease)
        ttl = await self.client.keep_alive(lease.backend_lease_id)
        now = _utc_now()
        lease.expires_at = now + timedelta(seconds=ttl)
        lease.heartbeat_at = now
        return lease

    async def release_lease(self, lease: LeaseRecord) -> None:
        try:
            await self._validate_lease(lease)
        except (LeaseConflictError, StaleFenceError):
            return
        await self._append_unfenced(
            LedgerEvent(
                event_type="lease_released",
                session_id=lease.session_id,
                payload={
                    "holder_id": lease.holder_id,
                    "fencing_token": lease.fencing_token,
                },
            ),
            lease,
        )
        await self.client.revoke_lease(lease.backend_lease_id)

    async def append_event(
        self,
        event: LedgerEvent,
        lease: LeaseRecord | None = None,
    ) -> LedgerEvent:
        compare: list[JsonObject] = []
        if lease is not None:
            if event.session_id != lease.session_id:
                raise StoreInvariantError("Lease session does not match event session.")
            await self._validate_lease(lease)
            compare.append(
                compare_mod_revision(
                    self._session_key(event.session_id, "owner"),
                    "EQUAL",
                    lease.fencing_token,
                )
            )

        writes = [request_put(self._event_key(event.session_id), _dumps(event.model_dump(mode="json")))]
        if event.event_type == "session_started":
            writes.append(
                request_put(
                    self._session_key(event.session_id, "metadata"),
                    _dumps(event.payload),
                )
            )
        elif event.event_type == "plan_authorized":
            plan_result = dict(event.payload.get("plan_result") or {})
            generation = int(plan_result.get("plan_generation") or 1)
            plan_key = self._session_key(event.session_id, f"plans/{generation}")
            compare.append(compare_version(plan_key, "EQUAL", 0))
            writes.append(request_put(plan_key, _dumps(plan_result)))
        elif event.event_type == "task_attempt_started":
            attempt_key = self._attempt_key(event.session_id, event.attempt_id)
            task_key = self._session_key(event.session_id, f"tasks/{event.task_id}")
            compare.append(compare_version(attempt_key, "EQUAL", 0))
            writes.append(request_put(attempt_key, _dumps(event.payload)))
            writes.append(
                request_put(
                    task_key,
                    _dumps(
                        {
                            "state": "started",
                            "attempt_id": event.attempt_id,
                            "task": event.payload.get("task") or {},
                        }
                    ),
                )
            )
            idempotency_key = str(event.payload.get("idempotency_key") or "")
            if idempotency_key:
                writes.append(
                    request_put(
                        self._idempotency_key(idempotency_key),
                        _dumps({"attempt_key": attempt_key.decode(), "result": None}),
                    )
                )
        elif event.event_type in self.RESULT_EVENTS:
            attempt_key = self._attempt_key(event.session_id, event.attempt_id)
            attempt = await self.client.range(attempt_key)
            if not attempt.values:
                raise StoreInvariantError(
                    f"Task result {event.attempt_id} has no started attempt."
                )
            attempt_payload = _loads(attempt.values[0].value)
            compare.append(compare_mod_revision(attempt_key, "EQUAL", attempt.values[0].mod_revision))
            result_payload = dict(event.payload.get("task_result") or {})
            writes.append(
                request_put(
                    attempt_key,
                    _dumps({**attempt_payload, "result": result_payload, "state": event.event_type}),
                )
            )
            writes.append(
                request_put(
                    self._session_key(event.session_id, f"tasks/{event.task_id}"),
                    _dumps(
                        {
                            "state": event.event_type,
                            "attempt_id": event.attempt_id,
                            "result": result_payload,
                        }
                    ),
                )
            )
            idempotency_key = str(attempt_payload.get("idempotency_key") or "")
            if idempotency_key:
                writes.append(
                    request_put(
                        self._idempotency_key(idempotency_key),
                        _dumps({"attempt_key": attempt_key.decode(), "result": result_payload}),
                    )
                )
        elif event.event_type in self.TERMINAL_EVENTS:
            terminal_key = self._session_key(event.session_id, "terminal")
            compare.append(compare_version(terminal_key, "EQUAL", 0))
            writes.append(request_put(terminal_key, _dumps(event.payload.get("run_result") or {})))

        result = await self.client.transaction(compare=compare, success=writes)
        if not result.get("succeeded"):
            if event.event_type in self.TERMINAL_EVENTS:
                raise StoreInvariantError(
                    f"Session {event.session_id} already has a terminal result."
                )
            if event.event_type in self.RESULT_EVENTS:
                raise StoreInvariantError(
                    f"Task result {event.attempt_id} raced or has no started attempt."
                )
            if lease is not None:
                raise StaleFenceError(
                    f"Session {event.session_id} rejected stale fencing token "
                    f"{lease.fencing_token}."
                )
            raise CoordinationStoreError("etcd transaction comparison failed.")
        return event

    async def events(self, session_id: str) -> list[LedgerEvent]:
        values = await self.client.range(self._events_prefix(session_id), prefix=True)
        ordered = sorted(values.values, key=lambda item: item.create_revision)
        return [LedgerEvent.model_validate(_loads(item.value)) for item in ordered]

    async def session_state(self, session_id: str) -> CoordinationSessionState:
        return _fold_session(session_id, await self.events(session_id))

    async def task_result_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> JsonObject | None:
        result = await self.client.range(self._idempotency_key(idempotency_key))
        if not result.values:
            return None
        payload = _loads(result.values[0].value)
        value = payload.get("result")
        return dict(value) if isinstance(value, dict) else None

    async def close(self) -> None:
        await self.client.close()

    async def ready(self) -> bool:
        await self.client.status()
        await self.client.sync_endpoints()
        return True

    async def _validate_lease(self, lease: LeaseRecord) -> EtcdKeyValue:
        current = await self.client.range(self._session_key(lease.session_id, "owner"))
        if not current.values:
            raise LeaseConflictError(f"Session {lease.session_id} has no active lease.")
        item = current.values[0]
        owner = _loads(item.value)
        if owner.get("holder_id") != lease.holder_id:
            raise LeaseConflictError(
                f"Session {lease.session_id} is leased by {owner.get('holder_id', 'unknown')}."
            )
        if item.mod_revision != lease.fencing_token:
            raise StaleFenceError(
                f"Session {lease.session_id} rejected stale fencing token "
                f"{lease.fencing_token}; current token is {item.mod_revision}."
            )
        return item

    async def _append_unfenced(self, event: LedgerEvent, lease: LeaseRecord) -> None:
        result = await self.client.transaction(
            compare=[
                compare_mod_revision(
                    self._session_key(event.session_id, "owner"),
                    "EQUAL",
                    lease.fencing_token,
                )
            ],
            success=[request_put(self._event_key(event.session_id), _dumps(event.model_dump(mode="json")))],
        )
        if not result.get("succeeded"):
            raise StaleFenceError(f"Session {event.session_id} lease is stale.")

    def _session_prefix(self, session_id: str) -> bytes:
        return self.root + f"/sessions/{session_id}".encode()

    def _session_key(self, session_id: str, suffix: str) -> bytes:
        return self._session_prefix(session_id) + f"/{suffix}".encode()

    def _events_prefix(self, session_id: str) -> bytes:
        return self._session_key(session_id, "events/")

    def _event_key(self, session_id: str) -> bytes:
        return self._events_prefix(session_id) + uuid.uuid4().hex.encode()

    def _attempt_key(self, session_id: str, attempt_id: str) -> bytes:
        return self._session_key(session_id, f"attempts/{attempt_id}")

    def _idempotency_key(self, idempotency_key: str) -> bytes:
        digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
        return self.root + f"/idempotency/{digest}".encode()


def etcd_endpoints_from_url(url: str) -> list[str]:
    raw = url.removeprefix("etcd://")
    return [
        endpoint if endpoint.startswith(("http://", "https://")) else f"http://{endpoint}"
        for endpoint in raw.split(",")
        if endpoint.strip()
    ]


def etcd_store_from_url(url: str) -> EtcdCoordinationStore:
    return EtcdCoordinationStore(
        etcd_endpoints_from_url(url),
        cluster_id=os.getenv("COORDINATION_CLUSTER_ID", "default"),
    )


def _dumps(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _loads(value: bytes) -> JsonObject:
    loaded = json.loads(value.decode())
    return loaded if isinstance(loaded, dict) else {"value": loaded}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
