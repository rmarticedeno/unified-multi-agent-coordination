"""Durable coordination store abstractions for replicated coordinators."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from .coordination_ledger import (
    CoordinationLedger,
    CoordinationSessionState,
    InMemoryCoordinationLedger,
    JsonlCoordinationLedger,
    LedgerEvent,
)
from .models import LeaseRecord, TaskState


JsonObject = dict[str, Any]


class CoordinationStoreError(RuntimeError):
    """Base class for durable store errors."""


class LeaseConflictError(CoordinationStoreError):
    """Raised when another coordinator currently owns a session lease."""


class StaleFenceError(CoordinationStoreError):
    """Raised when a stale coordinator writes with an old fencing token."""


class StoreInvariantError(CoordinationStoreError):
    """Raised when a write would violate coordination invariants."""


class CoordinationStore(Protocol):
    """Async durable coordination store used by replicated coordinators."""

    async def acquire_lease(
        self,
        session_id: str,
        holder_id: str,
        ttl_s: float,
    ) -> LeaseRecord:
        ...

    async def renew_lease(self, lease: LeaseRecord, ttl_s: float) -> LeaseRecord:
        ...

    async def release_lease(self, lease: LeaseRecord) -> None:
        ...

    async def append_event(
        self,
        event: LedgerEvent,
        lease: LeaseRecord | None = None,
    ) -> LedgerEvent:
        ...

    async def events(self, session_id: str) -> list[LedgerEvent]:
        ...

    async def session_state(self, session_id: str) -> CoordinationSessionState:
        ...

    async def task_result_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> JsonObject | None:
        ...

    async def close(self) -> None:
        ...


class AuditProjectingCoordinationStore:
    """Authoritative store with a serialized, best-effort audit projection."""

    def __init__(self, authoritative: CoordinationStore, audit: CoordinationStore) -> None:
        self.authoritative = authoritative
        self.audit = audit
        self.last_projection_error = ""
        self._queue: asyncio.Queue[LedgerEvent] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def acquire_lease(
        self, session_id: str, holder_id: str, ttl_s: float
    ) -> LeaseRecord:
        return await self.authoritative.acquire_lease(session_id, holder_id, ttl_s)

    async def renew_lease(self, lease: LeaseRecord, ttl_s: float) -> LeaseRecord:
        return await self.authoritative.renew_lease(lease, ttl_s)

    async def release_lease(self, lease: LeaseRecord) -> None:
        await self.authoritative.release_lease(lease)

    async def append_event(
        self, event: LedgerEvent, lease: LeaseRecord | None = None
    ) -> LedgerEvent:
        written = await self.authoritative.append_event(event, lease)
        self._ensure_worker()
        self._queue.put_nowait(written)
        return written

    async def events(self, session_id: str) -> list[LedgerEvent]:
        return await self.authoritative.events(session_id)

    async def session_state(self, session_id: str) -> CoordinationSessionState:
        return await self.authoritative.session_state(session_id)

    async def task_result_by_idempotency_key(
        self, idempotency_key: str
    ) -> JsonObject | None:
        return await self.authoritative.task_result_by_idempotency_key(idempotency_key)

    async def ready(self) -> bool:
        ready = getattr(self.authoritative, "ready", None)
        if ready is not None:
            return bool(await ready())
        return True

    async def close(self) -> None:
        await self.authoritative.close()
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=1.0)
            except TimeoutError:
                pass
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
        await self.audit.close()

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._project_events())

    async def _project_events(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self.audit.append_event(event)
                self.last_projection_error = ""
            except Exception as exc:
                # Projection is observability only and cannot affect authority
                # or readiness. The retained etcd event remains replayable.
                self.last_projection_error = str(exc)
            finally:
                self._queue.task_done()


class JsonlCoordinationStore:
    """Compatibility store backed by the existing JSONL/in-memory ledger."""

    def __init__(self, ledger: CoordinationLedger | None = None) -> None:
        self.ledger = ledger or InMemoryCoordinationLedger()
        self._leases: dict[str, LeaseRecord] = {}
        self._attempt_key_by_attempt: dict[str, str] = {}
        self._result_by_idempotency_key: dict[str, JsonObject] = {}
        self._lock = asyncio.Lock()

    async def acquire_lease(
        self,
        session_id: str,
        holder_id: str,
        ttl_s: float,
    ) -> LeaseRecord:
        async with self._lock:
            now = _utc_now()
            current = self._leases.get(session_id)
            if (
                current is not None
                and current.expires_at > now
                and current.holder_id != holder_id
            ):
                raise LeaseConflictError(
                    f"Session {session_id} is leased by {current.holder_id}."
                )
            token = (current.fencing_token + 1) if current is not None else 1
            lease = LeaseRecord(
                session_id=session_id,
                holder_id=holder_id,
                fencing_token=token,
                expires_at=now + timedelta(seconds=ttl_s),
                heartbeat_at=now,
            )
            self._leases[session_id] = lease
            self.ledger.append(
                LedgerEvent(
                    event_type="lease_acquired",
                    session_id=session_id,
                    payload={
                        "holder_id": holder_id,
                        "fencing_token": token,
                        "expires_at": lease.expires_at.isoformat(),
                    },
                )
            )
            return lease

    async def renew_lease(self, lease: LeaseRecord, ttl_s: float) -> LeaseRecord:
        async with self._lock:
            self._validate_lease_locked(lease)
            now = _utc_now()
            renewed = lease.model_copy(
                update={
                    "expires_at": now + timedelta(seconds=ttl_s),
                    "heartbeat_at": now,
                }
            )
            self._leases[lease.session_id] = renewed
            return renewed

    async def release_lease(self, lease: LeaseRecord) -> None:
        async with self._lock:
            current = self._leases.get(lease.session_id)
            if (
                current is not None
                and current.holder_id == lease.holder_id
                and current.fencing_token == lease.fencing_token
            ):
                now = _utc_now()
                expired = current.model_copy(
                    update={"expires_at": now, "heartbeat_at": now}
                )
                self._leases[lease.session_id] = expired
                self.ledger.append(
                    LedgerEvent(
                        event_type="lease_released",
                        session_id=lease.session_id,
                        payload={
                            "holder_id": lease.holder_id,
                            "fencing_token": lease.fencing_token,
                        },
                    )
                )

    async def append_event(
        self,
        event: LedgerEvent,
        lease: LeaseRecord | None = None,
    ) -> LedgerEvent:
        async with self._lock:
            if lease is not None:
                self._validate_lease_locked(lease)
            self._validate_event_invariants_locked(event)
            written = self.ledger.append(event)
            self._index_event_locked(written)
            return written

    async def events(self, session_id: str) -> list[LedgerEvent]:
        return self.ledger.events(session_id)

    async def session_state(self, session_id: str) -> CoordinationSessionState:
        return self.ledger.session_state(session_id)

    async def task_result_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> JsonObject | None:
        return self._result_by_idempotency_key.get(idempotency_key)

    async def close(self) -> None:
        return None

    def _validate_lease_locked(self, lease: LeaseRecord) -> None:
        current = self._leases.get(lease.session_id)
        if current is None:
            raise LeaseConflictError(f"Session {lease.session_id} has no active lease.")
        if current.expires_at <= _utc_now():
            raise LeaseConflictError(f"Session {lease.session_id} lease expired.")
        if current.holder_id != lease.holder_id:
            raise LeaseConflictError(
                f"Session {lease.session_id} is leased by {current.holder_id}."
            )
        if current.fencing_token != lease.fencing_token:
            raise StaleFenceError(
                f"Session {lease.session_id} rejected stale fencing token "
                f"{lease.fencing_token}; current token is {current.fencing_token}."
            )

    def _validate_event_invariants_locked(self, event: LedgerEvent) -> None:
        state = self.ledger.session_state(event.session_id)
        if event.event_type in {"run_completed", "run_failed", "plan_infeasible"}:
            if state.terminal_result is not None:
                raise StoreInvariantError(
                    f"Session {event.session_id} already has a terminal result."
                )
        if event.event_type in {
            "task_attempt_completed",
            "task_attempt_failed",
            "task_attempt_timeout",
            "task_attempt_unknown",
        }:
            attempts = state.task_attempts_by_task.get(event.task_id, [])
            if event.attempt_id not in attempts:
                raise StoreInvariantError(
                    f"Task result {event.attempt_id} has no started attempt."
                )

    def _index_event_locked(self, event: LedgerEvent) -> None:
        if event.event_type == "task_attempt_started":
            key = str(event.payload.get("idempotency_key") or "")
            if key:
                self._attempt_key_by_attempt[event.attempt_id] = key
        if event.event_type in {
            "task_attempt_completed",
            "task_attempt_failed",
            "task_attempt_timeout",
            "task_attempt_unknown",
        }:
            result_key = self._attempt_key_by_attempt.get(event.attempt_id)
            result = event.payload.get("task_result")
            if result_key and isinstance(result, dict):
                self._result_by_idempotency_key[result_key] = result


class PostgresCoordinationStore:
    """PostgreSQL-backed coordination store using asyncpg."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: Any | None = None

    async def acquire_lease(
        self,
        session_id: str,
        holder_id: str,
        ttl_s: float,
    ) -> LeaseRecord:
        await self._ensure_pool()
        now = _utc_now()
        expires_at = now + timedelta(seconds=ttl_s)
        async with self._required_pool().acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO coordination_sessions (session_id, state, updated_at)
                    VALUES ($1, 'new', $2)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    session_id,
                    now,
                )
                row = await conn.fetchrow(
                    "SELECT holder_id, fencing_token, expires_at "
                    "FROM coordination_leases WHERE session_id = $1 FOR UPDATE",
                    session_id,
                )
                if row and row["expires_at"] > now and row["holder_id"] != holder_id:
                    raise LeaseConflictError(
                        f"Session {session_id} is leased by {row['holder_id']}."
                    )
                token = int(row["fencing_token"]) + 1 if row else 1
                await conn.execute(
                    """
                    INSERT INTO coordination_leases
                        (session_id, holder_id, fencing_token, expires_at, heartbeat_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (session_id) DO UPDATE SET
                        holder_id = EXCLUDED.holder_id,
                        fencing_token = EXCLUDED.fencing_token,
                        expires_at = EXCLUDED.expires_at,
                        heartbeat_at = EXCLUDED.heartbeat_at
                    """,
                    session_id,
                    holder_id,
                    token,
                    expires_at,
                    now,
                )
                await self._insert_event(
                    conn,
                    LedgerEvent(
                        event_type="lease_acquired",
                        session_id=session_id,
                        payload={
                            "holder_id": holder_id,
                            "fencing_token": token,
                            "expires_at": expires_at.isoformat(),
                        },
                    ),
                )
        return LeaseRecord(
            session_id=session_id,
            holder_id=holder_id,
            fencing_token=token,
            expires_at=expires_at,
            heartbeat_at=now,
        )

    async def renew_lease(self, lease: LeaseRecord, ttl_s: float) -> LeaseRecord:
        await self._ensure_pool()
        now = _utc_now()
        expires_at = now + timedelta(seconds=ttl_s)
        async with self._required_pool().acquire() as conn:
            async with conn.transaction():
                await self._validate_lease(conn, lease)
                await conn.execute(
                    """
                    UPDATE coordination_leases
                    SET expires_at = $4, heartbeat_at = $5
                    WHERE session_id = $1 AND holder_id = $2 AND fencing_token = $3
                    """,
                    lease.session_id,
                    lease.holder_id,
                    lease.fencing_token,
                    expires_at,
                    now,
                )
        return lease.model_copy(update={"expires_at": expires_at, "heartbeat_at": now})

    async def release_lease(self, lease: LeaseRecord) -> None:
        await self._ensure_pool()
        now = _utc_now()
        async with self._required_pool().acquire() as conn:
            await conn.execute(
                """
                UPDATE coordination_leases
                SET expires_at = $4, heartbeat_at = $4
                WHERE session_id = $1 AND holder_id = $2 AND fencing_token = $3
                """,
                lease.session_id,
                lease.holder_id,
                lease.fencing_token,
                now,
            )

    async def append_event(
        self,
        event: LedgerEvent,
        lease: LeaseRecord | None = None,
    ) -> LedgerEvent:
        await self._ensure_pool()
        async with self._required_pool().acquire() as conn:
            async with conn.transaction():
                if lease is not None:
                    await self._validate_lease(conn, lease)
                await self._apply_event(conn, event)
                await self._insert_event(conn, event)
        return event

    async def events(self, session_id: str) -> list[LedgerEvent]:
        await self._ensure_pool()
        async with self._required_pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_type, session_id, plan_id, task_id, attempt_id,
                       payload, timestamp
                FROM coordination_events
                WHERE session_id = $1
                ORDER BY id
                """,
                session_id,
            )
        return [
            LedgerEvent(
                event_type=row["event_type"],
                session_id=row["session_id"],
                plan_id=row["plan_id"] or "",
                task_id=row["task_id"] or "",
                attempt_id=row["attempt_id"] or "",
                payload=_loads(row["payload"]),
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    async def session_state(self, session_id: str) -> CoordinationSessionState:
        return _fold_session_from_events(session_id, await self.events(session_id))

    async def task_result_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> JsonObject | None:
        await self._ensure_pool()
        async with self._required_pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT result
                FROM coordination_task_attempts
                WHERE idempotency_key = $1 AND result IS NOT NULL
                """,
                idempotency_key,
            )
        return _loads(row["result"]) if row else None

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        try:
            import asyncpg
        except ImportError as exc:
            raise CoordinationStoreError(
                "asyncpg is required for PostgreSQL coordination storage."
            ) from exc
        self._pool = await asyncpg.create_pool(self.dsn)
        async with self._required_pool().acquire() as conn:
            await apply_postgres_migrations(conn)

    def _required_pool(self) -> Any:
        if self._pool is None:
            raise CoordinationStoreError("PostgreSQL pool has not been initialized.")
        return self._pool

    async def _validate_lease(self, conn: Any, lease: LeaseRecord) -> None:
        row = await conn.fetchrow(
            """
            SELECT holder_id, fencing_token, expires_at
            FROM coordination_leases
            WHERE session_id = $1
            """,
            lease.session_id,
        )
        if row is None:
            raise LeaseConflictError(f"Session {lease.session_id} has no active lease.")
        if row["expires_at"] <= _utc_now():
            raise LeaseConflictError(f"Session {lease.session_id} lease expired.")
        if row["holder_id"] != lease.holder_id:
            raise LeaseConflictError(
                f"Session {lease.session_id} is leased by {row['holder_id']}."
            )
        if int(row["fencing_token"]) != lease.fencing_token:
            raise StaleFenceError(
                f"Session {lease.session_id} rejected stale fencing token "
                f"{lease.fencing_token}; current token is {row['fencing_token']}."
            )

    async def _apply_event(self, conn: Any, event: LedgerEvent) -> None:
        if event.event_type == "session_started":
            await conn.execute(
                """
                INSERT INTO coordination_sessions
                    (session_id, state, payload, context, updated_at)
                VALUES ($1, 'planning', $2, $3, $4)
                ON CONFLICT (session_id) DO UPDATE SET
                    state = coordination_sessions.state,
                    updated_at = EXCLUDED.updated_at
                """,
                event.session_id,
                _dumps(event.payload.get("payload") or {}),
                _dumps(event.payload.get("context") or {}),
                event.timestamp,
            )
        elif event.event_type == "registry_snapshot_recorded":
            snapshot = event.payload.get("registry_snapshot") or []
            await conn.execute(
                """
                INSERT INTO coordination_registry_snapshots
                    (session_id, plan_id, snapshot_hash, snapshot)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                """,
                event.session_id,
                event.plan_id,
                registry_snapshot_hash(snapshot),
                _dumps(snapshot),
            )
        elif event.event_type == "plan_authorized":
            plan_result = event.payload.get("plan_result") or {}
            generation = int(plan_result.get("plan_generation") or 1)
            snapshot = plan_result.get("registry_snapshot") or []
            await conn.execute(
                """
                UPDATE coordination_plan_generations
                SET active = false
                WHERE session_id = $1 AND active
                """,
                event.session_id,
            )
            await conn.execute(
                """
                INSERT INTO coordination_plan_generations
                    (session_id, plan_id, generation, registry_snapshot_hash,
                     authorized, active, plan_result)
                VALUES ($1, $2, $3, $4, true, true, $5)
                ON CONFLICT (session_id, generation) DO NOTHING
                """,
                event.session_id,
                event.plan_id,
                generation,
                registry_snapshot_hash(snapshot),
                _dumps(plan_result),
            )
            await conn.execute(
                """
                UPDATE coordination_sessions
                SET state = 'authorized', updated_at = $2
                WHERE session_id = $1
                """,
                event.session_id,
                event.timestamp,
            )
            for task in (plan_result.get("proposal") or {}).get("tasks", []):
                await conn.execute(
                    """
                    INSERT INTO coordination_task_commitments
                        (session_id, plan_id, generation, task_id, requirement_name,
                         assigned_to, state, depends_on, expected_artifacts,
                         side_effect_class)
                    VALUES ($1, $2, $3, $4, $5, $6, 'pending', $7, $8, $9)
                    ON CONFLICT (session_id, plan_id, task_id) DO NOTHING
                    """,
                    event.session_id,
                    event.plan_id,
                    generation,
                    str(task.get("task_id") or ""),
                    str(task.get("requirement_name") or ""),
                    str(task.get("assigned_to") or ""),
                    _dumps(task.get("depends_on") or []),
                    _dumps(task.get("expected_artifacts") or []),
                    _task_side_effect(plan_result, str(task.get("requirement_name") or "")),
                )
        elif event.event_type == "task_attempt_started":
            await conn.execute(
                """
                INSERT INTO coordination_task_attempts
                    (session_id, plan_id, task_id, attempt_id, idempotency_key,
                     coordinator_id, fencing_token, state, task_payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'running', $8)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                event.session_id,
                event.plan_id,
                event.task_id,
                event.attempt_id,
                str(event.payload.get("idempotency_key") or ""),
                str(event.payload.get("coordinator_id") or ""),
                int(event.payload.get("fencing_token") or 0),
                _dumps(event.payload),
            )
            await conn.execute(
                """
                UPDATE coordination_task_commitments
                SET state = 'running'
                WHERE session_id = $1 AND plan_id = $2 AND task_id = $3
                """,
                event.session_id,
                event.plan_id,
                event.task_id,
            )
        elif event.event_type in {
            "task_attempt_completed",
            "task_attempt_failed",
            "task_attempt_timeout",
            "task_attempt_unknown",
        }:
            result = event.payload.get("task_result") or {}
            state = _task_state_for_event(event.event_type)
            updated = await conn.execute(
                """
                UPDATE coordination_task_attempts
                SET state = $5, result = $6, completed_at = $7
                WHERE session_id = $1 AND plan_id = $2
                  AND task_id = $3 AND attempt_id = $4
                """,
                event.session_id,
                event.plan_id,
                event.task_id,
                event.attempt_id,
                state.value,
                _dumps(result),
                event.timestamp,
            )
            if updated.endswith(" 0"):
                raise StoreInvariantError(
                    f"Task result {event.attempt_id} has no started attempt."
                )
            await conn.execute(
                """
                UPDATE coordination_task_commitments
                SET state = $4
                WHERE session_id = $1 AND plan_id = $2 AND task_id = $3
                """,
                event.session_id,
                event.plan_id,
                event.task_id,
                state.value,
            )
            for artifact in result.get("artifacts") or []:
                await conn.execute(
                    """
                    INSERT INTO coordination_artifacts
                        (session_id, plan_id, task_id, attempt_id, artifact)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    event.session_id,
                    event.plan_id,
                    event.task_id,
                    event.attempt_id,
                    _dumps(artifact),
                )
        elif event.event_type == "task_skipped":
            result = event.payload.get("task_result") or {}
            await conn.execute(
                """
                UPDATE coordination_task_commitments
                SET state = 'skipped'
                WHERE session_id = $1 AND plan_id = $2 AND task_id = $3
                """,
                event.session_id,
                event.plan_id,
                event.task_id,
            )
        elif event.event_type in {"run_completed", "run_failed", "plan_infeasible"}:
            result = event.payload.get("run_result") or {}
            status = result.get("status") or (
                "infeasible" if event.event_type == "plan_infeasible" else "failed"
            )
            await conn.execute(
                """
                INSERT INTO coordination_terminal_results
                    (session_id, plan_id, status, run_result)
                VALUES ($1, $2, $3, $4)
                """,
                event.session_id,
                event.plan_id,
                str(status),
                _dumps(result),
            )
            await conn.execute(
                """
                UPDATE coordination_sessions
                SET state = $2, updated_at = $3
                WHERE session_id = $1
                """,
                event.session_id,
                str(status),
                event.timestamp,
            )

    async def _insert_event(self, conn: Any, event: LedgerEvent) -> None:
        await conn.execute(
            """
            INSERT INTO coordination_events
                (event_type, session_id, plan_id, task_id, attempt_id, payload, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            event.event_type,
            event.session_id,
            event.plan_id,
            event.task_id,
            event.attempt_id,
            _dumps(event.payload),
            event.timestamp,
        )


POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS coordination_schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS coordination_sessions (
    session_id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'new',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS coordination_leases (
    session_id TEXT PRIMARY KEY REFERENCES coordination_sessions(session_id)
        ON DELETE CASCADE,
    holder_id TEXT NOT NULL,
    fencing_token BIGINT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    heartbeat_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS coordination_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    attempt_id TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS coordination_registry_snapshots (
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, plan_id, snapshot_hash)
);

CREATE TABLE IF NOT EXISTS coordination_plan_generations (
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    registry_snapshot_hash TEXT NOT NULL,
    authorized BOOLEAN NOT NULL DEFAULT false,
    active BOOLEAN NOT NULL DEFAULT true,
    plan_result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, generation),
    UNIQUE (session_id, plan_id)
);

CREATE TABLE IF NOT EXISTS coordination_task_commitments (
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    requirement_name TEXT NOT NULL,
    assigned_to TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'pending',
    depends_on JSONB NOT NULL DEFAULT '[]'::jsonb,
    expected_artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
    side_effect_class TEXT NOT NULL DEFAULT 'unknown',
    PRIMARY KEY (session_id, plan_id, task_id)
);

CREATE TABLE IF NOT EXISTS coordination_task_attempts (
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    coordinator_id TEXT NOT NULL,
    fencing_token BIGINT NOT NULL,
    state TEXT NOT NULL DEFAULT 'running',
    task_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    PRIMARY KEY (session_id, plan_id, task_id, attempt_id)
);

CREATE TABLE IF NOT EXISTS coordination_artifacts (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    plan_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    artifact JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    FOREIGN KEY (session_id, plan_id, task_id, attempt_id)
        REFERENCES coordination_task_attempts(session_id, plan_id, task_id, attempt_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS coordination_terminal_results (
    session_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    status TEXT NOT NULL,
    run_result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

POSTGRES_V2_MIGRATION_SQL = """
DROP INDEX IF EXISTS one_active_authorized_plan_generation;
WITH ranked_active AS (
    SELECT session_id, generation,
           row_number() OVER (PARTITION BY session_id ORDER BY generation DESC) AS position
    FROM coordination_plan_generations
    WHERE authorized AND active
)
UPDATE coordination_plan_generations AS plans
SET active = false
FROM ranked_active
WHERE plans.session_id = ranked_active.session_id
  AND plans.generation = ranked_active.generation
  AND ranked_active.position > 1;
CREATE UNIQUE INDEX one_active_authorized_plan_generation
ON coordination_plan_generations(session_id)
WHERE authorized AND active;
"""

POSTGRES_MIGRATIONS = (
    (1, POSTGRES_SCHEMA_SQL),
    (2, POSTGRES_V2_MIGRATION_SQL),
)


async def apply_postgres_migrations(conn: Any) -> None:
    """Apply every missing schema migration in order inside transactions."""
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS coordination_schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    current = int(
        await conn.fetchval(
            "SELECT COALESCE(MAX(version), 0) FROM coordination_schema_migrations"
        )
        or 0
    )
    for version, statement in POSTGRES_MIGRATIONS:
        if version <= current:
            continue
        async with conn.transaction():
            await conn.execute(statement)
            await conn.execute(
                "INSERT INTO coordination_schema_migrations (version) VALUES ($1)",
                version,
            )


async def migrate_postgres_database(dsn: str) -> None:
    """Create or update PostgreSQL coordination-store tables."""
    try:
        import asyncpg
    except ImportError as exc:
        raise CoordinationStoreError(
            "asyncpg is required for PostgreSQL coordination storage."
        ) from exc
    conn = await asyncpg.connect(dsn)
    try:
        await apply_postgres_migrations(conn)
    finally:
        await conn.close()


def registry_snapshot_hash(snapshot: Any) -> str:
    """Return a stable hash for a registry snapshot."""
    encoded = json.dumps(_jsonable(snapshot), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def store_from_url(url: str | None, ledger_path: str | None = None) -> CoordinationStore:
    """Create a coordination store from deployment configuration."""
    if url:
        if url.startswith("etcd://"):
            from .etcd_store import etcd_store_from_url

            return etcd_store_from_url(url)
        if url.startswith(("postgres://", "postgresql://")):
            return PostgresCoordinationStore(url)
        raise CoordinationStoreError(f"Unsupported coordination store URL: {url}")
    if ledger_path:
        return JsonlCoordinationStore(JsonlCoordinationLedger(ledger_path))
    return JsonlCoordinationStore()


def main() -> None:
    """Run PostgreSQL store migrations from COORDINATION_STORE_URL."""
    import os

    dsn = os.getenv("COORDINATION_STORE_URL")
    if not dsn:
        raise SystemExit("COORDINATION_STORE_URL is required.")
    if not dsn.startswith(("postgres://", "postgresql://")):
        raise SystemExit("Only PostgreSQL COORDINATION_STORE_URL values are supported.")
    asyncio.run(migrate_postgres_database(dsn))


def _fold_session_from_events(
    session_id: str,
    events: list[LedgerEvent],
) -> CoordinationSessionState:
    ledger = InMemoryCoordinationLedger()
    for event in events:
        ledger.append(event)
    return ledger.session_state(session_id)


def _task_state_for_event(event_type: str) -> TaskState:
    if event_type == "task_attempt_completed":
        return TaskState.COMPLETED
    if event_type == "task_attempt_timeout":
        return TaskState.TIMEOUT
    if event_type == "task_attempt_unknown":
        return TaskState.UNKNOWN
    return TaskState.FAILED


def _task_side_effect(plan_result: JsonObject, requirement_name: str) -> str:
    request = plan_result.get("request") or {}
    for requirement in request.get("requirements") or []:
        if requirement.get("name") == requirement_name:
            return str(requirement.get("side_effect_class") or "unknown")
    return "unknown"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)


def _loads(value: Any) -> JsonObject:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    return dict(value)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value
