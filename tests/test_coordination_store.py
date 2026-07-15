import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from unified_multi_agent_coordination.coordination_ledger import LedgerEvent
from unified_multi_agent_coordination.coordination_store import (
    AuditProjectingCoordinationStore,
    CoordinationStoreError,
    JsonlCoordinationStore,
    LeaseConflictError,
    PostgresCoordinationStore,
    StaleFenceError,
    StoreInvariantError,
    apply_postgres_migrations,
    registry_snapshot_hash,
    store_from_url,
)
from unified_multi_agent_coordination.models import LeaseRecord


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakeMigrationConnection:
    def __init__(self, current_version):
        self.current_version = current_version
        self.executed = []

    async def execute(self, statement, *args):
        self.executed.append((statement, args))

    async def fetchval(self, statement):
        return self.current_version

    def transaction(self):
        return _FakeTransaction()


@pytest.mark.asyncio
async def test_postgres_migration_upgrades_v1_with_partial_active_plan_index():
    conn = _FakeMigrationConnection(current_version=1)

    await apply_postgres_migrations(conn)

    statements = "\n".join(item[0] for item in conn.executed)
    assert "WHERE authorized AND active" in statements
    assert any(args == (2,) for _, args in conn.executed)
    assert not any(args == (1,) for _, args in conn.executed)


@pytest.mark.asyncio
async def test_postgres_migration_initializes_versions_in_order():
    conn = _FakeMigrationConnection(current_version=0)

    await apply_postgres_migrations(conn)

    versions = [args[0] for _, args in conn.executed if args]
    assert versions == [1, 2]


@pytest.mark.asyncio
async def test_jsonl_store_rejects_competing_lease_and_stale_fence():
    store = JsonlCoordinationStore()

    first = await store.acquire_lease("s1", "coordinator-a", ttl_s=0.01)

    with pytest.raises(LeaseConflictError):
        await store.acquire_lease("s1", "coordinator-b", ttl_s=30)

    await asyncio.sleep(0.02)
    second = await store.acquire_lease("s1", "coordinator-a", ttl_s=30)

    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(StaleFenceError):
        await store.append_event(
            LedgerEvent(event_type="session_started", session_id="s1"),
            lease=first,
        )


@pytest.mark.asyncio
async def test_jsonl_store_release_preserves_monotonic_fencing():
    store = JsonlCoordinationStore()

    first = await store.acquire_lease("released-session", "coordinator-a", ttl_s=30)
    await store.release_lease(first)
    second = await store.acquire_lease("released-session", "coordinator-a", ttl_s=30)

    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(StaleFenceError):
        await store.append_event(
            LedgerEvent(event_type="session_started", session_id="released-session"),
            lease=first,
        )


@pytest.mark.asyncio
async def test_jsonl_store_enforces_terminal_and_attempt_invariants():
    store = JsonlCoordinationStore()
    lease = await store.acquire_lease("s2", "coordinator-a", ttl_s=30)

    await store.append_event(
        LedgerEvent(event_type="session_started", session_id="s2"),
        lease=lease,
    )
    await store.append_event(
        LedgerEvent(
            event_type="run_failed",
            session_id="s2",
            payload={"run_result": {"status": "failed"}},
        ),
        lease=lease,
    )

    with pytest.raises(StoreInvariantError):
        await store.append_event(
            LedgerEvent(
                event_type="run_completed",
                session_id="s2",
                payload={"run_result": {"status": "completed"}},
            ),
            lease=lease,
        )

    lease = await store.acquire_lease("s3", "coordinator-a", ttl_s=30)
    with pytest.raises(StoreInvariantError):
        await store.append_event(
            LedgerEvent(
                event_type="task_attempt_completed",
                session_id="s3",
                task_id="t1",
                attempt_id="t1-attempt-1",
                payload={"task_result": {"status": "completed"}},
            ),
            lease=lease,
        )


@pytest.mark.asyncio
async def test_jsonl_store_indexes_attempt_result_by_idempotency_key():
    store = JsonlCoordinationStore()
    lease = await store.acquire_lease("s4", "coordinator-a", ttl_s=30)

    await store.append_event(
        LedgerEvent(
            event_type="task_attempt_started",
            session_id="s4",
            plan_id="p1",
            task_id="t1",
            attempt_id="t1-attempt-1",
            payload={"idempotency_key": "s4:p1:t1:t1-attempt-1"},
        ),
        lease=lease,
    )
    await store.append_event(
        LedgerEvent(
            event_type="task_attempt_completed",
            session_id="s4",
            plan_id="p1",
            task_id="t1",
            attempt_id="t1-attempt-1",
            payload={
                "task_result": {
                    "task_id": "t1",
                    "agent_id": "summarizer",
                    "status": "completed",
                }
            },
        ),
        lease=lease,
    )

    result = await store.task_result_by_idempotency_key("s4:p1:t1:t1-attempt-1")

    assert result is not None
    assert result["status"] == "completed"


class _FailingAuditStore(JsonlCoordinationStore):
    async def append_event(self, event, lease=None):
        raise RuntimeError("audit sink offline")


@pytest.mark.asyncio
async def test_audit_projection_is_best_effort_and_delegates_authority():
    authoritative = JsonlCoordinationStore()
    audit = _FailingAuditStore()
    store = AuditProjectingCoordinationStore(authoritative, audit)

    lease = await store.acquire_lease("audit-session", "coordinator-a", 30)
    renewed = await store.renew_lease(lease, 30)
    event = await store.append_event(
        LedgerEvent(event_type="session_started", session_id="audit-session"),
        renewed,
    )
    await asyncio.wait_for(store._queue.join(), timeout=1)

    assert (await store.events("audit-session"))[-1] == event
    assert (await store.session_state("audit-session")).session_id == "audit-session"
    assert await store.task_result_by_idempotency_key("missing") is None
    assert await store.ready() is True
    assert store.last_projection_error == "audit sink offline"
    await store.release_lease(renewed)
    await store.close()


class _FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _FakePool:
    def __init__(self, connection):
        self.connection = connection
        self.closed = False

    def acquire(self):
        return _FakeAcquire(self.connection)

    async def close(self):
        self.closed = True


class _FakePostgresConnection:
    def __init__(self):
        self.executed = []
        self.fetchrows = []
        self.fetchrow_results = []
        self.update_result = "UPDATE 1"

    async def execute(self, statement, *args):
        self.executed.append((statement, args))
        if "UPDATE coordination_task_attempts" in statement:
            return self.update_result
        return "OK"

    async def fetchrow(self, statement, *args):
        self.executed.append((statement, args))
        if self.fetchrow_results:
            return self.fetchrow_results.pop(0)
        return None

    async def fetch(self, statement, *args):
        return self.fetchrows

    def transaction(self):
        return _FakeTransaction()


def _lease(*, holder="coordinator-a", token=1, expired=False):
    now = datetime.now(timezone.utc)
    return LeaseRecord(
        session_id="pg-session",
        holder_id=holder,
        fencing_token=token,
        expires_at=now + timedelta(seconds=-1 if expired else 30),
        heartbeat_at=now,
    )


@pytest.mark.asyncio
async def test_postgres_store_lease_lifecycle_queries_and_close():
    conn = _FakePostgresConnection()
    pool = _FakePool(conn)
    store = PostgresCoordinationStore("postgresql://unused")
    store._pool = pool

    now = datetime.now(timezone.utc)
    conn.fetchrow_results.append(
        {
            "holder_id": "coordinator-a",
            "fencing_token": 1,
            "expires_at": now + timedelta(seconds=30),
            "heartbeat_at": now,
        }
    )
    acquired = await store.acquire_lease("pg-session", "coordinator-a", 30)
    assert acquired.fencing_token == 1

    conn.fetchrow_results.append(
        {
            "holder_id": acquired.holder_id,
            "fencing_token": acquired.fencing_token,
            "expires_at": acquired.expires_at,
        }
    )
    renewed = await store.renew_lease(acquired, 60)
    assert renewed.expires_at > acquired.expires_at
    await store.release_lease(renewed)

    conn.fetchrows = [
        {
            "event_type": "session_started",
            "session_id": "pg-session",
            "plan_id": None,
            "task_id": None,
            "attempt_id": None,
            "payload": json.dumps({"payload": {"x": 1}}),
            "timestamp": now,
        }
    ]
    assert (await store.events("pg-session"))[0].payload["payload"] == {"x": 1}
    assert (await store.session_state("pg-session")).session_id == "pg-session"

    conn.fetchrow_results.append({"result": json.dumps({"status": "completed"})})
    assert (await store.task_result_by_idempotency_key("key"))["status"] == "completed"
    assert await store.task_result_by_idempotency_key("missing") is None

    await store.close()
    assert pool.closed is True
    with pytest.raises(CoordinationStoreError):
        store._required_pool()


@pytest.mark.asyncio
async def test_postgres_store_rejects_conflicts_expiry_wrong_holder_and_fence():
    conn = _FakePostgresConnection()
    store = PostgresCoordinationStore("postgresql://unused")
    store._pool = _FakePool(conn)
    now = datetime.now(timezone.utc)

    conn.fetchrow_results.extend(
        [
            None,
            {"holder_id": "other"},
        ]
    )
    with pytest.raises(LeaseConflictError):
        await store.acquire_lease("pg-session", "coordinator-a", 30)

    statements = "\n".join(sql for sql, _ in conn.executed)
    assert "ON CONFLICT (session_id) DO UPDATE" in statements
    assert "coordination_leases.fencing_token + 1" in statements
    assert "WHERE coordination_leases.expires_at" in statements

    with pytest.raises(LeaseConflictError, match="no active lease"):
        await store._validate_lease(conn, _lease())
    conn.fetchrow_results.append(
        {"holder_id": "coordinator-a", "fencing_token": 1, "expires_at": now - timedelta(seconds=1)}
    )
    with pytest.raises(LeaseConflictError, match="expired"):
        await store._validate_lease(conn, _lease())
    conn.fetchrow_results.append(
        {"holder_id": "other", "fencing_token": 1, "expires_at": now + timedelta(seconds=30)}
    )
    with pytest.raises(LeaseConflictError, match="leased by other"):
        await store._validate_lease(conn, _lease())
    conn.fetchrow_results.append(
        {"holder_id": "coordinator-a", "fencing_token": 2, "expires_at": now + timedelta(seconds=30)}
    )
    with pytest.raises(StaleFenceError):
        await store._validate_lease(conn, _lease())


@pytest.mark.asyncio
async def test_postgres_event_projection_covers_every_state_transition():
    conn = _FakePostgresConnection()
    store = PostgresCoordinationStore("postgresql://unused")
    store._pool = _FakePool(conn)
    lease = _lease()
    valid = {
        "holder_id": lease.holder_id,
        "fencing_token": lease.fencing_token,
        "expires_at": lease.expires_at,
    }

    events = [
        LedgerEvent(
            event_type="session_started",
            session_id="pg-session",
            payload={"payload": {"goal": "x"}, "context": {"source": "test"}},
        ),
        LedgerEvent(
            event_type="registry_snapshot_recorded",
            session_id="pg-session",
            plan_id="p1",
            payload={"registry_snapshot": [{"agent_id": "a"}]},
        ),
        LedgerEvent(
            event_type="plan_authorized",
            session_id="pg-session",
            plan_id="p1",
            payload={
                "plan_result": {
                    "plan_generation": 2,
                    "registry_snapshot": [{"agent_id": "a"}],
                    "requirements": [{"name": "summarize", "side_effect": "none"}],
                    "proposal": {
                        "tasks": [
                            {
                                "task_id": "t1",
                                "requirement_name": "summarize",
                                "assigned_to": "a",
                                "depends_on": [],
                                "expected_artifacts": ["summary"],
                            }
                        ]
                    },
                }
            },
        ),
        LedgerEvent(
            event_type="task_attempt_started",
            session_id="pg-session",
            plan_id="p1",
            task_id="t1",
            attempt_id="a1",
            payload={"idempotency_key": "key", "coordinator_id": "c", "fencing_token": 1},
        ),
        LedgerEvent(
            event_type="task_attempt_completed",
            session_id="pg-session",
            plan_id="p1",
            task_id="t1",
            attempt_id="a1",
            payload={"task_result": {"artifacts": [{"artifact_type": "summary"}]}},
        ),
        LedgerEvent(
            event_type="task_skipped",
            session_id="pg-session",
            plan_id="p1",
            task_id="t2",
            payload={"task_result": {"status": "skipped"}},
        ),
        LedgerEvent(
            event_type="run_completed",
            session_id="pg-session",
            plan_id="p1",
            payload={"run_result": {"status": "completed"}},
        ),
    ]
    for event in events:
        conn.fetchrow_results.append(valid)
        assert await store.append_event(event, lease) == event

    for event_type in ("task_attempt_failed", "task_attempt_timeout", "task_attempt_unknown"):
        await store._apply_event(
            conn,
            LedgerEvent(
                event_type=event_type,
                session_id="pg-session",
                plan_id="p1",
                task_id="t1",
                attempt_id="a1",
                payload={"task_result": {}},
            ),
        )
    for event_type in ("run_failed", "plan_infeasible"):
        await store._apply_event(
            conn,
            LedgerEvent(event_type=event_type, session_id="other", plan_id="p2"),
        )

    conn.update_result = "UPDATE 0"
    with pytest.raises(StoreInvariantError, match="no started attempt"):
        await store._apply_event(
            conn,
            LedgerEvent(
                event_type="task_attempt_completed",
                session_id="pg-session",
                plan_id="p1",
                task_id="missing",
                attempt_id="missing",
            ),
        )

    assert any("coordination_terminal_results" in sql for sql, _ in conn.executed)
    assert registry_snapshot_hash([{"b": 2, "a": 1}]) == registry_snapshot_hash(
        [{"a": 1, "b": 2}]
    )


def test_store_url_selection_and_validation(tmp_path):
    assert isinstance(store_from_url(None), JsonlCoordinationStore)
    assert isinstance(store_from_url(None, str(tmp_path / "ledger.jsonl")), JsonlCoordinationStore)
    assert isinstance(store_from_url("postgresql://db/test"), PostgresCoordinationStore)
    for unsupported in ("memory://", "jsonl://", "redis://localhost"):
        with pytest.raises(CoordinationStoreError, match="Unsupported"):
            store_from_url(unsupported)
