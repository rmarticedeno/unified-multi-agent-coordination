import asyncio

import pytest

from unified_multi_agent_coordination.coordination_ledger import LedgerEvent
from unified_multi_agent_coordination.coordination_store import (
    JsonlCoordinationStore,
    LeaseConflictError,
    StaleFenceError,
    StoreInvariantError,
    apply_postgres_migrations,
)


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
