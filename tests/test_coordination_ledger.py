from unified_multi_agent_coordination import JsonlCoordinationLedger, LedgerEvent


def test_jsonl_ledger_appends_and_reconstructs_session(tmp_path):
    path = tmp_path / "coordination.jsonl"
    ledger = JsonlCoordinationLedger(path)

    ledger.append(
        LedgerEvent(
            event_type="session_started",
            session_id="s1",
            plan_id="p1",
            payload={"payload": {"text": "hello"}, "context": {"source": "test"}},
        )
    )
    ledger.append(
        LedgerEvent(
            event_type="task_attempt_completed",
            session_id="s1",
            plan_id="p1",
            task_id="t1",
            attempt_id="t1-attempt-1",
            payload={
                "task_result": {
                    "task_id": "t1",
                    "agent_id": "summarizer",
                    "status": "completed",
                    "artifacts": [{"kind": "text", "text": "hello"}],
                }
            },
        )
    )

    reopened = JsonlCoordinationLedger(path)
    state = reopened.session_state("s1")

    assert state.payload == {"text": "hello"}
    assert state.context == {"source": "test"}
    assert state.completed_task_ids == {"t1"}
    assert len(state.events) == 2
