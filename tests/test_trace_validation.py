from unified_multi_agent_coordination.models import TraceEvent
from unified_multi_agent_coordination.trace_validation import validate_trace


def _event(event_type: str, *, session: str = "session", attempt: str = "") -> TraceEvent:
    return TraceEvent(
        event_type=event_type,
        message=event_type,
        session_id=session,
        attempt_id=attempt,
    )


def test_trace_rejects_empty_or_identity_invalid_events() -> None:
    assert validate_trace([]).violations == ["trace is empty"]
    report = validate_trace(
        [_event("plan_authorized", session=""), _event("run_completed", session="other")]
    )
    assert report.complete is False
    assert "trace has no session identity" in report.violations
    assert "trace contains cross-session events" in report.violations


def test_trace_rejects_dispatch_without_or_before_authorization() -> None:
    no_authorization = validate_trace([_event("task_attempt_started", attempt="a1")])
    assert "dispatch exists without persisted authorization" in no_authorization.violations
    assert "attempt a1 lacks a terminal observation" in no_authorization.violations

    reversed_trace = validate_trace(
        [_event("sdk_task_started"), _event("plan_authorized")]
    )
    assert reversed_trace.violations == ["dispatch precedes authorization"]


def test_trace_accepts_each_terminal_attempt_observation() -> None:
    terminal_types = (
        "task_attempt_completed",
        "task_attempt_failed",
        "task_attempt_timeout",
        "task_attempt_unknown",
    )
    for terminal in terminal_types:
        report = validate_trace(
            [
                _event("plan_authorized"),
                _event("task_attempt_started", attempt="a1"),
                _event(terminal, attempt="a1"),
            ],
            "session",
        )
        assert report.complete is True
        assert report.violations == []
