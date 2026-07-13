"""Deterministic validation for persisted coordination traces."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import TraceEvent


class TraceValidationReport(BaseModel):
    complete: bool
    violations: list[str] = Field(default_factory=list)


def validate_trace(events: list[TraceEvent], session_id: str = "") -> TraceValidationReport:
    """Check identity, authorization ordering, and terminal attempt observations."""
    violations: list[str] = []
    if not events:
        return TraceValidationReport(complete=False, violations=["trace is empty"])
    expected_session = session_id or events[0].session_id
    if not expected_session:
        violations.append("trace has no session identity")
    if any(event.session_id != expected_session for event in events):
        violations.append("trace contains cross-session events")

    authorization_positions = [
        index for index, event in enumerate(events) if event.event_type == "plan_authorized"
    ]
    dispatch_positions = [
        index
        for index, event in enumerate(events)
        if event.event_type in {"task_attempt_started", "sdk_task_started", "sdk_auxiliary_task_started"}
    ]
    if dispatch_positions and not authorization_positions:
        violations.append("dispatch exists without persisted authorization")
    elif dispatch_positions and min(dispatch_positions) < min(authorization_positions):
        violations.append("dispatch precedes authorization")

    started_attempts = {
        event.attempt_id
        for event in events
        if event.event_type == "task_attempt_started" and event.attempt_id
    }
    terminal_attempts = {
        event.attempt_id
        for event in events
        if event.event_type
        in {
            "task_attempt_completed",
            "task_attempt_failed",
            "task_attempt_timeout",
            "task_attempt_unknown",
        }
        and event.attempt_id
    }
    for attempt_id in sorted(started_attempts - terminal_attempts):
        violations.append(f"attempt {attempt_id} lacks a terminal observation")
    return TraceValidationReport(complete=not violations, violations=violations)
