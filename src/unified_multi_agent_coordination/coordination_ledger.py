"""Durable coordination ledger for crash recovery.

The JSONL backend is intentionally small: every coordination observation is one
append-only JSON record, and recovery folds the records for a session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from pydantic import BaseModel, Field


JsonObject = dict[str, Any]


class LedgerEvent(BaseModel):
    """One durable coordination observation."""

    event_type: str
    session_id: str
    plan_id: str = ""
    task_id: str = ""
    attempt_id: str = ""
    payload: JsonObject = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoordinationSessionState(BaseModel):
    """Recovered view of one coordination session."""

    session_id: str
    events: list[LedgerEvent] = Field(default_factory=list)
    payload: JsonObject = Field(default_factory=dict)
    context: JsonObject = Field(default_factory=dict)
    plan_result: JsonObject | None = None
    terminal_result: JsonObject | None = None
    task_results: dict[str, JsonObject] = Field(default_factory=dict)
    task_attempt_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def completed_task_ids(self) -> set[str]:
        return {
            task_id
            for task_id, result in self.task_results.items()
            if result.get("status") == "completed"
        }


class CoordinationLedger(Protocol):
    """Append-only ledger interface used by the coordinator."""

    def append(self, event: LedgerEvent) -> LedgerEvent:
        ...

    def events(self, session_id: str) -> list[LedgerEvent]:
        ...

    def session_state(self, session_id: str) -> CoordinationSessionState:
        ...


class InMemoryCoordinationLedger:
    """Process-local ledger for tests and deployments without persistence."""

    def __init__(self) -> None:
        self._events: list[LedgerEvent] = []

    def append(self, event: LedgerEvent) -> LedgerEvent:
        self._events.append(event)
        return event

    def events(self, session_id: str) -> list[LedgerEvent]:
        return [event for event in self._events if event.session_id == session_id]

    def session_state(self, session_id: str) -> CoordinationSessionState:
        return _fold_session(session_id, self.events(session_id))


class JsonlCoordinationLedger:
    """JSON Lines ledger backend."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, event: LedgerEvent) -> LedgerEvent:
        line = event.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        return event

    def events(self, session_id: str) -> list[LedgerEvent]:
        records: list[LedgerEvent] = []
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("session_id") == session_id:
                        records.append(LedgerEvent.model_validate(payload))
        return records

    def session_state(self, session_id: str) -> CoordinationSessionState:
        return _fold_session(session_id, self.events(session_id))


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry configuration for registry and task dispatch."""

    registry_retries: int = 2
    task_retries: int = 1
    backoff_s: float = 0.05


def _fold_session(
    session_id: str,
    events: list[LedgerEvent],
) -> CoordinationSessionState:
    state = CoordinationSessionState(session_id=session_id, events=list(events))
    for event in events:
        if event.event_type == "session_started":
            state.payload = dict(event.payload.get("payload") or {})
            state.context = dict(event.payload.get("context") or {})
        elif event.event_type == "plan_authorized":
            state.plan_result = event.payload.get("plan_result")
        elif event.event_type == "plan_infeasible":
            state.plan_result = event.payload.get("plan_result")
            state.terminal_result = event.payload.get("run_result")
        elif event.event_type in {
            "task_attempt_completed",
            "task_attempt_failed",
            "task_attempt_timeout",
        }:
            result = event.payload.get("task_result")
            if isinstance(result, dict):
                state.task_results[event.task_id] = result
            state.task_attempt_counts[event.task_id] = (
                state.task_attempt_counts.get(event.task_id, 0) + 1
            )
        elif event.event_type == "task_attempt_started":
            state.task_attempt_counts[event.task_id] = max(
                state.task_attempt_counts.get(event.task_id, 0),
                _attempt_number(event.attempt_id),
            )
        elif event.event_type in {"run_completed", "run_failed"}:
            state.terminal_result = event.payload.get("run_result")
    return state


def _attempt_number(attempt_id: str) -> int:
    try:
        return int(attempt_id.rsplit("-", 1)[-1])
    except (TypeError, ValueError):
        return 0
