"""A2A boundary adapter used by the prototype coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .models import AgentRegistryEntry, CapabilityRequirement, FeasibilityReport, TaskSpec, TraceEvent


class AuthorizationError(RuntimeError):
    """Raised when code attempts to delegate without an authorized plan."""


class A2AAdapter:
    """Normalize Agent Cards and delegate tasks after symbolic authorization."""

    def __init__(
        self,
        card_fetcher: Callable[[str], Awaitable[Any]],
        task_sender: Callable[[str, dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self.card_fetcher = card_fetcher
        self.task_sender = task_sender
        self.trace: list[TraceEvent] = []

    async def register_from_card_url(self, url: str) -> AgentRegistryEntry:
        card = await self.card_fetcher(url)
        entry = self.normalize_card(card, fallback_url=url)
        self.trace.append(
            TraceEvent(
                event_type="agent_card_registered",
                message=f"Registered {entry.agent_id}",
                data={"url": url},
            )
        )
        return entry

    def normalize_card(self, card: Any, fallback_url: str = "") -> AgentRegistryEntry:
        """Convert a dict/object Agent Card into the local registry shape."""
        raw = self._as_dict(card)
        name = str(raw.get("name") or raw.get("id") or raw.get("agent_id"))
        endpoint = str(raw.get("url") or raw.get("service_endpoint") or fallback_url)
        skills = [
            self._normalize_skill(skill)
            for skill in raw.get("skills", [])
            if skill is not None
        ]
        input_modes = self._string_list(
            raw.get("defaultInputModes") or raw.get("input_modes") or raw.get("inputModes")
        )
        output_modes = self._string_list(
            raw.get("defaultOutputModes")
            or raw.get("output_modes")
            or raw.get("outputModes")
        )
        return AgentRegistryEntry(
            agent_id=name,
            name=name,
            description=str(raw.get("description") or ""),
            service_endpoint=endpoint,
            skills=skills,
            input_modes=input_modes,
            output_modes=output_modes,
            status="available",
            trust_level=str(raw.get("trust_level") or "standard"),
            source_card=raw,
        )

    async def send_task_after_authorization(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> Any:
        """Send a task only when the feasibility report authorized it."""
        agent_id = task.assigned_to or report.matched_agents.get(task.task_id)
        if not report.feasible or not agent_id:
            self.trace.append(
                TraceEvent(
                    event_type="delegation_refused",
                    message=f"Task {task.task_id} was not authorized.",
                    data={"task_id": task.task_id},
                )
            )
            raise AuthorizationError(f"Task {task.task_id} is not authorized.")

        try:
            result = await asyncio.wait_for(
                self.task_sender(agent_id, payload), timeout=timeout_s
            )
        except TimeoutError:
            self.trace.append(
                TraceEvent(
                    event_type="delegation_timeout",
                    message=f"Task {task.task_id} timed out.",
                    data={"task_id": task.task_id, "agent_id": agent_id},
                )
            )
            raise
        except Exception as exc:
            self.trace.append(
                TraceEvent(
                    event_type="delegation_failed",
                    message=f"Task {task.task_id} failed.",
                    data={"task_id": task.task_id, "agent_id": agent_id, "error": str(exc)},
                )
            )
            raise

        self.trace.append(
            TraceEvent(
                event_type="delegation_completed",
                message=f"Task {task.task_id} completed.",
                data={"task_id": task.task_id, "agent_id": agent_id},
            )
        )
        return result

    def convert_artifact_parts(self, parts: list[Any]) -> list[dict[str, Any]]:
        """Normalize common A2A text, data, and file parts for validators."""
        converted: list[dict[str, Any]] = []
        for part in parts:
            raw = self._as_dict(getattr(part, "root", part))
            kind = raw.get("kind") or raw.get("type")
            if kind == "text":
                converted.append({"kind": "text", "text": raw.get("text", "")})
            elif kind == "data":
                converted.append({"kind": "data", "data": raw.get("data", {})})
            elif kind == "file":
                file_data = self._as_dict(raw.get("file", {}))
                converted.append(
                    {
                        "kind": "file",
                        "name": file_data.get("name"),
                        "mime_type": file_data.get("mime_type"),
                        "bytes": file_data.get("bytes"),
                    }
                )
            else:
                converted.append({"kind": "unknown", "raw": raw})
        return converted

    def _normalize_skill(self, skill: Any) -> CapabilityRequirement:
        raw = self._as_dict(skill)
        name = str(raw.get("name") or raw.get("id") or raw.get("skill_id"))
        return CapabilityRequirement(
            name=name,
            description=str(raw.get("description") or ""),
            input_modes=self._string_list(raw.get("inputModes") or raw.get("input_modes")),
            output_modes=self._string_list(raw.get("outputModes") or raw.get("output_modes")),
            auxiliary_eligible=False,
        )

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        return {}

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]
