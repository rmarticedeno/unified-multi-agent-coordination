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
        endpoint = str(
            raw.get("url")
            or raw.get("service_endpoint")
            or raw.get("serviceEndpoint")
            or self._first_interface_url(raw)
            or fallback_url
        )
        name = str(
            raw.get("name")
            or raw.get("id")
            or raw.get("agent_id")
            or raw.get("agentId")
            or endpoint
        )
        agent_id = str(raw.get("agent_id") or raw.get("agentId") or raw.get("id") or name)
        skills = [
            self._normalize_skill(skill)
            for skill in raw.get("skills", [])
            if skill is not None
        ]
        input_modes = self._string_list(
            raw.get("defaultInputModes")
            or raw.get("default_input_modes")
            or raw.get("input_modes")
            or raw.get("inputModes")
        )
        output_modes = self._string_list(
            raw.get("defaultOutputModes")
            or raw.get("default_output_modes")
            or raw.get("output_modes")
            or raw.get("outputModes")
        )
        status = raw.get("status")
        if status not in {"available", "unavailable"}:
            status = "available"
        return AgentRegistryEntry(
            agent_id=agent_id,
            name=name,
            description=str(raw.get("description") or ""),
            service_endpoint=endpoint,
            invocation_endpoint=endpoint,
            skills=skills,
            input_modes=input_modes,
            output_modes=output_modes,
            status=status,
            trust_level=str(raw.get("trust_level") or raw.get("trustLevel") or "standard"),
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
        name = str(
            raw.get("name")
            or raw.get("id")
            or raw.get("skill_id")
            or raw.get("skillId")
        )
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
        try:
            from google.protobuf.json_format import MessageToDict
            from google.protobuf.message import Message as ProtobufMessage
        except ImportError:
            ProtobufMessage = None  # type: ignore[assignment]
        if ProtobufMessage is not None and isinstance(value, ProtobufMessage):
            return MessageToDict(value, preserving_proto_field_name=True)
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

    @staticmethod
    def _first_interface_url(raw: dict[str, Any]) -> str:
        interfaces = raw.get("supported_interfaces") or raw.get("supportedInterfaces") or []
        if not isinstance(interfaces, list):
            return ""
        for item in interfaces:
            interface = A2AAdapter._as_dict(item)
            url = interface.get("url")
            if url:
                return str(url)
        return ""
