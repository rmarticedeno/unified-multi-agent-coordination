"""Configurable A2A fixture agent for Docker system tests."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI

from a2a.helpers import new_data_artifact, new_text_artifact, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol

from .models import stable_identifier


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class FixtureConfig:
    agent_id: str
    name: str
    skill: str
    description: str
    input_modes: list[str]
    output_modes: list[str]
    base_url: str
    artifact_name: str
    artifact_kind: str
    artifact_text: str
    artifact_data: JsonObject
    delay_s: float
    failure_mode: str
    include_coordination_summary: bool

    @classmethod
    def from_env(cls) -> "FixtureConfig":
        agent_id = os.getenv("A2A_AGENT_ID", "fixture-agent")
        skill = os.getenv("A2A_AGENT_SKILL", "echo")
        return cls(
            agent_id=agent_id,
            name=os.getenv("A2A_AGENT_NAME", agent_id),
            skill=skill,
            description=os.getenv("A2A_AGENT_DESCRIPTION", f"Fixture agent for {skill}."),
            input_modes=_csv("A2A_AGENT_INPUT_MODES", ["text"]),
            output_modes=_csv("A2A_AGENT_OUTPUT_MODES", ["data"]),
            base_url=os.getenv("A2A_AGENT_BASE_URL", "http://127.0.0.1:8000"),
            artifact_name=os.getenv("A2A_AGENT_ARTIFACT_NAME", skill),
            artifact_kind=os.getenv("A2A_AGENT_ARTIFACT_KIND", "data"),
            artifact_text=os.getenv("A2A_AGENT_ARTIFACT_TEXT", ""),
            artifact_data=_json_env("A2A_AGENT_ARTIFACT_JSON", {}),
            delay_s=float(os.getenv("A2A_AGENT_DELAY_S", "0")),
            failure_mode=os.getenv("A2A_AGENT_FAILURE_MODE", "none"),
            include_coordination_summary=_bool_env(
                "A2A_AGENT_INCLUDE_COORDINATION_SUMMARY"
            ),
        )


class FixtureAgentExecutor(AgentExecutor):
    """Small deterministic executor used only by the Docker system harness."""

    def __init__(self, config: FixtureConfig) -> None:
        self.config = config
        self._completed_by_idempotency_key: dict[str, Any] = {}
        self._request_count_by_idempotency_key: dict[str, int] = {}
        self._effectful_count_by_idempotency_key: dict[str, int] = {}
        self._request_count_by_session_task: dict[str, int] = {}
        self._effectful_count_by_session_task: dict[str, int] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        payload = _payload_from_context(context)
        idempotency_key = _idempotency_key(payload)
        session_task_key = _session_task_key(payload)
        if idempotency_key:
            self._request_count_by_idempotency_key[idempotency_key] = (
                self._request_count_by_idempotency_key.get(idempotency_key, 0) + 1
            )
        if session_task_key:
            self._request_count_by_session_task[session_task_key] = (
                self._request_count_by_session_task.get(session_task_key, 0) + 1
            )
        if idempotency_key and idempotency_key in self._completed_by_idempotency_key:
            artifact = self._completed_by_idempotency_key[idempotency_key]
            updater = TaskUpdater(
                event_queue,
                task_id=str(context.task_id),
                context_id=str(context.context_id),
            )
            await updater.add_artifact(
                list(artifact.parts),
                name=artifact.name,
                artifact_id=artifact.artifact_id,
            )
            await updater.complete(
                new_text_message(
                    f"{self.config.agent_id} returned duplicate-safe result.",
                    task_id=context.task_id,
                    context_id=context.context_id,
                )
            )
            return

        if self.config.delay_s > 0:
            await asyncio.sleep(self.config.delay_s)
        if self.config.failure_mode == "exception":
            raise RuntimeError(f"{self.config.agent_id} fixture exception")
        if idempotency_key:
            self._effectful_count_by_idempotency_key[idempotency_key] = (
                self._effectful_count_by_idempotency_key.get(idempotency_key, 0) + 1
            )
        if session_task_key:
            self._effectful_count_by_session_task[session_task_key] = (
                self._effectful_count_by_session_task.get(session_task_key, 0) + 1
            )

        updater = TaskUpdater(
            event_queue,
            task_id=str(context.task_id),
            context_id=str(context.context_id),
        )
        if self.config.failure_mode == "reject":
            await updater.reject(
                new_text_message(
                    f"{self.config.agent_id} rejected the fixture request.",
                    task_id=context.task_id,
                    context_id=context.context_id,
                )
            )
            return

        artifact = self._artifact(payload)
        if idempotency_key:
            self._completed_by_idempotency_key[idempotency_key] = artifact
        await updater.add_artifact(
            list(artifact.parts),
            name=artifact.name,
            artifact_id=artifact.artifact_id,
        )
        if self.config.failure_mode == "failed":
            await updater.failed(
                new_text_message(
                    f"{self.config.agent_id} fixture failed.",
                    task_id=context.task_id,
                    context_id=context.context_id,
                )
            )
            return
        await updater.complete(
            new_text_message(
                f"{self.config.agent_id} completed.",
                task_id=context.task_id,
                context_id=context.context_id,
            )
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(
            event_queue,
            task_id=str(context.task_id),
            context_id=str(context.context_id),
        )
        await updater.update_status(TaskState.TASK_STATE_CANCELED)

    def stats(self) -> JsonObject:
        """Return agent-side invocation counters for harness assertions."""
        duplicate_idempotency_requests = sum(
            max(count - 1, 0)
            for count in self._request_count_by_idempotency_key.values()
        )
        repeated_task_executions = sum(
            max(count - 1, 0)
            for count in self._effectful_count_by_session_task.values()
        )
        return {
            "agent_id": self.config.agent_id,
            "total_requests": sum(self._request_count_by_session_task.values()),
            "total_effectful_executions": sum(
                self._effectful_count_by_session_task.values()
            ),
            "duplicate_idempotency_key_requests": duplicate_idempotency_requests,
            "repeated_session_task_effectful_executions": repeated_task_executions,
            "requests_by_idempotency_key": dict(self._request_count_by_idempotency_key),
            "effectful_executions_by_idempotency_key": dict(
                self._effectful_count_by_idempotency_key
            ),
            "requests_by_session_task": dict(self._request_count_by_session_task),
            "effectful_executions_by_session_task": dict(
                self._effectful_count_by_session_task
            ),
        }

    def _artifact(self, payload: JsonObject):
        if self.config.artifact_kind == "text":
            text = self.config.artifact_text or json.dumps(
                self.config.artifact_data, sort_keys=True
            )
            return new_text_artifact(self.config.artifact_name, text)

        data = dict(self.config.artifact_data)
        if self.config.include_coordination_summary:
            coordination = payload.get("_coordination")
            if isinstance(coordination, dict):
                previous_artifacts = coordination.get("previous_artifacts")
                inputs_by_task = coordination.get("inputs_by_task")
                data["received_previous_artifact_count"] = (
                    len(previous_artifacts) if isinstance(previous_artifacts, list) else 0
                )
                data["received_dependency_task_count"] = (
                    len(inputs_by_task) if isinstance(inputs_by_task, dict) else 0
                )
        return new_data_artifact(self.config.artifact_name, data)


def create_app(config: FixtureConfig | None = None) -> FastAPI:
    """Create the fixture A2A FastAPI application."""
    config = config or FixtureConfig.from_env()
    app = FastAPI(title=f"A2A Fixture Agent: {config.agent_id}")
    card = _agent_card(config)
    executor = FixtureAgentExecutor(config)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
    )

    @app.get("/health")
    async def health() -> JsonObject:
        return {"status": "ok", "agent_id": config.agent_id, "skill": config.skill}

    @app.get("/fixture-stats")
    async def fixture_stats() -> JsonObject:
        return executor.stats()

    return app


def _agent_card(config: FixtureConfig) -> AgentCard:
    return AgentCard(
        name=config.agent_id,
        description=config.description,
        supported_interfaces=[
            AgentInterface(
                url=f"{config.base_url.rstrip('/')}/",
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=config.input_modes,
        default_output_modes=config.output_modes,
        skills=[
            AgentSkill(
                id=stable_identifier(config.skill),
                name=config.skill,
                description=config.description,
                input_modes=config.input_modes,
                output_modes=config.output_modes,
            )
        ],
    )


def _payload_from_context(context: RequestContext) -> JsonObject:
    try:
        value = json.loads(context.get_user_input())
    except json.JSONDecodeError:
        return {"text": context.get_user_input()}
    return value if isinstance(value, dict) else {"value": value}


def _idempotency_key(payload: JsonObject) -> str:
    coordination = payload.get("_coordination")
    if not isinstance(coordination, dict):
        return ""
    return str(coordination.get("idempotency_key") or "")


def _session_task_key(payload: JsonObject) -> str:
    coordination = payload.get("_coordination")
    if not isinstance(coordination, dict):
        return ""
    session_id = str(coordination.get("session_id") or "")
    task_id = str(coordination.get("task_id") or "")
    if not session_id or not task_id:
        return ""
    return f"{session_id}:{task_id}"


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _json_env(name: str, default: JsonObject) -> JsonObject:
    raw = os.getenv(name)
    if not raw:
        return dict(default)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    """Run the fixture A2A service."""
    host = os.getenv("A2A_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("A2A_AGENT_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
