"""SDK facade for registry discovery and authorized task dispatch."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

from .a2a_adapter import A2AAdapter, AuthorizationError
from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    FeasibilityReport,
    GeneratedNlpAgentSpec,
    TaskExecutionResult,
    TaskSpec,
    TraceEvent,
)


class RemoteRegistryError(RuntimeError):
    """Raised when a remote registry cannot be read or normalized."""


class CoordinationSdk:
    """Communication facade used by the coordination agent."""

    _CARD_KEYS = ("agents", "agent_cards", "cards", "items", "results")
    _URL_KEYS = ("card_urls", "urls")

    def __init__(
        self,
        remote_registry_url: str | None = None,
        *,
        registry_endpoint: str | None = None,
        registry_addr: str | None = None,
        self_agent_id: str | None = None,
        registry_headers: Mapping[str, str] | None = None,
        request_timeout_s: float = 10.0,
        card_fetcher: Callable[[str], Awaitable[Any]] | None = None,
        task_sender: Callable[[str, dict[str, Any]], Awaitable[Any]] | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.remote_registry_url = self._select_registry_url(
            remote_registry_url, registry_endpoint, registry_addr
        )
        self.self_agent_id = self_agent_id
        self.registry_headers = dict(registry_headers or {})
        self.request_timeout_s = request_timeout_s
        self.http_client = http_client or httpx.AsyncClient()
        self._trace: list[TraceEvent] = []
        self._local_registry: dict[str, AgentRegistryEntry] = {}
        self._remote_registry: dict[str, AgentRegistryEntry] = {}
        self._local_handlers: dict[str, Callable[..., Any]] = {}
        self._linguistic_handlers: dict[str, Callable[..., Any]] = {}
        self.a2a_adapter = A2AAdapter(
            card_fetcher or self._fetch_agent_card,
            task_sender or self._send_remote_a2a_task,
        )

    async def refresh_registry(self) -> list[AgentRegistryEntry]:
        """Refresh remote registry entries and return the visible snapshot."""
        if not self.remote_registry_url:
            self._record(
                "registry_refresh_skipped",
                "No remote registry URL configured.",
            )
            return self._visible_registry()

        self._record(
            "registry_refresh_started",
            "Refreshing remote registry.",
            url=self.remote_registry_url,
        )
        try:
            response = await self._maybe_await(
                self.http_client.get(
                    self.remote_registry_url,
                    headers=self.registry_headers or None,
                    timeout=self.request_timeout_s,
                )
            )
            self._raise_for_status(response)
            payload = await self._maybe_await(response.json())
            entries = await self._entries_from_registry_payload(payload)
        except RemoteRegistryError as exc:
            self._record(
                "registry_refresh_failed",
                "Remote registry refresh failed.",
                error=str(exc),
            )
            raise
        except ValueError as exc:
            error = RemoteRegistryError(f"Remote registry returned invalid JSON: {exc}")
            self._record(
                "registry_refresh_failed",
                "Remote registry refresh failed.",
                error=str(error),
            )
            raise error from exc
        except httpx.HTTPError as exc:
            error = RemoteRegistryError(f"Remote registry request failed: {exc}")
            self._record(
                "registry_refresh_failed",
                "Remote registry refresh failed.",
                error=str(error),
            )
            raise error from exc

        self._remote_registry = {entry.agent_id: entry for entry in entries}
        self._record(
            "registry_refresh_completed",
            "Remote registry refreshed.",
            count=len(entries),
        )
        return self._visible_registry()

    async def registry_snapshot(self, refresh: bool = False) -> list[AgentRegistryEntry]:
        """Return the current admitted agent records."""
        if refresh:
            return await self.refresh_registry()
        return self._visible_registry()

    async def capability_index(
        self, refresh: bool = False
    ) -> dict[str, list[AgentRegistryEntry]]:
        """Return agents grouped by advertised capability name."""
        registry = await self.registry_snapshot(refresh=refresh)
        index: dict[str, list[AgentRegistryEntry]] = {}
        for agent in registry:
            for skill in agent.skills:
                index.setdefault(skill.name, []).append(agent)
        return index

    async def register_a2a_agent(
        self, agent_card_url: str, trust_level: str = "standard"
    ) -> AgentRegistryEntry:
        """Admit a remote A2A agent from an Agent Card URL."""
        entry = await self.a2a_adapter.register_from_card_url(agent_card_url)
        entry.agent_kind = "remote_a2a"
        entry.trust_level = trust_level
        entry.invocation_endpoint = entry.invocation_endpoint or entry.service_endpoint
        self._local_registry[entry.agent_id] = entry
        self._record(
            "sdk_agent_registered",
            f"Registered {entry.agent_id}.",
            agent_id=entry.agent_id,
            url=agent_card_url,
        )
        return entry

    def register_local_agent(
        self,
        name: str,
        capabilities: list[CapabilityRequirement],
        handler: Callable[..., Any],
        *,
        agent_id: str | None = None,
        description: str = "",
        trust_level: str = "standard",
        status: str = "available",
        validation_contract: dict[str, Any] | None = None,
    ) -> AgentRegistryEntry:
        """Register a local Python handler as an SDK-managed agent."""
        entry = self._handler_entry(
            name=name,
            capabilities=capabilities,
            handler=handler,
            agent_kind="local_python",
            scheme="local",
            agent_id=agent_id,
            description=description,
            trust_level=trust_level,
            status=status,
            validation_contract=validation_contract,
        )
        self._local_handlers[entry.agent_id] = handler
        self._record(
            "sdk_local_agent_registered",
            f"Registered local agent {entry.agent_id}.",
            agent_id=entry.agent_id,
        )
        return entry

    def register_linguistic_agent(
        self,
        name: str,
        capabilities: list[CapabilityRequirement],
        handler: Callable[..., Any],
        *,
        agent_id: str | None = None,
        description: str = "",
        trust_level: str = "standard",
        status: str = "available",
        validation_contract: dict[str, Any] | None = None,
    ) -> AgentRegistryEntry:
        """Register a linguistic runtime as an SDK-managed capability provider."""
        entry = self._handler_entry(
            name=name,
            capabilities=capabilities,
            handler=handler,
            agent_kind="linguistic",
            scheme="linguistic",
            agent_id=agent_id,
            description=description,
            trust_level=trust_level,
            status=status,
            validation_contract=validation_contract,
        )
        self._linguistic_handlers[entry.agent_id] = handler
        self._record(
            "sdk_linguistic_agent_registered",
            f"Registered linguistic agent {entry.agent_id}.",
            agent_id=entry.agent_id,
        )
        return entry

    async def invoke_agent(
        self,
        agent_id: str,
        task: TaskSpec,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> TaskExecutionResult:
        """Invoke one admitted agent through its SDK-managed runtime path."""
        entry = self._agent_by_id(agent_id)
        if entry is None:
            result = TaskExecutionResult(
                task_id=task.task_id,
                agent_id=agent_id,
                status="failed",
                error=f"Unknown agent {agent_id}.",
            )
            self._record(
                "sdk_task_failed",
                f"Task {task.task_id} failed before dispatch.",
                task_id=task.task_id,
                agent_id=agent_id,
                error=result.error,
            )
            return result

        self._record(
            "sdk_task_started",
            f"Task {task.task_id} started.",
            task_id=task.task_id,
            agent_id=agent_id,
            agent_kind=entry.agent_kind,
        )
        try:
            output = await asyncio.wait_for(
                self._invoke_entry(entry, task, payload),
                timeout=timeout_s,
            )
        except TimeoutError:
            result = TaskExecutionResult(
                task_id=task.task_id,
                agent_id=agent_id,
                agent_kind=entry.agent_kind,
                status="timeout",
                error=f"Task {task.task_id} timed out.",
            )
            self._record(
                "sdk_task_timeout",
                result.error,
                task_id=task.task_id,
                agent_id=agent_id,
            )
            return result
        except Exception as exc:
            result = TaskExecutionResult(
                task_id=task.task_id,
                agent_id=agent_id,
                agent_kind=entry.agent_kind,
                status="failed",
                error=str(exc),
            )
            self._record(
                "sdk_task_failed",
                f"Task {task.task_id} failed.",
                task_id=task.task_id,
                agent_id=agent_id,
                error=str(exc),
            )
            return result

        result = self._execution_result(entry, task, output)
        validation_errors = self._validate_execution_result(entry, task, result)
        if validation_errors:
            result = result.model_copy(
                update={
                    "status": "failed",
                    "error": "; ".join(validation_errors),
                    "metadata": {
                        **result.metadata,
                        "validation_errors": validation_errors,
                    },
                }
            )
            self._record(
                "sdk_task_validation_failed",
                f"Task {task.task_id} failed artifact validation.",
                task_id=task.task_id,
                agent_id=agent_id,
                errors=validation_errors,
            )
            return result
        self._record(
            "sdk_task_completed",
            f"Task {task.task_id} completed.",
            task_id=task.task_id,
            agent_id=agent_id,
            artifact_count=len(result.artifacts),
        )
        return result

    async def send_task(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> TaskExecutionResult:
        """Delegate a task only after symbolic authorization."""
        agent_id = task.assigned_to or report.matched_agents.get(task.task_id)
        aux = self._authorized_auxiliary(report, task)
        if report.feasible and aux is not None:
            return self._invoke_auxiliary(aux, task, payload)
        if not report.feasible or not agent_id:
            self._record(
                "sdk_delegation_refused",
                f"Task {task.task_id} was not authorized.",
                task_id=task.task_id,
            )
            raise AuthorizationError(f"Task {task.task_id} is not authorized.")
        return await self.invoke_agent(
            agent_id,
            task,
            payload,
            timeout_s=timeout_s,
        )

    async def send_task_after_authorization(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> TaskExecutionResult:
        """Compatibility wrapper for the newer send_task API."""
        return await self.send_task(report, task, payload, timeout_s=timeout_s)

    def _authorized_auxiliary(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
    ) -> GeneratedNlpAgentSpec | None:
        if not task.auxiliary_spec_id:
            return None
        for spec in report.generated_nlp_agents:
            if spec.spec_id == task.auxiliary_spec_id:
                return spec
        return None

    def _invoke_auxiliary(
        self,
        spec: GeneratedNlpAgentSpec,
        task: TaskSpec,
        payload: dict[str, Any],
    ) -> TaskExecutionResult:
        self._record(
            "sdk_auxiliary_task_started",
            f"Auxiliary task {task.task_id} started.",
            task_id=task.task_id,
            auxiliary_spec_id=spec.spec_id,
            method=spec.method,
        )
        artifact = self._auxiliary_artifact(spec, payload)
        missing = self._missing_required_fields(spec.output_schema, artifact)
        if missing:
            result = TaskExecutionResult(
                task_id=task.task_id,
                agent_id=spec.spec_id,
                agent_kind="auxiliary",
                status="failed",
                output=artifact,
                artifacts=[artifact],
                error=f"Auxiliary output missing required fields: {', '.join(missing)}.",
                metadata={"method": spec.method, "validation_rule": spec.validation_rule},
            )
            self._record(
                "sdk_auxiliary_task_validation_failed",
                result.error,
                task_id=task.task_id,
                auxiliary_spec_id=spec.spec_id,
                missing=missing,
            )
            return result

        result = TaskExecutionResult(
            task_id=task.task_id,
            agent_id=spec.spec_id,
            agent_kind="auxiliary",
            output=artifact,
            artifacts=[artifact],
            metadata={"method": spec.method, "validation_rule": spec.validation_rule},
        )
        self._record(
            "sdk_auxiliary_task_completed",
            f"Auxiliary task {task.task_id} completed.",
            task_id=task.task_id,
            auxiliary_spec_id=spec.spec_id,
        )
        return result

    def _auxiliary_artifact(
        self,
        spec: GeneratedNlpAgentSpec,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        source = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if spec.method == "schema_extraction":
            required = spec.output_schema.get("required", [])
            if isinstance(required, list) and required:
                data = {str(key): source.get(str(key)) for key in required if str(key) in source}
            else:
                data = dict(source)
            return {"kind": "data", "data": data, **data}
        if spec.method == "normalization":
            value = payload.get("value", payload.get("text", payload.get("input", "")))
            normalized = str(value).strip().lower()
            return {"kind": "data", "data": {"normalized": normalized}, "normalized": normalized}
        label = payload.get("label") or payload.get("value") or payload.get("text")
        return {"kind": "data", "data": {"label": label}, "label": label}

    @staticmethod
    def _missing_required_fields(
        schema: dict[str, Any],
        artifact: dict[str, Any],
    ) -> list[str]:
        required = schema.get("required", [])
        if not isinstance(required, list):
            return []
        data = artifact.get("data")
        data_keys = set(data) if isinstance(data, dict) else set()
        artifact_keys = set(artifact)
        return [
            str(field)
            for field in required
            if str(field) not in artifact_keys and str(field) not in data_keys
        ]

    def trace(self) -> list[TraceEvent]:
        """Return SDK and adapter trace events."""
        return [*self._trace, *self.a2a_adapter.trace]

    def reset_session(self) -> None:
        """Clear trace state without unregistering admitted agents."""
        self._trace.clear()
        self.a2a_adapter.trace.clear()

    async def _fetch_agent_card(self, url: str) -> Any:
        response = await self._maybe_await(
            self.http_client.get(url, timeout=self.request_timeout_s)
        )
        self._raise_for_status(response)
        return await self._maybe_await(response.json())

    async def _send_remote_a2a_task(self, agent_id: str, payload: dict[str, Any]) -> Any:
        entry = self._agent_by_id(agent_id)
        if entry is None:
            raise RuntimeError(f"Unknown remote A2A agent {agent_id}.")

        try:
            from a2a.client import ClientConfig, create_client
            from a2a.helpers import new_text_message
            from a2a.types.a2a_pb2 import Role, SendMessageRequest
        except ImportError as exc:
            raise RuntimeError("A2A SDK client support is not available.") from exc

        client = await create_client(
            agent=entry.invocation_endpoint or entry.service_endpoint,
            client_config=ClientConfig(streaming=False),
        )
        try:
            message = new_text_message(
                self._payload_to_text(payload),
                role=Role.ROLE_USER,
            )
            request = SendMessageRequest(message=message)
            chunks: list[Any] = []
            async for chunk in client.send_message(request):
                chunks.append(self._jsonable_output(chunk))
            return {"chunks": chunks}
        finally:
            await client.close()

    async def _entries_from_registry_payload(
        self, payload: Any
    ) -> list[AgentRegistryEntry]:
        if isinstance(payload, list):
            return await self._entries_from_registry_items(payload)
        if not isinstance(payload, dict):
            raise RemoteRegistryError("Remote registry payload must be a JSON object or list.")

        for key in self._CARD_KEYS:
            if key in payload:
                return await self._entries_from_registry_items(
                    self._require_list(payload[key], key)
                )
        for key in self._URL_KEYS:
            if key in payload:
                return await self._entries_from_card_urls(
                    self._require_list(payload[key], key)
                )
        if self._looks_like_card(payload):
            return [self.a2a_adapter.normalize_card(payload)]
        raise RemoteRegistryError("Remote registry JSON shape is not supported.")

    async def _entries_from_registry_items(
        self, items: list[Any]
    ) -> list[AgentRegistryEntry]:
        entries: list[AgentRegistryEntry] = []
        for item in items:
            entries.extend(await self._entries_from_registry_item(item))
        return entries

    async def _entries_from_registry_item(
        self, item: Any
    ) -> list[AgentRegistryEntry]:
        if isinstance(item, str):
            return [await self._entry_from_card_url(item)]
        if not isinstance(item, dict):
            raise RemoteRegistryError("Remote registry item is not supported.")

        if "card" in item:
            return [self.a2a_adapter.normalize_card(item["card"])]
        if "agent_card" in item:
            return [self.a2a_adapter.normalize_card(item["agent_card"])]
        card_url = item.get("card_url") or item.get("agent_card_url")
        if card_url:
            return [await self._entry_from_card_url(str(card_url))]
        if self._looks_like_card(item):
            return [self.a2a_adapter.normalize_card(item)]
        raise RemoteRegistryError("Remote registry item shape is not supported.")

    async def _entries_from_card_urls(self, urls: list[Any]) -> list[AgentRegistryEntry]:
        entries: list[AgentRegistryEntry] = []
        for url in urls:
            if not isinstance(url, str):
                raise RemoteRegistryError("Registry card URLs must be strings.")
            entries.append(await self._entry_from_card_url(url))
        return entries

    async def _entry_from_card_url(self, url: str) -> AgentRegistryEntry:
        return await self.a2a_adapter.register_from_card_url(url)

    def _visible_registry(self) -> list[AgentRegistryEntry]:
        registry = dict(self._remote_registry)
        registry.update(self._local_registry)
        return [
            entry
            for entry in registry.values()
            if not self.self_agent_id or entry.agent_id != self.self_agent_id
        ]

    def _agent_by_id(self, agent_id: str) -> AgentRegistryEntry | None:
        for entry in self._visible_registry():
            if entry.agent_id == agent_id:
                return entry
        return None

    def _handler_entry(
        self,
        *,
        name: str,
        capabilities: list[CapabilityRequirement],
        handler: Callable[..., Any],
        agent_kind: str,
        scheme: str,
        agent_id: str | None,
        description: str,
        trust_level: str,
        status: str,
        validation_contract: dict[str, Any] | None,
    ) -> AgentRegistryEntry:
        normalized_id = agent_id or self._normalize_agent_id(name)
        entry = AgentRegistryEntry(
            agent_id=normalized_id,
            name=name,
            agent_kind=agent_kind,  # type: ignore[arg-type]
            description=description,
            service_endpoint=f"{scheme}://{normalized_id}",
            invocation_endpoint=f"{scheme}://{normalized_id}",
            skills=list(capabilities),
            input_modes=self._capability_modes(capabilities, "input_modes"),
            output_modes=self._capability_modes(capabilities, "output_modes"),
            status=status,  # type: ignore[arg-type]
            trust_level=trust_level,
            validation_contract=dict(validation_contract or {}),
            source_card={"handler": self._handler_name(handler)},
        )
        self._local_registry[entry.agent_id] = entry
        return entry

    async def _invoke_entry(
        self,
        entry: AgentRegistryEntry,
        task: TaskSpec,
        payload: dict[str, Any],
    ) -> Any:
        if entry.agent_kind == "local_python":
            return await self._call_handler(
                self._local_handlers[entry.agent_id],
                task,
                payload,
            )
        if entry.agent_kind == "linguistic":
            return await self._call_handler(
                self._linguistic_handlers[entry.agent_id],
                task,
                payload,
            )
        return await self.a2a_adapter.task_sender(entry.agent_id, payload)

    async def _call_handler(
        self,
        handler: Callable[..., Any],
        task: TaskSpec,
        payload: dict[str, Any],
    ) -> Any:
        signature = inspect.signature(handler)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
            and parameter.default is inspect.Parameter.empty
        ]
        if len(positional) >= 2:
            result = handler(task, payload)
        elif len(positional) == 1:
            result = handler(payload)
        else:
            result = handler()
        return await self._maybe_await(result)

    def _execution_result(
        self,
        entry: AgentRegistryEntry,
        task: TaskSpec,
        output: Any,
    ) -> TaskExecutionResult:
        normalized = self._jsonable_output(output)
        return TaskExecutionResult(
            task_id=task.task_id,
            agent_id=entry.agent_id,
            agent_kind=entry.agent_kind,
            output=normalized,
            artifacts=self._extract_artifacts(normalized),
            metadata={"invocation_endpoint": entry.invocation_endpoint},
        )

    def _validate_execution_result(
        self,
        entry: AgentRegistryEntry,
        task: TaskSpec,
        result: TaskExecutionResult,
    ) -> list[str]:
        contract = self._validation_contract_for(entry, task)
        if not contract:
            return []

        errors: list[str] = []
        required_kinds = contract.get("artifact_kinds", [])
        if isinstance(required_kinds, list):
            available_kinds = {str(artifact.get("kind")) for artifact in result.artifacts}
            missing_kinds = [
                str(kind) for kind in required_kinds if str(kind) not in available_kinds
            ]
            if missing_kinds:
                errors.append(f"missing artifact kinds: {', '.join(missing_kinds)}")

        required_artifacts = contract.get("required_artifacts", [])
        if isinstance(required_artifacts, list):
            missing_artifacts = [
                str(name)
                for name in required_artifacts
                if not self._artifact_named(result.artifacts, str(name))
            ]
            if missing_artifacts:
                errors.append(f"missing artifacts: {', '.join(missing_artifacts)}")

        required_fields = contract.get("required_fields", [])
        if isinstance(required_fields, list):
            missing_fields = [
                str(field)
                for field in required_fields
                if not self._field_present(result.artifacts, str(field))
            ]
            if missing_fields:
                errors.append(f"missing artifact fields: {', '.join(missing_fields)}")

        return errors

    def _validation_contract_for(
        self,
        entry: AgentRegistryEntry,
        task: TaskSpec,
    ) -> dict[str, Any]:
        contract: dict[str, Any] = {}
        contract.update(entry.validation_contract)
        for skill in entry.skills:
            if self._normalize_agent_id(skill.name) == self._normalize_agent_id(
                task.requirement_name
            ):
                contract.update(skill.validation_contract)
                break
        contract.update(task.validation_contract)
        return contract

    @staticmethod
    def _artifact_named(artifacts: list[dict[str, Any]], name: str) -> bool:
        wanted = name.strip().lower()
        for artifact in artifacts:
            values = [
                artifact.get("artifact_id"),
                artifact.get("id"),
                artifact.get("name"),
                artifact.get("kind"),
                artifact.get("type"),
            ]
            if any(str(value).strip().lower() == wanted for value in values if value):
                return True
            data = artifact.get("data")
            if isinstance(data, dict) and wanted in {str(key).lower() for key in data}:
                return True
        return False

    @staticmethod
    def _field_present(artifacts: list[dict[str, Any]], field: str) -> bool:
        wanted = field.strip().lower()
        for artifact in artifacts:
            if wanted in {str(key).lower() for key in artifact}:
                return True
            data = artifact.get("data")
            if isinstance(data, dict) and wanted in {str(key).lower() for key in data}:
                return True
        return False

    def _extract_artifacts(self, output: Any) -> list[dict[str, Any]]:
        if output is None:
            return []
        if isinstance(output, dict):
            artifacts = output.get("artifacts")
            if isinstance(artifacts, list):
                return [self._artifact_record(item) for item in artifacts]
            parts = output.get("parts")
            if isinstance(parts, list):
                return self.a2a_adapter.convert_artifact_parts(parts)
            artifact = output.get("artifact")
            if artifact is not None:
                return [self._artifact_record(artifact)]
            return [output]
        if isinstance(output, list):
            return [self._artifact_record(item) for item in output]
        return [{"kind": "value", "value": output}]

    def _artifact_record(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {"kind": "value", "value": self._jsonable_output(value)}

    def _record(self, event_type: str, message: str, **data: Any) -> TraceEvent:
        event = TraceEvent(event_type=event_type, message=message, data=data)
        self._trace.append(event)
        return event

    @staticmethod
    def _select_registry_url(*values: str | None) -> str | None:
        configured = [value for value in values if value]
        if not configured:
            return None
        if len(set(configured)) > 1:
            raise ValueError("Only one remote registry URL/endpoint/address may be configured.")
        return configured[0]

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _jsonable_output(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [CoordinationSdk._jsonable_output(item) for item in value]
        if isinstance(value, tuple):
            return [CoordinationSdk._jsonable_output(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): CoordinationSdk._jsonable_output(item)
                for key, item in value.items()
            }
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        try:
            from google.protobuf.json_format import MessageToDict
            from google.protobuf.message import Message as ProtobufMessage
        except ImportError:
            ProtobufMessage = None  # type: ignore[assignment]
        if ProtobufMessage is not None and isinstance(value, ProtobufMessage):
            return MessageToDict(value, preserving_proto_field_name=True)
        if hasattr(value, "__dict__"):
            return {
                str(key): CoordinationSdk._jsonable_output(item)
                for key, item in value.__dict__.items()
                if not key.startswith("_")
            }
        return str(value)

    @staticmethod
    def _payload_to_text(payload: dict[str, Any]) -> str:
        for key in ("input", "text", "message", "prompt"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return str(CoordinationSdk._jsonable_output(payload))

    @staticmethod
    def _capability_modes(
        capabilities: list[CapabilityRequirement],
        field_name: str,
    ) -> list[str]:
        modes: list[str] = []
        for capability in capabilities:
            for mode in getattr(capability, field_name):
                if mode not in modes:
                    modes.append(mode)
        return modes

    @staticmethod
    def _normalize_agent_id(name: str) -> str:
        normalized = "".join(
            character.lower() if character.isalnum() else "-"
            for character in name.strip()
        ).strip("-")
        return normalized or "agent"

    @staticmethod
    def _handler_name(handler: Callable[..., Any]) -> str:
        return getattr(handler, "__qualname__", getattr(handler, "__name__", repr(handler)))

    @staticmethod
    def _require_list(value: Any, key: str) -> list[Any]:
        if not isinstance(value, list):
            raise RemoteRegistryError(f"Remote registry field {key!r} must be a list.")
        return value

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        if hasattr(response, "raise_for_status"):
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise RemoteRegistryError(f"Remote registry HTTP error: {exc}") from exc
            return
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            raise RemoteRegistryError(f"Remote registry HTTP error: {status_code}")

    @staticmethod
    def _looks_like_card(value: dict[str, Any]) -> bool:
        card_keys = {
            "agentId",
            "agent_id",
            "id",
            "name",
            "serviceEndpoint",
            "service_endpoint",
            "skills",
            "supportedInterfaces",
            "supported_interfaces",
            "url",
        }
        return bool(card_keys & set(value))
