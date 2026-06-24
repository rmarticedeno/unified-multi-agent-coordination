"""SDK facade for registry discovery and authorized task dispatch."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

from .a2a_adapter import A2AAdapter
from .models import AgentRegistryEntry, FeasibilityReport, TaskSpec, TraceEvent


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
        self.a2a_adapter = A2AAdapter(
            card_fetcher or self._fetch_agent_card,
            task_sender or self._missing_task_sender,
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
        entry.trust_level = trust_level
        self._local_registry[entry.agent_id] = entry
        self._record(
            "sdk_agent_registered",
            f"Registered {entry.agent_id}.",
            agent_id=entry.agent_id,
            url=agent_card_url,
        )
        return entry

    async def send_task_after_authorization(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        payload: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> Any:
        """Delegate a task through the A2A adapter after authorization."""
        return await self.a2a_adapter.send_task_after_authorization(
            report, task, payload, timeout_s=timeout_s
        )

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

    async def _missing_task_sender(self, agent_id: str, payload: dict[str, Any]) -> Any:
        raise NotImplementedError(
            f"No task sender configured for agent {agent_id}."
        )

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
