"""Agent registry abstractions and the etcd lease-backed implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol

from pydantic import BaseModel

from .etcd_client import EtcdClient
from .models import AgentRegistryEntry


class RegisteredAgent(BaseModel):
    entry: AgentRegistryEntry
    supports_fencing: bool = False
    owner_node_id: str = ""
    availability_scope: str = "remote"
    trust_result: str = "admitted"
    registration_revision: int = 0
    expires_at: datetime | None = None
    backend_lease_id: int = 0


class AgentRegistry(Protocol):
    revision: int

    async def snapshot(self) -> list[AgentRegistryEntry]: ...

    async def register(
        self,
        record: RegisteredAgent,
        *,
        ttl_s: float = 30.0,
    ) -> RegisteredAgent: ...

    async def heartbeat(self, agent_id: str) -> int: ...

    async def remove(self, agent_id: str) -> None: ...

    async def close(self) -> None: ...


class EtcdAgentRegistry:
    """Authoritative registry whose entries disappear with their leases."""

    def __init__(
        self,
        client: EtcdClient,
        *,
        cluster_id: str,
    ) -> None:
        self.client = client
        self.prefix = f"/umac/{cluster_id}/agents/".encode()
        self.revision = 0

    async def snapshot(self) -> list[AgentRegistryEntry]:
        await self.client.sync_endpoints()
        result = await self.client.range(self.prefix, prefix=True)
        self.revision = result.revision
        records: list[RegisteredAgent] = []
        for item in result.values:
            records.append(RegisteredAgent.model_validate_json(item.value))
        return [
            record.entry
            for record in records
            if record.entry.status == "available"
            and (
                record.availability_scope != "node_local"
                or record.owner_node_id
            )
        ]

    async def records(self) -> list[RegisteredAgent]:
        result = await self.client.range(self.prefix, prefix=True)
        self.revision = result.revision
        return [
            RegisteredAgent.model_validate_json(item.value).model_copy(
                update={"registration_revision": item.create_revision}
            )
            for item in result.values
        ]

    async def register(
        self,
        record: RegisteredAgent,
        *,
        ttl_s: float = 30.0,
    ) -> RegisteredAgent:
        if record.availability_scope == "node_local":
            raise ValueError(
                "Node-local agents cannot enter the fault-tolerant shared registry."
            )
        lease_id = await self.client.grant_lease(ttl_s)
        stored = record.model_copy(
            update={
                "backend_lease_id": lease_id,
                "supports_fencing": record.supports_fencing
                or record.entry.supports_fencing,
                "owner_node_id": record.owner_node_id or record.entry.owner_node_id,
                "availability_scope": record.availability_scope
                or record.entry.availability_scope,
                "expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl_s),
            }
        )
        self.revision = await self.client.put(
            self._key(record.entry.agent_id),
            stored.model_dump_json().encode(),
            lease=lease_id,
        )
        return stored.model_copy(update={"registration_revision": self.revision})

    async def heartbeat(self, agent_id: str) -> int:
        result = await self.client.range(self._key(agent_id))
        if not result.values:
            raise KeyError(f"Agent {agent_id} is not registered.")
        record = RegisteredAgent.model_validate_json(result.values[0].value)
        if not record.backend_lease_id:
            raise ValueError(f"Agent {agent_id} has no renewable registry lease.")
        ttl = await self.client.keep_alive(record.backend_lease_id)
        refreshed = record.model_copy(
            update={"expires_at": datetime.now(timezone.utc) + timedelta(seconds=ttl)}
        )
        self.revision = await self.client.put(
            self._key(agent_id),
            refreshed.model_dump_json().encode(),
            lease=record.backend_lease_id,
        )
        return ttl

    async def remove(self, agent_id: str) -> None:
        result = await self.client.range(self._key(agent_id))
        await self.client.delete(self._key(agent_id))
        if result.values:
            record = RegisteredAgent.model_validate_json(result.values[0].value)
            if record.backend_lease_id:
                await self.client.revoke_lease(record.backend_lease_id)

    async def close(self) -> None:
        await self.client.close()

    def _key(self, agent_id: str) -> bytes:
        if not agent_id or "/" in agent_id:
            raise ValueError("Agent ID must be a non-empty path segment.")
        return self.prefix + agent_id.encode()


class InMemoryAgentRegistry:
    """Small registry used by isolated unit tests."""

    def __init__(self) -> None:
        self._records: dict[str, RegisteredAgent] = {}
        self.revision = 0

    async def snapshot(self) -> list[AgentRegistryEntry]:
        return [record.entry for record in self._records.values()]

    async def register(
        self,
        record: RegisteredAgent,
        *,
        ttl_s: float = 30.0,
    ) -> RegisteredAgent:
        del ttl_s
        self.revision += 1
        self._records[record.entry.agent_id] = record
        return record

    async def heartbeat(self, agent_id: str) -> int:
        if agent_id not in self._records:
            raise KeyError(agent_id)
        return 30

    async def remove(self, agent_id: str) -> None:
        self._records.pop(agent_id, None)
        self.revision += 1

    async def close(self) -> None:
        return None
