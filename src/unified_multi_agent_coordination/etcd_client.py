"""Small asynchronous etcd v3 HTTP-gateway client.

The project deliberately uses the stable JSON gateway rather than depending on
an unofficial Python etcd binding.  Values exposed by this module are raw bytes;
serialization belongs to the coordination and registry adapters.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


JsonObject = dict[str, Any]


class EtcdError(RuntimeError):
    """Base error returned by the etcd gateway."""


class EtcdUnavailableError(EtcdError):
    """Raised when no configured endpoint can serve a request."""


class EtcdQuorumUnavailableError(EtcdUnavailableError):
    """Raised when authoritative etcd operations cannot reach a serving quorum."""


@dataclass(frozen=True)
class EtcdKeyValue:
    key: bytes
    value: bytes
    create_revision: int
    mod_revision: int
    version: int
    lease: int


@dataclass(frozen=True)
class EtcdRange:
    values: list[EtcdKeyValue]
    revision: int


@dataclass(frozen=True)
class EtcdWatchEvent:
    event_type: str
    value: EtcdKeyValue
    revision: int


class EtcdClient:
    """Endpoint-failing etcd v3 client using the built-in gRPC gateway."""

    def __init__(
        self,
        endpoints: Iterable[str],
        *,
        timeout_s: float = 5.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        normalized = [_normalize_endpoint(endpoint) for endpoint in endpoints]
        if not normalized:
            raise ValueError("At least one etcd endpoint is required.")
        self.endpoints = normalized
        self.timeout_s = timeout_s
        self.http_client = http_client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = http_client is None
        self._next_endpoint = 0

    async def close(self) -> None:
        if self._owns_client:
            await self.http_client.aclose()

    async def range(self, key: bytes, *, prefix: bool = False) -> EtcdRange:
        payload: JsonObject = {"key": _b64(key)}
        if prefix:
            payload["range_end"] = _b64(_prefix_end(key))
        result = await self.post("/v3/kv/range", payload)
        header = result.get("header") or {}
        return EtcdRange(
            values=[_decode_kv(item) for item in result.get("kvs") or []],
            revision=int(header.get("revision") or 0),
        )

    async def put(self, key: bytes, value: bytes, *, lease: int = 0) -> int:
        payload: JsonObject = {"key": _b64(key), "value": _b64(value)}
        if lease:
            payload["lease"] = str(lease)
        result = await self.post("/v3/kv/put", payload)
        return int((result.get("header") or {}).get("revision") or 0)

    async def delete(self, key: bytes, *, prefix: bool = False) -> int:
        payload: JsonObject = {"key": _b64(key)}
        if prefix:
            payload["range_end"] = _b64(_prefix_end(key))
        result = await self.post("/v3/kv/deleterange", payload)
        return int(result.get("deleted") or 0)

    async def transaction(
        self,
        *,
        compare: list[JsonObject],
        success: list[JsonObject],
        failure: list[JsonObject] | None = None,
    ) -> JsonObject:
        return await self.post(
            "/v3/kv/txn",
            {"compare": compare, "success": success, "failure": failure or []},
        )

    async def grant_lease(self, ttl_s: float) -> int:
        result = await self.post("/v3/lease/grant", {"TTL": max(int(ttl_s), 1)})
        lease_id = int(result.get("ID") or result.get("id") or 0)
        if not lease_id:
            raise EtcdError(f"etcd returned no lease ID: {result}")
        return lease_id

    async def keep_alive(self, lease_id: int) -> int:
        """Renew a lease and return the server-reported TTL.

        The gateway exposes keep-alive as a streaming response.  Reading the
        first response and closing the stream performs one bounded renewal.
        """

        last_error: Exception | None = None
        for endpoint in self._ordered_endpoints():
            try:
                async with self.http_client.stream(
                    "POST",
                    f"{endpoint}/v3/lease/keepalive",
                    json={"ID": str(lease_id)},
                    timeout=self.timeout_s,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        decoded = json.loads(line)
                        body = decoded.get("result") or decoded
                        ttl = int(body.get("TTL") or body.get("ttl") or 0)
                        if ttl <= 0:
                            raise EtcdError(f"Lease {lease_id} no longer exists.")
                        self._remember(endpoint)
                        return ttl
                raise EtcdError("etcd keep-alive stream closed without a response.")
            except (httpx.HTTPError, ValueError, EtcdError) as exc:
                last_error = exc
        raise EtcdQuorumUnavailableError(
            f"No etcd quorum endpoint renewed lease: {last_error}"
        )

    async def revoke_lease(self, lease_id: int) -> None:
        await self.post("/v3/lease/revoke", {"ID": str(lease_id)})

    async def status(self) -> JsonObject:
        return await self.post("/v3/maintenance/status", {})

    async def member_list(self) -> JsonObject:
        result = await self.post("/v3/cluster/member/list", {})
        discovered = [
            url
            for member in result.get("members") or []
            for url in (member.get("clientURLs") or member.get("client_urls") or [])
        ]
        self._merge_endpoints(discovered)
        return result

    async def sync_endpoints(self) -> list[str]:
        """Refresh client endpoints from authoritative etcd membership."""
        await self.member_list()
        return list(self.endpoints)

    async def member_add(self, peer_url: str, *, learner: bool = True) -> JsonObject:
        return await self.post(
            "/v3/cluster/member/add",
            {"peerURLs": [peer_url], "isLearner": learner},
        )

    async def member_promote(self, member_id: int) -> JsonObject:
        return await self.post(
            "/v3/cluster/member/promote",
            {"ID": str(member_id)},
        )

    async def member_remove(self, member_id: int) -> JsonObject:
        return await self.post(
            "/v3/cluster/member/remove",
            {"ID": str(member_id)},
        )

    async def watch(
        self,
        key: bytes,
        *,
        prefix: bool = False,
        start_revision: int = 0,
    ):
        """Yield watch events and resume from the last observed revision."""
        next_revision = max(start_revision, 0)
        while True:
            for endpoint in self._ordered_endpoints():
                create_request: JsonObject = {"key": _b64(key)}
                if prefix:
                    create_request["range_end"] = _b64(_prefix_end(key))
                if next_revision:
                    create_request["start_revision"] = str(next_revision)
                try:
                    async with self.http_client.stream(
                        "POST",
                        f"{endpoint}/v3/watch",
                        json={"create_request": create_request},
                        timeout=None,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            message = json.loads(line)
                            result = message.get("result") or message
                            compact_revision = int(result.get("compact_revision") or 0)
                            if compact_revision:
                                next_revision = compact_revision + 1
                                break
                            header_revision = int(
                                (result.get("header") or {}).get("revision") or 0
                            )
                            for event in result.get("events") or []:
                                item = _decode_kv(event.get("kv") or {})
                                revision = max(item.mod_revision, header_revision)
                                next_revision = max(next_revision, revision + 1)
                                yield EtcdWatchEvent(
                                    event_type=str(event.get("type") or "PUT"),
                                    value=item,
                                    revision=revision,
                                )
                        else:
                            raise EtcdError("etcd watch stream closed.")
                        self._remember(endpoint)
                        break
                except (httpx.HTTPError, ValueError, EtcdError):
                    continue
            else:
                await asyncio.sleep(min(self.timeout_s, 1.0))

    async def post(self, path: str, payload: JsonObject) -> JsonObject:
        last_error: Exception | None = None
        for endpoint in self._ordered_endpoints():
            try:
                response = await self.http_client.post(
                    f"{endpoint}{path}", json=payload, timeout=self.timeout_s
                )
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise EtcdError(f"Invalid etcd response for {path}.")
                if result.get("error"):
                    raise EtcdError(str(result["error"]))
                self._remember(endpoint)
                return result
            except (httpx.HTTPError, ValueError, EtcdError) as exc:
                last_error = exc
        raise EtcdQuorumUnavailableError(
            f"No etcd quorum endpoint served {path}: {last_error}"
        )

    def _ordered_endpoints(self) -> list[str]:
        return [
            self.endpoints[(self._next_endpoint + index) % len(self.endpoints)]
            for index in range(len(self.endpoints))
        ]

    def _remember(self, endpoint: str) -> None:
        self._next_endpoint = self.endpoints.index(endpoint)

    def _merge_endpoints(self, endpoints: Iterable[str]) -> None:
        for endpoint in endpoints:
            normalized = _normalize_endpoint(endpoint)
            if normalized not in self.endpoints:
                self.endpoints.append(normalized)


def compare_version(key: bytes, result: str, version: int) -> JsonObject:
    return {
        "key": _b64(key),
        "target": "VERSION",
        "result": result,
        "version": str(version),
    }


def compare_mod_revision(key: bytes, result: str, revision: int) -> JsonObject:
    return {
        "key": _b64(key),
        "target": "MOD",
        "result": result,
        "mod_revision": str(revision),
    }


def request_put(key: bytes, value: bytes, *, lease: int = 0) -> JsonObject:
    request: JsonObject = {"key": _b64(key), "value": _b64(value)}
    if lease:
        request["lease"] = str(lease)
    return {"request_put": request}


def request_delete(key: bytes) -> JsonObject:
    return {"request_delete_range": {"key": _b64(key)}}


def _decode_kv(value: JsonObject) -> EtcdKeyValue:
    return EtcdKeyValue(
        key=base64.b64decode(value.get("key") or ""),
        value=base64.b64decode(value.get("value") or ""),
        create_revision=int(value.get("create_revision") or 0),
        mod_revision=int(value.get("mod_revision") or 0),
        version=int(value.get("version") or 0),
        lease=int(value.get("lease") or 0),
    )


def _normalize_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    if not normalized:
        raise ValueError("Empty etcd endpoint.")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized


def _prefix_end(prefix: bytes) -> bytes:
    if not prefix:
        return b"\0"
    mutable = bytearray(prefix)
    for index in range(len(mutable) - 1, -1, -1):
        if mutable[index] < 0xFF:
            mutable[index] += 1
            return bytes(mutable[: index + 1])
    return b"\0"


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")
