"""DNS/direct coordinator discovery with authenticated multicast fallback."""

from __future__ import annotations

import asyncio
import json
import random
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from .cluster import HmacAuthenticator, SignedEnvelope


JsonObject = dict[str, Any]


class ClusterDiscovery:
    def __init__(
        self,
        *,
        cluster_id: str,
        node_id: str,
        authenticator: HmacAuthenticator,
        seeds: list[str] | None = None,
        multicast_group: str = "239.255.42.99",
        multicast_port: int = 7947,
        multicast_hops: int = 1,
        timeout_s: float = 3.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.cluster_id = cluster_id
        self.node_id = node_id
        self.authenticator = authenticator
        self.seeds = [_normalize_seed(seed) for seed in (seeds or []) if seed.strip()]
        self.multicast_group = multicast_group
        self.multicast_port = multicast_port
        self.multicast_hops = multicast_hops
        self.timeout_s = timeout_s
        self.http_client = http_client or httpx.AsyncClient(timeout=1.0)
        self._owns_client = http_client is None

    async def close(self) -> None:
        if self._owns_client:
            await self.http_client.aclose()

    async def discover(self) -> tuple[str | None, str]:
        for seed in self.seeds:
            if await self._reachable(seed):
                return seed, "dns"
        response = await asyncio.to_thread(self._multicast_discover)
        if response is None:
            return None, "none"
        api_url = str(response.payload.get("api_url") or "")
        if not api_url:
            return None, "none"
        return api_url.rstrip("/"), "multicast"

    async def _reachable(self, seed: str) -> bool:
        try:
            response = await self.http_client.get(f"{seed}/health/live")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _multicast_discover(self) -> SignedEnvelope | None:
        request = self.authenticator.sign(
            SignedEnvelope(
                message_type="coordinator_discovery",
                cluster_id=self.cluster_id,
                node_id=self.node_id,
            )
        )
        encoded = request.model_dump_json().encode()
        deadline = time.monotonic() + self.timeout_s
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, self.multicast_hops)
            sock.bind(("", 0))
            for delay in (0.0, 0.5, 1.0):
                if delay:
                    time.sleep(delay + random.uniform(0, 0.1))
                sock.sendto(encoded, (self.multicast_group, self.multicast_port))
            while time.monotonic() < deadline:
                sock.settimeout(max(deadline - time.monotonic(), 0.05))
                try:
                    payload, _ = sock.recvfrom(65535)
                except TimeoutError:
                    break
                try:
                    response = SignedEnvelope.model_validate_json(payload)
                    if response.message_type != "coordinator_discovery_response":
                        continue
                    self.authenticator.verify(
                        response,
                        expected_cluster_id=self.cluster_id,
                    )
                    return response
                except (ValueError, json.JSONDecodeError):
                    continue
        return None


class MulticastDiscoveryResponder:
    """Local-network responder that answers valid multicast requests by unicast."""

    def __init__(
        self,
        *,
        cluster_id: str,
        node_id: str,
        authenticator: HmacAuthenticator,
        response_payload: Callable[[], JsonObject],
        multicast_group: str = "239.255.42.99",
        multicast_port: int = 7947,
    ) -> None:
        self.cluster_id = cluster_id
        self.node_id = node_id
        self.authenticator = authenticator
        self.response_payload = response_payload
        self.multicast_group = multicast_group
        self.multicast_port = multicast_port
        self._stop = threading.Event()
        self._socket: socket.socket | None = None
        self._task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(asyncio.to_thread(self._serve))

    async def stop(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)

    def _serve(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._socket = sock
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", self.multicast_port))
            membership = socket.inet_aton(self.multicast_group) + socket.inet_aton("0.0.0.0")
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
            sock.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    payload, address = sock.recvfrom(65535)
                except TimeoutError:
                    continue
                except OSError:
                    break
                try:
                    request = SignedEnvelope.model_validate_json(payload)
                    if request.message_type != "coordinator_discovery":
                        continue
                    self.authenticator.verify(
                        request,
                        expected_cluster_id=self.cluster_id,
                    )
                    response = self.authenticator.sign(
                        SignedEnvelope(
                            message_type="coordinator_discovery_response",
                            cluster_id=self.cluster_id,
                            node_id=self.node_id,
                            payload=self.response_payload(),
                        )
                    )
                    sock.sendto(response.model_dump_json().encode(), address)
                except (ValueError, json.JSONDecodeError):
                    continue
        finally:
            sock.close()
            self._socket = None


def _normalize_seed(seed: str) -> str:
    normalized = seed.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized
