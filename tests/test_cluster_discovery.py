import asyncio
import socket
from types import SimpleNamespace

import httpx

from unified_multi_agent_coordination.cluster import HmacAuthenticator, SignedEnvelope
from unified_multi_agent_coordination.cluster_discovery import (
    ClusterDiscovery,
    MulticastDiscoveryResponder,
    _normalize_seed,
)
from unified_multi_agent_coordination import cluster_discovery


def test_seed_normalization_and_dns_discovery_paths():
    assert _normalize_seed(" host:8000/ ") == "http://host:8000"
    assert _normalize_seed("https://host/") == "https://host"

    async def run():
        def transport(request: httpx.Request) -> httpx.Response:
            status = 200 if request.url.host == "live" else 503
            return httpx.Response(status, request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        discovery = ClusterDiscovery(
            cluster_id="c",
            node_id="n",
            authenticator=HmacAuthenticator("secret"),
            seeds=["dead", "live"],
            http_client=client,
        )
        assert await discovery.discover() == ("http://live", "dns")
        await discovery.close()  # externally owned clients are not closed
        assert not client.is_closed
        await client.aclose()

    asyncio.run(run())


def test_multicast_discovery_none_missing_url_and_success(monkeypatch):
    auth = HmacAuthenticator("secret")
    response = auth.sign(
        SignedEnvelope(
            message_type="coordinator_discovery_response",
            cluster_id="c",
            node_id="peer",
            payload={"api_url": "http://peer/"},
        )
    )

    async def run():
        discovery = ClusterDiscovery(
            cluster_id="c", node_id="n", authenticator=auth, http_client=httpx.AsyncClient()
        )
        monkeypatch.setattr(discovery, "_multicast_discover", lambda: None)
        assert await discovery.discover() == (None, "none")
        monkeypatch.setattr(
            discovery,
            "_multicast_discover",
            lambda: response.model_copy(update={"payload": {}}),
        )
        assert await discovery.discover() == (None, "none")
        monkeypatch.setattr(discovery, "_multicast_discover", lambda: response)
        assert await discovery.discover() == ("http://peer", "multicast")
        await discovery.http_client.aclose()

    asyncio.run(run())


class _DiscoverySocket:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def setsockopt(self, *_args):
        pass

    def bind(self, *_args):
        pass

    def settimeout(self, *_args):
        pass

    def sendto(self, payload, address):
        self.sent.append((payload, address))

    def recvfrom(self, _size):
        if not self.payloads:
            raise TimeoutError
        return self.payloads.pop(0), ("127.0.0.1", 1)


def test_multicast_socket_filters_invalid_and_wrong_messages(monkeypatch):
    auth = HmacAuthenticator("secret")
    wrong = auth.sign(
        SignedEnvelope(message_type="wrong", cluster_id="c", node_id="peer")
    ).model_dump_json().encode()
    valid = auth.sign(
        SignedEnvelope(
            message_type="coordinator_discovery_response",
            cluster_id="c",
            node_id="peer",
            payload={"api_url": "http://peer"},
        )
    ).model_dump_json().encode()
    fake = _DiscoverySocket([b"invalid", wrong, valid])
    fake_socket_module = SimpleNamespace(
        AF_INET=socket.AF_INET,
        SOCK_DGRAM=socket.SOCK_DGRAM,
        IPPROTO_UDP=socket.IPPROTO_UDP,
        IPPROTO_IP=socket.IPPROTO_IP,
        IP_MULTICAST_TTL=socket.IP_MULTICAST_TTL,
        socket=lambda *_args: fake,
    )
    monkeypatch.setattr(cluster_discovery, "socket", fake_socket_module)
    monkeypatch.setattr(cluster_discovery.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(cluster_discovery.random, "uniform", lambda *_args: 0)
    discovery = ClusterDiscovery(
        cluster_id="c", node_id="n", authenticator=auth, timeout_s=10, http_client=object()
    )

    assert discovery._multicast_discover().payload["api_url"] == "http://peer"
    assert len(fake.sent) == 3


class _ResponderSocket:
    def __init__(self, request):
        self.events = [(b"bad", ("127.0.0.1", 1)), (request, ("127.0.0.1", 2))]
        self.sent = []
        self.closed = False

    def setsockopt(self, *_args):
        pass

    def bind(self, *_args):
        pass

    def settimeout(self, *_args):
        pass

    def recvfrom(self, _size):
        if self.events:
            return self.events.pop(0)
        raise OSError("stop")

    def sendto(self, payload, address):
        self.sent.append((payload, address))

    def close(self):
        self.closed = True


def test_multicast_responder_serves_authenticated_request_and_stops(monkeypatch):
    auth = HmacAuthenticator("secret")
    request = auth.sign(
        SignedEnvelope(message_type="coordinator_discovery", cluster_id="c", node_id="peer")
    ).model_dump_json().encode()
    responder = MulticastDiscoveryResponder(
        cluster_id="c",
        node_id="node",
        authenticator=auth,
        response_payload=lambda: {"api_url": "http://node"},
    )

    async def lifecycle():
        responder._serve = lambda: None
        await responder.start()
        await responder.stop()

    asyncio.run(lifecycle())

    fake = _ResponderSocket(request)
    responder = MulticastDiscoveryResponder(
        cluster_id="c",
        node_id="node",
        authenticator=auth,
        response_payload=lambda: {"api_url": "http://node"},
    )
    fake_socket_module = SimpleNamespace(
        AF_INET=socket.AF_INET,
        SOCK_DGRAM=socket.SOCK_DGRAM,
        IPPROTO_UDP=socket.IPPROTO_UDP,
        SOL_SOCKET=socket.SOL_SOCKET,
        SO_REUSEADDR=socket.SO_REUSEADDR,
        IPPROTO_IP=socket.IPPROTO_IP,
        IP_ADD_MEMBERSHIP=socket.IP_ADD_MEMBERSHIP,
        inet_aton=lambda _value: b"1234",
        socket=lambda *_args: fake,
    )
    monkeypatch.setattr(cluster_discovery, "socket", fake_socket_module)

    responder._serve()
    payload = SignedEnvelope.model_validate_json(fake.sent[0][0])
    assert payload.message_type == "coordinator_discovery_response"
    assert fake.sent[0][1] == ("127.0.0.1", 2)
    assert fake.closed
