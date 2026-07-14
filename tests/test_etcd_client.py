import base64
from unittest.mock import AsyncMock

import httpx
import pytest

from unified_multi_agent_coordination.etcd_client import (
    EtcdClient,
    EtcdError,
    EtcdQuorumUnavailableError,
    _is_quorum_error,
    _normalize_endpoint,
    _prefix_end,
    compare_mod_revision,
    compare_version,
    request_delete,
    request_put,
)


@pytest.mark.asyncio
async def test_etcd_crud_lease_membership_and_endpoint_discovery(monkeypatch):
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    client = EtcdClient(["one:2379"], http_client=http_client)
    responses = [
        {
            "header": {"revision": "7"},
            "kvs": [
                {
                    "key": base64.b64encode(b"/x/a").decode(),
                    "value": base64.b64encode(b"value").decode(),
                    "create_revision": "2",
                    "mod_revision": "7",
                    "version": "3",
                    "lease": "4",
                }
            ],
        },
        {"header": {"revision": "8"}},
        {"deleted": "2"},
        {"succeeded": True},
        {"ID": "55"},
        {},
        {"leader": "1"},
        {"members": [{"clientURLs": ["http://two:2379"]}]},
        {"members": [{"client_urls": ["three:2379"]}]},
        {"member": {"ID": "2"}},
        {},
        {},
    ]
    mocked_post = AsyncMock(side_effect=responses)
    monkeypatch.setattr(client, "post", mocked_post)

    result = await client.range(b"/x/", prefix=True)
    assert result.revision == 7
    assert result.values[0].value == b"value"
    assert await client.put(b"k", b"v", lease=4) == 8
    assert await client.delete(b"/x/", prefix=True) == 2
    assert (await client.transaction(compare=[], success=[]))["succeeded"] is True
    assert await client.grant_lease(0.1) == 55
    await client.revoke_lease(55)
    assert (await client.status())["leader"] == "1"
    await client.member_list()
    assert await client.sync_endpoints() == [
        "http://one:2379",
        "http://two:2379",
        "http://three:2379",
    ]
    assert (await client.member_add("http://peer:2380"))["member"]["ID"] == "2"
    await client.member_promote(2)
    await client.member_remove(2)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_grant_lease_requires_server_id(monkeypatch):
    client = EtcdClient(["one"], http_client=httpx.AsyncClient())
    monkeypatch.setattr(client, "post", AsyncMock(return_value={}))
    with pytest.raises(EtcdError, match="no lease ID"):
        await client.grant_lease(30)
    await client.http_client.aclose()


@pytest.mark.asyncio
async def test_post_handles_invalid_semantic_and_quorum_responses():
    responses = {
        "/semantic": httpx.Response(400, json={"error": "bad compare"}),
        "/quorum": httpx.Response(503, json={"error": "etcdserver: no leader"}),
        "/invalid": httpx.Response(200, json=["not", "an", "object"]),
        "/embedded": httpx.Response(200, json={"error": "permission denied"}),
        "/text": httpx.Response(200, text="not-json"),
    }

    def handler(request):
        response = responses[request.url.path]
        response.request = request
        return response

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = EtcdClient(["http://one"], http_client=http_client)
    with pytest.raises(EtcdError, match="bad compare"):
        await client.post("/semantic", {})
    with pytest.raises(EtcdQuorumUnavailableError):
        await client.post("/quorum", {})
    with pytest.raises(EtcdError, match="Invalid etcd response"):
        await client.post("/invalid", {})
    with pytest.raises(EtcdError, match="permission denied"):
        await client.post("/embedded", {})
    assert await client.post("/text", {}) == {}
    await http_client.aclose()


class _StreamResponse:
    def __init__(self, lines, *, status=200):
        self.lines = lines
        self.status_code = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://one")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failure", request=request, response=response)

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _StreamClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def stream(self, *args, **kwargs):
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_keep_alive_success_and_semantic_failures():
    successful = EtcdClient(
        ["one"],
        http_client=_StreamClient([_StreamResponse(["", '{"result":{"TTL":"9"}}'])]),
    )
    assert await successful.keep_alive(5) == 9

    missing = EtcdClient(
        ["one"],
        http_client=_StreamClient([_StreamResponse(['{"result":{"TTL":"0"}}'])]),
    )
    with pytest.raises(EtcdError, match="no longer exists"):
        await missing.keep_alive(5)

    closed = EtcdClient(["one"], http_client=_StreamClient([_StreamResponse([])]))
    with pytest.raises(EtcdError, match="closed without"):
        await closed.keep_alive(5)


@pytest.mark.asyncio
async def test_owned_http_client_is_closed():
    client = EtcdClient(["one"])
    assert not client.http_client.is_closed
    await client.close()
    assert client.http_client.is_closed


def test_etcd_request_helpers_and_endpoint_validation():
    assert compare_version(b"k", "EQUAL", 1)["version"] == "1"
    assert compare_mod_revision(b"k", "GREATER", 2)["mod_revision"] == "2"
    assert request_put(b"k", b"v")["request_put"].get("lease") is None
    assert request_put(b"k", b"v", lease=3)["request_put"]["lease"] == "3"
    assert request_delete(b"k")["request_delete_range"]
    assert _normalize_endpoint("host:2379/") == "http://host:2379"
    with pytest.raises(ValueError, match="Empty"):
        _normalize_endpoint("  ")
    assert _prefix_end(b"") == b"\0"
    assert _prefix_end(b"a\xff") == b"b"
    assert _prefix_end(b"\xff") == b"\0"
    assert _is_quorum_error("context deadline exceeded") is True
    assert _is_quorum_error("permission denied") is False
    with pytest.raises(ValueError, match="At least one"):
        EtcdClient([])
