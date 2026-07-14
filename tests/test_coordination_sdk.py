import json
from types import SimpleNamespace

import httpx
import pytest

from unified_multi_agent_coordination import (
    AuthorizationError,
    CapabilityRequirement,
    CoordinationSdk,
    FeasibilityAnalyzer,
    ProblemRequest,
    RemoteRegistryError,
    SolutionProposal,
    TaskSpec,
    ValidationContract,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://registry.example")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("registry failed", request=request, response=response)

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


def _card(name, skill="summarize", url=None):
    return {
        "name": name,
        "url": url or f"http://{name}.example",
        "skills": [{"id": skill, "description": f"{skill} things"}],
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
    }


@pytest.mark.asyncio
async def test_refresh_registry_accepts_full_cards():
    http_client = FakeHttpClient(FakeResponse({"agents": [_card("summarizer")]}))
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example/agents",
        http_client=http_client,
    )

    snapshot = await sdk.refresh_registry()

    assert [agent.agent_id for agent in snapshot] == ["summarizer"]
    assert snapshot[0].skills[0].name == "summarize"
    assert http_client.requests[0][0] == "http://registry.example/agents"
    assert sdk.trace()[-1].event_type == "registry_refresh_completed"


@pytest.mark.asyncio
async def test_refresh_registry_accepts_wrapped_cards():
    http_client = FakeHttpClient(
        FakeResponse(
            {
                "agent_cards": [
                    {"card": _card("summarizer")},
                    {"agent_card": _card("calculator", skill="calculate")},
                ]
            }
        )
    )
    sdk = CoordinationSdk(registry_endpoint="http://registry.example", http_client=http_client)

    snapshot = await sdk.registry_snapshot(refresh=True)

    assert [agent.agent_id for agent in snapshot] == ["summarizer", "calculator"]


@pytest.mark.asyncio
async def test_refresh_registry_accepts_card_url_lists():
    fetched = []

    async def fetcher(url):
        fetched.append(url)
        return _card("summarizer", url=url)

    http_client = FakeHttpClient(
        FakeResponse({"card_urls": ["http://summarizer.example/.well-known/agent-card.json"]})
    )
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        card_fetcher=fetcher,
        http_client=http_client,
    )

    snapshot = await sdk.refresh_registry()

    assert fetched == ["http://summarizer.example/.well-known/agent-card.json"]
    assert snapshot[0].service_endpoint == fetched[0]


@pytest.mark.asyncio
async def test_refresh_registry_accepts_mixed_items():
    async def fetcher(url):
        return _card("remote-url-agent", skill="extract", url=url)

    http_client = FakeHttpClient(
        FakeResponse(
            {
                "items": [
                    _card("summarizer"),
                    {"card_url": "http://url-agent.example/card.json"},
                    {"agent_card": _card("calculator", skill="calculate")},
                    "http://string-agent.example/card.json",
                ]
            }
        )
    )
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        card_fetcher=fetcher,
        http_client=http_client,
    )

    snapshot = await sdk.refresh_registry()

    assert [agent.agent_id for agent in snapshot] == [
        "summarizer",
        "remote-url-agent",
        "calculator",
    ]


@pytest.mark.asyncio
async def test_refresh_registry_rejects_unknown_shapes():
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        http_client=FakeHttpClient(FakeResponse({"unknown": "shape"})),
    )

    with pytest.raises(RemoteRegistryError):
        await sdk.refresh_registry()

    assert sdk.trace()[-1].event_type == "registry_refresh_failed"


@pytest.mark.asyncio
async def test_refresh_registry_rejects_invalid_json():
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        http_client=FakeHttpClient(FakeResponse(json_error=ValueError("not json"))),
    )

    with pytest.raises(RemoteRegistryError, match="invalid JSON"):
        await sdk.refresh_registry()


def test_registry_locator_aliases_and_conflicts():
    sdk = CoordinationSdk(registry_addr="http://registry.example", http_client=FakeHttpClient(FakeResponse({})))

    assert sdk.remote_registry_url == "http://registry.example"

    with pytest.raises(ValueError):
        CoordinationSdk(
            remote_registry_url="http://one.example",
            registry_endpoint="http://two.example",
            http_client=FakeHttpClient(FakeResponse({})),
        )


def test_remote_a2a_text_payload_preserves_coordination_metadata():
    payload = {
        "text": "summarize this",
        "_coordination": {
            "session_id": "s1",
            "task_id": "t1",
            "idempotency_key": "s1:p1:t1:t1-attempt-1",
        },
    }

    encoded = CoordinationSdk._payload_to_text(payload)

    assert json.loads(encoded) == payload


@pytest.mark.asyncio
async def test_registry_snapshot_filters_self_and_indexes_capabilities():
    http_client = FakeHttpClient(
        FakeResponse({"agents": [_card("coordinator"), _card("summarizer")]})
    )
    sdk = CoordinationSdk(
        remote_registry_url="http://registry.example",
        self_agent_id="coordinator",
        http_client=http_client,
    )

    snapshot = await sdk.registry_snapshot(refresh=True)
    index = await sdk.capability_index()

    assert [agent.agent_id for agent in snapshot] == ["summarizer"]
    assert [agent.agent_id for agent in index["summarize"]] == ["summarizer"]


@pytest.mark.asyncio
async def test_register_a2a_agent_and_reset_session_preserve_registry():
    async def fetcher(url):
        return _card("summarizer", url=url)

    sdk = CoordinationSdk(card_fetcher=fetcher, http_client=FakeHttpClient(FakeResponse({})))

    await sdk.register_a2a_agent("http://summarizer.example/card.json")
    assert sdk.trace()

    sdk.reset_session()

    assert sdk.trace() == []
    assert [agent.agent_id for agent in await sdk.registry_snapshot()] == ["summarizer"]


@pytest.mark.asyncio
async def test_register_local_agent_and_dispatch_normalizes_artifacts():
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    requirement = CapabilityRequirement(name="summarize", output_modes=["text"], validation_contract={"json_schema": {"type": "object"}})

    def handler(task, payload):
        return {
            "artifacts": [
                {
                    "kind": "text",
                    "text": f"{task.requirement_name}: {payload['text']}",
                }
            ]
        }

    entry = sdk.register_local_agent("Summarizer", [requirement], handler)
    task = TaskSpec(task_id="t1", requirement_name="summarize", assigned_to=entry.agent_id)
    report = FeasibilityAnalyzer().check(
        ProblemRequest(
            user_goal="Summarize.",
            requirements=[requirement],
            required_artifacts=["summary"],
        ),
        [entry],
        SolutionProposal(
            tasks=[task],
            execution_order=["t1"],
            expected_artifacts=["summary"],
            completion_criteria=["summary exists"],
        ),
    )

    result = await sdk.send_task(report, task, {"text": "hello"})

    assert result.status == "completed"
    assert result.agent_kind == "local_python"
    assert result.artifacts == [{"kind": "text", "text": "summarize: hello"}]
    assert sdk.trace()[-1].event_type == "sdk_task_completed"


@pytest.mark.asyncio
async def test_register_linguistic_agent_and_dispatch_async_handler():
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    requirement = CapabilityRequirement(name="classify", output_modes=["json"], validation_contract={"json_schema": {"type": "object"}})

    async def handler(payload):
        return {"artifacts": [{"kind": "data", "data": {"label": payload["label"]}}]}

    entry = sdk.register_linguistic_agent("Classifier", [requirement], handler)
    task = TaskSpec(task_id="t1", requirement_name="classify", assigned_to=entry.agent_id)
    report = FeasibilityAnalyzer().check(
        ProblemRequest(
            user_goal="Classify.",
            requirements=[requirement],
            required_artifacts=["label"],
        ),
        [entry],
        SolutionProposal(
            tasks=[task],
            execution_order=["t1"],
            expected_artifacts=["label"],
            completion_criteria=["label exists"],
        ),
    )

    result = await sdk.send_task(report, task, {"label": "finance"})

    assert result.status == "completed"
    assert result.agent_kind == "linguistic"
    assert result.artifacts[0]["data"] == {"label": "finance"}


@pytest.mark.asyncio
async def test_send_task_blocks_without_authorization():
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    requirement = CapabilityRequirement(name="summarize", validation_contract={"json_schema": {"type": "object"}})
    entry = sdk.register_local_agent("Summarizer", [requirement], lambda payload: {})
    task = TaskSpec(task_id="t1", requirement_name="summarize", assigned_to=entry.agent_id)
    report = FeasibilityAnalyzer().check(
        ProblemRequest(user_goal="x", requirements=[requirement]),
        [],
        SolutionProposal(tasks=[task]),
    )

    with pytest.raises(AuthorizationError):
        await sdk.send_task(report, task, {})

    assert sdk.trace()[-1].event_type == "sdk_delegation_refused"


@pytest.mark.asyncio
async def test_runtime_failure_is_returned_and_traced():
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    requirement = CapabilityRequirement(name="summarize", validation_contract={"json_schema": {"type": "object"}})

    def handler(payload):
        raise RuntimeError("handler unavailable")

    entry = sdk.register_local_agent("Summarizer", [requirement], handler)
    task = TaskSpec(task_id="t1", requirement_name="summarize", assigned_to=entry.agent_id)

    result = await sdk.invoke_agent(entry.agent_id, task, {})

    assert result.status == "failed"
    assert result.error == "handler unavailable"
    assert sdk.trace()[-1].event_type == "sdk_task_failed"


@pytest.mark.asyncio
async def test_explicit_validation_contract_can_fail_runtime_artifacts():
    sdk = CoordinationSdk(http_client=FakeHttpClient(FakeResponse({})))
    requirement = CapabilityRequirement(
        name="classify",
        validation_contract={"required_fields": ["label"]},
    )

    entry = sdk.register_local_agent(
        "Classifier",
        [requirement],
        lambda payload: {"artifacts": [{"kind": "data", "data": {"score": 0.8}}]},
    )
    task = TaskSpec(task_id="t1", requirement_name="classify", assigned_to=entry.agent_id)
    report = FeasibilityAnalyzer().check(
        ProblemRequest(
            user_goal="Classify.",
            requirements=[requirement],
            required_artifacts=["label"],
        ),
        [entry],
        SolutionProposal(
            tasks=[task],
            execution_order=["t1"],
            expected_artifacts=["label"],
            completion_criteria=["label exists"],
        ),
    )

    result = await sdk.send_task(report, task, {})

    assert result.status == "failed"
    assert "label" in result.error
    assert sdk.trace()[-1].event_type == "sdk_task_validation_failed"


@pytest.mark.asyncio
async def test_handler_arity_json_conversion_and_artifact_helpers():
    sdk = CoordinationSdk()
    task = TaskSpec(task_id="t", requirement_name="work")

    assert await sdk._call_handler(lambda: "zero", task, {}) == "zero"
    assert await sdk._call_handler(lambda payload: payload["x"], task, {"x": "one"}) == "one"
    assert await sdk._call_handler(lambda received, payload: received.task_id + payload["x"], task, {"x": "two"}) == "ttwo"

    model = ValidationContract(required_artifacts=["summary"])
    assert sdk._jsonable_output(model)["required_artifacts"] == ["summary"]
    assert sdk._jsonable_output((1, {"x": 2})) == [1, {"x": 2}]
    assert sdk._jsonable_output(SimpleNamespace(public="yes", _private="no")) == {"public": "yes"}
    assert sdk._extract_artifacts(None) == []
    assert sdk._extract_artifacts({"artifact": "value"}) == [{"kind": "value", "value": "value"}]
    assert sdk._extract_artifacts([1, {"kind": "text"}]) == [
        {"kind": "value", "value": 1},
        {"kind": "text"},
    ]
    assert sdk._extract_artifacts("scalar") == [{"kind": "value", "value": "scalar"}]
    assert sdk._artifact_named([{"data": {"Summary": "x"}}], "summary") is True
    assert sdk._artifact_named([{"name": "other"}], "summary") is False
    assert sdk._field_present([{"data": {"Score": 1}}], "score") is True
    assert sdk._field_present([{"other": 1}], "score") is False


def test_auxiliary_derivation_payload_and_low_level_helpers():
    sdk = CoordinationSdk()
    payload = {"text": "Email a@example.com, phone +1 (212) 555-0101, revenue: 1,234.50"}
    assert sdk._derive_field("email", {}, payload) == "a@example.com"
    assert "212" in sdk._derive_field("phone", {}, payload)
    assert sdk._derive_field("revenue", {}, payload) == 1234.5
    assert sdk._derive_field("unknown", {}, payload) is None
    assert sdk._derive_field("known", {"known": 4}, payload) == 4
    assert sdk._missing_required_fields({"required": "invalid"}, {}) == []
    assert sdk._missing_required_fields(
        {"required": ["a", "b"]}, {"a": 1, "data": {"b": 2}}
    ) == []

    assert sdk._payload_to_text({"text": "plain"}) == "plain"
    assert sdk._payload_to_text({"value": 1}) == '{"value": 1}'
    encoded = sdk._payload_to_text({"text": "x", "_coordination": {"session_id": "s"}})
    assert "_coordination" in encoded
    assert sdk._idempotency_key_from_payload({}) == ""
    assert sdk._idempotency_key_from_payload({"_coordination": "invalid"}) == ""
    assert sdk._idempotency_key_from_payload(
        {"_coordination": {"idempotency_key": "key"}}
    ) == "key"
    assert sdk._coordination_identity({"_coordination": "invalid"}) == {}
    assert sdk._normalize_agent_id("  Weird Agent!  ") == "weird-agent"
    assert sdk._normalize_agent_id("!!!") == "agent"
    assert sdk._capability_modes(
        [
            CapabilityRequirement(name="a", input_modes=["text", "json"]),
            CapabilityRequirement(name="b", input_modes=["json", "audio"]),
        ],
        "input_modes",
    ) == ["text", "json", "audio"]


class _ProtoLike:
    def HasField(self, field):
        if field == "bad":
            raise ValueError("not a message field")
        return field == "present"


def test_protocol_registry_and_http_error_helpers():
    assert CoordinationSdk._has_proto_field(_ProtoLike(), "present") is True
    assert CoordinationSdk._has_proto_field(_ProtoLike(), "bad") is False
    assert CoordinationSdk._has_proto_field(SimpleNamespace(value=1), "value") is True
    assert CoordinationSdk._has_proto_field(SimpleNamespace(), "value") is False
    assert CoordinationSdk._select_registry_url(None, None) is None
    assert CoordinationSdk._select_registry_url("same", "same") == "same"
    with pytest.raises(RemoteRegistryError, match="must be a list"):
        CoordinationSdk._require_list({}, "agents")
    with pytest.raises(RemoteRegistryError, match="HTTP error: 503"):
        CoordinationSdk._raise_for_status(SimpleNamespace(status_code=503))
    assert CoordinationSdk._looks_like_card({"name": "agent"}) is True
    assert CoordinationSdk._looks_like_card({"unrelated": True}) is False


@pytest.mark.asyncio
async def test_unknown_agent_timeout_and_local_idempotency_paths():
    sdk = CoordinationSdk()
    task = TaskSpec(task_id="t", requirement_name="work")
    unknown = await sdk.invoke_agent("missing", task, {})
    assert unknown.status == "failed"

    async def slow(payload):
        del payload
        await __import__("asyncio").sleep(0.05)
        return {"artifacts": []}

    entry = sdk.register_local_agent("Slow", [], slow)
    timeout = await sdk.invoke_agent(entry.agent_id, task, {}, timeout_s=0.001)
    assert timeout.status == "timeout"

    calls = []
    cached = sdk.register_local_agent(
        "Cached",
        [],
        lambda payload: calls.append(payload) or {"artifacts": []},
    )
    payload = {"_coordination": {"idempotency_key": "stable"}}
    first = await sdk.invoke_agent(cached.agent_id, task, payload)
    second = await sdk.invoke_agent(cached.agent_id, task, payload)
    assert first == second
    assert len(calls) == 1
    assert sdk.trace()[-1].event_type == "sdk_duplicate_task_returned"
