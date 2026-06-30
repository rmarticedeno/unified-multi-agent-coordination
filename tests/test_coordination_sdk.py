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
    requirement = CapabilityRequirement(name="summarize", output_modes=["text"])

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
    requirement = CapabilityRequirement(name="classify", output_modes=["json"])

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
    requirement = CapabilityRequirement(name="summarize")
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
    requirement = CapabilityRequirement(name="summarize")

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
