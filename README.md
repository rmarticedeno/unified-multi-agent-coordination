# Unified Multi-Agent Coordination

Prototype implementation for the thesis _Unified Multi-Agent Coordination Bridging Large Language Models and Symbolic AI for Autonomous Systems_.

The package separates linguistic proposal from symbolic authorization. `CoordinationAgent` owns planning, feasibility, authorization, and aggregation. `CoordinationSdk` owns registry discovery, remote A2A admission, local and linguistic runtime wrappers, dispatch, artifact normalization, and traces.

## Local Development

Install dependencies with uv:

```powershell
uv sync
```

Run the non-Docker test suite:

```powershell
uv run pytest
```

Pytest is scoped to `tests/` so the vendored `a2a-samples/` tree is not collected by default.

## End-to-End Thesis Scenarios

Run deterministic local scenarios that exercise the full coordination path without external services:

```powershell
uv run --with-editable . unified-coordination-scenarios
```

The runner writes `demo_runs/end_to_end_scenarios.json` by default. Each scenario records the user request, fixed local interpretation/proposal, symbolic authorization result, task dispatches, artifacts, SDK trace events, and ledger events. To run only the three primary thesis showcase scenarios:

```powershell
uv run --with-editable . unified-coordination-scenarios --showcase
```

## Thesis Evidence Analysis

Generate thesis-ready summaries from the preserved deterministic, Docker, and local-LLM reports:

```powershell
uv run --with-editable . unified-thesis-results
```

The command writes `demo_runs/thesis_analysis/summary.json`,
`demo_runs/thesis_analysis/tables.md`, and
`demo_runs/thesis_analysis/tables.tex`.

Local LLM reference checks use the OpenAI-compatible local endpoint at
`http://127.0.0.1:1234` and are deliberately run one model at a time. Only
`qwen/qwen3-1.7b` and `google/gemma-4-e2b` are accepted:

```powershell
lms unload --all
lms load qwen/qwen3-1.7b --identifier qwen/qwen3-1.7b -y
uv run --with-editable . unified-local-llm-reference --model qwen/qwen3-1.7b

lms unload --all
lms load google/gemma-4-e2b --identifier google/gemma-4-e2b -y
uv run --with-editable . unified-local-llm-reference --model google/gemma-4-e2b
```

Each model batch writes a model-specific report under `demo_runs/local_llm/`.
The runner verifies that the endpoint advertises the requested model and that
the completion response identifies the requested model before preserving
outputs.

## Service

Start the FastAPI service locally:

```powershell
uv run unified-coordination-service
```

Useful environment variables:

- `COORDINATION_REGISTRY_URL`: optional remote registry URL.
- `COORDINATION_SELF_AGENT_ID`: optional coordinator agent id to filter from snapshots.
- `COORDINATION_REQUEST_TIMEOUT_S`: registry request timeout in seconds.
- `COORDINATION_LEDGER_PATH`: optional JSONL ledger path for crash recovery.
- `COORDINATION_REGISTRY_RETRIES`: registry refresh retries, default `2`.
- `COORDINATION_TASK_RETRIES`: task dispatch retries, default `1`.
- `COORDINATION_RETRY_BACKOFF_S`: retry backoff in seconds, default `0.05`.
- `COORDINATION_SERVICE_HOST`: service bind host, default `0.0.0.0`.
- `COORDINATION_SERVICE_PORT`: service bind port, default `8000`.

Endpoints:

- `GET /health`
- `GET /registry`
- `POST /plan`
- `POST /coordinate`
- `POST /sessions/{session_id}/resume`
- `POST /feasibility`

Example plan request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/plan `
  -ContentType application/json `
  -Body '{
    "problem": {
      "user_goal": "Summarize the report.",
      "requirements": [{"name": "summarize"}],
      "required_artifacts": ["summary"]
    }
  }'
```

## Docker

Docker support is provided for the service wrapper and for a deterministic
multi-container A2A system harness. The ordinary compose file starts only the
coordinator service:

```powershell
docker build -t unified-multi-agent-coordination:local .
docker compose up coordination-service
```

The system harness starts one coordinator, one card registry, multiple
independent A2A fixture-agent containers, a coordinator pointed at a missing
registry, and a test-runner container. All system checks call HTTP endpoints
inside the Docker network; they do not inject SDK objects in-process.

Acceptance command:

```powershell
docker compose -f docker-compose.system.yml up --build --abort-on-container-exit --exit-code-from system-tests
```

The test runner writes `demo_runs/docker_system_report.json`. The harness is
deliberately deterministic: it uses local fixture services, no external LLM or
vendor cloud calls, and no Kubernetes, service mesh, or database.
