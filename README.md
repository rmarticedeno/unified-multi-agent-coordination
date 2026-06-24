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

## Service

Start the FastAPI service locally:

```powershell
uv run unified-coordination-service
```

Useful environment variables:

- `COORDINATION_REGISTRY_URL`: optional remote registry URL.
- `COORDINATION_SELF_AGENT_ID`: optional coordinator agent id to filter from snapshots.
- `COORDINATION_REQUEST_TIMEOUT_S`: registry request timeout in seconds.
- `COORDINATION_SERVICE_HOST`: service bind host, default `0.0.0.0`.
- `COORDINATION_SERVICE_PORT`: service bind port, default `8000`.

Endpoints:

- `GET /health`
- `GET /registry`
- `POST /plan`
- `POST /coordinate`
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

Docker support is provided for the service wrapper. The implementation in this revision was not built or tested inside Docker.

Available commands for a later Docker verification pass:

```powershell
docker build -t unified-multi-agent-coordination:local .
docker compose up coordination-service
```

The Docker image starts `unified-coordination-service` and exposes port `8000`.
