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

## Version 0.2 Defense Study

Reports outside `demo_runs/v0.2/` are legacy evidence and cannot supply final
thesis numbers. Generate and validate the blinded 36-case corpus with:

```powershell
uv run unified-generate-defense-corpus --output corpus/v0.2
uv run unified-defense-study --check
```

Collection is fail-closed until a qualified independent reviewer completes
`corpus/v0.2/label-signoff.json` for the exact corpus hash. After sign-off,
`uv run unified-defense-study --collect` runs 36 cases at five seeds for each of
`qwen/qwen3-1.7b`, `google/gemma-4-e2b`, and `qwen/qwen3-8b`. Raw outputs are
written to a new immutable run-ID directory. Hidden labels are not placed in
prompts. Scoring is a separate command:

```powershell
uv run unified-analyze-defense-study demo_runs/v0.2/<run-id> `
  --output demo_runs/v0.2/<run-id>/analysis.json
```

See `REPRODUCING.md` and `evidence-manifest.json` for the complete gates.

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
- `COORDINATION_STORE_URL`: optional coordination store URL. PostgreSQL URLs enable the shared durable store; omitted values keep the JSONL/testing backend.
- `COORDINATION_COORDINATOR_ID`: optional stable id for a coordinator replica. A generated id is used when omitted.
- `COORDINATION_LEASE_TTL_S`: session lease TTL in seconds, default `30.0`.
- `COORDINATION_LEASE_RENEW_INTERVAL_S`: in-flight dispatch renewal interval in seconds, default half the lease TTL.
- `COORDINATION_REGISTRY_RETRIES`: registry refresh retries, default `2`.
- `COORDINATION_TASK_RETRIES`: task dispatch retries, default `1`.
- `COORDINATION_ALLOW_INSECURE_A2A`: explicit development/test exception for HTTP Agent Cards; production admission requires HTTPS by default.
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

When a live lease is held by another coordinator replica, `POST /coordinate` and
`POST /sessions/{session_id}/resume` return HTTP `409` instead of silently
taking over the session.

Run the PostgreSQL store migration explicitly when using an externally managed
database:

```powershell
$env:COORDINATION_STORE_URL = "postgresql://postgres:postgres@localhost:5432/coordination"
uv run --with-editable . unified-coordination-store-migrate
```

Example plan request:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8000/plan `
  -ContentType application/json `
  -Body '{
    "problem": {
      "user_goal": "Summarize the report.",
      "requirements": [{
        "name": "summarize",
        "validation_contract": {"required_artifacts": ["summary"]}
      }],
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

The test runner writes to the configured immutable version 0.2 preflight path. The harness is
deliberately deterministic: it uses local fixture services, no external LLM or
vendor cloud calls, and no Kubernetes or service mesh.

The distributed coordinator harness adds PostgreSQL as a shared coordination
store, three coordinator replicas with distinct coordinator ids, and controlled
lease/fencing recovery checks:

```powershell
docker compose -f docker-compose.distributed.yml up --build --abort-on-container-exit --exit-code-from distributed-system-tests
```

The runner writes to the configured immutable version 0.2 preflight path with lease
conflicts, process-exit recovery, stale-fence rejection, PostgreSQL invariant
checks, fixture-agent duplicate-dispatch counters, recovery latency, and
terminal correctness. It targets crash-fault-tolerant replicated execution; it
does not claim Byzantine consensus or malicious-registry tolerance.
