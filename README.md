# Unified Multi-Agent Coordination

Prototype implementation for the thesis _Unified Multi-Agent Coordination Bridging Large Language Models and Symbolic AI for Autonomous Systems_.

The package separates linguistic semantic selection from symbolic authorization. Raw language is grounded against an authoritative `SemanticCatalog`; typed `ProblemRequest` inputs bypass the LLM. `SymbolicPlanCompiler` owns dependencies, plan construction, bounded provider search, and deterministic ranking. `CoordinationAgent` owns admission, feasibility, authorization, and aggregation. `CoordinationSdk` owns registry discovery, remote A2A admission, local and linguistic runtime wrappers, dispatch, artifact normalization, and traces.

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

## Version 0.3 Defense Study

Version 0.2 is preserved as historical evidence. Final study numbers must come
from `demo_runs/v0.3/`. Generate the historical author-labelled, pre-specified corpus with:

```powershell
uv run unified-generate-defense-corpus --output corpus/v0.3
uv run unified-defense-study --check
```

Collection is fail-closed until `corpus/v0.3/label-provenance.json` records
frozen author labels for the exact corpus hash. No independent adjudicator was
available. Labels are hidden from inference prompts and scoring occurs only
after collection completes. The resulting study measures conformance to the
author's declared framework criteria, not neutral benchmark truth. After freeze,
`uv run unified-defense-study --collect` runs 36 cases at five seeds for each of
`qwen/qwen3-1.7b`, `google/gemma-4-e2b`, and `qwen/qwen3-8b`. Raw outputs are
written to a new immutable run-ID directory. Hidden labels are not placed in
prompts. An interrupted, incomplete directory can be continued without
rewriting validated case files:

```powershell
uv run unified-defense-study --resume demo_runs/v0.3/<run-id>
```

Scoring is a separate command and requires the completion marker:

```powershell
uv run unified-analyze-defense-study demo_runs/v0.3/<run-id> `
  --output demo_runs/v0.3/<run-id>/analysis.json
```

See `REPRODUCING.md` and `evidence-manifest.json` for the complete gates.

The accepted run `20260713T013624Z-b9fd8b6f83` contains all 540 outputs. Its
deterministic analysis is deliberately negative: the hybrid refused every
feasible observation, while LLM-only produced extensive false acceptance and
the structured oracle satisfied the declared criteria. These results support
the fail-closed boundary but not the effectiveness of the current linguistic
plan generator or any general-superiority claim.

## Version 0.4 Bridge and Runtime Evidence

Version 0.3 remains immutable historical evidence. Version 0.4 evaluates the
redesigned boundary: a model emits only a `LinguisticPlanDraft` containing
declared requirement/capability references and requirement-level dependencies.
`PlanHydrator` deterministically selects admitted providers and copies task IDs,
contracts, artifacts, and completion conditions from authoritative inputs. One
repair may receive public hydration errors; labels are never available.

```powershell
uv run unified-generate-defense-corpus-v04 --output corpus/v0.4
uv run unified-defense-study-v04 --corpus corpus/v0.4 --check
uv run unified-defense-study-v04 --corpus corpus/v0.4 --collect
uv run unified-analyze-defense-study-v04 --run demo_runs/v0.4/<run-id>
uv run unified-runtime-ablation-study --output-dir demo_runs/runtime_ablation/<run-id>
```

Runtime-only dependency, auxiliary-admission, and trace-evidence controls are
reported separately from planning comparisons. Unsafe controls are explicit
experimental classes; production constructors retain secure defaults.

## Version 0.7 Production-Path Comparison

The primary comparison is the frozen v0.7 production-path study. Qwen3-1.7B
first receives 48 development-only cases across paraphrase, ambiguity,
negation, trust, artifact contracts, dependencies, provider recovery, and
adversarial wording. These runs may motivate corrections and are never used as
confirmatory accuracy evidence. After the production hashes and protocol are
locked, Qwen3-1.7B, Gemma-4-E2B, and Qwen3-8B run in that order on 64 held-out
cases (32 feasible/infeasible pairs), two seeds, and two arms. The complete
confirmatory matrix contains 768 immutable observations.

```powershell
uv run unified-generate-defense-corpus-v07 --output corpus/v0.7
uv run unified-symbolic-benchmark-v07 --output demo_runs/v0.7/deterministic-benchmark.json
uv run unified-defense-study-v07 --corpus corpus/v0.7 --freeze-protocol
uv run unified-defense-study-v07 --corpus corpus/v0.7 --check
uv run unified-defense-study-v07 --corpus corpus/v0.7 --phase confirmatory --model qwen/qwen3-1.7b
uv run unified-defense-study-v07 --corpus corpus/v0.7 --phase confirmatory --model google/gemma-4-e2b --resume demo_runs/v0.7/<run-id>
uv run unified-defense-study-v07 --corpus corpus/v0.7 --phase confirmatory --model qwen/qwen3-8b --resume demo_runs/v0.7/<run-id>
uv run unified-analyze-defense-study-v07 --run demo_runs/v0.7/<run-id> --corpus corpus/v0.7 --phase confirmatory --benchmark demo_runs/v0.7/deterministic-benchmark.json
```

Confirmatory inference uses temperature 0.2, top-p 1, an 800-token limit, seed
11 as primary, and seed 29 as a stability replication. Invalid schema is a
failed observation without repair. Version 0.5 remains the experiment that
motivated this redesign, and the unexecuted study-specific v0.6 protocol is
preserved as superseded.

## Version 0.8 Follow-up

The versioned v0.8 path adds evidence-grounded semantic admission,
deterministically derived ambiguity, Unicode-aware lexical retrieval, and
constraint-directed provider search while preserving v0.7. Its development
study uses 72 cases, two arms, seeds 11 and 29, and only Qwen3-1.7B and
Gemma-4-E2B (576 observations). Confirmatory collection is intentionally
blocked until the two output-blind human author-review worksheets under
`corpus/v0.8/review/` are completed and reconciled.

The completed primary-seed hybrid has 71.88% balanced accuracy, 44.79%
feasible recall, and one unsafe acceptance, compared with 53.13%, 98.96%, and
89 unsafe acceptances for the direct arm. Only safety superiority and the
deterministic symbolic invariants pass; four of six declared gates fail. The
exact-name/alias control solves all 64 designed cases, so the result does not
establish that an LLM is necessary or generally superior.

The associated consensus-v4 campaign completed all 45 trials. Forty-three
passed, two reconfiguration trials ended in infrastructure timeouts, and no
executed primary invariant failed (140/150 checks executed). Its provenance
records the intentionally preserved dirty worktree, so it is descriptive
complete evidence and is not promoted as accepted clean release evidence.

## Version 0.5 Historical Comparison

The historical defense comparison is the frozen v0.5 dual-arm study. It contains 48
held-out cases (24 feasible/infeasible matched pairs), eight unscored
development examples, and 12 separately scored runtime cases. The hybrid and
direct-LLM arms receive identical public request, registry, payload, and schema
inputs and at most one schema/referential repair call. Hidden author labels are
not opened until the exact 1,440-output collection is complete.

```powershell
uv run unified-generate-defense-corpus-v05 --output corpus/v0.5
uv run unified-defense-study-v05 --corpus corpus/v0.5 --check
uv run unified-defense-study-v05 --corpus corpus/v0.5 --collect
uv run unified-analyze-defense-study-v05 --run demo_runs/v0.5/<run-id> --corpus corpus/v0.5
```

v0.3 and v0.4 remain immutable historical development evidence. A failed
pre-specified v0.5 criterion remains valid evidence when source provenance,
matrix completeness, artifact hashes, and protocol identity all validate.

The accepted v0.5 run is packaged under `e/v5`. It contains all 1,440 outputs
and a deterministic analysis digest of
`5bead15a76a84f66671b9bc14dd7ac6ead9c9a5ce4b434757d94a5e86c093f82`.
The outcome is mixed: the repaired hybrid made zero false acceptances but only
1.39% feasible recall, while the direct arm made 345 false acceptances. The
comparative-advantage criterion is therefore unsupported rather than rerun.

The accepted consensus report is
`demo_runs/consensus/20260715T022727Z-3f8093a-v2/campaign.json`. The evidence is
valid but the outcome failed: 20 of 33 trials passed, seven violated invariants,
and six had infrastructure errors. The earlier complete failed campaign is also
preserved. Current deterministic, Docker A2A, PostgreSQL, and upstream-derived
A2A checks pass 7/7, 11/11, 5/5, and 1/1 respectively.

## Versionless Hybrid Strategy Validation

The production redesign is checked separately from the immutable versioned
studies. The fixed eight-case raw-language subset uses the strict production
semantic schema and shared compiler. Qwen3 1.7B runs all eight cases; after the
implementation gate passes, Gemma E2B and Qwen3 8B run the same four sentinels.
The final developmental run is
`demo_runs/hybrid_strategy_validation/20260717T031509Z-3a9e1f9123`: all 16
outputs were schema-valid without repair, all feasibility decisions were
correct, and no false acceptance or refusal occurred. This tiny,
author-designed result is not evidence of model or pipeline superiority.

```powershell
uv run unified-hybrid-strategy-validation collect --model qwen/qwen3-1.7b --case-set all
uv run unified-hybrid-strategy-validation collect --model google/gemma-4-e2b --case-set sentinel --run <run>
uv run unified-hybrid-strategy-validation collect --model qwen/qwen3-8b --case-set sentinel --run <run>
uv run unified-hybrid-strategy-validation analyze --run <run>
```

The unchanged 48-case corpus behind the accepted 1.39% hybrid recall was also
replayed through the new typed-request production path. Because every public
case already contains an authoritative `ProblemRequest`, this path deliberately
makes zero model calls. The label-hidden replay at
`demo_runs/hybrid_strategy_validation/20260717T035308Z-typed-corpus-f3d214432a`
was 48/48 correct with 100% feasible recall and zero false acceptances or
refusals. It is a descriptive architecture comparison, not a new v0.x study.

```powershell
uv run unified-hybrid-strategy-validation replay-typed-corpus --corpus-root corpus/v0.5
uv run unified-hybrid-strategy-validation analyze-typed-corpus --run <run>
```

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
- `COORDINATION_STORE_URL=etcd://host-a:2379,host-b:2379`: enables the consensus-backed authoritative store and distributed registry.
- `COORDINATION_AUDIT_STORE_URL`: optional PostgreSQL URL for best-effort asynchronous audit projection when etcd is authoritative.
- `COORDINATION_COORDINATOR_ID`: optional stable id for a coordinator replica. A generated id is used when omitted.
- `COORDINATION_LEASE_TTL_S`: session lease TTL in seconds, default `30.0`.
- `COORDINATION_LEASE_RENEW_INTERVAL_S`: in-flight dispatch renewal interval in seconds, default half the lease TTL.
- `COORDINATION_REGISTRY_RETRIES`: registry refresh retries, default `2`.
- `COORDINATION_TASK_RETRIES`: task dispatch retries, default `1`.
- `COORDINATION_SEMANTIC_CATALOG_PATH`: optional default authoritative catalog for raw-language planning.
- `COORDINATION_SEMANTIC_MODEL`: exact model identifier used by strict semantic admission.
- `COORDINATION_SEMANTIC_ENDPOINT`: OpenAI-compatible semantic endpoint, default `http://127.0.0.1:1234/v1`.
- `COORDINATION_SEMANTIC_SEED`: semantic-selection seed, default `11`.
- `COORDINATION_ALLOW_INSECURE_A2A`: explicit development/test exception for HTTP Agent Cards; production admission requires HTTPS by default.
- `COORDINATION_RETRY_BACKOFF_S`: retry backoff in seconds, default `0.05`.
- `COORDINATION_SERVICE_HOST`: service bind host, default `0.0.0.0`.
- `COORDINATION_SERVICE_PORT`: service bind port, default `8000`.
- `COORDINATION_MAX_CONCURRENT_DISPATCHES`: bounds concurrent task dispatches so coordinator work cannot exhaust the process, default `16`.
- `COORDINATION_LEAVE_ON_SHUTDOWN`: submits authenticated graceful leave on shutdown by default; set `false` for restart-oriented supervisors and invoke the leave API explicitly for deliberate scale-down.

Endpoints:

- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /cluster/status`
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
$env:EVIDENCE_RUN_DIR="demo_runs/system/<new-run-id>"
docker compose -f docker-compose.system.yml up --build --abort-on-container-exit --exit-code-from system-tests
```

The test runner requires a new output directory and refuses overwrite. The harness is
deliberately deterministic: it uses local fixture services, no external LLM or
vendor cloud calls, and no Kubernetes or service mesh.

The distributed coordinator harness adds PostgreSQL as a shared coordination
store, three coordinator replicas with distinct coordinator ids, and controlled
lease/fencing recovery checks:

```powershell
$env:EVIDENCE_RUN_DIR="demo_runs/postgres/<new-run-id>"
docker compose -f docker-compose.distributed.yml up --build --abort-on-container-exit --exit-code-from distributed-system-tests
```

The runner writes immutable evidence with lease
conflicts, process-exit recovery, stale-fence rejection, PostgreSQL invariant
checks, fixture-agent duplicate-dispatch counters, recovery latency, and
terminal correctness. It targets crash-fault-tolerant replicated execution; it
does not claim Byzantine consensus or malicious-registry tolerance.

## Consensus-backed coordinator nodes

`docker-compose.etcd-distributed.yml` runs the fault-tolerant topology. Each
coordinator container supervises one Python coordinator process and, when the
authoritative membership assignment requires it, one etcd `v3.6.13` process.
The first node bootstraps only with `COORDINATION_BOOTSTRAP=true`; every later
node joins through DNS/direct API seeds and falls back to authenticated UDP
multicast on `239.255.42.99:7947` when no seed is reachable.

The authoritative voter target is configured with
`COORDINATION_VOTER_TARGET`. Only `1`, `3`, `5`, and `7` are accepted. The
default is `3`, which is also the minimum production recommendation. Target
`1` is a development/degraded profile with no failure tolerance. New members
enter as learners and are promoted only after etcd accepts promotion; nodes
beyond the target remain coordinator-only. Authenticated target changes add or
remove one member per reconciliation cycle. Majority loss is deliberately not
auto-repaired.

Required security variables are:

- `COORDINATION_CLUSTER_ID`: stable logical cluster identity.
- `COORDINATION_CLUSTER_SECRET`: HMAC-SHA256 secret for discovery and coordinator membership.
- `COORDINATION_AGENT_REGISTRATION_SECRET`: separate HMAC-SHA256 secret for agent registration and heartbeats.
- `COORDINATION_DISCOVERY_SEEDS`: comma-separated coordinator API URLs; DNS resolution is performed by the HTTP client.

Etcd client and peer ports (`2379` and `2380`) are intended only for the private
container network. Discovery and membership envelopes reject incompatible
versions, wrong clusters, signatures outside a 30-second timestamp window, and
replayed nonces. The explicit insecure mode exists only for isolated fixtures.

The shared registry uses renewable etcd leases. Remote agents receive logical
operation keys, attempt keys, coordinator fencing tokens, and registry
revisions. Effectful agents must durably retain the highest fence and completed
operation results; agents without fence support are limited to read-only or
explicitly idempotent work. Process-local handlers are rejected from
fault-tolerant plans unless represented as replicated providers.

Run the repeatable 3/5/7 campaign with:

```powershell
uv run unified-consensus-matrix `
  --output-dir demo_runs/consensus/<new-run-id> --trials 3 `
  --promotion-candidate
```

The campaign creates isolated Compose project names, fresh volumes, unique
evidence directories, and separate network faults. It exercises formation at
three, five, and seven voters, reconfiguration, leader and partition faults,
quorum restoration, restart and replacement, audit-sink failure, concurrent
ownership, and controlled crash windows. It records source and dependency
hashes, image/container identities, topology, timing, fences, terminal state,
and duplicate-effect observations. Dirty-source output is retained as
historical but cannot be accepted. The existing JSONL and PostgreSQL harnesses
remain supported as non-consensus adapters and audit projections.
