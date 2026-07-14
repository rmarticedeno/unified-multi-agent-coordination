# Distributed Coordinator Migration Fine-Grained Step-by-Step Plan

This plan converts the distributed-coordinator migration idea into an
implementation and thesis-update checklist. The goal is to move from one
coordinator process as a single point of execution to multiple coordinator
replicas governed by durable ownership, recovery, and idempotency rules.

## Target Outcome

The migration is successful when multiple coordinator processes can safely
serve the same logical coordination system while preserving one authoritative
decision path per session.

The target is crash-fault-tolerant coordination through shared durable state,
leases, fencing tokens, idempotent dispatch, and explicit recovery rules. It is
not Byzantine fault tolerance unless a consensus protocol and adversarial tests
are added later.

## Must-Follow Rules

1. Preserve the symbolic authorization boundary.
2. Preserve the SDK as the controlled dispatch boundary.
3. Do not dispatch any task before an authorization record exists.
4. Do not allow two active coordinators to dispatch work for the same session.
5. Do not accept writes from stale coordinator replicas.
6. Do not silently repeat unsafe external work after failover.
7. Do not mutate an authorized plan invisibly; replanning must create a new plan
   generation.
8. Preserve registry snapshots used for authorization.
9. Preserve traceability from user request to plan, authorization, task attempt,
   artifact, and terminal result.
10. Keep JSONL as a local/testing backend only after a durable shared store is
    introduced.
11. Treat distributed availability claims as incomplete until a multi-replica
    harness proves failover behavior.
12. Do not claim Byzantine fault tolerance, adversarial resistance, or
    production-grade reliability without corresponding protocol and evidence.

## Step-by-Step Outline

### Step 0. Baseline The Current Single-Coordinator Behavior

Description:
Before changing the architecture, preserve the current behavior and make sure
all existing evidence still passes. This gives the migration a regression
baseline.

Must-follow steps:

1. Run the non-Docker test suite:

   ```powershell
   uv run pytest
   ```

2. Regenerate deterministic scenarios:

   ```powershell
   uv run --with-editable . unified-coordination-scenarios
   ```

3. Run the existing Docker system harness if Docker is available:

   ```powershell
   docker compose -f docker-compose.system.yml up --build --abort-on-container-exit --exit-code-from system-tests
   ```

4. Save the current reports under `demo_runs/`.
5. Record current behavior that must not regress:
   - explicit symbolic authorization
   - explicit symbolic refusal
   - SDK-mediated dispatch
   - trace event preservation
   - ledger event preservation
   - runtime failure classification
   - session resume behavior

Deliverables:

- Baseline passing test and evidence reports.
- Regression list for behaviors that must remain true.

Verification evidence:

- Passing pytest summary.
- Fresh `demo_runs/end_to_end_scenarios.json`.
- Fresh `demo_runs/docker_system_report.json`, if Docker is available.

### Step 1. Define Distributed Coordination Terms

Description:
Clarify the architecture vocabulary before implementation so the code and
thesis do not overclaim.

Must-follow steps:

1. Define "single logical coordinator" as one authoritative decision path per
   session.
2. Define "coordinator replica" as a process that can acquire a session lease
   and continue coordination.
3. Define "distributed coordinator" as replicated coordinator processes using a
   shared durable coordination store.
4. Define "decentralized consensus" as a stronger design that is out of scope
   unless a consensus protocol is implemented.
5. Add these definitions to developer notes before updating the thesis.

Deliverables:

- Terminology note or design section.

Verification evidence:

- A written distinction among logical coordinator, coordinator replica,
  distributed coordinator, and decentralized consensus.

### Step 2. Make Session And Task State Explicit

Description:
Replace implicit event interpretation with explicit state models that can be
checked during recovery and failover.

Must-follow steps:

1. Add a `SessionState` enum with at least:
   - `new`
   - `planning`
   - `authorized`
   - `dispatching`
   - `recovering`
   - `completed`
   - `failed`
   - `infeasible`
2. Add a `TaskState` enum with at least:
   - `pending`
   - `leased`
   - `running`
   - `completed`
   - `failed`
   - `timeout`
   - `skipped`
3. Define allowed state transitions.
4. Reject invalid transitions in code.
5. Fold persisted events into the same state model used by live coordination.
6. Add invariant checks:
   - one terminal result per session
   - one authorized plan per plan generation
   - no dispatch before authorization
   - no task completion without an attempt record
   - no artifact without task-attempt provenance

Deliverables:

- State enums.
- Transition validator.
- State-folding tests.
- Invariant tests.

Verification evidence:

- Unit tests for valid transitions.
- Unit tests for rejected invalid transitions.
- Tests proving event replay reconstructs the expected session state.

### Step 3. Extract A Coordination Store Interface

Description:
Separate durable coordination state from the current JSONL ledger so the system
can support both local testing and shared distributed storage.

Must-follow steps:

1. Define a `CoordinationStore` interface.
2. Keep append-only event recording as one responsibility.
3. Add transactional state responsibilities:
   - create session
   - get session state
   - append event conditionally
   - acquire lease
   - renew lease
   - release lease
   - create plan generation
   - record authorization decision
   - create task commitment
   - create task attempt
   - record task result
   - record terminal result
4. Implement a JSONL-backed adapter for local compatibility.
5. Move code that assumes a file ledger behind the interface.
6. Ensure all coordinator code depends on the interface, not the concrete JSONL
   implementation.

Deliverables:

- `CoordinationStore` interface.
- JSONL compatibility backend.
- Coordinator code using the interface.

Verification evidence:

- Existing tests pass through the store interface.
- No coordinator code imports the concrete ledger for state decisions except
  through the adapter boundary.

### Step 4. Add A PostgreSQL Durable Store Backend

Description:
Introduce a shared store with transactions, uniqueness constraints, row-level
locking, and conditional updates.

Must-follow steps:

1. Add PostgreSQL schema migrations.
2. Create tables for:
   - sessions
   - plans
   - registry snapshots
   - task commitments
   - task attempts
   - artifacts
   - leases
   - events
   - terminal results
3. Add unique constraints for:
   - one active lease per session
   - one terminal result per session
   - one task-attempt result per attempt ID
   - one active authorized plan per plan generation
4. Add foreign keys linking artifacts to task attempts.
5. Add indexes for session lookup and recovery scans.
6. Add transaction boundaries around authorization, dispatch commitment, and
   terminal result writes.
7. Keep append-only events for audit, but use transactional tables for current
   state.

Deliverables:

- PostgreSQL schema.
- Migration scripts.
- PostgreSQL implementation of `CoordinationStore`.
- Store tests that run against PostgreSQL.

Verification evidence:

- Migration command succeeds.
- Store interface tests pass against JSONL and PostgreSQL backends.
- Constraint tests prove duplicates are rejected.

### Step 5. Add Coordinator Identity

Description:
Every coordinator process must have a stable identity so leases, heartbeats,
and stale writes can be attributed.

Must-follow steps:

1. Add a required or generated `coordinator_id`.
2. Include `coordinator_id` in:
   - lease records
   - task attempt metadata
   - trace events
   - store events
   - recovery logs
3. Expose the current coordinator ID in service diagnostics.
4. Ensure every replica has a unique ID in Docker.

Deliverables:

- Coordinator identity configuration.
- Coordinator ID propagation in events and traces.

Verification evidence:

- Tests show coordinator IDs in persisted events.
- Docker logs show distinct coordinator IDs for each replica.

### Step 6. Add Session Leases

Description:
Ensure that only one coordinator replica owns a session at a time.

Must-follow steps:

1. Add lease fields:
   - `session_id`
   - `holder_id`
   - `fencing_token`
   - `expires_at`
   - `heartbeat_at`
2. Require a valid lease before:
   - planning
   - authorization
   - dispatch
   - retry
   - recovery
   - terminal completion
3. Implement lease acquisition with an atomic conditional update.
4. Implement lease renewal.
5. Implement lease expiry.
6. Implement lease release on clean completion.
7. Ensure a standby can acquire an expired lease.
8. Ensure a non-holder cannot write session state.

Deliverables:

- Lease data model.
- Lease acquisition and renewal logic.
- Lease enforcement in coordinator operations.

Verification evidence:

- Two coordinators competing for one session cannot both acquire an active
  lease.
- A standby acquires the session only after expiry.
- A non-holder write is rejected.

### Step 7. Add Fencing Tokens

Description:
Prevent stale coordinator replicas from writing after a newer owner has taken
over.

Must-follow steps:

1. Increment `fencing_token` on every successful lease acquisition.
2. Store the current token in session state.
3. Include the token in every conditional write.
4. Include the token in every task dispatch metadata payload.
5. Reject writes with stale tokens.
6. Record stale-write rejections as traceable events.

Deliverables:

- Fencing token lifecycle.
- Conditional write enforcement.
- Stale-write trace events.

Verification evidence:

- Test where coordinator A loses lease, coordinator B takes over, and A's later
  write is rejected.
- Test where stale dispatch completion cannot overwrite the terminal result.

### Step 8. Persist Registry Snapshots And Plan Generations

Description:
Keep authorization tied to the exact capabilities and trust facts used when the
plan was approved.

Must-follow steps:

1. Persist a registry snapshot for each planning attempt.
2. Compute and store a registry snapshot hash.
3. Link each plan generation to its registry snapshot.
4. Link each authorization decision to its plan generation.
5. Reject dispatch for tasks not present in the authorized plan generation.
6. On registry change, require a new plan generation before using new agents.

Deliverables:

- Registry snapshot table or store record.
- Plan generation records.
- Authorization-to-snapshot linkage.

Verification evidence:

- Test where an agent disappears after authorization.
- Test where a new compatible agent appears after authorization but is not used
  unless replanning occurs.
- Trace output identifies the registry snapshot hash.

### Step 9. Add Idempotency Keys

Description:
Prevent duplicate unsafe work when requests are retried or a standby recovers
after an uncertain dispatch.

Must-follow steps:

1. Define the idempotency key format:

   ```text
   session_id + plan_id + task_id + attempt_id
   ```

2. Persist idempotency keys before dispatch.
3. Send idempotency keys through SDK adapters when the protocol supports
   metadata.
4. Add SDK-side idempotency handling for local and linguistic agents.
5. Define duplicate behavior:
   - return prior completed result when safe
   - return duplicate-detected refusal when unsafe
   - never silently repeat unsafe external work
6. Store duplicate detection outcomes.

Deliverables:

- Idempotency key generation.
- Idempotency persistence.
- SDK adapter support.
- Duplicate-attempt policy.

Verification evidence:

- Duplicate local-agent dispatch test.
- Duplicate remote-fixture dispatch test.
- Recovery test where a coordinator crashes after dispatch and standby does not
  blindly repeat unsafe work.

### Step 10. Add Task Commitments And Artifact Provenance

Description:
Make every authorized task, attempt, result, and artifact traceable.

Must-follow steps:

1. Create task commitments when a plan is authorized.
2. Link every commitment to:
   - session ID
   - plan generation
   - task ID
   - required capability
   - selected agent
   - dependency set
3. Create task attempts before dispatch.
4. Link every attempt to one commitment.
5. Link every artifact to one attempt.
6. Reject terminal aggregation if required artifacts are missing.
7. Reject artifact reuse across unrelated attempts unless explicitly marked as
   cached and safe.

Deliverables:

- Commitment records.
- Attempt records.
- Artifact provenance records.
- Aggregation validation.

Verification evidence:

- Trace completeness tests.
- Tests for missing artifact rejection.
- Tests for artifact-to-attempt linkage.

### Step 11. Implement Takeover Recovery

Description:
Allow a standby coordinator to continue a session after lease expiry without
losing the authorization boundary or duplicating completed work.

Must-follow steps:

1. On takeover, read session state from the durable store.
2. Reconstruct task commitments and attempts.
3. Skip completed tasks.
4. Resume pending tasks whose dependencies are satisfied.
5. Mark expired in-flight attempts according to policy:
   - `unknown` if the external side effect may have happened
   - `timeout` if no dispatch occurred or the operation is known safe to retry
6. Require idempotency checks before retrying uncertain work.
7. Record recovery decisions as events.
8. Continue only while holding a valid lease and current fencing token.

Deliverables:

- Takeover recovery routine.
- Recovery decision policy.
- Recovery trace events.

Verification evidence:

- Crash before authorization: no dispatch after restart without authorization.
- Crash after authorization before dispatch: standby dispatches once.
- Crash after partial completion: standby skips completed work.
- Crash after dispatch before result persistence: standby uses idempotency
  policy before retry.

### Step 12. Implement Replanning Rules

Description:
Allow a recovered coordinator to generate a new plan only when the prior plan
cannot continue safely.

Must-follow steps:

1. Replan only when the original task has no terminal result.
2. Replan only when the original executor is unavailable or incompatible.
3. Replan only when a compatible replacement exists in a fresh registry
   snapshot.
4. Re-run feasibility checks for the new plan generation.
5. Persist the new plan generation.
6. Preserve the old plan generation for audit.
7. Record why replanning occurred.
8. Do not silently mutate old task commitments.

Deliverables:

- Replanning policy.
- New plan-generation flow.
- Replanning trace events.

Verification evidence:

- Executor unavailable after authorization: standby replans with evidence or
  fails explicitly.
- New plan generation is visible in the store.
- Old plan generation remains queryable.

### Step 13. Add Fault-Injection Hooks

Description:
Controlled crash points are needed to prove distributed recovery behavior.

Must-follow steps:

1. Add opt-in fault-injection hooks at these points:
   - before plan authorization write
   - after plan authorization write
   - after task attempt start
   - after external dispatch
   - after task result received but before persistence
   - during aggregation
2. Keep hooks disabled by default.
3. Ensure hooks are safe in production-like runs unless explicitly enabled.
4. Record injected failure labels in test output.

Deliverables:

- Fault-injection hook mechanism.
- Named failure points.
- Test-only configuration.

Verification evidence:

- Unit test or integration test proving each hook can trigger.
- Default run proving hooks are inactive unless enabled.

### Step 14. Build A Multi-Coordinator Docker Harness

Description:
Prove that the distributed design works across real processes and containers,
not only in unit tests.

Must-follow steps:

1. Extend Docker compose to run three coordinator replicas.
2. Add a PostgreSQL container or selected durable store container.
3. Give each coordinator a unique `coordinator_id`.
4. Route requests through a load balancer or test runner.
5. Add fixture A2A agents.
6. Add a test runner that can kill a coordinator at controlled points.
7. Collect reports for:
   - lease holder transitions
   - fencing token values
   - duplicate dispatch count
   - recovery time
   - terminal correctness
   - stale-write rejections

Deliverables:

- Multi-coordinator compose file.
- Distributed system test runner.
- Distributed report JSON.

Verification evidence:

- Existing single-coordinator Docker checks still pass.
- New multi-coordinator failover matrix passes.
- Report contains the required distributed metrics.

### Step 15. Add Distributed Analysis Tables

Description:
Convert distributed test output into thesis-ready evidence.

Must-follow steps:

1. Extend the analysis script to consume the distributed report.
2. Generate tables for:
   - failover scenario outcomes
   - lease transitions
   - stale-write rejections
   - duplicate dispatch observations
   - recovery latency
3. Link every table row to a report field.
4. Do not hand-write distributed results without preserved JSON evidence.

Deliverables:

- Updated analysis script.
- Distributed summary JSON.
- Markdown and LaTeX table drafts.

Verification evidence:

- Generated tables trace back to preserved distributed report JSON.
- Missing distributed report causes a clear analysis warning rather than fake
  success.

### Step 16. Update The Thesis Architecture Chapter

Description:
Revise the design chapter only after the implementation has corresponding
evidence.

Must-follow steps:

1. Distinguish:
   - single logical coordinator
   - replicated coordinator processes
   - decentralized consensus
2. Add the durable coordination store.
3. Add leases and fencing tokens.
4. Add idempotent dispatch.
5. Add recovery and replanning flow.
6. Add diagrams for the distributed architecture and recovery state machine.
7. Keep the symbolic authorization boundary central.

Deliverables:

- Updated implementation/design chapter.
- Architecture diagrams.

Verification evidence:

- Architecture chapter terminology matches the implemented behavior.
- No figure implies consensus if the system only uses lease-based crash
  recovery.

### Step 17. Update The Thesis Results Chapter

Description:
Report distributed evidence only after the multi-coordinator harness exists and
passes.

Must-follow steps:

1. Add a distributed-coordinator results subsection.
2. Add tables generated from the distributed report.
3. Explain what the failover matrix proves.
4. Explain what remains unproven.
5. Compare distributed behavior to the single-coordinator baseline.
6. Avoid claiming production safety or Byzantine tolerance.

Deliverables:

- Updated results chapter with distributed evidence.

Verification evidence:

- Every distributed result has a preserved JSON source.
- Results include both success cases and limitations.

### Step 18. Update Discussion, Limitations, And Conclusion

Description:
Ensure the manuscript claims exactly what the distributed migration proves.

Must-follow steps:

1. State that the implementation is crash-fault tolerant only if the harness
   proves that property.
2. State that Byzantine tolerance is out of scope unless implemented.
3. State remaining limits:
   - durable-store dependency
   - external side-effect uncertainty
   - fixture-agent external validity
   - latency overhead from leases and transactions
4. State the achieved contribution precisely:
   - replicated coordinator processes
   - durable ownership state
   - stale-write rejection
   - bounded recovery behavior
5. Keep future work separate from completed results.

Deliverables:

- Updated discussion, limitations, and conclusion.

Verification evidence:

- Search results show no unsupported distributed-systems claims.
- Manual review confirms claims match the test report.

## Minimum First Sprint

If this migration is split into one first practical sprint, implement only this
slice:

1. Add explicit session and task state enums.
2. Add invariant tests for authorization-before-dispatch and one terminal
   result per session.
3. Define `CoordinationStore`.
4. Keep the JSONL backend through the new interface.
5. Add PostgreSQL-backed session and event storage.
6. Add session leases.
7. Add fencing tokens.
8. Add a two-coordinator Docker test where one coordinator loses its lease and
   stale writes are rejected.

The sprint is complete only when the system has crossed the architectural
boundary from "one process plus a recovery log" to "multiple coordinator
processes governed by durable ownership state."

## Final Completion Gate

The distributed-coordinator migration is complete only when all of the
following are true:

1. Existing single-coordinator tests still pass.
2. Existing deterministic scenarios still pass.
3. Existing Docker A2A checks still pass.
4. A shared durable coordination store exists.
5. Coordinator replicas have unique identities.
6. Session leases prevent simultaneous ownership.
7. Fencing tokens reject stale writes.
8. Dispatch is idempotent or duplicate-safe.
9. Recovery skips completed work.
10. Recovery handles uncertain in-flight attempts explicitly.
11. Replanning creates new plan generations.
12. Multi-coordinator Docker failover tests pass.
13. Distributed report JSON is preserved.
14. Thesis analysis consumes distributed evidence.
15. Thesis chapters distinguish implemented crash-fault tolerance from stronger
    unimplemented consensus or Byzantine claims.
