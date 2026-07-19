"""Coordination-agent facade for registry-aware plan construction."""

from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import suppress
from typing import Any, Literal
from uuid import uuid4

from .auxiliary import BoundedAuxiliaryCapabilityFactory
from .coordination_ledger import (
    CoordinationLedger,
    InMemoryCoordinationLedger,
    LedgerEvent,
    RetryPolicy,
)
from .coordination_sdk import CoordinationSdk, RemoteRegistryError
from .coordination_store import (
    CoordinationStore,
    JsonlCoordinationStore,
    registry_snapshot_hash,
)
from .feasibility import FeasibilityAnalyzer
from .lingo_coordinator import LingoLinguisticCoordinator
from .models import (
    AgentRegistryEntry,
    CoordinationPlanResult,
    CoordinationRunResult,
    FeasibilityReport,
    LeaseRecord,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskExecutionResult,
    TaskSpec,
    TraceEvent,
)
from .plan_hydration import PlanHydrator
from .runtime_policies import (
    DependencyDispatchPolicy,
    DurableTraceEvidencePolicy,
    StrictDependencyDispatchPolicy,
    TraceEvidencePolicy,
)
from .semantic_admission import (
    OpenAICompatibleSemanticInterpreter,
    SemanticAdmissionIssue,
    SemanticCatalog,
    SemanticInterpretationResult,
    SemanticRequestAdmitter,
)
from .symbolic_plan_compiler import SymbolicPlanCompiler


class CoordinationAgent:
    """Build candidate plans from the SDK registry and authorize them."""

    def __init__(
        self,
        sdk: CoordinationSdk,
        *,
        feasibility_analyzer: FeasibilityAnalyzer | None = None,
        linguistic_coordinator: LingoLinguisticCoordinator | None = None,
        auxiliary_factory: BoundedAuxiliaryCapabilityFactory | None = None,
        ledger: CoordinationLedger | None = None,
        store: CoordinationStore | None = None,
        retry_policy: RetryPolicy | None = None,
        self_agent_id: str | None = None,
        coordinator_id: str | None = None,
        lease_ttl_s: float = 30.0,
        lease_renew_interval_s: float | None = None,
        plan_hydrator: PlanHydrator | None = None,
        semantic_catalog: SemanticCatalog | None = None,
        semantic_interpreter: OpenAICompatibleSemanticInterpreter | Any | None = None,
        symbolic_plan_compiler: SymbolicPlanCompiler | None = None,
        dependency_dispatch_policy: DependencyDispatchPolicy | None = None,
        trace_evidence_policy: TraceEvidencePolicy | None = None,
        max_concurrent_dispatches: int = 16,
    ) -> None:
        self.sdk = sdk
        self.feasibility_analyzer = feasibility_analyzer or FeasibilityAnalyzer()
        self._linguistic_coordinator = linguistic_coordinator
        self.auxiliary_factory = auxiliary_factory or BoundedAuxiliaryCapabilityFactory()
        self.store = store or JsonlCoordinationStore(ledger or InMemoryCoordinationLedger())
        self.ledger = getattr(self.store, "ledger", ledger or InMemoryCoordinationLedger())
        self.retry_policy = retry_policy or RetryPolicy()
        self.self_agent_id = self_agent_id or sdk.self_agent_id
        self.coordinator_id = coordinator_id or self._new_id("coordinator")
        self.lease_ttl_s = lease_ttl_s
        self.lease_renew_interval_s = lease_renew_interval_s or max(
            lease_ttl_s / 2,
            0.1,
        )
        self._abandon_current_lease = False
        # Retained for historical v0.4/v0.5 compatibility; production planning
        # now uses semantic admission followed by SymbolicPlanCompiler.
        self.plan_hydrator = plan_hydrator or PlanHydrator()
        self.semantic_catalog = semantic_catalog
        self.semantic_interpreter = semantic_interpreter
        self.symbolic_plan_compiler = symbolic_plan_compiler or SymbolicPlanCompiler(
            self.feasibility_analyzer
        )
        self.semantic_request_admitter = SemanticRequestAdmitter(
            self.feasibility_analyzer.trust_order
        )
        self.dependency_dispatch_policy = (
            dependency_dispatch_policy or StrictDependencyDispatchPolicy()
        )
        self.trace_evidence_policy = trace_evidence_policy or DurableTraceEvidencePolicy()
        if max_concurrent_dispatches < 1:
            raise ValueError("max_concurrent_dispatches must be positive.")
        self._dispatch_semaphore = asyncio.Semaphore(max_concurrent_dispatches)

    async def build_solution_plan(
        self,
        user_request: str | ProblemRequest,
        context: dict[str, Any] | None = None,
        *,
        session_id: str = "",
        plan_id: str = "",
        semantic_catalog: SemanticCatalog | None = None,
    ) -> CoordinationPlanResult:
        """Build and check a candidate solution plan against the live registry."""
        try:
            registry = await self._registry_snapshot_with_retry()
        except RemoteRegistryError as exc:
            return self._registry_failure_result(user_request, context, exc).model_copy(
                update={"session_id": session_id, "plan_id": plan_id}
            )

        registry = self._without_self(registry)
        request, interpretation, admission_issues = await self._admit_request(
            user_request,
            registry,
            context,
            semantic_catalog or self.semantic_catalog,
        )
        if request is None:
            return self._semantic_failure_result(
                user_request,
                context,
                registry,
                interpretation,
                admission_issues,
                session_id=session_id,
                plan_id=plan_id,
            )

        compilation = self.symbolic_plan_compiler.compile(request, registry)
        proposal, report = compilation.proposal, compilation.report
        if not report.feasible:
            candidate = self._with_auxiliary_specs(
                request, proposal, report, lifecycle_scope=plan_id
            )
            if candidate != proposal:
                candidate_report = self.feasibility_analyzer.check(
                    request, registry, candidate
                )
                proposal, report = candidate, candidate_report
        if self._linguistic_coordinator is not None:
            self._linguistic_coordinator.record_feasibility(report)
        return CoordinationPlanResult(
            session_id=session_id,
            plan_id=plan_id,
            request=request,
            proposal=proposal,
            feasibility_report=report,
            registry_snapshot=registry,
            registry_revision=self.sdk.registry_revision,
            registry_snapshot_hash=registry_snapshot_hash(registry),
            semantic_intent=(
                interpretation.intent.model_dump(mode="json")
                if interpretation is not None and interpretation.intent is not None
                else {}
            ),
            admission_issues=[item.model_dump(mode="json") for item in admission_issues],
            planning_diagnostics={
                **compilation.diagnostics.model_dump(mode="json"),
                "semantic_interpretation": (
                    {
                        "model_id": interpretation.model_id,
                        "call_count": interpretation.call_count,
                        "repair_attempted": interpretation.repair_attempted,
                        "latency_ms": interpretation.latency_ms,
                        "prompt_tokens": interpretation.prompt_tokens,
                        "completion_tokens": interpretation.completion_tokens,
                    }
                    if interpretation is not None
                    else {"bypassed_for_typed_request": True}
                ),
            },
        )

    async def coordinate(
        self,
        user_request: str | ProblemRequest,
        context: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
        session_id: str | None = None,
        semantic_catalog: SemanticCatalog | None = None,
    ) -> CoordinationRunResult:
        """Plan, authorize, dispatch, and aggregate one coordination attempt."""
        session_id = session_id or self._new_id("session")
        lease = await self.store.acquire_lease(
            session_id,
            self.coordinator_id,
            self.lease_ttl_s,
        )
        self._abandon_current_lease = False
        try:
            state = await self.store.session_state(session_id)
            if state.terminal_result is not None:
                return await self._with_recovered_trace(
                    CoordinationRunResult.model_validate(state.terminal_result),
                    session_id,
                )

            plan_result: CoordinationPlanResult
            if state.plan_result is not None:
                plan_result = CoordinationPlanResult.model_validate(state.plan_result)
                plan_id = plan_result.plan_id or self._new_id("plan")
            else:
                plan_id = self._new_id("plan")
                payload = dict(payload or {})
                context = dict(context or {})
                await self._append(
                    "session_started",
                    session_id=session_id,
                    plan_id=plan_id,
                    payload={
                        "user_request": self._jsonable(user_request),
                        "context": context,
                        "payload": payload,
                    },
                    lease=lease,
                )
                plan_kwargs: dict[str, Any] = {
                    "context": context,
                    "session_id": session_id,
                    "plan_id": plan_id,
                }
                if semantic_catalog is not None:
                    plan_kwargs["semantic_catalog"] = semantic_catalog
                plan_result = await self.build_solution_plan(
                    user_request,
                    **plan_kwargs,
                )
                semantic = plan_result.planning_diagnostics.get(
                    "semantic_interpretation", {}
                )
                await self._append(
                    "semantic_admission_completed",
                    session_id=session_id,
                    plan_id=plan_id,
                    payload={
                        "admitted": not plan_result.admission_issues,
                        "issues": plan_result.admission_issues,
                        "metrics": semantic,
                    },
                    lease=lease,
                )
                await self._append(
                    "symbolic_plan_compiled",
                    session_id=session_id,
                    plan_id=plan_id,
                    payload={
                        "task_count": len(plan_result.proposal.tasks),
                        "diagnostics": plan_result.planning_diagnostics,
                    },
                    lease=lease,
                )
                await self._append(
                    "symbolic_authorization_completed",
                    session_id=session_id,
                    plan_id=plan_id,
                    payload={
                        "authorized": plan_result.feasibility_report.feasible,
                        "evidence": [
                            item.model_dump(mode="json")
                            for item in plan_result.feasibility_report.evidence
                        ],
                    },
                    lease=lease,
                )
                await self._append(
                    "registry_snapshot_recorded",
                    session_id=session_id,
                    plan_id=plan_id,
                    payload={
                        "registry_snapshot": [
                            agent.model_dump(mode="json")
                            for agent in plan_result.registry_snapshot
                        ]
                    },
                    lease=lease,
                )

            report = plan_result.feasibility_report
            if not report.feasible:
                result = CoordinationRunResult(
                    session_id=session_id,
                    plan_id=plan_result.plan_id,
                    status="infeasible",
                    plan_result=plan_result,
                    explanation=report.explanation,
                    trace=[],
                )
                await self._append(
                    "plan_infeasible",
                    session_id=session_id,
                    plan_id=plan_result.plan_id,
                    payload={
                        "plan_result": plan_result.model_dump(mode="json"),
                        "run_result": result.model_dump(mode="json"),
                    },
                    lease=lease,
                )
                return result.model_copy(
                    update={"trace": await self._session_trace(session_id)}
                )

            if state.plan_result is None:
                self._inject_fault("before_plan_authorization_write")
                await self._append(
                    "plan_authorized",
                    session_id=session_id,
                    plan_id=plan_result.plan_id,
                    payload={"plan_result": plan_result.model_dump(mode="json")},
                    lease=lease,
                )
                self._inject_fault("after_plan_authorization_write")
                state = await self.store.session_state(session_id)

            return await self._execute_authorized_plan(
                plan_result,
                payload=payload if payload is not None else state.payload,
                timeout_s=timeout_s,
                recovered_state=state,
                lease=lease,
            )
        finally:
            if not self._abandon_current_lease:
                await self.store.release_lease(lease)
            self._abandon_current_lease = False

    async def resume_session(
        self,
        session_id: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> CoordinationRunResult:
        """Resume a previously authorized session from the ledger."""
        lease = await self.store.acquire_lease(
            session_id,
            self.coordinator_id,
            self.lease_ttl_s,
        )
        self._abandon_current_lease = False
        try:
            state = await self.store.session_state(session_id)
            if state.terminal_result is not None:
                return await self._with_recovered_trace(
                    CoordinationRunResult.model_validate(state.terminal_result),
                    session_id,
                )
            if state.plan_result is None:
                raise ValueError(f"Session {session_id} has no authorized plan to resume.")
            plan_result = CoordinationPlanResult.model_validate(state.plan_result)
            return await self._execute_authorized_plan(
                plan_result,
                payload=payload if payload is not None else state.payload,
                timeout_s=timeout_s,
                recovered_state=state,
                lease=lease,
            )
        finally:
            if not self._abandon_current_lease:
                await self.store.release_lease(lease)
            self._abandon_current_lease = False

    async def _execute_authorized_plan(
        self,
        plan_result: CoordinationPlanResult,
        *,
        payload: dict[str, Any] | None,
        timeout_s: float,
        recovered_state,
        lease: LeaseRecord,
    ) -> CoordinationRunResult:
        report = plan_result.feasibility_report
        session_id = plan_result.session_id
        plan_id = plan_result.plan_id

        task_results: list[TaskExecutionResult] = []
        results_by_task: dict[str, TaskExecutionResult] = {}
        artifacts_by_task: dict[str, list[dict[str, Any]]] = {}
        for task in self._ordered_tasks(plan_result.proposal):
            lease = await self.store.renew_lease(lease, self.lease_ttl_s)
            blocked_by = self.dependency_dispatch_policy.blocked_dependencies(
                task, results_by_task
            )
            if blocked_by:
                skipped = TaskExecutionResult(
                    session_id=session_id,
                    plan_id=plan_id,
                    task_id=task.task_id,
                    agent_id=task.assigned_to or "",
                    status="skipped",
                    error=(
                        "Dependency did not complete successfully: "
                        + ", ".join(blocked_by)
                    ),
                    metadata={"blocked_by": blocked_by},
                )
                await self._append(
                    "task_skipped",
                    session_id=session_id,
                    plan_id=plan_id,
                    task_id=task.task_id,
                    payload={"task_result": skipped.model_dump(mode="json")},
                    lease=lease,
                )
                task_results.append(skipped)
                results_by_task[task.task_id] = skipped
                continue
            recovered_result = recovered_state.task_results.get(task.task_id)
            if recovered_result and recovered_result.get("status") == "completed":
                recovered_task_result = TaskExecutionResult.model_validate(recovered_result)
                task_results.append(recovered_task_result)
                results_by_task[task.task_id] = recovered_task_result
                artifacts_by_task[task.task_id] = list(recovered_task_result.artifacts)
                continue

            prior_attempts = recovered_state.task_attempt_counts.get(task.task_id, 0)
            if task.task_id in recovered_state.running_task_ids:
                side_effect_class = self._side_effect_class(plan_result, task)
                if side_effect_class not in {"read_only", "idempotent"}:
                    unknown = self._unknown_attempt_result(
                        plan_result,
                        task,
                        recovered_state,
                    )
                    await self._append(
                        "task_attempt_unknown",
                        session_id=session_id,
                        plan_id=plan_id,
                        task_id=task.task_id,
                        attempt_id=unknown.attempt_id,
                        payload={"task_result": unknown.model_dump(mode="json")},
                        lease=lease,
                    )
                    task_results.append(unknown)
                    results_by_task[task.task_id] = unknown
                    continue

            max_attempts = self.retry_policy.task_retries + 1
            if self._side_effect_class(plan_result, task) not in {
                "read_only",
                "idempotent",
            }:
                max_attempts = max(prior_attempts + 1, 1)
            if recovered_result and prior_attempts >= max_attempts:
                task_results.append(TaskExecutionResult.model_validate(recovered_result))
                results_by_task[task.task_id] = task_results[-1]
                continue

            task_result = await self._dispatch_with_retry(
                report,
                task,
                plan_result,
                self._with_dependency_artifacts(payload or {}, task, artifacts_by_task),
                timeout_s=timeout_s,
                session_id=session_id,
                plan_id=plan_id,
                starting_attempt=prior_attempts + 1,
                max_attempts=max_attempts,
                lease=lease,
            )
            task_results.append(task_result)
            results_by_task[task.task_id] = task_result
            if task_result.status == "completed":
                artifacts_by_task[task.task_id] = list(task_result.artifacts)

        artifacts = [
            artifact
            for task_result in task_results
            for artifact in task_result.artifacts
        ]
        status: Literal["completed", "failed"] = (
            "completed"
            if self._completion_contract_satisfied(
                plan_result.proposal, task_results, artifacts
            )
            else "failed"
        )
        explanation = (
            "Authorized coordination completed."
            if status == "completed"
            else "Authorized coordination encountered runtime failures."
        )
        result = CoordinationRunResult(
            session_id=session_id,
            plan_id=plan_id,
            status=status,
            plan_result=plan_result,
            task_results=task_results,
            artifacts=artifacts,
            explanation=explanation,
            trace=[],
        )
        self._inject_fault("during_aggregation")
        await self._append(
            "run_completed" if status == "completed" else "run_failed",
            session_id=session_id,
            plan_id=plan_id,
            payload={"run_result": result.model_dump(mode="json")},
            lease=lease,
        )
        return result.model_copy(update={"trace": await self._session_trace(session_id)})

    async def _dispatch_with_retry(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        plan_result: CoordinationPlanResult,
        payload: dict[str, Any],
        *,
        timeout_s: float,
        session_id: str,
        plan_id: str,
        starting_attempt: int,
        max_attempts: int,
        lease: LeaseRecord,
    ) -> TaskExecutionResult:
        result: TaskExecutionResult | None = None
        for attempt_number in range(starting_attempt, max_attempts + 1):
            attempt_id = f"{task.task_id}-attempt-{attempt_number}"
            idempotency_key = self._idempotency_key(
                session_id,
                plan_id,
                task.task_id,
                attempt_id,
            )
            operation_key = self._operation_key(
                session_id,
                plan_result.plan_generation,
                task.task_id,
            )
            prior_result = await self.store.task_result_by_idempotency_key(
                idempotency_key
            )
            if prior_result is not None:
                return TaskExecutionResult.model_validate(prior_result)

            await self._append(
                "task_attempt_started",
                session_id=session_id,
                plan_id=plan_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                payload={
                    "task": task.model_dump(mode="json"),
                    "coordinator_id": self.coordinator_id,
                    "fencing_token": lease.fencing_token,
                    "idempotency_key": idempotency_key,
                    "operation_key": operation_key,
                    "attempt_key": idempotency_key,
                },
                lease=lease,
            )
            self._inject_fault("after_task_attempt_start")
            try:
                result = await self._send_task_with_lease_renewal(
                    report,
                    task,
                    self._payload_for_task(
                        self._with_coordination_metadata(
                            payload,
                            session_id=session_id,
                            plan_id=plan_id,
                            plan_generation=plan_result.plan_generation,
                            task_id=task.task_id,
                            attempt_id=attempt_id,
                            coordinator_id=self.coordinator_id,
                            fencing_token=lease.fencing_token,
                            idempotency_key=idempotency_key,
                            operation_key=operation_key,
                            registry_revision=plan_result.registry_revision,
                        ),
                        task,
                    ),
                    timeout_s=timeout_s,
                    lease=lease,
                )
            except Exception as exc:
                agent_id = task.assigned_to or report.matched_agents.get(task.task_id) or ""
                result = TaskExecutionResult(
                    session_id=session_id,
                    plan_id=plan_id,
                    attempt_id=attempt_id,
                    task_id=task.task_id,
                    agent_id=agent_id,
                    status="failed",
                    error=str(exc),
                )

            self._inject_fault("after_external_dispatch")
            result = result.model_copy(
                update={
                    "session_id": session_id,
                    "plan_id": plan_id,
                    "attempt_id": attempt_id,
                    "metadata": {
                        **result.metadata,
                        "coordinator_id": self.coordinator_id,
                        "fencing_token": lease.fencing_token,
                        "plan_generation": plan_result.plan_generation,
                        "idempotency_key": idempotency_key,
                        "operation_key": operation_key,
                        "attempt_key": idempotency_key,
                        "registry_revision": plan_result.registry_revision,
                    },
                }
            )
            await self._append(
                "sdk_observation",
                session_id=session_id,
                plan_id=plan_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                payload={
                    "status": result.status,
                    "agent_id": result.agent_id,
                    "agent_kind": result.agent_kind,
                    "error": result.error,
                    "metadata": result.metadata,
                },
                lease=lease,
            )
            for sdk_event in self.sdk.trace_for_attempt(
                session_id, plan_id, task.task_id, attempt_id
            ):
                await self._append(
                    sdk_event.event_type,
                    session_id=session_id,
                    plan_id=plan_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    payload={
                        "message": sdk_event.message,
                        "source": sdk_event.source,
                        "data": sdk_event.data,
                    },
                    lease=lease,
                )
            await self._append(
                self._task_event_type(result),
                session_id=session_id,
                plan_id=plan_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                payload={"task_result": result.model_dump(mode="json")},
                lease=lease,
            )
            if result.status == "completed":
                return result
            if (
                attempt_number < max_attempts
                and self._side_effect_class(plan_result, task)
                in {"read_only", "idempotent"}
            ):
                await asyncio.sleep(self.retry_policy.backoff_s)
            else:
                break
        assert result is not None
        return result

    async def _send_task_with_lease_renewal(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        payload: dict[str, Any],
        *,
        timeout_s: float,
        lease: LeaseRecord,
    ) -> TaskExecutionResult:
        dispatch_task = asyncio.create_task(
            self._bounded_send_task(report, task, payload, timeout_s=timeout_s)
        )
        renewal_task = asyncio.create_task(self._lease_renewal_loop(lease))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {dispatch_task, renewal_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if dispatch_task in done:
                    return await dispatch_task
                if renewal_task in done:
                    exc = renewal_task.exception()
                    if exc is not None:
                        dispatch_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await dispatch_task
                        raise exc
        finally:
            renewal_task.cancel()
            with suppress(asyncio.CancelledError):
                await renewal_task

    async def _bounded_send_task(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        payload: dict[str, Any],
        *,
        timeout_s: float,
    ) -> TaskExecutionResult:
        async with self._dispatch_semaphore:
            return await self.sdk.send_task(report, task, payload, timeout_s=timeout_s)

    async def _lease_renewal_loop(self, lease: LeaseRecord) -> None:
        interval = max(self.lease_renew_interval_s, 0.05)
        while True:
            await asyncio.sleep(interval)
            await self.store.renew_lease(lease, self.lease_ttl_s)

    def _inject_fault(self, point: str) -> None:
        if os.getenv("COORDINATION_FAULT_AT") != point:
            return
        mode = os.getenv("COORDINATION_FAULT_MODE", "raise")
        if mode == "exit":
            os._exit(int(os.getenv("COORDINATION_FAULT_EXIT_CODE", "137")))
        if mode == "abandon_lease":
            self._abandon_current_lease = True
        raise RuntimeError(f"Injected coordinator fault at {point}.")

    def trace(self) -> list[TraceEvent]:
        """Return linguistic and SDK trace events for the current session."""
        linguistic_events: list[TraceEvent] = []
        if self._linguistic_coordinator is not None:
            linguistic_events = [
                TraceEvent.model_validate(event)
                for event in self._linguistic_coordinator.state.trace
            ]
        return [*linguistic_events, *self.sdk.trace()]

    async def _session_trace(self, session_id: str) -> list[TraceEvent]:
        ledger_events = [
            TraceEvent(
                event_type=event.event_type,
                message=f"Ledger event {event.event_type}.",
                session_id=event.session_id,
                plan_id=event.plan_id,
                task_id=event.task_id,
                attempt_id=event.attempt_id,
                source=(
                    "sdk"
                    if event.event_type == "sdk_observation"
                    or event.payload.get("source") == "sdk"
                    else "store"
                ),
                data={
                    "session_id": event.session_id,
                    "plan_id": event.plan_id,
                    "task_id": event.task_id,
                    "attempt_id": event.attempt_id,
                    "payload": event.payload,
                },
                timestamp=event.timestamp,
            )
            for event in await self.store.events(session_id)
        ]
        return self.trace_evidence_policy.expose(ledger_events)

    async def _with_recovered_trace(
        self,
        result: CoordinationRunResult,
        session_id: str,
    ) -> CoordinationRunResult:
        return result.model_copy(update={"trace": await self._session_trace(session_id)})

    async def _registry_snapshot_with_retry(self) -> list[AgentRegistryEntry]:
        attempts = self.retry_policy.registry_retries + 1
        last_error: RemoteRegistryError | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self.sdk.registry_snapshot(refresh=True)
            except RemoteRegistryError as exc:
                last_error = exc
                if attempt < attempts:
                    await asyncio.sleep(self.retry_policy.backoff_s)
        assert last_error is not None
        raise last_error

    async def _append(
        self,
        event_type: str,
        *,
        session_id: str,
        plan_id: str = "",
        task_id: str = "",
        attempt_id: str = "",
        payload: dict[str, Any] | None = None,
        lease: LeaseRecord | None = None,
    ) -> LedgerEvent:
        return await self.store.append_event(
            LedgerEvent(
                event_type=event_type,
                session_id=session_id,
                plan_id=plan_id,
                task_id=task_id,
                attempt_id=attempt_id,
                payload=payload or {},
            ),
            lease=lease,
        )

    @staticmethod
    def _task_event_type(result: TaskExecutionResult) -> str:
        if result.status == "unknown":
            return "task_attempt_unknown"
        if result.status == "completed":
            return "task_attempt_completed"
        if result.status == "timeout":
            return "task_attempt_timeout"
        return "task_attempt_failed"

    def _unknown_attempt_result(
        self,
        plan_result: CoordinationPlanResult,
        task: TaskSpec,
        recovered_state,
    ) -> TaskExecutionResult:
        attempts = recovered_state.task_attempts_by_task.get(task.task_id, [])
        attempt_id = attempts[-1] if attempts else f"{task.task_id}-attempt-unknown"
        agent_id = task.assigned_to or plan_result.feasibility_report.matched_agents.get(
            task.task_id,
            "",
        )
        return TaskExecutionResult(
            session_id=plan_result.session_id,
            plan_id=plan_result.plan_id,
            attempt_id=attempt_id,
            task_id=task.task_id,
            agent_id=agent_id,
            status="unknown",
            error=(
                "Recovered an in-flight attempt with unknown side effects; "
                "refusing blind duplicate dispatch."
            ),
            metadata={
                "coordinator_id": self.coordinator_id,
                "recovery_policy": "no_blind_duplicate_dispatch",
            },
        )

    def _side_effect_class(
        self,
        plan_result: CoordinationPlanResult,
        task: TaskSpec,
    ) -> str:
        wanted = self._norm(task.requirement_name)
        for requirement in plan_result.request.requirements:
            if self._norm(requirement.name) == wanted:
                return requirement.side_effect_class
        return "unknown"

    @staticmethod
    def _idempotency_key(
        session_id: str,
        plan_id: str,
        task_id: str,
        attempt_id: str,
    ) -> str:
        return f"{session_id}:{plan_id}:{task_id}:{attempt_id}"

    @staticmethod
    def _operation_key(
        session_id: str,
        plan_generation: int,
        task_id: str,
    ) -> str:
        return f"{session_id}:{plan_generation}:{task_id}"

    @staticmethod
    def _with_coordination_metadata(
        payload: dict[str, Any],
        *,
        session_id: str,
        plan_id: str,
        plan_generation: int,
        task_id: str,
        attempt_id: str,
        coordinator_id: str,
        fencing_token: int,
        idempotency_key: str,
        operation_key: str,
        registry_revision: int,
    ) -> dict[str, Any]:
        enriched = dict(payload)
        metadata = dict(enriched.get("_coordination") or {})
        metadata.update(
            {
                "session_id": session_id,
                "plan_id": plan_id,
                "plan_generation": plan_generation,
                "task_id": task_id,
                "attempt_id": attempt_id,
                "coordinator_id": coordinator_id,
                "fencing_token": fencing_token,
                "idempotency_key": idempotency_key,
                "attempt_key": idempotency_key,
                "operation_key": operation_key,
                "registry_revision": registry_revision,
            }
        )
        enriched["_coordination"] = metadata
        return enriched

    @staticmethod
    def _with_dependency_artifacts(
        payload: dict[str, Any],
        task: TaskSpec,
        artifacts_by_task: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        enriched = dict(payload)
        metadata = dict(enriched.get("_coordination") or {})
        inputs_by_task = {
            dependency: list(artifacts_by_task.get(dependency, []))
            for dependency in task.depends_on
        }
        previous_artifacts = [
            artifact
            for dependency in task.depends_on
            for artifact in artifacts_by_task.get(dependency, [])
        ]
        metadata.update(
            {
                "inputs_by_task": inputs_by_task,
                "previous_artifacts": previous_artifacts,
            }
        )
        enriched["_coordination"] = metadata
        return enriched

    @classmethod
    def _completion_contract_satisfied(
        cls,
        proposal: SolutionProposal,
        task_results: list[TaskExecutionResult],
        artifacts: list[dict[str, Any]],
    ) -> bool:
        allowed_states = set(proposal.completion_contract.required_task_states)
        if not task_results or any(result.status not in allowed_states for result in task_results):
            return False
        return all(
            cls._artifact_named(artifacts, name)
            for name in proposal.completion_contract.required_artifacts
        )

    @staticmethod
    def _artifact_named(artifacts: list[dict[str, Any]], name: str) -> bool:
        wanted = name.strip().lower()
        for artifact in artifacts:
            values = [
                artifact.get("artifact_id"),
                artifact.get("id"),
                artifact.get("name"),
                artifact.get("kind"),
                artifact.get("type"),
            ]
            if any(str(value).strip().lower() == wanted for value in values if value):
                return True
            data = artifact.get("data")
            if isinstance(data, dict) and wanted in {str(key).lower() for key in data}:
                return True
        return False

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex}"

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    async def _admit_request(
        self,
        user_request: str | ProblemRequest,
        registry: list[AgentRegistryEntry],
        context: dict[str, Any] | None,
        semantic_catalog: SemanticCatalog | None,
    ) -> tuple[
        ProblemRequest | None,
        SemanticInterpretationResult | None,
        list[SemanticAdmissionIssue],
    ]:
        if isinstance(user_request, ProblemRequest):
            request = user_request
        else:
            if semantic_catalog is None:
                issue = SemanticAdmissionIssue(
                    code="missing_catalog",
                    message="Raw-language coordination requires an admitted semantic catalog.",
                )
                return None, None, [issue]
            if self.semantic_interpreter is None:
                issue = SemanticAdmissionIssue(
                    code="missing_interpreter",
                    message="No strict semantic interpreter is configured.",
                )
                return None, None, [issue]
            try:
                interpretation = await self.semantic_interpreter.interpret(
                    user_request, semantic_catalog, registry
                )
            except Exception as exc:
                issue = SemanticAdmissionIssue(
                    code="schema_invalid",
                    message=f"Semantic interpretation failed closed: {exc}",
                )
                return None, None, [issue]
            if interpretation.intent is None:
                issue = SemanticAdmissionIssue(
                    code="schema_invalid",
                    message=(
                        "Semantic output remained invalid after the bounded repair: "
                        + "; ".join(interpretation.issues)
                    ),
                )
                return None, interpretation, [issue]
            admission = self.semantic_request_admitter.admit(
                user_request, semantic_catalog, interpretation.intent, registry
            )
            if not admission.admitted:
                return None, interpretation, admission.issues
            assert admission.request is not None
            request = admission.request
            if context:
                request = request.model_copy(
                    update={"context": {**request.context, **context}}
                )
            return request, interpretation, []
        if not context:
            return request, None, []
        merged_context = {**request.context, **context}
        return request.model_copy(update={"context": merged_context}), None, []

    def _semantic_failure_result(
        self,
        user_request: str | ProblemRequest,
        context: dict[str, Any] | None,
        registry: list[AgentRegistryEntry],
        interpretation: SemanticInterpretationResult | None,
        issues: list[SemanticAdmissionIssue],
        *,
        session_id: str,
        plan_id: str,
    ) -> CoordinationPlanResult:
        request = (
            user_request
            if isinstance(user_request, ProblemRequest)
            else ProblemRequest(
                user_goal=user_request,
                context=dict(context or {}),
            )
        )
        report = FeasibilityReport(
            feasible=False,
            risks=[item.message for item in issues],
            explanation="Raw-language semantic admission failed closed.",
            evidence=[
                PredicateEvidence(
                    name="semantic_admission",
                    passed=False,
                    details={
                        "issues": [item.model_dump(mode="json") for item in issues]
                    },
                )
            ],
        )
        return CoordinationPlanResult(
            session_id=session_id,
            plan_id=plan_id,
            request=request,
            proposal=SolutionProposal(),
            feasibility_report=report,
            registry_snapshot=registry,
            registry_revision=self.sdk.registry_revision,
            registry_snapshot_hash=registry_snapshot_hash(registry),
            semantic_intent=(
                request.context.get("semantic_intent", {})
                if interpretation is not None
                else {}
            ),
            admission_issues=[item.model_dump(mode="json") for item in issues],
            planning_diagnostics={
                "semantic_interpretation": (
                    interpretation.model_dump(mode="json")
                    if interpretation is not None
                    else {}
                )
            },
        )

    def _without_self(
        self, registry: list[AgentRegistryEntry]
    ) -> list[AgentRegistryEntry]:
        if not self.self_agent_id:
            return registry
        return [agent for agent in registry if agent.agent_id != self.self_agent_id]

    def _with_auxiliary_specs(
        self,
        request: ProblemRequest,
        proposal: SolutionProposal,
        report: FeasibilityReport,
        *,
        lifecycle_scope: str = "",
    ) -> SolutionProposal:
        if not report.missing_capabilities:
            return proposal

        req_by_name = {self._norm(req.name): req for req in request.requirements}
        generated = list(proposal.generated_nlp_agents)
        changed_tasks: list[TaskSpec] = []
        changed = False
        scope = lifecycle_scope or hashlib.sha256(
            request.model_dump_json().encode("utf-8")
        ).hexdigest()[:16]

        for task in proposal.tasks:
            if task.assigned_to or task.auxiliary_spec_id:
                changed_tasks.append(task)
                continue
            requirement = req_by_name.get(self._norm(task.requirement_name))
            if requirement is None or requirement.name not in report.missing_capabilities:
                changed_tasks.append(task)
                continue
            lifecycle = f"plan:{scope}:task:{task.task_id}"
            spec = self.auxiliary_factory.specify(requirement, lifecycle=lifecycle)
            if spec is None:
                changed_tasks.append(task)
                continue
            generated.append(spec)
            changed_tasks.append(task.model_copy(update={"auxiliary_spec_id": spec.spec_id}))
            changed = True

        if not changed:
            return proposal
        return proposal.model_copy(
            update={
                "tasks": changed_tasks,
                "generated_nlp_agents": generated,
            }
        )

    def _registry_failure_result(
        self,
        user_request: str | ProblemRequest,
        context: dict[str, Any] | None,
        exc: RemoteRegistryError,
    ) -> CoordinationPlanResult:
        if isinstance(user_request, ProblemRequest):
            request = user_request
            if context:
                request = request.model_copy(
                    update={"context": {**request.context, **context}}
                )
        else:
            request = ProblemRequest(user_goal=user_request, context=dict(context or {}))

        report = FeasibilityReport(
            feasible=False,
            risks=[str(exc)],
            explanation="Remote registry refresh failed.",
            evidence=[
                PredicateEvidence(
                    name="registry_available",
                    passed=False,
                    details={"error": str(exc)},
                )
            ],
        )
        return CoordinationPlanResult(
            request=request,
            proposal=SolutionProposal(),
            feasibility_report=report,
            registry_snapshot=[],
        )

    def _ordered_tasks(self, proposal: SolutionProposal) -> list[TaskSpec]:
        task_by_id = {task.task_id: task for task in proposal.tasks}
        ordered: list[TaskSpec] = []
        for task_id in proposal.execution_order:
            task = task_by_id.get(task_id)
            if task is not None:
                ordered.append(task)
        if ordered:
            return ordered
        return list(proposal.tasks)

    @staticmethod
    def _payload_for_task(
        payload: dict[str, Any] | None,
        task: TaskSpec,
    ) -> dict[str, Any]:
        if not payload:
            return {}
        task_payload = payload.get(task.task_id)
        if isinstance(task_payload, dict):
            result = dict(task_payload)
            if "_coordination" in payload:
                result["_coordination"] = payload["_coordination"]
            return result
        return dict(payload)

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().lower().replace("_", " ")
