"""Coordination-agent facade for registry-aware plan construction."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from .auxiliary import BoundedAuxiliaryCapabilityFactory
from .coordination_ledger import (
    CoordinationLedger,
    InMemoryCoordinationLedger,
    LedgerEvent,
    RetryPolicy,
)
from .coordination_sdk import CoordinationSdk, RemoteRegistryError
from .feasibility import FeasibilityAnalyzer
from .lingo_coordinator import LingoLinguisticCoordinator
from .models import (
    AgentRegistryEntry,
    CoordinationPlanResult,
    CoordinationRunResult,
    FeasibilityReport,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskExecutionResult,
    TaskSpec,
    TraceEvent,
)


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
        retry_policy: RetryPolicy | None = None,
        self_agent_id: str | None = None,
    ) -> None:
        self.sdk = sdk
        self.feasibility_analyzer = feasibility_analyzer or FeasibilityAnalyzer()
        self._linguistic_coordinator = linguistic_coordinator
        self.auxiliary_factory = auxiliary_factory or BoundedAuxiliaryCapabilityFactory()
        self.ledger = ledger or InMemoryCoordinationLedger()
        self.retry_policy = retry_policy or RetryPolicy()
        self.self_agent_id = self_agent_id or sdk.self_agent_id

    async def build_solution_plan(
        self,
        user_request: str | ProblemRequest,
        context: dict[str, Any] | None = None,
        *,
        session_id: str = "",
        plan_id: str = "",
    ) -> CoordinationPlanResult:
        """Build and check a candidate solution plan against the live registry."""
        try:
            registry = await self._registry_snapshot_with_retry()
        except RemoteRegistryError as exc:
            return self._registry_failure_result(user_request, context, exc).model_copy(
                update={"session_id": session_id, "plan_id": plan_id}
            )

        registry = self._without_self(registry)
        request = await self._admit_request(user_request, registry, context)
        proposal = self._direct_solution_plan(request, registry)
        if proposal is None:
            if isinstance(user_request, ProblemRequest) and self._linguistic_coordinator is None:
                proposal = self._unassigned_solution_plan(request)
            else:
                proposal = await self._linguistic().propose_solution(
                    request, registry
                )
        report = self.feasibility_analyzer.check(request, registry, proposal)
        if not report.feasible:
            proposal = self._with_auxiliary_specs(request, proposal, report)
            report = self.feasibility_analyzer.check(request, registry, proposal)
        if self._linguistic_coordinator is not None:
            self._linguistic_coordinator.record_feasibility(report)
        return CoordinationPlanResult(
            session_id=session_id,
            plan_id=plan_id,
            request=request,
            proposal=proposal,
            feasibility_report=report,
            registry_snapshot=registry,
        )

    async def coordinate(
        self,
        user_request: str | ProblemRequest,
        context: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
        session_id: str | None = None,
    ) -> CoordinationRunResult:
        """Plan, authorize, dispatch, and aggregate one coordination attempt."""
        session_id = session_id or self._new_id("session")
        state = self.ledger.session_state(session_id)
        if state.terminal_result is not None:
            return self._with_recovered_trace(
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
            self._append(
                "session_started",
                session_id=session_id,
                plan_id=plan_id,
                payload={
                    "user_request": self._jsonable(user_request),
                    "context": context,
                    "payload": payload,
                },
            )
            plan_result = await self.build_solution_plan(
                user_request,
                context=context,
                session_id=session_id,
                plan_id=plan_id,
            )
            self._append(
                "registry_snapshot_recorded",
                session_id=session_id,
                plan_id=plan_id,
                payload={
                    "registry_snapshot": [
                        agent.model_dump(mode="json")
                        for agent in plan_result.registry_snapshot
                    ]
                },
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
            self._append(
                "plan_infeasible",
                session_id=session_id,
                plan_id=plan_result.plan_id,
                payload={
                    "plan_result": plan_result.model_dump(mode="json"),
                    "run_result": result.model_dump(mode="json"),
                },
            )
            return result.model_copy(update={"trace": self._session_trace(session_id)})

        if state.plan_result is None:
            self._append(
                "plan_authorized",
                session_id=session_id,
                plan_id=plan_result.plan_id,
                payload={"plan_result": plan_result.model_dump(mode="json")},
            )
            state = self.ledger.session_state(session_id)

        return await self._execute_authorized_plan(
            plan_result,
            payload=payload if payload is not None else state.payload,
            timeout_s=timeout_s,
            recovered_state=state,
        )

    async def resume_session(
        self,
        session_id: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> CoordinationRunResult:
        """Resume a previously authorized session from the ledger."""
        state = self.ledger.session_state(session_id)
        if state.terminal_result is not None:
            return self._with_recovered_trace(
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
        )

    async def _execute_authorized_plan(
        self,
        plan_result: CoordinationPlanResult,
        *,
        payload: dict[str, Any] | None,
        timeout_s: float,
        recovered_state,
    ) -> CoordinationRunResult:
        report = plan_result.feasibility_report
        session_id = plan_result.session_id
        plan_id = plan_result.plan_id

        task_results: list[TaskExecutionResult] = []
        artifacts_by_task: dict[str, list[dict[str, Any]]] = {}
        for task in self._ordered_tasks(plan_result.proposal):
            recovered_result = recovered_state.task_results.get(task.task_id)
            if recovered_result and recovered_result.get("status") == "completed":
                recovered_task_result = TaskExecutionResult.model_validate(recovered_result)
                task_results.append(recovered_task_result)
                artifacts_by_task[task.task_id] = list(recovered_task_result.artifacts)
                continue

            prior_attempts = recovered_state.task_attempt_counts.get(task.task_id, 0)
            max_attempts = self.retry_policy.task_retries + 1
            if recovered_result and prior_attempts >= max_attempts:
                task_results.append(TaskExecutionResult.model_validate(recovered_result))
                continue

            task_result = await self._dispatch_with_retry(
                report,
                task,
                self._with_dependency_artifacts(payload or {}, task, artifacts_by_task),
                timeout_s=timeout_s,
                session_id=session_id,
                plan_id=plan_id,
                starting_attempt=prior_attempts + 1,
                max_attempts=max_attempts,
            )
            task_results.append(task_result)
            if task_result.status == "completed":
                artifacts_by_task[task.task_id] = list(task_result.artifacts)

        artifacts = [
            artifact
            for task_result in task_results
            for artifact in task_result.artifacts
        ]
        status = (
            "completed"
            if all(result.status == "completed" for result in task_results)
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
        self._append(
            "run_completed" if status == "completed" else "run_failed",
            session_id=session_id,
            plan_id=plan_id,
            payload={"run_result": result.model_dump(mode="json")},
        )
        return result.model_copy(update={"trace": self._session_trace(session_id)})

    async def _dispatch_with_retry(
        self,
        report: FeasibilityReport,
        task: TaskSpec,
        payload: dict[str, Any],
        *,
        timeout_s: float,
        session_id: str,
        plan_id: str,
        starting_attempt: int,
        max_attempts: int,
    ) -> TaskExecutionResult:
        result: TaskExecutionResult | None = None
        for attempt_number in range(starting_attempt, max_attempts + 1):
            attempt_id = f"{task.task_id}-attempt-{attempt_number}"
            self._append(
                "task_attempt_started",
                session_id=session_id,
                plan_id=plan_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                payload={"task": task.model_dump(mode="json")},
            )
            try:
                result = await self.sdk.send_task(
                    report,
                    task,
                    self._payload_for_task(
                        self._with_coordination_metadata(
                            payload,
                            session_id=session_id,
                            plan_id=plan_id,
                            task_id=task.task_id,
                            attempt_id=attempt_id,
                        ),
                        task,
                    ),
                    timeout_s=timeout_s,
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

            result = result.model_copy(
                update={
                    "session_id": session_id,
                    "plan_id": plan_id,
                    "attempt_id": attempt_id,
                }
            )
            self._append(
                self._task_event_type(result),
                session_id=session_id,
                plan_id=plan_id,
                task_id=task.task_id,
                attempt_id=attempt_id,
                payload={"task_result": result.model_dump(mode="json")},
            )
            if result.status == "completed":
                return result
            if attempt_number < max_attempts:
                await asyncio.sleep(self.retry_policy.backoff_s)
        assert result is not None
        return result

    def trace(self) -> list[TraceEvent]:
        """Return linguistic and SDK trace events for the current session."""
        linguistic_events: list[TraceEvent] = []
        if self._linguistic_coordinator is not None:
            linguistic_events = [
                TraceEvent.model_validate(event)
                for event in self._linguistic_coordinator.state.trace
            ]
        return [*linguistic_events, *self.sdk.trace()]

    def _session_trace(self, session_id: str) -> list[TraceEvent]:
        ledger_events = [
            TraceEvent(
                event_type=event.event_type,
                message=f"Ledger event {event.event_type}.",
                data={
                    "session_id": event.session_id,
                    "plan_id": event.plan_id,
                    "task_id": event.task_id,
                    "attempt_id": event.attempt_id,
                    "payload": event.payload,
                },
                timestamp=event.timestamp,
            )
            for event in self.ledger.events(session_id)
        ]
        return [*ledger_events, *self.trace()]

    def _with_recovered_trace(
        self,
        result: CoordinationRunResult,
        session_id: str,
    ) -> CoordinationRunResult:
        return result.model_copy(update={"trace": self._session_trace(session_id)})

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

    def _append(
        self,
        event_type: str,
        *,
        session_id: str,
        plan_id: str = "",
        task_id: str = "",
        attempt_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> LedgerEvent:
        return self.ledger.append(
            LedgerEvent(
                event_type=event_type,
                session_id=session_id,
                plan_id=plan_id,
                task_id=task_id,
                attempt_id=attempt_id,
                payload=payload or {},
            )
        )

    @staticmethod
    def _task_event_type(result: TaskExecutionResult) -> str:
        if result.status == "completed":
            return "task_attempt_completed"
        if result.status == "timeout":
            return "task_attempt_timeout"
        return "task_attempt_failed"

    @staticmethod
    def _with_coordination_metadata(
        payload: dict[str, Any],
        *,
        session_id: str,
        plan_id: str,
        task_id: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        enriched = dict(payload)
        metadata = dict(enriched.get("_coordination") or {})
        metadata.update(
            {
                "session_id": session_id,
                "plan_id": plan_id,
                "task_id": task_id,
                "attempt_id": attempt_id,
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
            for artifacts in artifacts_by_task.values()
            for artifact in artifacts
        ]
        metadata.update(
            {
                "inputs_by_task": inputs_by_task,
                "previous_artifacts": previous_artifacts,
            }
        )
        enriched["_coordination"] = metadata
        return enriched

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
    ) -> ProblemRequest:
        if isinstance(user_request, ProblemRequest):
            request = user_request
        else:
            request = await self._linguistic().interpret_request(
                user_request, registry
            )
        if not context:
            return request
        merged_context = {**request.context, **context}
        return request.model_copy(update={"context": merged_context})

    def _direct_solution_plan(
        self,
        request: ProblemRequest,
        registry: list[AgentRegistryEntry],
    ) -> SolutionProposal | None:
        if not request.requirements:
            return None

        tasks: list[TaskSpec] = []
        selected_agents: dict[str, str] = {}
        for index, requirement in enumerate(request.requirements, start=1):
            agent = self._first_exact_skill_agent(requirement.name, registry)
            if agent is None:
                return None
            task_id = f"t{index}"
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    requirement_name=requirement.name,
                    assigned_to=agent.agent_id,
                    expected_artifacts=self._task_expected_artifacts(requirement, request),
                    validation_contract=dict(requirement.validation_contract),
                )
            )
            selected_agents[task_id] = agent.agent_id

        return SolutionProposal(
            tasks=tasks,
            selected_agents=selected_agents,
            execution_order=[task.task_id for task in tasks],
            expected_artifacts=list(request.required_artifacts),
            completion_criteria=self._completion_criteria(request),
        )

    def _unassigned_solution_plan(self, request: ProblemRequest) -> SolutionProposal:
        tasks = [
            TaskSpec(
                task_id=f"t{index}",
                requirement_name=requirement.name,
                expected_artifacts=self._task_expected_artifacts(requirement, request),
                validation_contract=dict(requirement.validation_contract),
            )
            for index, requirement in enumerate(request.requirements, start=1)
        ]
        return SolutionProposal(
            tasks=tasks,
            execution_order=[task.task_id for task in tasks],
            expected_artifacts=list(request.required_artifacts),
            completion_criteria=self._completion_criteria(request),
        )

    def _first_exact_skill_agent(
        self,
        requirement_name: str,
        registry: list[AgentRegistryEntry],
    ) -> AgentRegistryEntry | None:
        wanted = self._norm(requirement_name)
        for agent in registry:
            if agent.status != "available":
                continue
            if any(self._norm(skill.name) == wanted for skill in agent.skills):
                return agent
        return None

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
    ) -> SolutionProposal:
        if not report.missing_capabilities:
            return proposal

        req_by_name = {self._norm(req.name): req for req in request.requirements}
        generated = list(proposal.generated_nlp_agents)
        changed_tasks: list[TaskSpec] = []
        changed = False
        lifecycle = f"plan:{abs(hash(request.user_goal))}"

        for task in proposal.tasks:
            if task.assigned_to or task.auxiliary_spec_id:
                changed_tasks.append(task)
                continue
            requirement = req_by_name.get(self._norm(task.requirement_name))
            if requirement is None or requirement.name not in report.missing_capabilities:
                changed_tasks.append(task)
                continue
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

    def _linguistic(self) -> LingoLinguisticCoordinator:
        if self._linguistic_coordinator is None:
            self._linguistic_coordinator = LingoLinguisticCoordinator()
        return self._linguistic_coordinator

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
    def _completion_criteria(request: ProblemRequest) -> list[str]:
        if request.required_artifacts:
            return [
                f"{artifact} artifact exists"
                for artifact in request.required_artifacts
            ]
        return ["all task validators pass"]

    @staticmethod
    def _task_expected_artifacts(
        requirement,
        request: ProblemRequest,
    ) -> list[str]:
        if len(request.requirements) == 1:
            return list(request.required_artifacts)
        if requirement.name in request.required_artifacts:
            return [requirement.name]
        return []

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().lower().replace("_", " ")
