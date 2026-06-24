"""Coordination-agent facade for registry-aware plan construction."""

from __future__ import annotations

from typing import Any

from .coordination_sdk import CoordinationSdk, RemoteRegistryError
from .feasibility import FeasibilityAnalyzer
from .lingo_coordinator import LingoLinguisticCoordinator
from .models import (
    AgentRegistryEntry,
    CoordinationPlanResult,
    FeasibilityReport,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


class CoordinationAgent:
    """Build candidate plans from the SDK registry and authorize them."""

    def __init__(
        self,
        sdk: CoordinationSdk,
        *,
        feasibility_analyzer: FeasibilityAnalyzer | None = None,
        linguistic_coordinator: LingoLinguisticCoordinator | None = None,
        self_agent_id: str | None = None,
    ) -> None:
        self.sdk = sdk
        self.feasibility_analyzer = feasibility_analyzer or FeasibilityAnalyzer()
        self._linguistic_coordinator = linguistic_coordinator
        self.self_agent_id = self_agent_id or sdk.self_agent_id

    async def build_solution_plan(
        self,
        user_request: str | ProblemRequest,
        context: dict[str, Any] | None = None,
    ) -> CoordinationPlanResult:
        """Build and check a candidate solution plan against the live registry."""
        try:
            registry = await self.sdk.registry_snapshot(refresh=True)
        except RemoteRegistryError as exc:
            return self._registry_failure_result(user_request, context, exc)

        registry = self._without_self(registry)
        request = await self._admit_request(user_request, registry, context)
        proposal = self._direct_solution_plan(request, registry)
        if proposal is None:
            proposal = await self._linguistic().propose_solution(
                request, registry
            )
        report = self.feasibility_analyzer.check(request, registry, proposal)
        return CoordinationPlanResult(
            request=request,
            proposal=proposal,
            feasibility_report=report,
            registry_snapshot=registry,
        )

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

    @staticmethod
    def _completion_criteria(request: ProblemRequest) -> list[str]:
        if request.required_artifacts:
            return [
                f"{artifact} artifact exists"
                for artifact in request.required_artifacts
            ]
        return ["all task validators pass"]

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().lower().replace("_", " ")
