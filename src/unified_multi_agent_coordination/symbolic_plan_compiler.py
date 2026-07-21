"""Deterministic plan compilation and bounded provider-assignment search."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

from .feasibility import FeasibilityAnalyzer
from .models import (
    AgentRegistryEntry,
    CompletionContract,
    FeasibilityReport,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


class PlanCompilationIssue(BaseModel):
    code: Literal[
        "duplicate_requirement",
        "unknown_dependency",
        "dependency_cycle",
        "assignment_search_exhausted",
    ]
    message: str


class PlanCompilationDiagnostics(BaseModel):
    provider_candidates: dict[str, list[str]] = Field(default_factory=dict)
    provider_rejections: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    assignments_total: int = 0
    assignments_considered: int = 0
    branches_explored: int = 0
    branches_pruned: int = 0
    recovered_alternative_provider: bool = False
    search_exhausted: bool = False
    failed_predicates: list[str] = Field(default_factory=list)
    conflict_requirements: list[str] = Field(default_factory=list)
    issues: list[PlanCompilationIssue] = Field(default_factory=list)


class PlanCompilationResult(BaseModel):
    proposal: SolutionProposal
    report: FeasibilityReport
    diagnostics: PlanCompilationDiagnostics


class SymbolicPlanCompiler:
    """Compile authoritative requests and search deterministic provider assignments."""

    AVAILABILITY_RANK = {"replicated": 0, "remote": 1, "node_local": 2}

    def __init__(
        self,
        feasibility_analyzer: FeasibilityAnalyzer | None = None,
        *,
        max_assignment_evaluations: int = 4096,
        prefilter_providers: bool = True,
    ) -> None:
        if max_assignment_evaluations < 1:
            raise ValueError("max_assignment_evaluations must be positive.")
        self.feasibility_analyzer = feasibility_analyzer or FeasibilityAnalyzer()
        self.max_assignment_evaluations = max_assignment_evaluations
        self.prefilter_providers = prefilter_providers

    def compile(
        self,
        request: ProblemRequest,
        registry: list[AgentRegistryEntry],
    ) -> PlanCompilationResult:
        issues = self._request_issues(request)
        if issues:
            proposal = self._proposal(request, [None] * len(request.requirements))
            return PlanCompilationResult(
                proposal=proposal,
                report=self._invalid_report(issues),
                diagnostics=PlanCompilationDiagnostics(issues=issues),
            )

        request = request.model_copy(
            update={"requirements": self._topological_requirements(request)}
        )
        provider_sets: dict[str, list[AgentRegistryEntry | None]] = {}
        candidate_ids: dict[str, list[str]] = {}
        rejection_diagnostics: dict[str, dict[str, list[str]]] = {}
        initial_assignment: list[AgentRegistryEntry | None] = []
        for requirement in request.requirements:
            advertised = [
                agent
                for agent in registry
                if agent.status == "available"
                and any(
                    skill.capability_id == requirement.capability_id
                    for skill in agent.skills
                )
            ]
            advertised.sort(key=lambda item: self._provider_rank(requirement, item))
            initial_assignment.append(advertised[0] if advertised else None)
            exact: list[AgentRegistryEntry] = []
            rejected: dict[str, list[str]] = {}
            for agent in registry:
                reasons = self._provider_rejection_reasons(request, requirement, agent)
                if reasons:
                    rejected[agent.agent_id] = reasons
                elif self.prefilter_providers:
                    exact.append(agent)
            if not self.prefilter_providers:
                exact = list(advertised)
            exact.sort(key=lambda item: self._provider_rank(requirement, item))
            candidate_ids[requirement.requirement_id] = [
                item.agent_id for item in exact
            ]
            rejection_diagnostics[requirement.requirement_id] = rejected
            provider_sets[requirement.requirement_id] = list(exact) or [None]

        assignments_total = math.prod(len(item) for item in provider_sets.values())
        considered = 0
        branches_explored = 0
        branches_pruned = 0
        best_proposal: SolutionProposal | None = None
        best_report: FeasibilityReport | None = None
        best_score = -1
        selected_assignment: tuple[AgentRegistryEntry | None, ...] | None = None
        requirement_index = {
            item.requirement_id: index for index, item in enumerate(request.requirements)
        }
        search_order = sorted(
            request.requirements,
            key=lambda item: (len(provider_sets[item.requirement_id]), requirement_index[item.requirement_id]),
        )
        partial: dict[str, AgentRegistryEntry | None] = {}

        def search(depth: int) -> bool:
            nonlocal considered, branches_explored, branches_pruned
            nonlocal best_proposal, best_report, best_score, selected_assignment
            if considered >= self.max_assignment_evaluations:
                return False
            if depth == len(search_order):
                considered += 1
                ordered_assignment = tuple(
                    partial[item.requirement_id] for item in request.requirements
                )
                proposal = self._proposal(request, list(ordered_assignment))
                report = self.feasibility_analyzer.check(request, registry, proposal)
                score = sum(item.passed for item in report.evidence)
                if score > best_score:
                    best_proposal, best_report, best_score = proposal, report, score
                if report.feasible:
                    best_proposal, best_report = proposal, report
                    selected_assignment = ordered_assignment
                    return True
                return False
            requirement = search_order[depth]
            candidates = provider_sets[requirement.requirement_id]
            if candidates == [None]:
                branches_pruned += 1
            for candidate in candidates:
                if considered >= self.max_assignment_evaluations:
                    return False
                branches_explored += 1
                partial[requirement.requirement_id] = candidate
                if search(depth + 1):
                    return True
            partial.pop(requirement.requirement_id, None)
            return False

        search(0)

        assert best_proposal is not None and best_report is not None
        exhausted = (
            not best_report.feasible
            and assignments_total > self.max_assignment_evaluations
            and considered >= self.max_assignment_evaluations
        )
        compilation_issues: list[PlanCompilationIssue] = []
        if exhausted:
            compilation_issues.append(PlanCompilationIssue(
                code="assignment_search_exhausted",
                message=(
                    f"Provider search reached {self.max_assignment_evaluations} "
                    "assignments without authorization."
                ),
            ))
            best_report = best_report.model_copy(update={
                "feasible": False,
                "risks": [
                    *best_report.risks,
                    compilation_issues[0].message,
                ],
                "evidence": [
                    *best_report.evidence,
                    PredicateEvidence(name="assignment_search_complete", passed=False),
                ],
            })
        recovered = False
        if selected_assignment is not None:
            initial_report = self.feasibility_analyzer.check(
                request,
                registry,
                self._proposal(request, initial_assignment),
            )
            recovered = not initial_report.feasible and any(
                initial is not None
                and selected is not None
                and initial.agent_id != selected.agent_id
                for initial, selected in zip(
                    initial_assignment, selected_assignment, strict=True
                )
            )
        failed_predicates = sorted(
            item.name for item in best_report.evidence if not item.passed
        )
        conflict_requirements = sorted(
            requirement_id
            for requirement_id, candidates in provider_sets.items()
            if candidates == [None]
        )
        return PlanCompilationResult(
            proposal=best_proposal,
            report=best_report,
            diagnostics=PlanCompilationDiagnostics(
                provider_candidates=candidate_ids,
                provider_rejections=rejection_diagnostics,
                assignments_total=assignments_total,
                assignments_considered=considered,
                branches_explored=branches_explored,
                branches_pruned=branches_pruned,
                recovered_alternative_provider=recovered,
                search_exhausted=exhausted,
                failed_predicates=failed_predicates,
                conflict_requirements=conflict_requirements,
                issues=compilation_issues,
            ),
        )

    def _provider_rejection_reasons(
        self,
        request: ProblemRequest,
        requirement,
        agent: AgentRegistryEntry,
    ) -> list[str]:
        reasons: list[str] = []
        if agent.status != "available":
            reasons.append("unavailable")
        skill = next(
            (item for item in agent.skills if item.capability_id == requirement.capability_id),
            None,
        )
        if skill is None:
            reasons.append("capability")
            return reasons
        if not self.feasibility_analyzer._contract_modes_compatible(skill, requirement):
            reasons.append("modes")
        if not self.feasibility_analyzer._schema_compatible(
            skill.input_schema, requirement.input_schema
        ) or not self.feasibility_analyzer._schema_compatible(
            skill.output_schema, requirement.output_schema
        ):
            reasons.append("schema")
        if not self.feasibility_analyzer._trust_satisfies(
            agent.trust_level, requirement.required_trust_level
        ):
            reasons.append("trust")
        if (
            self.feasibility_analyzer.require_effect_fencing
            and requirement.side_effect_class in {"unsafe", "unknown"}
            and not agent.supports_fencing
        ):
            reasons.append("fencing")
        for constraint in request.constraints:
            if constraint.source != "agent" or constraint.requirement_id not in {
                "", requirement.requirement_id
            }:
                continue
            if constraint.path != "/agent_id":
                continue
            if not self.feasibility_analyzer._compare_constraint(
                constraint, True, agent.agent_id
            ):
                reasons.append(f"constraint:{constraint.constraint_id}")
        return sorted(set(reasons))

    def _provider_rank(self, requirement, agent: AgentRegistryEntry) -> tuple[int, int, int, str]:
        trust_order = self.feasibility_analyzer.trust_order
        try:
            actual = trust_order.index(agent.trust_level)
            required = trust_order.index(requirement.required_trust_level)
            trust_surplus = actual - required if actual >= required else 999
        except ValueError:
            trust_surplus = 0 if agent.trust_level == requirement.required_trust_level else 999
        fencing_rank = (
            0
            if requirement.side_effect_class in {"unsafe", "unknown"}
            and agent.supports_fencing
            else 1
        )
        return (
            trust_surplus,
            fencing_rank,
            self.AVAILABILITY_RANK.get(agent.availability_scope, 99),
            agent.agent_id,
        )

    def _proposal(
        self,
        request: ProblemRequest,
        assignment: list[AgentRegistryEntry | None],
    ) -> SolutionProposal:
        task_ids = {
            item.requirement_id: f"t{index}"
            for index, item in enumerate(request.requirements, start=1)
        }
        tasks = [
            TaskSpec(
                task_id=task_ids[requirement.requirement_id],
                requirement_name=requirement.name,
                requirement_id=requirement.requirement_id,
                capability_id=requirement.capability_id,
                assigned_to=agent.agent_id if agent is not None else None,
                depends_on=[
                    task_ids[item]
                    for item in requirement.depends_on_requirement_ids
                    if item in task_ids
                ],
                expected_artifacts=(
                    list(requirement.validation_contract.required_artifacts)
                    or (
                        list(request.required_artifacts)
                        if len(request.requirements) == 1
                        else []
                    )
                ),
                validation_contract=requirement.validation_contract.model_copy(deep=True),
            )
            for requirement, agent in zip(
                request.requirements, assignment, strict=True
            )
        ]
        return SolutionProposal(
            tasks=tasks,
            selected_agents={
                task.task_id: task.assigned_to
                for task in tasks
                if task.assigned_to is not None
            },
            execution_order=[task_ids[item.requirement_id] for item in request.requirements],
            expected_artifacts=list(request.required_artifacts),
            completion_criteria=[
                f"artifact:{item}" for item in request.required_artifacts
            ],
            completion_contract=CompletionContract(
                required_artifacts=list(request.required_artifacts)
            ),
        )

    def _request_issues(self, request: ProblemRequest) -> list[PlanCompilationIssue]:
        identifiers = [item.requirement_id for item in request.requirements]
        issues: list[PlanCompilationIssue] = []
        if len(identifiers) != len(set(identifiers)):
            issues.append(PlanCompilationIssue(
                code="duplicate_requirement",
                message="Requirement identifiers must be unique.",
            ))
        known = set(identifiers)
        for requirement in request.requirements:
            unknown = set(requirement.depends_on_requirement_ids) - known
            if unknown:
                issues.append(PlanCompilationIssue(
                    code="unknown_dependency",
                    message=(
                        f"Requirement {requirement.requirement_id!r} references unknown "
                        f"dependencies {sorted(unknown)!r}."
                    ),
                ))
        if not issues and self._has_cycle(request):
            issues.append(PlanCompilationIssue(
                code="dependency_cycle",
                message="The admitted request dependency graph contains a cycle.",
            ))
        return issues

    @staticmethod
    def _has_cycle(request: ProblemRequest) -> bool:
        dependencies = {
            item.requirement_id: item.depends_on_requirement_ids
            for item in request.requirements
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(requirement_id: str) -> bool:
            if requirement_id in visiting:
                return True
            if requirement_id in visited:
                return False
            visiting.add(requirement_id)
            if any(visit(item) for item in dependencies[requirement_id]):
                return True
            visiting.remove(requirement_id)
            visited.add(requirement_id)
            return False

        return any(visit(item) for item in dependencies)

    @staticmethod
    def _topological_requirements(
        request: ProblemRequest,
    ) -> list:
        requirements = {
            item.requirement_id: item
            for item in request.requirements
        }
        request_order = {
            item.requirement_id: index
            for index, item in enumerate(request.requirements)
        }
        indegree = {
            identifier: len(item.depends_on_requirement_ids)
            for identifier, item in requirements.items()
        }
        dependants: dict[str, list[str]] = {identifier: [] for identifier in requirements}
        for identifier, item in requirements.items():
            for dependency in item.depends_on_requirement_ids:
                dependants[dependency].append(identifier)
        ready = sorted(
            (identifier for identifier, degree in indegree.items() if degree == 0),
            key=request_order.__getitem__,
        )
        ordered = []
        while ready:
            identifier = ready.pop(0)
            ordered.append(requirements[identifier])
            for dependant in sorted(
                dependants[identifier],
                key=request_order.__getitem__,
            ):
                indegree[dependant] -= 1
                if indegree[dependant] == 0:
                    ready.append(dependant)
                    ready.sort(key=request_order.__getitem__)
        return ordered

    @staticmethod
    def _invalid_report(issues: list[PlanCompilationIssue]) -> FeasibilityReport:
        return FeasibilityReport(
            feasible=False,
            risks=[item.message for item in issues],
            explanation="The authoritative request could not be compiled.",
            evidence=[PredicateEvidence(name="request_compilable", passed=False)],
        )
