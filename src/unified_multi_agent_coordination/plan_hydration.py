"""Fail-closed hydration of non-executable linguistic plan drafts."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import (
    AgentRegistryEntry,
    CompletionContract,
    DraftRequirementSelection,
    HydrationIssue,
    LinguisticPlanDraft,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


class PlanHydrationResult(BaseModel):
    proposal: SolutionProposal | None = None
    issues: list[HydrationIssue] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.proposal is not None and not self.issues


class PlanHydrator:
    """Construct executable plans solely from admitted request and registry data."""

    def hydrate(
        self,
        request: ProblemRequest,
        registry: list[AgentRegistryEntry],
        draft: LinguisticPlanDraft,
    ) -> PlanHydrationResult:
        requirements = {item.requirement_id: item for item in request.requirements}
        issues: list[HydrationIssue] = []
        selections = {}
        for selection in draft.selections:
            rid = selection.requirement_id
            if rid not in requirements:
                issues.append(HydrationIssue(
                    code="unknown_requirement",
                    requirement_id=rid,
                    message=f"Draft references unknown requirement {rid!r}.",
                ))
            elif rid in selections:
                issues.append(HydrationIssue(
                    code="duplicate_requirement",
                    requirement_id=rid,
                    message=f"Requirement {rid!r} is selected more than once.",
                ))
            else:
                selections[rid] = selection

        for rid in requirements:
            if rid not in selections:
                issues.append(HydrationIssue(
                    code="missing_requirement",
                    requirement_id=rid,
                    message=f"Draft omits requirement {rid!r}.",
                ))
        for term in draft.unresolved_terms:
            issues.append(HydrationIssue(
                code="unresolved_term",
                message=f"Unresolved linguistic term: {term}",
            ))

        agents_by_capability: dict[str, list[AgentRegistryEntry]] = {}
        for agent in registry:
            if agent.status != "available":
                continue
            for skill in agent.skills:
                agents_by_capability.setdefault(skill.capability_id, []).append(agent)
        for candidates in agents_by_capability.values():
            candidates.sort(key=lambda item: item.agent_id)

        for rid, selection in selections.items():
            requirement = requirements[rid]
            if selection.capability_id != requirement.capability_id:
                issues.append(HydrationIssue(
                    code="unknown_capability",
                    requirement_id=rid,
                    message=(f"Capability {selection.capability_id!r} is not the declared "
                             f"capability for requirement {rid!r}."),
                ))
            elif not agents_by_capability.get(selection.capability_id):
                issues.append(HydrationIssue(
                    code="unavailable_capability",
                    requirement_id=rid,
                    message=f"No available agent advertises {selection.capability_id!r}.",
                ))
            for dependency in selection.depends_on_requirement_ids:
                if dependency == rid or dependency not in requirements:
                    issues.append(HydrationIssue(
                        code="invalid_dependency",
                        requirement_id=rid,
                        message=f"Invalid dependency {dependency!r} for {rid!r}.",
                    ))
            declared = set(requirement.depends_on_requirement_ids)
            proposed = set(selection.depends_on_requirement_ids)
            if (
                "depends_on_requirement_ids" in requirement.model_fields_set
                and declared != proposed
            ):
                issues.append(HydrationIssue(
                    code="invalid_dependency",
                    requirement_id=rid,
                    message=(f"Dependencies for {rid!r} must equal the admitted set "
                             f"{sorted(declared)!r}."),
                ))

        if not issues and self._has_cycle(selections):
            issues.append(HydrationIssue(
                code="dependency_cycle",
                message="Requirement dependencies contain a cycle.",
            ))
        if issues:
            return PlanHydrationResult(issues=issues)

        task_ids = {
            requirement.requirement_id: f"t{index}"
            for index, requirement in enumerate(request.requirements, start=1)
        }
        tasks: list[TaskSpec] = []
        selected_agents: dict[str, str] = {}
        for requirement in request.requirements:
            selection = selections[requirement.requirement_id]
            task_id = task_ids[requirement.requirement_id]
            agent = agents_by_capability[selection.capability_id][0]
            artifacts = list(requirement.validation_contract.required_artifacts)
            tasks.append(TaskSpec(
                task_id=task_id,
                requirement_name=requirement.name,
                requirement_id=requirement.requirement_id,
                capability_id=requirement.capability_id,
                assigned_to=agent.agent_id,
                depends_on=[task_ids[item] for item in selection.depends_on_requirement_ids],
                expected_artifacts=artifacts,
                validation_contract=requirement.validation_contract.model_copy(deep=True),
            ))
            selected_agents[task_id] = agent.agent_id

        order = self._topological_order(tasks)
        return PlanHydrationResult(proposal=SolutionProposal(
            tasks=tasks,
            selected_agents=selected_agents,
            execution_order=order,
            expected_artifacts=list(request.required_artifacts),
            completion_criteria=[f"artifact:{item}" for item in request.required_artifacts],
            completion_contract=CompletionContract(
                required_artifacts=list(request.required_artifacts)
            ),
        ))

    @staticmethod
    def _has_cycle(selections: dict[str, DraftRequirementSelection]) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(rid: str) -> bool:
            if rid in visiting:
                return True
            if rid in visited:
                return False
            visiting.add(rid)
            selection = selections[rid]
            if any(visit(dep) for dep in selection.depends_on_requirement_ids):
                return True
            visiting.remove(rid)
            visited.add(rid)
            return False

        return any(visit(rid) for rid in selections)

    @staticmethod
    def _topological_order(tasks: list[TaskSpec]) -> list[str]:
        remaining = {task.task_id: set(task.depends_on) for task in tasks}
        order: list[str] = []
        while remaining:
            ready = sorted(task_id for task_id, deps in remaining.items() if not deps)
            for task_id in ready:
                order.append(task_id)
                remaining.pop(task_id)
                for dependencies in remaining.values():
                    dependencies.discard(task_id)
        return order
