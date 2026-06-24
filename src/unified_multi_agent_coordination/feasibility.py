"""Deterministic feasibility checks for candidate coordination plans."""

from __future__ import annotations

from collections import defaultdict, deque

from .models import (
    AgentRegistryEntry,
    FeasibilityReport,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


class FeasibilityAnalyzer:
    """Authorize or reject candidate plans against a registry snapshot."""

    def __init__(self, trust_order: list[str] | None = None) -> None:
        self.trust_order = trust_order or ["standard", "elevated", "admin"]

    def check(
        self,
        request: ProblemRequest,
        registry: list[AgentRegistryEntry],
        proposal: SolutionProposal,
    ) -> FeasibilityReport:
        evidence: list[PredicateEvidence] = []
        missing: list[str] = []
        risks: list[str] = []
        matched: dict[str, str] = {}

        task_by_id = {task.task_id: task for task in proposal.tasks}
        req_by_name = {self._norm(req.name): req for req in request.requirements}
        agent_by_id = {agent.agent_id: agent for agent in registry}
        aux_by_id = {aux.spec_id: aux for aux in proposal.generated_nlp_agents}

        coverage_ok = True
        authority_ok = True
        compatibility_ok = True

        for task in proposal.tasks:
            req = req_by_name.get(self._norm(task.requirement_name))
            if req is None:
                coverage_ok = False
                missing.append(task.requirement_name)
                risks.append(f"Task {task.task_id} references an unknown requirement.")
                continue

            if task.assigned_to:
                agent = agent_by_id.get(task.assigned_to)
                if not agent:
                    coverage_ok = False
                    missing.append(req.name)
                    risks.append(f"Task {task.task_id} selects an unknown agent.")
                    continue
                if agent.status != "available":
                    coverage_ok = False
                    missing.append(req.name)
                    risks.append(f"Agent {agent.agent_id} is unavailable.")
                    continue
                if not self._agent_covers(agent, req.name):
                    coverage_ok = False
                    missing.append(req.name)
                    risks.append(f"Agent {agent.agent_id} does not advertise {req.name}.")
                    continue
                if not self._trust_satisfies(agent.trust_level, req.required_trust_level):
                    authority_ok = False
                    risks.append(
                        f"Agent {agent.agent_id} has trust {agent.trust_level}, "
                        f"but {req.required_trust_level} is required."
                    )
                matched[task.task_id] = agent.agent_id
                continue

            if task.auxiliary_spec_id:
                aux = aux_by_id.get(task.auxiliary_spec_id)
                if aux and req.auxiliary_eligible and not aux.persists:
                    continue
                coverage_ok = False
                missing.append(req.name)
                risks.append(f"Auxiliary task {task.task_id} is not admissible.")
                continue

            coverage_ok = False
            missing.append(req.name)
            risks.append(f"Task {task.task_id} has no accountable executor.")

        acyclic_ok = self._is_acyclic(proposal.tasks)
        if not acyclic_ok:
            risks.append("The candidate dependency graph contains a cycle.")

        for task in proposal.tasks:
            consumer_req = req_by_name.get(self._norm(task.requirement_name))
            if not consumer_req:
                continue
            for dependency in task.depends_on:
                producer = task_by_id.get(dependency)
                if not producer:
                    compatibility_ok = False
                    risks.append(f"Task {task.task_id} depends on unknown task {dependency}.")
                    continue
                producer_req = req_by_name.get(self._norm(producer.requirement_name))
                if producer_req and not self._modes_compatible(
                    producer_req.output_modes, consumer_req.input_modes
                ):
                    compatibility_ok = False
                    risks.append(
                        f"Artifact modes from {producer.task_id} are incompatible "
                        f"with {task.task_id}."
                    )

        artifacts_ok = all(
            artifact in proposal.expected_artifacts
            for artifact in request.required_artifacts
        )
        if not artifacts_ok:
            risks.append("The candidate plan does not promise all required artifacts.")

        verifiable_ok = bool(proposal.completion_criteria)
        if not verifiable_ok:
            risks.append("The candidate plan lacks decidable completion criteria.")

        evidence.extend(
            [
                PredicateEvidence(name="covered", passed=coverage_ok),
                PredicateEvidence(name="authorized", passed=authority_ok),
                PredicateEvidence(name="compatible", passed=compatibility_ok),
                PredicateEvidence(name="acyclic", passed=acyclic_ok),
                PredicateEvidence(name="complete", passed=artifacts_ok),
                PredicateEvidence(name="verifiable", passed=verifiable_ok),
            ]
        )

        feasible = all(item.passed for item in evidence)
        explanation = (
            "Plan authorized by deterministic feasibility checks."
            if feasible
            else "Plan rejected by deterministic feasibility checks."
        )
        return FeasibilityReport(
            feasible=feasible,
            matched_agents=matched,
            generated_nlp_agents=proposal.generated_nlp_agents,
            missing_capabilities=sorted(set(missing)),
            risks=risks,
            explanation=explanation,
            evidence=evidence,
        )

    def _agent_covers(self, agent: AgentRegistryEntry, requirement_name: str) -> bool:
        wanted = self._norm(requirement_name)
        return any(self._norm(skill.name) == wanted for skill in agent.skills)

    def _trust_satisfies(self, actual: str, required: str) -> bool:
        try:
            return self.trust_order.index(actual) >= self.trust_order.index(required)
        except ValueError:
            return actual == required

    @staticmethod
    def _modes_compatible(outputs: list[str], inputs: list[str]) -> bool:
        if not outputs or not inputs:
            return True
        return bool(set(outputs) & set(inputs))

    @staticmethod
    def _is_acyclic(tasks: list[TaskSpec]) -> bool:
        task_ids = {task.task_id for task in tasks}
        indegree = {task_id: 0 for task_id in task_ids}
        outgoing: dict[str, list[str]] = defaultdict(list)

        for task in tasks:
            for dependency in task.depends_on:
                if dependency not in task_ids:
                    return False
                outgoing[dependency].append(task.task_id)
                indegree[task.task_id] += 1

        queue = deque(task_id for task_id, degree in indegree.items() if degree == 0)
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for neighbor in outgoing[current]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        return visited == len(task_ids)

    @staticmethod
    def _norm(value: str) -> str:
        return value.strip().lower().replace("_", " ")
