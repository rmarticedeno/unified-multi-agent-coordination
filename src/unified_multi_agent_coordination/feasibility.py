"""Deterministic feasibility checks for candidate coordination plans."""

from __future__ import annotations

from collections import defaultdict, deque

from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    FeasibilityReport,
    GeneratedNlpAgentSpec,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


class FeasibilityAnalyzer:
    """Authorize or reject candidate plans against a registry snapshot."""

    APPROVED_AUXILIARY_METHODS = {
        "schema_extraction",
        "label_classification",
        "normalization",
    }

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

        structure_ok = self._structure_ok(proposal.tasks, risks)
        order_ok = self._execution_order_ok(proposal, risks)
        total_assignment_ok = True
        coverage_ok = True
        authority_ok = True
        compatibility_ok = True

        for task in proposal.tasks:
            has_agent = bool(task.assigned_to)
            has_auxiliary = bool(task.auxiliary_spec_id)
            if has_agent == has_auxiliary:
                total_assignment_ok = False
                coverage_ok = False
                if has_agent:
                    risks.append(
                        f"Task {task.task_id} selects both an agent and an auxiliary spec."
                    )
                else:
                    missing.append(task.requirement_name)
                    risks.append(f"Task {task.task_id} has no accountable executor.")

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
                skill = self._agent_skill(agent, req.name)
                if skill is None:
                    coverage_ok = False
                    missing.append(req.name)
                    risks.append(f"Agent {agent.agent_id} does not advertise {req.name}.")
                    continue
                if not self._contract_modes_compatible(skill, req):
                    compatibility_ok = False
                    risks.append(
                        f"Agent {agent.agent_id} advertises {req.name}, "
                        "but its modes are incompatible with the requirement."
                    )
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
                if aux and self._auxiliary_admissible(aux, req, risks):
                    continue
                coverage_ok = False
                missing.append(req.name)
                risks.append(f"Auxiliary task {task.task_id} is not admissible.")
                continue

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

        required_names = {self._norm(req.name) for req in request.requirements}
        planned_names = {self._norm(task.requirement_name) for task in proposal.tasks}
        requirements_ok = required_names <= planned_names
        if not requirements_ok:
            for requirement in request.requirements:
                if self._norm(requirement.name) not in planned_names:
                    missing.append(requirement.name)
                    risks.append(f"Requirement {requirement.name} is not planned.")

        artifacts_ok = all(
            self._norm(artifact)
            in {self._norm(item) for item in proposal.expected_artifacts}
            for artifact in request.required_artifacts
        )
        if not artifacts_ok:
            risks.append("The candidate plan does not promise all required artifacts.")

        verifiable_ok = bool(proposal.completion_criteria) and all(
            self._task_verifiable(task, req_by_name) for task in proposal.tasks
        )
        if not verifiable_ok:
            risks.append("The candidate plan lacks decidable validation evidence.")

        evidence.extend(
            [
                PredicateEvidence(name="well_formed", passed=structure_ok),
                PredicateEvidence(name="covered", passed=coverage_ok),
                PredicateEvidence(name="total_assignment", passed=total_assignment_ok),
                PredicateEvidence(name="authorized", passed=authority_ok),
                PredicateEvidence(name="compatible", passed=compatibility_ok),
                PredicateEvidence(name="acyclic", passed=acyclic_ok),
                PredicateEvidence(name="ordered", passed=order_ok),
                PredicateEvidence(name="requirements_complete", passed=requirements_ok),
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
        return self._agent_skill(agent, requirement_name) is not None

    def _agent_skill(
        self, agent: AgentRegistryEntry, requirement_name: str
    ) -> CapabilityRequirement | None:
        wanted = self._norm(requirement_name)
        for skill in agent.skills:
            if self._norm(skill.name) == wanted:
                return skill
        return None

    def _contract_modes_compatible(
        self,
        skill: CapabilityRequirement,
        requirement: CapabilityRequirement,
    ) -> bool:
        return self._modes_compatible(
            skill.input_modes,
            requirement.input_modes,
        ) and self._modes_compatible(
            skill.output_modes,
            requirement.output_modes,
        )

    def _auxiliary_admissible(
        self,
        aux: GeneratedNlpAgentSpec,
        requirement: CapabilityRequirement,
        risks: list[str],
    ) -> bool:
        ok = True
        if not requirement.auxiliary_eligible:
            risks.append(f"Requirement {requirement.name} is not auxiliary eligible.")
            ok = False
        if aux.method not in self.APPROVED_AUXILIARY_METHODS:
            risks.append(f"Auxiliary method {aux.method} is not approved.")
            ok = False
        if aux.persists:
            risks.append(f"Auxiliary spec {aux.spec_id} persists beyond the plan.")
            ok = False
        if not aux.lifecycle:
            risks.append(f"Auxiliary spec {aux.spec_id} lacks a lifecycle token.")
            ok = False
        if not aux.validation_rule:
            risks.append(f"Auxiliary spec {aux.spec_id} lacks a validation rule.")
            ok = False
        if "read_only" not in aux.authority_bounds:
            risks.append(f"Auxiliary spec {aux.spec_id} is not read-only bounded.")
            ok = False
        if not self._schema_compatible(aux.output_schema, requirement.output_schema):
            risks.append(
                f"Auxiliary spec {aux.spec_id} does not promise the required output schema."
            )
            ok = False
        return ok

    def _task_verifiable(
        self,
        task: TaskSpec,
        req_by_name: dict[str, CapabilityRequirement],
    ) -> bool:
        if task.validation_contract:
            return True
        requirement = req_by_name.get(self._norm(task.requirement_name))
        if requirement is None:
            return False
        return bool(
            task.expected_artifacts
            or requirement.expected_evidence
            or requirement.validation_contract
            or not (task.validation_contract or requirement.validation_contract)
        )

    def _structure_ok(self, tasks: list[TaskSpec], risks: list[str]) -> bool:
        ok = True
        if not tasks:
            risks.append("The candidate plan contains no tasks.")
            return False
        seen: set[str] = set()
        for task in tasks:
            if task.task_id in seen:
                risks.append(f"Duplicate task id {task.task_id}.")
                ok = False
            seen.add(task.task_id)
            if not task.requirement_name.strip():
                risks.append(f"Task {task.task_id} lacks a requirement name.")
                ok = False
        return ok

    def _execution_order_ok(
        self,
        proposal: SolutionProposal,
        risks: list[str],
    ) -> bool:
        task_ids = [task.task_id for task in proposal.tasks]
        if not proposal.execution_order:
            return True
        if set(proposal.execution_order) != set(task_ids):
            risks.append("Execution order does not include exactly the planned tasks.")
            return False
        positions = {task_id: index for index, task_id in enumerate(proposal.execution_order)}
        for task in proposal.tasks:
            for dependency in task.depends_on:
                if dependency in positions and positions[dependency] > positions[task.task_id]:
                    risks.append(
                        f"Execution order places {task.task_id} before dependency {dependency}."
                    )
                    return False
        return True

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
    def _schema_compatible(provided: dict, required: dict) -> bool:
        required_fields = required.get("required", [])
        provided_fields = provided.get("required", [])
        if not isinstance(required_fields, list) or not required_fields:
            return True
        if not isinstance(provided_fields, list):
            return False
        return set(required_fields) <= set(provided_fields)

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
