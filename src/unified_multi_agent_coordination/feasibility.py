"""Fail-closed deterministic feasibility checks for candidate plans."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    ConstraintSpec,
    FeasibilityReport,
    GeneratedNlpAgentSpec,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
    ValidationContract,
)


class FeasibilityAnalyzer:
    """Authorize only plans whose explicit contracts can be checked."""

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
        missing: list[str] = []
        risks: list[str] = []
        matched: dict[str, str] = {}
        validation_gaps: list[dict[str, Any]] = []
        schema_violations: list[dict[str, Any]] = []
        capability_resolution: list[dict[str, Any]] = []
        task_contexts: dict[str, dict[str, Any]] = {}

        task_by_id = {task.task_id: task for task in proposal.tasks}
        req_by_id = {req.requirement_id: req for req in request.requirements}
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
                if not has_agent:
                    missing.append(task.requirement_name)
                message = (
                    f"Task {task.task_id} selects both an agent and an auxiliary spec."
                    if has_agent
                    else f"Task {task.task_id} has no accountable executor."
                )
                risks.append(message)

            requirement = self._requirement_for_task(task, req_by_id, req_by_name)
            if requirement is None:
                coverage_ok = False
                missing.append(task.requirement_name)
                risks.append(f"Task {task.task_id} references an unknown requirement ID.")
                continue

            if task.assigned_to:
                agent = agent_by_id.get(task.assigned_to)
                if agent is None:
                    coverage_ok = False
                    missing.append(requirement.name)
                    risks.append(f"Task {task.task_id} selects an unknown agent.")
                    continue
                if agent.status != "available":
                    coverage_ok = False
                    missing.append(requirement.name)
                    risks.append(f"Agent {agent.agent_id} is unavailable.")
                    continue
                skill = self._agent_skill(agent, task.capability_id)
                capability_resolution.append(
                    {
                        "task_id": task.task_id,
                        "requirement_id": requirement.requirement_id,
                        "requested_capability_id": task.capability_id,
                        "agent_id": agent.agent_id,
                        "resolved": skill is not None,
                        "resolved_capability_id": skill.capability_id if skill else "",
                    }
                )
                if skill is None:
                    coverage_ok = False
                    missing.append(requirement.name)
                    risks.append(
                        f"Agent {agent.agent_id} does not advertise capability ID "
                        f"{task.capability_id}."
                    )
                    continue
                if not self._contract_modes_compatible(skill, requirement):
                    compatibility_ok = False
                    risks.append(
                        f"Agent {agent.agent_id} does not cover every required mode for "
                        f"{requirement.name}."
                    )
                if not self._schema_compatible(skill.input_schema, requirement.input_schema):
                    compatibility_ok = False
                    schema_violations.append(
                        {"task_id": task.task_id, "location": "input", "agent_id": agent.agent_id}
                    )
                if not self._schema_compatible(skill.output_schema, requirement.output_schema):
                    compatibility_ok = False
                    schema_violations.append(
                        {"task_id": task.task_id, "location": "output", "agent_id": agent.agent_id}
                    )
                if not self._trust_satisfies(agent.trust_level, requirement.required_trust_level):
                    authority_ok = False
                    risks.append(
                        f"Agent {agent.agent_id} has trust {agent.trust_level}, but "
                        f"{requirement.required_trust_level} is required."
                    )
                matched[task.task_id] = agent.agent_id
                task_contexts[task.task_id] = {
                    "task": task.model_dump(mode="json"),
                    "agent": agent.model_dump(mode="json"),
                    "capability": skill.model_dump(mode="json"),
                    "skill_contract": skill.validation_contract,
                    "requirement": requirement,
                }
            elif task.auxiliary_spec_id:
                auxiliary = aux_by_id.get(task.auxiliary_spec_id)
                if auxiliary and self._auxiliary_admissible(auxiliary, requirement, risks):
                    task_contexts[task.task_id] = {
                        "task": task.model_dump(mode="json"),
                        "capability": auxiliary.model_dump(mode="json"),
                        "requirement": requirement,
                    }
                else:
                    coverage_ok = False
                    missing.append(requirement.name)
                    risks.append(f"Auxiliary task {task.task_id} is not admissible.")

        acyclic_ok = self._is_acyclic(proposal.tasks)
        if not acyclic_ok:
            risks.append("The candidate dependency graph contains a cycle.")

        for task in proposal.tasks:
            consumer = self._requirement_for_task(task, req_by_id, req_by_name)
            if consumer is None:
                continue
            for dependency in task.depends_on:
                producer = task_by_id.get(dependency)
                if producer is None:
                    compatibility_ok = False
                    risks.append(f"Task {task.task_id} depends on unknown task {dependency}.")
                    continue
                producer_req = self._requirement_for_task(producer, req_by_id, req_by_name)
                if producer_req and not self._dependency_modes_compatible(
                    producer_req.output_modes, consumer.input_modes
                ):
                    compatibility_ok = False
                    risks.append(
                        f"Artifact modes from {producer.task_id} are incompatible with {task.task_id}."
                    )

        required_ids = {req.requirement_id for req in request.requirements}
        planned_ids = {task.requirement_id for task in proposal.tasks}
        requirements_ok = required_ids <= planned_ids
        for requirement in request.requirements:
            if requirement.requirement_id not in planned_ids:
                missing.append(requirement.name)
                risks.append(f"Requirement {requirement.requirement_id} is not planned.")

        promised = {self._norm(item) for item in proposal.expected_artifacts}
        artifacts_ok = all(self._norm(item) in promised for item in request.required_artifacts)
        if not artifacts_ok:
            risks.append("The candidate plan does not promise all required artifacts.")

        completion_ok = bool(proposal.completion_contract.required_task_states)
        completion_ok = completion_ok and all(
            self._norm(item)
            in {self._norm(value) for value in proposal.completion_contract.required_artifacts}
            for item in request.required_artifacts
        )
        if not completion_ok:
            risks.append("The plan lacks a decidable completion contract.")

        verifiable_ok = True
        for task in proposal.tasks:
            requirement = self._requirement_for_task(task, req_by_id, req_by_name)
            auxiliary = aux_by_id.get(task.auxiliary_spec_id or "")
            skill_contract = (task_contexts.get(task.task_id) or {}).get("skill_contract")
            if not self._task_verifiable(task, requirement, auxiliary, skill_contract):
                verifiable_ok = False
                validation_gaps.append(
                    {
                        "task_id": task.task_id,
                        "requirement_id": task.requirement_id,
                        "reason": "no enforceable validation contract",
                    }
                )
        if not verifiable_ok:
            risks.append("The candidate plan lacks deterministic validation evidence.")

        constraints = [*request.constraints]
        for requirement in request.requirements:
            constraints.extend(requirement.constraints)
        constraint_violations = self._evaluate_constraints(
            constraints, request, proposal, task_contexts
        )
        constraints_ok = not constraint_violations
        if not constraints_ok:
            risks.extend(
                f"Constraint {item['constraint_id']} failed: {item['reason']}."
                for item in constraint_violations
            )

        evidence = [
            PredicateEvidence(name="well_formed", passed=structure_ok),
            PredicateEvidence(name="covered", passed=coverage_ok),
            PredicateEvidence(name="total_assignment", passed=total_assignment_ok),
            PredicateEvidence(name="authorized", passed=authority_ok),
            PredicateEvidence(name="compatible", passed=compatibility_ok),
            PredicateEvidence(name="acyclic", passed=acyclic_ok),
            PredicateEvidence(name="ordered", passed=order_ok),
            PredicateEvidence(name="requirements_complete", passed=requirements_ok),
            PredicateEvidence(name="complete", passed=artifacts_ok and completion_ok),
            PredicateEvidence(name="verifiable", passed=verifiable_ok),
            PredicateEvidence(
                name="constraints_satisfied",
                passed=constraints_ok,
                details={"violations": constraint_violations},
            ),
        ]
        feasible = all(item.passed for item in evidence)
        return FeasibilityReport(
            feasible=feasible,
            matched_agents=matched,
            generated_nlp_agents=proposal.generated_nlp_agents,
            missing_capabilities=sorted(set(missing)),
            risks=risks,
            explanation=(
                "Plan authorized by deterministic feasibility checks."
                if feasible
                else "Plan rejected by deterministic feasibility checks."
            ),
            evidence=evidence,
            constraint_violations=constraint_violations,
            validation_gaps=validation_gaps,
            schema_violations=schema_violations,
            capability_resolution=capability_resolution,
        )

    @staticmethod
    def _requirement_for_task(
        task: TaskSpec,
        req_by_id: dict[str, CapabilityRequirement],
        req_by_name: dict[str, CapabilityRequirement],
    ) -> CapabilityRequirement | None:
        return req_by_id.get(task.requirement_id) or req_by_name.get(
            FeasibilityAnalyzer._norm(task.requirement_name)
        )

    @staticmethod
    def _agent_skill(
        agent: AgentRegistryEntry, capability_id: str
    ) -> CapabilityRequirement | None:
        return next(
            (skill for skill in agent.skills if skill.capability_id == capability_id),
            None,
        )

    @staticmethod
    def _contract_modes_compatible(
        skill: CapabilityRequirement, requirement: CapabilityRequirement
    ) -> bool:
        return set(requirement.input_modes) <= set(skill.input_modes) and set(
            requirement.output_modes
        ) <= set(skill.output_modes)

    def _auxiliary_admissible(
        self,
        auxiliary: GeneratedNlpAgentSpec,
        requirement: CapabilityRequirement,
        risks: list[str],
    ) -> bool:
        checks = [
            (requirement.auxiliary_eligible, f"Requirement {requirement.name} is not auxiliary eligible."),
            (auxiliary.method in self.APPROVED_AUXILIARY_METHODS, f"Auxiliary method {auxiliary.method} is not approved."),
            (not auxiliary.persists, f"Auxiliary spec {auxiliary.spec_id} persists beyond the plan."),
            (bool(auxiliary.lifecycle), f"Auxiliary spec {auxiliary.spec_id} lacks a lifecycle token."),
            (bool(auxiliary.validation_rule), f"Auxiliary spec {auxiliary.spec_id} lacks a validation rule."),
            ("read_only" in auxiliary.authority_bounds, f"Auxiliary spec {auxiliary.spec_id} is not read-only bounded."),
            (self._schema_compatible(auxiliary.output_schema, requirement.output_schema), f"Auxiliary spec {auxiliary.spec_id} does not cover the required output schema."),
        ]
        for passed, message in checks:
            if not passed:
                risks.append(message)
        return all(passed for passed, _ in checks)

    @staticmethod
    def _task_verifiable(
        task: TaskSpec,
        requirement: CapabilityRequirement | None,
        auxiliary: GeneratedNlpAgentSpec | None,
        skill_contract: ValidationContract | None,
    ) -> bool:
        if auxiliary is not None:
            return bool(auxiliary.validation_rule and auxiliary.output_schema)
        contracts: list[ValidationContract | dict[str, Any]] = [task.validation_contract]
        if requirement is not None:
            contracts.append(requirement.validation_contract)
        if skill_contract is not None:
            contracts.append(skill_contract)
        return any(
            (
                contract
                if isinstance(contract, ValidationContract)
                else ValidationContract.model_validate(contract)
            ).enforceable()
            for contract in contracts
        )

    def _evaluate_constraints(
        self,
        constraints: list[ConstraintSpec],
        request: ProblemRequest,
        proposal: SolutionProposal,
        task_contexts: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        for constraint in constraints:
            if constraint.source == "unresolved" or constraint.operator == "unresolved":
                violations.append(
                    {"constraint_id": constraint.constraint_id, "reason": "unresolved legacy constraint"}
                )
                continue
            context = self._constraint_context(constraint, request, proposal, task_contexts)
            found, actual = self._json_pointer(context, constraint.path)
            passed = self._compare_constraint(constraint, found, actual)
            if not passed:
                violations.append(
                    {
                        "constraint_id": constraint.constraint_id,
                        "reason": "missing value" if not found else "predicate evaluated false",
                        "actual": actual if found else None,
                        "expected": constraint.expected,
                    }
                )
        return violations

    @staticmethod
    def _constraint_context(
        constraint: ConstraintSpec,
        request: ProblemRequest,
        proposal: SolutionProposal,
        task_contexts: dict[str, dict[str, Any]],
    ) -> Any:
        if constraint.source == "request_context":
            return request.context
        if constraint.source == "proposal_evidence":
            return proposal.constraint_evidence
        task_context = task_contexts.get(constraint.task_id)
        if task_context is None and constraint.requirement_id:
            task_context = next(
                (
                    value
                    for value in task_contexts.values()
                    if value["requirement"].requirement_id == constraint.requirement_id
                ),
                None,
            )
        return (task_context or {}).get(constraint.source, {})

    @staticmethod
    def _json_pointer(value: Any, path: str) -> tuple[bool, Any]:
        if path in {"", "/"}:
            return True, value
        if not path.startswith("/"):
            return False, None
        current = value
        for raw in path.lstrip("/").split("/"):
            key = raw.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
                current = current[int(key)]
            else:
                return False, None
        return True, current

    @staticmethod
    def _compare_constraint(constraint: ConstraintSpec, found: bool, actual: Any) -> bool:
        if constraint.operator == "exists":
            return found is bool(constraint.expected)
        if not found:
            return False
        try:
            operations = {
                "eq": lambda: actual == constraint.expected,
                "ne": lambda: actual != constraint.expected,
                "lt": lambda: actual < constraint.expected,
                "lte": lambda: actual <= constraint.expected,
                "gt": lambda: actual > constraint.expected,
                "gte": lambda: actual >= constraint.expected,
                "in": lambda: actual in constraint.expected,
                "not_in": lambda: actual not in constraint.expected,
                "contains": lambda: constraint.expected in actual,
                "matches": lambda: re.fullmatch(str(constraint.expected), str(actual)) is not None,
            }
            return bool(operations[constraint.operator]())
        except (KeyError, TypeError, ValueError, re.error):
            return False

    @staticmethod
    def _structure_ok(tasks: list[TaskSpec], risks: list[str]) -> bool:
        if not tasks:
            risks.append("The candidate plan contains no tasks.")
            return False
        ids = [task.task_id for task in tasks]
        ok = len(ids) == len(set(ids))
        if not ok:
            risks.append("The candidate plan contains duplicate task IDs.")
        if any(not task.requirement_id or not task.capability_id for task in tasks):
            risks.append("Every task requires stable requirement and capability IDs.")
            ok = False
        return ok

    @staticmethod
    def _execution_order_ok(proposal: SolutionProposal, risks: list[str]) -> bool:
        task_ids = [task.task_id for task in proposal.tasks]
        if set(proposal.execution_order) != set(task_ids):
            risks.append("Execution order does not include exactly the planned tasks.")
            return False
        positions = {task_id: index for index, task_id in enumerate(proposal.execution_order)}
        for task in proposal.tasks:
            for dependency in task.depends_on:
                if dependency not in positions or positions[dependency] > positions[task.task_id]:
                    risks.append(
                        f"Execution order does not place dependency {dependency} before {task.task_id}."
                    )
                    return False
        return True

    def _trust_satisfies(self, actual: str, required: str) -> bool:
        try:
            return self.trust_order.index(actual) >= self.trust_order.index(required)
        except ValueError:
            return False

    @staticmethod
    def _dependency_modes_compatible(outputs: list[str], inputs: list[str]) -> bool:
        if not inputs:
            return True
        return bool(set(outputs) & set(inputs))

    @classmethod
    def _schema_compatible(cls, provided: dict[str, Any], required: dict[str, Any]) -> bool:
        if not required:
            return True
        if not provided:
            return False
        required_type = required.get("type")
        if required_type and provided.get("type") not in {None, required_type}:
            return False
        if not set(required.get("required", [])) <= set(provided.get("required", [])):
            return False
        provided_properties = provided.get("properties", {})
        for key, required_property in required.get("properties", {}).items():
            provided_property = provided_properties.get(key)
            if provided_property is None or not cls._schema_compatible(
                provided_property, required_property
            ):
                return False
        return True

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
