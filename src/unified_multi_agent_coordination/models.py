"""Typed coordination records shared by planning, authorization, and execution."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonObject = dict[str, Any]


def stable_identifier(value: str) -> str:
    """Return a stable wire identifier derived from a display label."""

    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "unnamed"


class ConstraintSpec(BaseModel):
    """One deterministic, fail-closed constraint checked before authorization."""

    constraint_id: str
    source: Literal[
        "request_context",
        "proposal_evidence",
        "agent",
        "capability",
        "task",
        "unresolved",
    ]
    path: str = ""
    operator: Literal[
        "eq",
        "ne",
        "lt",
        "lte",
        "gt",
        "gte",
        "in",
        "not_in",
        "contains",
        "exists",
        "matches",
        "unresolved",
    ] = "eq"
    expected: Any = None
    requirement_id: str = ""
    task_id: str = ""
    description: str = ""

    @classmethod
    def from_legacy_text(cls, value: str) -> "ConstraintSpec":
        return cls(
            constraint_id=f"legacy-{stable_identifier(value)}",
            source="unresolved",
            operator="unresolved",
            description=value,
        )


class ValidationContract(BaseModel):
    """Deterministic artifact and completion evidence enforced at runtime."""

    json_schema: JsonObject = Field(default_factory=dict)
    required_artifacts: list[str] = Field(default_factory=list)
    artifact_kinds: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    evidence_types: list[str] = Field(default_factory=list)
    validator_id: str = ""

    def enforceable(self) -> bool:
        return bool(
            self.json_schema
            or self.required_artifacts
            or self.artifact_kinds
            or self.required_fields
            or self.evidence_types
            or self.validator_id
        )


class CompletionContract(BaseModel):
    """Decidable terminal conditions for an authorized plan."""

    required_task_states: list[str] = Field(default_factory=lambda: ["completed"])
    required_artifacts: list[str] = Field(default_factory=list)
    require_all_task_validators: bool = True


class SessionState(StrEnum):
    NEW = "new"
    PLANNING = "planning"
    AUTHORIZED = "authorized"
    DISPATCHING = "dispatching"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    INFEASIBLE = "infeasible"


class TaskState(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    UNKNOWN = "unknown"


class CapabilityRequirement(BaseModel):
    """A structured required or advertised capability contract."""

    name: str
    requirement_id: str = ""
    capability_id: str = ""
    description: str = ""
    input_schema: JsonObject = Field(default_factory=dict)
    output_schema: JsonObject = Field(default_factory=dict)
    input_modes: list[str] = Field(default_factory=list)
    output_modes: list[str] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    auxiliary_eligible: bool = False
    required_trust_level: str = "standard"
    side_effect_class: Literal["read_only", "idempotent", "unsafe", "unknown"] = "unknown"
    expected_evidence: list[str] = Field(default_factory=list)
    validation_contract: ValidationContract = Field(default_factory=ValidationContract)

    @field_validator("constraints", mode="before")
    @classmethod
    def _migrate_constraints(cls, value: Any) -> Any:
        return [
            ConstraintSpec.from_legacy_text(item) if isinstance(item, str) else item
            for item in (value or [])
        ]

    @model_validator(mode="after")
    def _supply_ids(self) -> "CapabilityRequirement":
        stable = stable_identifier(self.name)
        self.requirement_id = self.requirement_id or stable
        self.capability_id = self.capability_id or stable
        return self


class AgentRegistryEntry(BaseModel):
    """A normalized, locally admitted agent record."""

    agent_id: str
    name: str
    agent_kind: Literal["remote_a2a", "local_python", "linguistic"] = "remote_a2a"
    description: str = ""
    service_endpoint: str
    invocation_endpoint: str = ""
    skills: list[CapabilityRequirement] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    output_modes: list[str] = Field(default_factory=list)
    status: Literal["available", "unavailable"] = "available"
    trust_level: str = "standard"
    validation_contract: ValidationContract = Field(default_factory=ValidationContract)
    security_schemes: JsonObject = Field(default_factory=dict)
    required_security_schemes: list[str] = Field(default_factory=list)
    source_card: JsonObject = Field(default_factory=dict)


class ProblemRequest(BaseModel):
    """The admitted symbolic representation of a user problem."""

    user_goal: str
    requirements: list[CapabilityRequirement] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    context: JsonObject = Field(default_factory=dict)

    @field_validator("constraints", mode="before")
    @classmethod
    def _migrate_constraints(cls, value: Any) -> Any:
        return [
            ConstraintSpec.from_legacy_text(item) if isinstance(item, str) else item
            for item in (value or [])
        ]


class DraftRequirementSelection(BaseModel):
    """A linguistic choice over identifiers already present in admitted input."""

    requirement_id: str
    capability_id: str
    depends_on_requirement_ids: list[str] = Field(default_factory=list)


class LinguisticPlanDraft(BaseModel):
    """Non-executable model output; it cannot create agents, contracts, or task IDs."""

    selections: list[DraftRequirementSelection] = Field(default_factory=list)
    unresolved_terms: list[str] = Field(default_factory=list)
    rationale: str = ""


class HydrationIssue(BaseModel):
    """One deterministic reason why a linguistic draft cannot become a proposal."""

    code: Literal[
        "duplicate_requirement",
        "missing_requirement",
        "unknown_requirement",
        "unknown_capability",
        "unavailable_capability",
        "invalid_dependency",
        "dependency_cycle",
        "unresolved_term",
    ]
    message: str
    requirement_id: str = ""


class GeneratedNlpAgentSpec(BaseModel):
    spec_id: str
    purpose: str
    input_schema: JsonObject = Field(default_factory=dict)
    output_schema: JsonObject = Field(default_factory=dict)
    method: Literal["schema_extraction", "label_classification", "normalization"]
    validation_rule: str
    lifecycle: str
    authority_bounds: list[str] = Field(default_factory=lambda: ["read_only"])
    persists: bool = False


class TaskSpec(BaseModel):
    """One accountable task in a candidate plan."""

    task_id: str
    requirement_name: str
    requirement_id: str = ""
    capability_id: str = ""
    description: str = ""
    assigned_to: str | None = None
    auxiliary_spec_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    validation_contract: ValidationContract = Field(default_factory=ValidationContract)

    @model_validator(mode="after")
    def _supply_ids(self) -> "TaskSpec":
        stable = stable_identifier(self.requirement_name)
        self.requirement_id = self.requirement_id or stable
        self.capability_id = self.capability_id or stable
        return self


class SolutionProposal(BaseModel):
    """A non-authoritative candidate coordination plan."""

    tasks: list[TaskSpec] = Field(default_factory=list)
    selected_agents: dict[str, str] = Field(default_factory=dict)
    generated_nlp_agents: list[GeneratedNlpAgentSpec] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    completion_contract: CompletionContract = Field(default_factory=CompletionContract)
    constraint_evidence: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _migrate_completion_contract(self) -> "SolutionProposal":
        if not self.completion_contract.required_artifacts and self.expected_artifacts:
            self.completion_contract.required_artifacts = list(self.expected_artifacts)
        return self


class PredicateEvidence(BaseModel):
    name: str
    passed: bool
    details: JsonObject = Field(default_factory=dict)


class FeasibilityReport(BaseModel):
    feasible: bool
    matched_agents: dict[str, str] = Field(default_factory=dict)
    generated_nlp_agents: list[GeneratedNlpAgentSpec] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence: list[PredicateEvidence] = Field(default_factory=list)
    constraint_violations: list[JsonObject] = Field(default_factory=list)
    validation_gaps: list[JsonObject] = Field(default_factory=list)
    schema_violations: list[JsonObject] = Field(default_factory=list)
    capability_resolution: list[JsonObject] = Field(default_factory=list)


class CoordinationPlanResult(BaseModel):
    session_id: str = ""
    plan_id: str = ""
    plan_generation: int = 1
    request: ProblemRequest
    proposal: SolutionProposal
    feasibility_report: FeasibilityReport
    registry_snapshot: list[AgentRegistryEntry] = Field(default_factory=list)


class PlanGeneration(BaseModel):
    session_id: str
    plan_id: str
    generation: int = 1
    registry_snapshot_hash: str = ""
    authorized: bool = False
    active: bool = True
    plan_result: JsonObject = Field(default_factory=dict)


class TaskCommitment(BaseModel):
    session_id: str
    plan_id: str
    plan_generation: int = 1
    task_id: str
    requirement_name: str
    assigned_to: str = ""
    state: TaskState = TaskState.PENDING
    depends_on: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    side_effect_class: Literal["read_only", "idempotent", "unsafe", "unknown"] = "unknown"


class TaskAttemptRecord(BaseModel):
    session_id: str
    plan_id: str
    task_id: str
    attempt_id: str
    idempotency_key: str
    coordinator_id: str
    fencing_token: int
    state: TaskState = TaskState.RUNNING
    result: JsonObject | None = None


class LeaseRecord(BaseModel):
    session_id: str
    holder_id: str
    fencing_token: int
    expires_at: datetime
    heartbeat_at: datetime


class TerminalResultRecord(BaseModel):
    session_id: str
    plan_id: str = ""
    status: Literal["completed", "infeasible", "failed"] = "completed"
    run_result: JsonObject = Field(default_factory=dict)


class TaskExecutionResult(BaseModel):
    session_id: str = ""
    plan_id: str = ""
    attempt_id: str = ""
    task_id: str
    agent_id: str
    agent_kind: str = "unknown"
    status: Literal["completed", "failed", "timeout", "refused", "skipped", "unknown"] = "completed"
    output: Any = None
    artifacts: list[JsonObject] = Field(default_factory=list)
    error: str = ""
    metadata: JsonObject = Field(default_factory=dict)


class TraceEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_type: str
    message: str
    session_id: str = ""
    plan_id: str = ""
    task_id: str = ""
    attempt_id: str = ""
    source: Literal["coordinator", "sdk", "store", "linguistic", "system"] = "system"
    data: JsonObject = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoordinationRunResult(BaseModel):
    session_id: str = ""
    plan_id: str = ""
    status: Literal["completed", "infeasible", "failed"] = "completed"
    plan_result: CoordinationPlanResult
    task_results: list[TaskExecutionResult] = Field(default_factory=list)
    artifacts: list[JsonObject] = Field(default_factory=list)
    explanation: str = ""
    trace: list[TraceEvent] = Field(default_factory=list)
