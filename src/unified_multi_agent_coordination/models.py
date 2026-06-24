"""Typed coordination records used by the prototype implementation.

The records mirror the thesis-level objects: requests, capability contracts,
candidate plans, auxiliary linguistic capabilities, feasibility reports, and
trace events. They are intentionally small enough for deterministic tests while
remaining explicit about the boundary between proposal and authorization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JsonObject = dict[str, Any]


class CapabilityRequirement(BaseModel):
    """A structured capability descriptor extracted from a user request."""

    name: str
    description: str = ""
    input_schema: JsonObject = Field(default_factory=dict)
    output_schema: JsonObject = Field(default_factory=dict)
    input_modes: list[str] = Field(default_factory=list)
    output_modes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    auxiliary_eligible: bool = False
    required_trust_level: str = "standard"


class AgentRegistryEntry(BaseModel):
    """A normalized view of an admitted remote agent."""

    agent_id: str
    name: str
    description: str = ""
    service_endpoint: str
    skills: list[CapabilityRequirement] = Field(default_factory=list)
    input_modes: list[str] = Field(default_factory=list)
    output_modes: list[str] = Field(default_factory=list)
    status: Literal["available", "unavailable"] = "available"
    trust_level: str = "standard"
    source_card: JsonObject = Field(default_factory=dict)


class ProblemRequest(BaseModel):
    """The admitted symbolic representation of a user problem."""

    user_goal: str
    requirements: list[CapabilityRequirement] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    context: JsonObject = Field(default_factory=dict)


class GeneratedNlpAgentSpec(BaseModel):
    """A task-local, bounded linguistic capability specification."""

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
    """One accountable task in a candidate coordination plan."""

    task_id: str
    requirement_name: str
    description: str = ""
    assigned_to: str | None = None
    auxiliary_spec_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)


class SolutionProposal(BaseModel):
    """A candidate plan produced by the linguistic coordinator."""

    tasks: list[TaskSpec] = Field(default_factory=list)
    selected_agents: dict[str, str] = Field(default_factory=dict)
    generated_nlp_agents: list[GeneratedNlpAgentSpec] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)


class PredicateEvidence(BaseModel):
    """Result of a deterministic feasibility predicate."""

    name: str
    passed: bool
    details: JsonObject = Field(default_factory=dict)


class FeasibilityReport(BaseModel):
    """The authorization verdict and its supporting evidence."""

    feasible: bool
    matched_agents: dict[str, str] = Field(default_factory=dict)
    generated_nlp_agents: list[GeneratedNlpAgentSpec] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    explanation: str = ""
    evidence: list[PredicateEvidence] = Field(default_factory=list)


class CoordinationPlanResult(BaseModel):
    """A coordination-agent planning result before task dispatch."""

    request: ProblemRequest
    proposal: SolutionProposal
    feasibility_report: FeasibilityReport
    registry_snapshot: list[AgentRegistryEntry] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """An auditable event in the coordination trace."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_type: str
    message: str
    data: JsonObject = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
