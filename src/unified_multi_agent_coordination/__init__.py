"""Prototype interfaces for the thesis coordination framework."""

from .a2a_adapter import A2AAdapter, AuthorizationError
from .auxiliary import BoundedAuxiliaryCapabilityFactory
from .coordination_agent import CoordinationAgent
from .coordination_sdk import CoordinationSdk, RemoteRegistryError
from .feasibility import FeasibilityAnalyzer
from .lingo_coordinator import CoordinatorState, LingoLinguisticCoordinator
from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    CoordinationPlanResult,
    CoordinationRunResult,
    FeasibilityReport,
    GeneratedNlpAgentSpec,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskExecutionResult,
    TaskSpec,
    TraceEvent,
)

__all__ = [
    "A2AAdapter",
    "AgentRegistryEntry",
    "AuthorizationError",
    "BoundedAuxiliaryCapabilityFactory",
    "CapabilityRequirement",
    "CoordinationAgent",
    "CoordinationPlanResult",
    "CoordinationRunResult",
    "CoordinationSdk",
    "CoordinatorState",
    "FeasibilityAnalyzer",
    "FeasibilityReport",
    "GeneratedNlpAgentSpec",
    "LingoLinguisticCoordinator",
    "PredicateEvidence",
    "ProblemRequest",
    "RemoteRegistryError",
    "SolutionProposal",
    "TaskExecutionResult",
    "TaskSpec",
    "TraceEvent",
]
