"""Prototype interfaces for the thesis coordination framework."""

from .a2a_adapter import A2AAdapter, AuthorizationError
from .auxiliary import BoundedAuxiliaryCapabilityFactory
from .feasibility import FeasibilityAnalyzer
from .lingo_coordinator import CoordinatorState, LingoLinguisticCoordinator
from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    FeasibilityReport,
    GeneratedNlpAgentSpec,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
    TraceEvent,
)

__all__ = [
    "A2AAdapter",
    "AgentRegistryEntry",
    "AuthorizationError",
    "BoundedAuxiliaryCapabilityFactory",
    "CapabilityRequirement",
    "CoordinatorState",
    "FeasibilityAnalyzer",
    "FeasibilityReport",
    "GeneratedNlpAgentSpec",
    "LingoLinguisticCoordinator",
    "PredicateEvidence",
    "ProblemRequest",
    "SolutionProposal",
    "TaskSpec",
    "TraceEvent",
]
