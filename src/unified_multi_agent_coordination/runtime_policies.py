"""Explicit secure-default runtime policies and isolated experimental controls."""

from __future__ import annotations

from typing import Protocol

from .models import GeneratedNlpAgentSpec, TaskExecutionResult, TaskSpec, TraceEvent


class DependencyDispatchPolicy(Protocol):
    def blocked_dependencies(
        self, task: TaskSpec, results: dict[str, TaskExecutionResult]
    ) -> list[str]: ...


class StrictDependencyDispatchPolicy:
    """Dispatch only after every declared dependency completed successfully."""

    def blocked_dependencies(
        self, task: TaskSpec, results: dict[str, TaskExecutionResult]
    ) -> list[str]:
        return [
            dependency for dependency in task.depends_on
            if dependency not in results or results[dependency].status != "completed"
        ]


class UnsafeIgnoreDependencyPolicy:
    """Experimental negative control; never use in production coordination."""

    def blocked_dependencies(
        self, task: TaskSpec, results: dict[str, TaskExecutionResult]
    ) -> list[str]:
        return []


class AuxiliaryAdmissionPolicy(Protocol):
    def admissible(self, spec: GeneratedNlpAgentSpec, eligible: bool) -> bool: ...


class StrictAuxiliaryAdmissionPolicy:
    def admissible(self, spec: GeneratedNlpAgentSpec, eligible: bool) -> bool:
        return bool(
            eligible
            and not spec.persists
            and spec.lifecycle
            and spec.validation_rule
            and "read_only" in spec.authority_bounds
        )


class UnsafePermissiveAuxiliaryPolicy:
    """Experimental control that demonstrates why all bounds are jointly needed."""

    def admissible(self, spec: GeneratedNlpAgentSpec, eligible: bool) -> bool:
        return True


class AuxiliaryLifecyclePolicy(Protocol):
    def claim(self, lifecycle: str, consumed: set[str]) -> bool: ...


class SingleUseAuxiliaryLifecyclePolicy:
    def claim(self, lifecycle: str, consumed: set[str]) -> bool:
        if not lifecycle or lifecycle in consumed:
            return False
        consumed.add(lifecycle)
        return True


class UnsafeReusableAuxiliaryLifecyclePolicy:
    """Experimental negative control allowing lifecycle-token reuse."""

    def claim(self, lifecycle: str, consumed: set[str]) -> bool:
        return True


class TraceEvidencePolicy(Protocol):
    def expose(self, events: list[TraceEvent]) -> list[TraceEvent]: ...


class DurableTraceEvidencePolicy:
    def expose(self, events: list[TraceEvent]) -> list[TraceEvent]:
        return events


class EphemeralTraceEvidencePolicy:
    """Runtime-only negative control; durable state remains untouched."""

    def expose(self, events: list[TraceEvent]) -> list[TraceEvent]:
        return []
