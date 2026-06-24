"""Bounded auxiliary linguistic capability factory."""

from __future__ import annotations

from itertools import count
from typing import Any, Literal, cast

from .models import CapabilityRequirement, GeneratedNlpAgentSpec

AuxiliaryMethod = Literal["schema_extraction", "label_classification", "normalization"]


class BoundedAuxiliaryCapabilityFactory:
    """Create task-local specs for approved narrow language operations."""

    APPROVED_METHODS = {
        "extract": "schema_extraction",
        "classif": "label_classification",
        "normal": "normalization",
    }

    def __init__(self) -> None:
        self._counter = count(1)

    def specify(
        self, gap: CapabilityRequirement, lifecycle: str
    ) -> GeneratedNlpAgentSpec | None:
        """Return an auxiliary spec when the requirement is explicitly eligible."""
        if not gap.auxiliary_eligible:
            return None

        method = self._select_method(gap)
        if method is None:
            return None

        return GeneratedNlpAgentSpec(
            spec_id=f"aux-{next(self._counter)}",
            purpose=gap.description or gap.name,
            input_schema=gap.input_schema,
            output_schema=gap.output_schema,
            method=method,
            validation_rule=f"{method}_contract_validator",
            lifecycle=lifecycle,
            authority_bounds=["read_only", "plan_local"],
            persists=False,
        )

    def validate_result(
        self, spec: GeneratedNlpAgentSpec, artifact: dict[str, Any]
    ) -> bool:
        """Check the minimum deterministic schema evidence for an artifact."""
        required = spec.output_schema.get("required", [])
        if not isinstance(required, list):
            return False
        return all(key in artifact for key in required)

    def _select_method(
        self, gap: CapabilityRequirement
    ) -> AuxiliaryMethod | None:
        text = f"{gap.name} {gap.description}".lower()
        for key, method in self.APPROVED_METHODS.items():
            if key in text:
                return cast(AuxiliaryMethod, method)
        return None
