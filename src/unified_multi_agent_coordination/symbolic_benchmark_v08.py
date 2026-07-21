"""Constraint-directed provider-search benchmark for study v0.8."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .models import ConstraintSpec
from .symbolic_benchmark_v07 import REPETITIONS, run_benchmark as run_legacy_matrix
from .symbolic_benchmark_v07 import _fixture
from .symbolic_plan_compiler import SymbolicPlanCompiler


def run_benchmark(repetitions: int = REPETITIONS) -> dict:
    legacy = run_legacy_matrix(repetitions)
    recovery_request, recovery_registry = _fixture(2, 4, valid_position="last")
    recovery = SymbolicPlanCompiler().compile(recovery_request, recovery_registry)
    exhaustion_request, exhaustion_registry = _fixture(6, 8, valid_position="first")
    requirements = {
        item.capability_id: item for item in exhaustion_request.requirements
    }
    exhaustion_registry = [
        agent.model_copy(deep=True, update={
            "skills": [requirements[agent.skills[0].capability_id].model_copy(deep=True)]
        })
        for agent in exhaustion_registry
    ]
    exhaustion_request.constraints.append(ConstraintSpec(
        constraint_id="impossible-global-evidence",
        source="proposal_evidence",
        path="/authorization-token",
        operator="eq",
        expected="present",
    ))
    exhaustion = SymbolicPlanCompiler(max_assignment_evaluations=4096).compile(
        exhaustion_request, exhaustion_registry
    )
    invariants = {
        "oracle_compilation": legacy["invariants"]["oracle_compilation"],
        "registry_permutation_invariance": legacy["invariants"]["registry_permutation_invariance"],
        "invalid_provider_prefiltering": (
            recovery.report.feasible
            and recovery.diagnostics.recovered_alternative_provider
            and recovery.diagnostics.branches_pruned >= 0
            and all(len(items) == 1 for items in recovery.diagnostics.provider_candidates.values())
        ),
        "bounded_search_refusal": (
            not exhaustion.report.feasible
            and exhaustion.diagnostics.search_exhausted
            and exhaustion.diagnostics.assignments_considered == 4096
        ),
        "conflict_diagnostics": bool(exhaustion.diagnostics.failed_predicates),
    }
    return {
        "schema_version": "0.8.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": legacy["rows"],
        "invariants": invariants,
        "all_invariants_passed": all(invariants.values()),
        "invariant_failures": legacy["invariant_failures"],
        "recovery_diagnostics": recovery.diagnostics.model_dump(mode="json"),
        "exhaustion_diagnostics": exhaustion.diagnostics.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("demo_runs/v0.8/deterministic-benchmark.json"),
    )
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    result = run_benchmark(args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
