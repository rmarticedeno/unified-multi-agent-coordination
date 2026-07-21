"""Deterministic scalability and invariant benchmark for study v0.7."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import AgentRegistryEntry, CapabilityRequirement, ProblemRequest, ValidationContract
from .symbolic_plan_compiler import SymbolicPlanCompiler

TASK_COUNTS = (1, 2, 4, 6)
PROVIDER_COUNTS = (1, 2, 4, 8)
REPETITIONS = 30


def _fixture(
    task_count: int,
    provider_count: int,
    *,
    valid_position: str = "first",
) -> tuple[ProblemRequest, list[AgentRegistryEntry]]:
    requirements = []
    registry = []
    for task_index in range(task_count):
        capability_id = f"cap-{task_index + 1}"
        input_mode = "text" if task_index == 0 else f"mode-{task_index}"
        output_mode = f"mode-{task_index + 1}"
        requirement = CapabilityRequirement(
            name=capability_id,
            requirement_id=capability_id,
            capability_id=capability_id,
            input_modes=[input_mode],
            output_modes=[output_mode],
            depends_on_requirement_ids=(
                [f"cap-{task_index}"] if task_index else []
            ),
            validation_contract=ValidationContract(
                json_schema={"type": "object"},
                required_artifacts=[f"artifact-{task_index + 1}"],
            ),
        )
        requirements.append(requirement)
        for provider_index in range(provider_count):
            valid_index = 0 if valid_position == "first" else provider_count - 1
            skill = requirement.model_copy(deep=True)
            if provider_index != valid_index:
                skill.output_modes = ["incompatible"]
            prefix = "a" if provider_index < provider_count - 1 else "z"
            registry.append(
                AgentRegistryEntry(
                    agent_id=f"{prefix}-{capability_id}-provider-{provider_index + 1}",
                    name=f"{capability_id} provider {provider_index + 1}",
                    service_endpoint=(
                        f"local://{capability_id}-provider-{provider_index + 1}"
                    ),
                    skills=[skill],
                )
            )
    return (
        ProblemRequest(
            user_goal="Execute the deterministic benchmark chain.",
            requirements=list(reversed(requirements)),
            required_artifacts=[f"artifact-{task_count}"],
        ),
        registry,
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def run_benchmark(repetitions: int = REPETITIONS) -> dict[str, Any]:
    rows = []
    invariant_failures = []
    for task_count in TASK_COUNTS:
        for provider_count in PROVIDER_COUNTS:
            request, registry = _fixture(task_count, provider_count)
            latencies = []
            reference = None
            for repetition in range(repetitions):
                shuffled = list(registry)
                random.Random(
                    20260718 + task_count * 1000 + provider_count * 100 + repetition
                ).shuffle(shuffled)
                started = time.perf_counter()
                result = SymbolicPlanCompiler(prefilter_providers=False).compile(request, shuffled)
                latencies.append((time.perf_counter() - started) * 1000)
                proposal = result.proposal.model_dump(mode="json")
                reference = proposal if reference is None else reference
                if not result.report.feasible or proposal != reference:
                    invariant_failures.append(
                        {
                            "task_count": task_count,
                            "provider_count": provider_count,
                            "repetition": repetition,
                            "feasible": result.report.feasible,
                            "permutation_invariant": proposal == reference,
                        }
                    )
            rows.append(
                {
                    "task_count": task_count,
                    "provider_count": provider_count,
                    "assignment_space": provider_count**task_count,
                    "repetitions": repetitions,
                    "latency_ms_p50": statistics.median(latencies),
                    "latency_ms_p95": _percentile(latencies, 0.95),
                }
            )
    recovery_request, recovery_registry = _fixture(2, 4, valid_position="last")
    recovery = SymbolicPlanCompiler(prefilter_providers=False).compile(
        recovery_request, recovery_registry
    )
    exhaustion_request, exhaustion_registry = _fixture(6, 8, valid_position="last")
    exhaustion = SymbolicPlanCompiler(
        max_assignment_evaluations=4096, prefilter_providers=False
    ).compile(
        exhaustion_request,
        exhaustion_registry,
    )
    invariants = {
        "oracle_compilation": not invariant_failures,
        "registry_permutation_invariance": not invariant_failures,
        "alternative_provider_recovery": (
            recovery.report.feasible
            and recovery.diagnostics.recovered_alternative_provider
        ),
        "bounded_search_refusal": (
            not exhaustion.report.feasible
            and exhaustion.diagnostics.search_exhausted
            and exhaustion.diagnostics.assignments_considered == 4096
        ),
    }
    return {
        "schema_version": "0.7.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "invariants": invariants,
        "all_invariants_passed": all(invariants.values()),
        "invariant_failures": invariant_failures,
        "recovery_diagnostics": recovery.diagnostics.model_dump(mode="json"),
        "exhaustion_diagnostics": exhaustion.diagnostics.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_runs/v0.7/deterministic-benchmark.json"),
    )
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    args = parser.parse_args()
    result = run_benchmark(args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
