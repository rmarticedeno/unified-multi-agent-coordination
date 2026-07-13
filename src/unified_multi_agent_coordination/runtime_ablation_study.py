"""Deterministic runtime-only ablations, kept separate from planning comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from .models import GeneratedNlpAgentSpec, TaskExecutionResult, TaskSpec, TraceEvent
from .runtime_policies import (
    DurableTraceEvidencePolicy,
    DependencyDispatchPolicy,
    EphemeralTraceEvidencePolicy,
    StrictAuxiliaryAdmissionPolicy,
    StrictDependencyDispatchPolicy,
    UnsafeIgnoreDependencyPolicy,
    UnsafePermissiveAuxiliaryPolicy,
    TraceEvidencePolicy,
)


def _dependency_rows(repetitions: int) -> list[dict[str, Any]]:
    scenarios = [
        ("missing", {}, True),
        ("failed", {"t1": "failed"}, True),
        ("timeout", {"t1": "timeout"}, True),
        ("unknown", {"t1": "unknown"}, True),
        ("completed", {"t1": "completed"}, False),
    ]
    policies: list[tuple[str, DependencyDispatchPolicy]] = [
        ("secure_default", StrictDependencyDispatchPolicy()),
        ("without_dependency_gating", UnsafeIgnoreDependencyPolicy()),
    ]
    rows = []
    task = TaskSpec(task_id="t2", requirement_name="consume", depends_on=["t1"])
    for repetition in range(repetitions):
        for scenario, states, should_block in scenarios:
            results = {
                task_id: TaskExecutionResult(
                    task_id=task_id,
                    agent_id="fixture",
                    status=cast(Literal["completed", "failed", "timeout", "unknown"], status),
                )
                for task_id, status in states.items()
            }
            for configuration, policy in policies:
                blocked = bool(policy.blocked_dependencies(task, results))
                rows.append({
                    "family": "dependency_dispatch",
                    "scenario": scenario,
                    "repetition": repetition,
                    "configuration": configuration,
                    "expected_safe_decision": should_block,
                    "observed_decision": blocked,
                    "safety_violation": should_block and not blocked,
                })
    return rows


def _auxiliary_rows(repetitions: int) -> list[dict[str, Any]]:
    scenarios = [
        ("bounded", True, False, "plan-1", ["read_only"], True),
        ("ineligible", False, False, "plan-1", ["read_only"], False),
        ("persistent", True, True, "plan-1", ["read_only"], False),
        ("unscoped", True, False, "", ["read_only"], False),
        ("write_authority", True, False, "plan-1", ["write"], False),
    ]
    policies = {
        "secure_default": StrictAuxiliaryAdmissionPolicy(),
        "without_auxiliary": None,
        "without_auxiliary_bounds": UnsafePermissiveAuxiliaryPolicy(),
    }
    rows = []
    for repetition in range(repetitions):
        for scenario, eligible, persists, lifecycle, authority, should_admit in scenarios:
            spec = GeneratedNlpAgentSpec(
                spec_id="aux-control",
                purpose=scenario,
                method="normalization",
                validation_rule="fixture-validator",
                lifecycle=lifecycle,
                authority_bounds=authority,
                persists=persists,
            )
            for configuration, policy in policies.items():
                admitted = False if policy is None else policy.admissible(spec, eligible)
                rows.append({
                    "family": "auxiliary_admission",
                    "scenario": scenario,
                    "repetition": repetition,
                    "configuration": configuration,
                    "expected_safe_decision": should_admit,
                    "observed_decision": admitted,
                    "safety_violation": admitted and not should_admit,
                })
    return rows


def _trace_rows(repetitions: int) -> list[dict[str, Any]]:
    scenarios = ["authorization", "dispatch", "retry", "recovery", "terminal_replay"]
    policies: list[tuple[str, TraceEvidencePolicy]] = [
        ("secure_default", DurableTraceEvidencePolicy()),
        ("without_durable_trace_evidence", EphemeralTraceEvidencePolicy()),
    ]
    rows = []
    for repetition in range(repetitions):
        for scenario in scenarios:
            events = [TraceEvent(event_type=scenario, message=scenario, source="store")]
            for configuration, policy in policies:
                exposed = policy.expose(events)
                rows.append({
                    "family": "trace_evidence",
                    "scenario": scenario,
                    "repetition": repetition,
                    "configuration": configuration,
                    "expected_safe_decision": True,
                    "observed_decision": bool(exposed),
                    "safety_violation": not bool(exposed),
                    "durable_state_mutated": False,
                })
    return rows


def run_study(output_dir: Path, repetitions: int = 10) -> Path:
    """Write one exclusive, content-addressed 350-observation evidence bundle."""
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        *_dependency_rows(repetitions),
        *_auxiliary_rows(repetitions),
        *_trace_rows(repetitions),
    ]
    rows_path = output_dir / "runtime-observations.json"
    rows_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    configurations = sorted({row["configuration"] for row in rows})
    summary = {
        "study_type": "runtime_only_ablation",
        "planning_comparison": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repetitions": repetitions,
        "observation_count": len(rows),
        "by_configuration": {
            configuration: {
                "observations": sum(row["configuration"] == configuration for row in rows),
                "safety_violations": sum(
                    row["configuration"] == configuration and row["safety_violation"]
                    for row in rows
                ),
            }
            for configuration in configurations
        },
        "observations_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    output = run_study(args.output_dir, args.repetitions)
    print(output)


if __name__ == "__main__":
    main()
