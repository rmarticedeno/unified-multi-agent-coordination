"""Enforce defense-critical module thresholds from coverage.py JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

THRESHOLDS = {
    "coordination_agent.py": 90.0,
    "feasibility.py": 90.0,
    "trace_validation.py": 90.0,
    "plan_hydration.py": 90.0,
    "cluster.py": 85.0,
    "coordinator_node.py": 85.0,
    "etcd_client.py": 85.0,
    "etcd_store.py": 85.0,
    "agent_registry.py": 85.0,
    "service.py": 85.0,
    "coordination_sdk.py": 85.0,
    "coordination_store.py": 80.0,
    "defense_study_v04.py": 80.0,
    "study_analysis_v04.py": 80.0,
    "cluster_discovery.py": 85.0,
    "consensus_campaign.py": 80.0,
    "evidence_preflight.py": 90.0,
}

PRODUCTION_MODULES = {
    "__init__.py",
    "a2a_adapter.py",
    "admission.py",
    "agent_registry.py",
    "auxiliary.py",
    "cluster.py",
    "cluster_discovery.py",
    "coordination_agent.py",
    "coordination_ledger.py",
    "coordination_sdk.py",
    "coordination_store.py",
    "coordinator_node.py",
    "etcd_client.py",
    "etcd_store.py",
    "feasibility.py",
    "lingo_coordinator.py",
    "models.py",
    "plan_hydration.py",
    "runtime_policies.py",
    "service.py",
    "trace_validation.py",
}


def evaluate(path: Path) -> tuple[list[str], float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    by_name = {Path(name).name: value for name, value in report["files"].items()}
    failures: list[str] = []
    for name, minimum in THRESHOLDS.items():
        if name not in by_name:
            failures.append(f"{name}: missing from coverage report")
            continue
        observed = float(by_name[name]["summary"]["percent_covered"])
        if observed + 1e-9 < minimum:
            failures.append(f"{name}: {observed:.2f}% < {minimum:.2f}%")
    production = [by_name[name]["summary"] for name in PRODUCTION_MODULES if name in by_name]
    missing_production = sorted(PRODUCTION_MODULES - by_name.keys())
    if missing_production:
        failures.append(f"production modules missing from coverage report: {missing_production}")
    covered = sum(item["covered_lines"] + item["covered_branches"] for item in production)
    opportunities = sum(item["num_statements"] + item["num_branches"] for item in production)
    total = 100.0 * covered / opportunities if opportunities else 0.0
    if total + 1e-9 < 85.0:
        failures.append(f"production total: {total:.2f}% < 85.00%")
    return failures, total


def check(path: Path) -> list[str]:
    """Compatibility wrapper used by tests and callers needing only failures."""
    return evaluate(path)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, nargs="?", default=Path("coverage.json"))
    args = parser.parse_args()
    failures, production_total = evaluate(args.report)
    if failures:
        raise SystemExit("Coverage gates failed:\n- " + "\n- ".join(failures))
    print(f"Coverage gates passed: production total {production_total:.2f}%.")


if __name__ == "__main__":
    main()
