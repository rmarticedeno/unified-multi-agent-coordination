"""Validation of historical v0.4 and runtime-ablation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def validate_study_evidence(repository_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    historical = manifest.get("historical_study_evidence") or {}
    bridge = historical.get("bridge_v04") or manifest.get("bridge_redesign_evidence")
    if not bridge:
        raise RuntimeError("The historical v0.4 study is missing from the manifest.")
    run_root = repository_root / "demo_runs/v0.4" / bridge["run_id"]
    completion = _json(run_root / "collection-complete.json")
    files = [path for path in run_root.glob("*/seed-*/*.json") if path.name != "sentinel.json"]
    if completion.get("complete") is not True or len(files) != 540:
        raise RuntimeError("Historical v0.4 collection is incomplete.")
    analysis_path = run_root / bridge.get("analysis_file", "analysis-v0.4.1-final.json")
    analysis = _json(analysis_path)
    digest = _sha256(analysis_path)
    if digest != str(bridge["analysis_sha256"]).lower():
        raise RuntimeError("v0.4.1 analysis digest differs from the evidence manifest.")
    if analysis.get("analysis_schema_version") != "0.4.1":
        raise RuntimeError("The current bridge analysis is not schema v0.4.1.")
    expected_counts = {
        "natural_language_rule_only": 36,
        "structured_oracle_upper_bound": 36,
        "hybrid_bridge": 540,
        "llm_only_no_symbolic_gate": 540,
    }
    for name, expected in expected_counts.items():
        if analysis["observation_counts"].get(name) != expected:
            raise RuntimeError(f"Unexpected observation count for {name}.")
    if analysis["preregistered_acceptance"]["every_model_recall_at_least_0_60"] is not False:
        raise RuntimeError("The failed small-model pre-specified criterion was not preserved.")
    replication = validate_clean_replication(repository_root, manifest)
    return {
        "historical_present": True,
        "historical_dirty_state": bool(bridge.get("dirty_state", True)),
        "run_id": bridge["run_id"],
        "case_outputs": len(files),
        "analysis_schema_version": analysis["analysis_schema_version"],
        "analysis_sha256": digest,
        "observation_counts": expected_counts,
        "clean_replication_present": replication["present"],
        "clean_replication": replication,
    }


def validate_clean_replication(repository_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    item = manifest.get("clean_replication_evidence")
    if not item:
        return {"present": False, "accepted": False}
    run_root = repository_root / "demo_runs/v0.4-r1" / item["run_id"]
    completion = _json(run_root / "collection-complete.json")
    provenance = _json(run_root / "provenance.json")
    outputs = [path for path in run_root.glob("*/seed-*/*.json") if path.name != "sentinel.json"]
    if completion.get("complete") is not True or len(outputs) != 540:
        raise RuntimeError("Clean v0.4-r1 replication is incomplete.")
    if provenance.get("dirty_state") is not False:
        raise RuntimeError("Clean v0.4-r1 replication records a dirty source tree.")
    if provenance.get("git_sha") != item.get("source_commit"):
        raise RuntimeError("Replication source SHA differs from the manifest.")
    analyses = item.get("analysis_files") or []
    if len(analyses) != 2:
        raise RuntimeError("Replication must record exactly two independent analysis files.")
    digests = [_sha256(run_root / name) for name in analyses]
    if len(set(digests)) != 1 or digests[0] != str(item.get("analysis_sha256", "")).lower():
        raise RuntimeError("Replication analysis reruns are not digest-identical.")
    return {
        "present": True,
        "accepted": True,
        "run_id": item["run_id"],
        "source_commit": provenance["git_sha"],
        "analysis_sha256": digests[0],
    }


def validate_runtime_ablation(repository_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    item = manifest.get("runtime_ablation_evidence") or {}
    path = repository_root / item.get("analysis_file", "demo_runs/runtime_ablation/v1/analysis.json")
    analysis = _json(path)
    if analysis.get("observation_count") != 350:
        raise RuntimeError("Runtime ablation bundle must contain 350 observations.")
    if analysis.get("planning_comparison") is not False:
        raise RuntimeError("Runtime ablations must not be represented as planning comparisons.")
    if item.get("observations_sha256") != analysis.get("observations_sha256"):
        raise RuntimeError("Runtime-ablation observation digest differs from the manifest.")
    return {"present": True, "observation_count": 350, "planning_comparison": False}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Evidence file is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
