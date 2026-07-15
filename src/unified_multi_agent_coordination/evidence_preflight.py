"""Fail-closed validation for thesis evidence, provenance, and release claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .consensus_campaign import expected_scenarios
from .corpus import audit_corpus
from .defense_study import validate_frozen_labels
from .legacy_evidence import validate_runtime_ablation, validate_study_evidence

FORBIDDEN_CURRENT_CLAIMS = (
    "qualified independent reviewer",
    "independent label approval",
    "independent corpus sign-off",
    "independently signed",
)
CURRENT_DOCUMENTS = (Path("README.md"), Path("REPRODUCING.md"), Path("thesis/chapters"))


def run_preflight(
    corpus_root: Path,
    repository_root: Path = Path("."),
    *,
    require_release: bool = False,
) -> dict[str, Any]:
    provenance = validate_frozen_labels(corpus_root)
    corpus_manifest = _json(corpus_root / "manifest.json")
    public = _json(corpus_root / "public/cases.json")
    hidden = _json(corpus_root / "hidden/reference-labels.json")
    audit = audit_corpus(
        public,
        hidden,
        expected_count=int(corpus_manifest.get("case_count", 36)),
    )
    violations = _terminology_violations(repository_root)
    if violations:
        raise RuntimeError("Terminology audit failed: " + "; ".join(violations))
    if not audit["passed"]:
        raise RuntimeError("Corpus audit failed: " + "; ".join(audit["errors"]))

    manifest = _json(repository_root / "evidence-manifest.json")
    study = validate_study_evidence(repository_root, manifest)
    study["v05"] = _validate_v05(repository_root, manifest)
    runtime = validate_runtime_ablation(repository_root, manifest)
    vendor = _validate_vendor(repository_root)
    consensus = _validate_consensus(repository_root, manifest)
    gates = list(manifest.get("open_gates") or [])
    administrative = _validate_release_metadata(repository_root, manifest)
    if not vendor["evidence_valid"]:
        gates.append("technical: vendored fixture files must be Git-tracked")
    if not study.get("v05", {}).get("evidence_valid", False):
        gates.append("technical: complete clean v0.5 evidence")
    if not consensus.get("evidence_valid", False):
        gates.append("technical: complete clean consensus campaign evidence")
    gates.extend(administrative["open_gates"])
    gates = list(dict.fromkeys(gates))
    release_ready = (
        manifest.get("release_status") == "release-ready"
        and not gates
        and consensus.get("evidence_valid") is True
        and study.get("v05", {}).get("evidence_valid") is True
        and vendor.get("evidence_valid") is True
        and administrative["passed"] is True
        and not manifest.get("dirty_state", True)
    )
    if require_release and not release_ready:
        raise RuntimeError("Release preflight is blocked: " + "; ".join(gates))
    return {
        "passed": True,
        "release_ready": release_ready,
        "corpus_hash": provenance["corpus_hash"],
        "labels_frozen": True,
        "annotation_type": provenance["annotation_type"],
        "independent_adjudication": False,
        "internal_consistency_audit": audit,
        "terminology_violations": [],
        "study_evidence": study,
        "runtime_ablation_evidence": runtime,
        "consensus_evidence": consensus,
        "vendor_evidence": vendor,
        "release_metadata": administrative,
        "open_gates": gates,
    }


def _terminology_violations(repository_root: Path) -> list[str]:
    violations: list[str] = []
    for relative in CURRENT_DOCUMENTS:
        target = repository_root / relative
        files = target.rglob("*.tex") if target.is_dir() else [target]
        for path in files:
            text = path.read_text(encoding="utf-8").lower()
            for phrase in FORBIDDEN_CURRENT_CLAIMS:
                if phrase in text:
                    violations.append(f"{path}: unsupported current claim {phrase!r}")
    return violations


def _validate_v05(repository_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    item = manifest.get("v05_study_evidence")
    if not item:
        return {"present": False, "evidence_valid": False, "outcome": "not_collected"}
    run_root = repository_root / item["run_root"]
    completion = _json(run_root / "collection-complete.json")
    provenance = _json(run_root / "provenance.json")
    outputs = list(run_root.glob("outputs/*/*/seed-*/*.json"))
    identities: set[tuple[str, str, str, int]] = set()
    for path in outputs:
        record = _json(path)
        identity = record.get("identity") or {}
        key = (
            str(identity.get("arm")),
            str(identity.get("case_id")),
            str(identity.get("model_id")),
            int(identity.get("seed") or 0),
        )
        if key in identities:
            raise RuntimeError("v0.5 collection contains a duplicate output identity.")
        identities.add(key)
        if record.get("call_count") not in {1, 2}:
            raise RuntimeError("v0.5 output exceeds the pre-specified two-call budget.")
    complete = (
        completion.get("complete") is True
        and completion.get("expected_outputs") == 1440
        and completion.get("observed_outputs") == 1440
        and len(outputs) == 1440
        and len(identities) == 1440
        and completion.get("labels_loaded_during_collection") is False
    )
    analysis_path = run_root / item.get("analysis_file", "analysis-v0.5.0.json")
    analysis = _json(analysis_path)
    analysis_hash = _sha256(analysis_path)
    if analysis_hash != str(item.get("analysis_sha256", "")).lower():
        raise RuntimeError("v0.5 analysis digest differs from the evidence manifest.")
    if analysis.get("analysis_schema_version") != "0.5.0":
        raise RuntimeError("v0.5 analysis uses an unsupported schema.")
    clean = provenance.get("dirty_state") is False
    source_matches = provenance.get("git_sha") == item.get("evaluated_source_commit")
    evidence_valid = complete and clean and source_matches
    if item.get("evidence_valid") is True and not evidence_valid:
        raise RuntimeError("Manifest marks incomplete or dirty v0.5 evidence as valid.")
    return {
        "present": True,
        "evidence_valid": evidence_valid,
        "outcome": analysis.get("outcome", "unknown"),
        "claim_status": analysis.get("claim_status", "unknown"),
        "planning_criteria_met": analysis.get("all_planning_criteria_met"),
        "output_count": len(outputs),
        "analysis_sha256": analysis_hash,
        "evaluated_source_commit": provenance.get("git_sha"),
    }


def _validate_vendor(repository_root: Path) -> dict[str, Any]:
    resolved_repository = repository_root.resolve()
    root = resolved_repository / "vendor/a2a-samples/helloworld"
    provenance = _json(root / "UPSTREAM.json")
    required = ("__main__.py", "agent_executor.py", "requirements.txt", "README.md")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Vendored A2A sample is incomplete: {missing}")
    if provenance.get("sdk_version") != "1.1.0" or len(provenance.get("commit", "")) != 40:
        raise RuntimeError("Vendored A2A provenance is not pinned to SDK 1.1.0 and a commit.")
    declared_hashes = provenance.get("local_sha256") or {}
    for name in required:
        if _sha256(root / name) != str(declared_hashes.get(name, "")).lower():
            raise RuntimeError(f"Vendored A2A file hash mismatch: {name}")
    license_path = (root / str(provenance.get("license_file", ""))).resolve()
    if not license_path.is_relative_to(repository_root.resolve()) or not license_path.is_file():
        raise RuntimeError("Vendored A2A license path is missing or leaves the repository.")
    if _sha256(license_path) != str(provenance.get("license_sha256", "")).lower():
        raise RuntimeError("Vendored A2A license hash mismatch.")
    project = (resolved_repository / "pyproject.toml").read_text(encoding="utf-8")
    if '"a2a-sdk==1.1.0"' not in project:
        raise RuntimeError("The project dependency is not pinned to a2a-sdk 1.1.0.")
    tracked_files = [root / "UPSTREAM.json", *(root / name for name in required), license_path]
    untracked = [
        str(path.relative_to(resolved_repository))
        for path in tracked_files
        if subprocess.run(
            [
                "git",
                "-C",
                str(resolved_repository),
                "ls-files",
                "--error-unmatch",
                "--",
                str(path.relative_to(resolved_repository)),
            ],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        != 0
    ]
    return {
        "present": True,
        "evidence_valid": not untracked,
        "commit": provenance["commit"],
        "sdk_version": "1.1.0",
        "hashes_verified": True,
        "git_tracked": not untracked,
        "untracked": untracked,
    }


def _validate_consensus(repository_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    item = manifest.get("consensus_campaign_evidence")
    if not item:
        return {"present": False, "accepted": False}
    path = repository_root / item["campaign_file"]
    campaign = _json(path)
    if _sha256(path) != str(item["sha256"]).lower():
        raise RuntimeError("Consensus campaign digest differs from the manifest.")
    schema_version = campaign.get("schema_version")
    if schema_version not in {"consensus-campaign-v2", "consensus-campaign-v3"}:
        raise RuntimeError("Consensus campaign uses an unsupported schema.")
    if schema_version == "consensus-campaign-v2":
        expected: list[tuple[str, int]] = []
        for trial in range(1, 4):
            expected.extend((f"formation-{topology}", trial) for topology in (3, 5, 7))
            expected.extend(
                (
                    ("reconfigure-3-5-3", trial),
                    ("reconfigure-5-7-5", trial),
                    ("leader-partition-quorum-concurrency", trial),
                    ("audit-sink-unavailable", trial),
                    ("failed-voter-replacement", trial),
                )
            )
            expected.extend(
                (f"crash-{fault_point}", trial)
                for fault_point in (
                    "after_task_attempt_start",
                    "after_external_dispatch",
                    "during_aggregation",
                )
            )
    else:
        expected = expected_scenarios(3, smoke=False)
    results = campaign.get("results") or []
    actual = [(result.get("scenario"), result.get("trial")) for result in results]
    accounting = all(
        set(result.get("expected_checks") or [])
        == set(result.get("executed_checks") or []) | set(result.get("unexecuted_checks") or [])
        and not (
            set(result.get("executed_checks") or []) & set(result.get("unexecuted_checks") or [])
        )
        for result in results
    )
    image_id = str(campaign.get("provenance", {}).get("image", {}).get("image_id", ""))
    clean = campaign.get("provenance", {}).get("dirty_state") is False
    v3_conditions_valid = True
    if schema_version == "consensus-campaign-v3":
        primary = [result for result in results if result.get("primary", True)]
        supplementary = [result for result in results if not result.get("primary", True)]
        reported_conditions = campaign.get("condition_results")
        derived_conditions: dict[str, dict[str, Any]] = {}
        for scenario in sorted({str(result.get("scenario")) for result in primary}):
            observations = [result for result in primary if result.get("scenario") == scenario]
            derived_conditions[scenario] = {
                "trial_count": len(observations),
                "passed_trials": sum(result.get("passed") is True for result in observations),
                "invariant_failed_trials": sum(
                    result.get("status") == "invariant_failed" for result in observations
                ),
                "infrastructure_failed_trials": sum(
                    result.get("status") == "infrastructure_error" for result in observations
                ),
                "checks_expected": sum(
                    len(result.get("expected_checks") or []) for result in observations
                ),
                "checks_executed": sum(
                    len(result.get("executed_checks") or []) for result in observations
                ),
                "violations": sum(
                    len(result.get("violated_checks") or []) for result in observations
                ),
                "supported": bool(observations)
                and all(result.get("passed") is True for result in observations),
            }
        v3_conditions_valid = (
            reported_conditions == derived_conditions
            and campaign.get("primary_trial_count") == len(primary) == 42
            and campaign.get("supplementary_trial_count") == len(supplementary) == 3
        )
    evidence_valid = (
        actual == expected
        and len(results) == len(expected)
        and accounting
        and image_id.startswith("sha256:")
        and clean
        and v3_conditions_valid
        and campaign.get("evidence_valid") is True
    )
    if item.get("evidence_valid") is True and not evidence_valid:
        raise RuntimeError(
            "Manifest accepts an incomplete, dirty, or wrong-image consensus campaign."
        )
    return {
        "present": True,
        "evidence_valid": evidence_valid,
        "accepted": item.get("accepted") is True and evidence_valid,
        "outcome": campaign.get("outcome"),
        "claim_status": campaign.get("claim_status"),
        "trial_count": campaign["trial_count"],
        "safety_checks_expected": campaign.get("safety_checks_expected"),
        "safety_checks_executed": campaign.get("safety_checks_executed"),
        "safety_violations": campaign.get("safety_violations"),
    }


def _validate_release_metadata(repository_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    gates: list[str] = []
    main = (repository_root / "thesis/main.tex").read_text(encoding="utf-8")
    if "[Tutor por confirmar]" in main:
        gates.append("administrative: advisor metadata")
    if "[Tribunal por confirmar]" in main:
        gates.append("administrative: tribunal metadata")
    evaluated = str(manifest.get("evaluated_source_commit") or "")
    packaging = str(manifest.get("packaging_commit") or "")
    if len(evaluated) != 40 or evaluated.upper() == "UNFROZEN":
        gates.append("administrative: evaluated source commit")
    if len(packaging) != 40 or packaging.upper() == "UNFROZEN":
        gates.append("administrative: packaging commit")
    return {"passed": not gates, "open_gates": gates}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required evidence file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Evidence file is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.3"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--require-release", "--release", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_preflight(
                args.corpus,
                args.repository,
                require_release=args.require_release,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
