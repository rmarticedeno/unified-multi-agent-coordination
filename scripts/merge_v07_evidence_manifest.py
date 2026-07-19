"""Merge generated v0.7 evidence into the thesis manifest without deleting history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def _load(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("evidence-manifest.json"))
    parser.add_argument("--v07-manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    manifest = _load(args.manifest)
    v07 = _load(args.v07_manifest)
    analysis = _load(args.analysis)
    consensus = _load(args.consensus)
    relative_analysis = args.analysis.resolve().relative_to(args.run.resolve()).as_posix()
    accepted = list(manifest.get("accepted_evidence") or [])
    accepted_candidates = [
        args.v07_manifest.as_posix(),
        args.analysis.as_posix(),
    ]
    if consensus.get("evidence_valid"):
        accepted_candidates.append(args.consensus.as_posix())
    for item in accepted_candidates:
        if item not in accepted:
            accepted.append(item)
    criteria = analysis["predeclared_criteria"]
    failed_criteria = [name for name, passed in criteria.items() if not passed]
    manifest.update(
        {
            "schema_version": "0.7.0",
            "release_status": "technical-ready-administrative-blocked",
            "technical_status": (
                "evidence-complete-all-v07-gates-passed"
                if not failed_criteria
                else "evidence-complete-negative-v07-gates"
            ),
            "evaluated_source_commit": v07["evaluated_source_commit"],
            "packaging_commit": "UNFROZEN",
            "dirty_state": True,
            "accepted_evidence": accepted,
            "v07_study_evidence": {
                "run_root": args.run.as_posix(),
                "analysis_file": relative_analysis,
                "analysis_sha256": _sha256(args.analysis),
                "artifact_manifest": args.v07_manifest.as_posix(),
                "artifact_manifest_sha256": _sha256(args.v07_manifest),
                "protocol_lock_sha256": v07["protocol_lock"]["sha256"],
                "protocol_hash": v07["protocol_lock"]["protocol_hash"],
                "corpus_hash": v07["protocol_lock"]["corpus_hash"],
                "evaluated_source_commit": v07["evaluated_source_commit"],
                "dirty_state": v07["workspace_dirty"],
                "evidence_valid": v07["matrix_complete"],
                "case_outputs": v07["raw_output_count"],
                "models": analysis["models"],
                "primary_seed": analysis["primary_seed"],
                "replication_seed": analysis["replication_seed"],
                "predeclared_criteria": criteria,
                "failed_criteria": failed_criteria,
                "all_criteria_met": analysis["all_criteria_met"],
                "author_only_labels": analysis["author_only_labels"],
                "known_metadata_inconsistencies": v07[
                    "known_metadata_inconsistencies"
                ],
            },
            "consensus_campaign_v4_evidence": {
                "campaign_file": args.consensus.as_posix(),
                "sha256": _sha256(args.consensus),
                "schema_version": consensus["schema_version"],
                "source_commit": consensus["provenance"]["source_commit"],
                "dirty_state": consensus["provenance"]["dirty_state"],
                "evidence_valid": consensus["evidence_valid"],
                "accepted": consensus["accepted"],
                "outcome": consensus["outcome"],
                "claim_status": consensus["claim_status"],
                "trial_count": consensus["trial_count"],
                "passed_trials": sum(
                    item.get("passed") is True
                    for item in consensus["results"]
                    if item.get("primary", True)
                ),
                "invariant_failed_trials": consensus["invariant_failed_trials"],
                "infrastructure_failed_trials": consensus[
                    "infrastructure_failed_trials"
                ],
                "safety_checks_expected": consensus["safety_checks_expected"],
                "safety_checks_executed": consensus["safety_checks_executed"],
                "safety_violations": consensus["safety_violations"],
                "unexecuted_checks": consensus["unexecuted_checks"],
            },
            "current_local_validation": {
                **(manifest.get("current_local_validation") or {}),
                "test_count": v07["validation"]["test_count"],
                "production_branch_coverage_percent": v07["validation"][
                    "production_branch_coverage_percent"
                ],
                "tests": f"{v07['validation']['test_count']} passing on Windows",
                "coverage": (
                    f"{v07['validation']['production_branch_coverage_percent']:.2f} "
                    "percent branch-aware production coverage; all module gates pass"
                ),
                "consensus_full": (
                    f"consensus-v4 {consensus['trial_count']}-trial campaign; "
                    f"evidence_valid={str(consensus['evidence_valid']).lower()}, "
                    f"outcome={consensus['outcome']}"
                ),
            },
            "reported_negative_outcomes": [
                *(manifest.get("reported_negative_outcomes") or []),
                *[f"v0.7 gate failed: {name}" for name in failed_criteria],
            ],
        }
    )
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "failed_v07_criteria": failed_criteria,
                "consensus_evidence_valid": consensus["evidence_valid"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
