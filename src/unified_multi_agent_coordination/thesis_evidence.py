"""Generate deterministic LaTeX evidence macros from the accepted manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def render(manifest_path: Path, repository_root: Path) -> str:
    manifest = _load(manifest_path)
    validation = manifest.get("current_local_validation") or {}
    v05 = manifest.get("v05_study_evidence")
    v07 = manifest.get("v07_study_evidence")
    consensus = (
        manifest.get("consensus_campaign_v4_evidence")
        or manifest.get("consensus_campaign_evidence")
    )
    values: dict[str, str] = {
        "EvidenceTestCount": str(validation.get("test_count", "pending")),
        "ProductionBranchCoverage": str(
            validation.get("production_branch_coverage_percent", "pending")
        ),
        "VFiveOutputCount": "0",
        "VFiveOutcome": "pending",
        "VFiveClaimStatus": "pending",
        "VSevenOutputCount": "0",
        "VSevenHybridRecall": "pending",
        "VSevenHybridUnsafe": "pending",
        "VSevenDirectUnsafe": "pending",
        "VSevenAliasBalancedAccuracy": "pending",
        "VSevenCriteriaPassed": "0",
        "VSevenCriteriaTotal": "0",
        "VSevenQwenSmallRecall": "pending",
        "VSevenGemmaRecall": "pending",
        "VSevenQwenLargeRecall": "pending",
        "ConsensusTrialCount": "0",
        "ConsensusOutcome": "pending",
        "ConsensusChecksExecuted": "0",
        "ConsensusChecksExpected": "0",
    }
    if v05:
        run_root = repository_root / v05["run_root"]
        completion = _load(run_root / "collection-complete.json")
        analysis = _load(run_root / v05.get("analysis_file", "analysis-v0.5.0.json"))
        values.update(
            {
                "VFiveOutputCount": str(completion["observed_outputs"]),
                "VFiveOutcome": str(analysis["outcome"]),
                "VFiveClaimStatus": str(analysis["claim_status"]),
            }
        )
    if v07:
        run_root = repository_root / v07["run_root"]
        analysis = _load(run_root / v07["analysis_file"])
        metrics = analysis["metrics_primary_seed"]
        by_model = analysis["metrics_by_model_primary_seed"]
        criteria = analysis["predeclared_criteria"]

        def percent(value: Any) -> str:
            return f"{100 * float(value):.2f}"

        values.update(
            {
                "VSevenOutputCount": str(v07["case_outputs"]),
                "VSevenHybridRecall": percent(metrics["hybrid"]["feasible_recall"]),
                "VSevenHybridUnsafe": str(metrics["hybrid"]["unsafe_acceptances"]),
                "VSevenDirectUnsafe": str(metrics["direct"]["unsafe_acceptances"]),
                "VSevenAliasBalancedAccuracy": percent(
                    metrics["alias"]["balanced_accuracy"]
                ),
                "VSevenCriteriaPassed": str(sum(bool(value) for value in criteria.values())),
                "VSevenCriteriaTotal": str(len(criteria)),
                "VSevenQwenSmallRecall": percent(
                    by_model["qwen/qwen3-1.7b"]["hybrid"]["feasible_recall"]
                ),
                "VSevenGemmaRecall": percent(
                    by_model["google/gemma-4-e2b"]["hybrid"]["feasible_recall"]
                ),
                "VSevenQwenLargeRecall": percent(
                    by_model["qwen/qwen3-8b"]["hybrid"]["feasible_recall"]
                ),
            }
        )
    if consensus:
        campaign = _load(repository_root / consensus["campaign_file"])
        values.update(
            {
                "ConsensusTrialCount": str(campaign["trial_count"]),
                "ConsensusOutcome": str(campaign["outcome"]),
                "ConsensusChecksExecuted": str(campaign["safety_checks_executed"]),
                "ConsensusChecksExpected": str(campaign["safety_checks_expected"]),
            }
        )
    lines = ["% Generated from evidence-manifest.json; do not edit by hand."]
    lines.extend(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items())
    return "\n".join(lines) + "\n"


def write_or_check(
    manifest_path: Path, repository_root: Path, output: Path, *, check: bool
) -> None:
    rendered = render(manifest_path, repository_root)
    if check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError("Generated thesis evidence macros differ from the manifest.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("evidence-manifest.json"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("thesis/generated/evidence_macros.tex")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    write_or_check(args.manifest, args.repository, args.output, check=args.check)


if __name__ == "__main__":
    main()
