"""Fail-closed checks for the current defense-evidence cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .corpus import audit_corpus
from .defense_study import validate_frozen_labels


FORBIDDEN_CURRENT_CLAIMS = (
    "qualified independent reviewer",
    "independent label approval",
    "independent corpus sign-off",
    "independently signed",
)
CURRENT_DOCUMENTS = (Path("README.md"), Path("REPRODUCING.md"), Path("thesis/chapters"))


def run_preflight(corpus_root: Path, repository_root: Path = Path(".")) -> dict[str, Any]:
    provenance = validate_frozen_labels(corpus_root)
    public = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    hidden = json.loads((corpus_root / "hidden/reference-labels.json").read_text(encoding="utf-8"))
    audit = audit_corpus(public, hidden)
    violations: list[str] = []
    for relative in CURRENT_DOCUMENTS:
        target = repository_root / relative
        files = target.rglob("*.tex") if target.is_dir() else [target]
        for path in files:
            text = path.read_text(encoding="utf-8").lower()
            for phrase in FORBIDDEN_CURRENT_CLAIMS:
                if phrase in text:
                    violations.append(f"{path}: unsupported current claim {phrase!r}")
    if violations:
        raise RuntimeError("Terminology audit failed: " + "; ".join(violations))
    if not audit["passed"]:
        raise RuntimeError("Corpus audit failed: " + "; ".join(audit["errors"]))
    study = _validate_study_evidence(repository_root)
    return {
        "passed": True,
        "corpus_hash": provenance["corpus_hash"],
        "labels_frozen": True,
        "annotation_type": provenance["annotation_type"],
        "independent_adjudication": False,
        "internal_consistency_audit": audit,
        "terminology_violations": [],
        "study_evidence": study,
    }


def _validate_study_evidence(repository_root: Path) -> dict[str, Any]:
    manifest = json.loads((repository_root / "evidence-manifest.json").read_text(encoding="utf-8"))
    study = manifest.get("study_evidence")
    if not study:
        return {"present": False}
    run_root = repository_root / "demo_runs/v0.3" / study["run_id"]
    completion = json.loads((run_root / "collection-complete.json").read_text(encoding="utf-8"))
    analysis_path = run_root / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    case_files = [
        path
        for path in run_root.glob("*/seed-*/*.json")
        if path.name != "sentinel.json"
    ]
    if completion.get("complete") is not True or len(case_files) != study["case_outputs"]:
        raise RuntimeError("Accepted study collection is incomplete.")
    if digest != study["analysis_sha256"]:
        raise RuntimeError("Accepted analysis digest differs from the evidence manifest.")
    if analysis.get("collection_identity_matrix_complete") is not True:
        raise RuntimeError("Accepted analysis lacks an identity-complete collection matrix.")
    headline = study["headline_results"]
    observed = {
        "hybrid_accuracy": analysis["metrics"]["hybrid"]["accuracy"],
        "hybrid_false_acceptance": analysis["metrics"]["hybrid"]["false_acceptance"],
        "hybrid_false_refusal": analysis["metrics"]["hybrid"]["false_refusal"],
        "llm_only_accuracy": analysis["metrics"]["llm_only_no_symbolic_gate"]["accuracy"],
        "llm_only_false_acceptance": analysis["metrics"]["llm_only_no_symbolic_gate"]["false_acceptance"],
        "llm_only_false_refusal": analysis["metrics"]["llm_only_no_symbolic_gate"]["false_refusal"],
        "structured_oracle_accuracy": analysis["metrics"]["structured_oracle_upper_bound"]["accuracy"],
    }
    if observed != headline:
        raise RuntimeError("Manifest headline results differ from the frozen analysis.")
    return {
        "present": True,
        "run_id": study["run_id"],
        "case_outputs": len(case_files),
        "analysis_sha256": digest,
        "headline_results_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.3"))
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.corpus, args.repository), indent=2))


if __name__ == "__main__":
    main()
