"""Create the v0.4 domain-specific held-out bridge-evaluation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import DEFAULT_AUTHOR, STRATA, _case, audit_corpus

VERSION = "0.4"
PROTOCOL = "author-conformance-bridge-v2"
DOMAINS = (
    "coastal flood response",
    "hospital supply routing",
    "wildfire sensor triage",
    "satellite imagery processing",
    "warehouse incident recovery",
    "municipal water monitoring",
)


def build_corpus() -> list[dict[str, Any]]:
    bases: list[dict[str, Any]] = []
    for index, stratum in enumerate(STRATA * 4, 1):
        case = _case(stratum, index)
        domain = DOMAINS[(index - 1) % len(DOMAINS)]
        goal = (
            f"Coordinate the declared services for {domain}; preserve the specified "
            "data flow, evidence contract, and authority limits."
        )
        case["request"]["user_goal"] = goal
        case["request_text"] = goal
        case["domain"] = domain
        # v0.3 encoded two graph-invalid labels only in an oracle proposal that the
        # linguistic model could not observe. v0.4 makes the invalidity intrinsic
        # to the admitted request/registry boundary.
        if stratum == "schema_modality_graph_invalidity" and (index - 1) // 6 >= 2:
            case["request"]["requirements"][0]["capability_id"] += "-required"
        bases.append(case)
    variations: list[dict[str, Any]] = []
    for offset in range(12):
        source = json.loads(json.dumps(bases[offset]))
        source["case_id"] = f"variation-{offset + 1:02d}"
        source["case_type"] = "variation"
        source["derived_from"] = bases[offset]["case_id"]
        if offset < 4:
            source["variation_type"] = "linguistic_paraphrase"
            source["request_text"] = (
                "Using only the listed providers, " + source["request_text"].lower()
            )
        elif offset < 8:
            source["variation_type"] = "availability_change"
            for agent in source["registry_snapshot"]:
                agent["status"] = "unavailable"
            source["reference"] = {
                "feasible": False,
                "status": "infeasible",
                "minimum_justification": "Every canonical provider is unavailable.",
            }
        else:
            source["variation_type"] = "canonical_identity_change"
            for agent in source["registry_snapshot"]:
                for skill in agent["skills"]:
                    skill["capability_id"] += "-changed"
            source["reference"] = {
                "feasible": False,
                "status": "infeasible",
                "minimum_justification": "No exact canonical capability identity remains.",
            }
        variations.append(source)
    return bases + variations


def _documents() -> tuple[dict[str, Any], dict[str, Any], str]:
    cases = build_corpus()
    public_cases = [{k: v for k, v in case.items() if k != "reference"} for case in cases]
    labels = [{"case_id": case["case_id"], **case["reference"]} for case in cases]
    canonical = json.dumps(
        {"cases": public_cases, "labels": labels}, sort_keys=True, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return (
        {"version": VERSION, "corpus_hash": digest, "cases": public_cases},
        {"version": VERSION, "corpus_hash": digest, "labels": labels},
        digest,
    )


def write_corpus(root: Path, author: str = DEFAULT_AUTHOR) -> dict[str, Any]:
    public_doc, labels_doc, digest = _documents()
    provenance_path = root / "label-provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("frozen") and provenance.get("corpus_hash") != digest:
            raise RuntimeError("Frozen v0.4 corpus cannot be modified; create a new version.")
        return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    public = root / "public"
    hidden = root / "hidden"
    review = root / "review"
    public.mkdir(parents=True, exist_ok=False)
    hidden.mkdir(parents=True, exist_ok=False)
    review.mkdir(parents=True, exist_ok=False)
    (public / "cases.json").write_text(json.dumps(public_doc, indent=2), encoding="utf-8")
    (hidden / "reference-labels.json").write_text(
        json.dumps(labels_doc, indent=2), encoding="utf-8"
    )
    # Development examples are explicitly outside the frozen held-out scoring units.
    dev_cases = [
        {"request": case["request"], "registry_snapshot": case["registry_snapshot"]}
        for case in public_doc["cases"][:6]
    ]
    (public / "development-examples.json").write_text(
        json.dumps({"scored": False, "cases": dev_cases}, indent=2), encoding="utf-8"
    )
    generated = datetime.now(timezone.utc).isoformat()
    provenance = {
        "annotation_type": "author_labeled",
        "author": author,
        "annotation_date": generated,
        "labeling_protocol_version": PROTOCOL,
        "corpus_version": VERSION,
        "corpus_hash": digest,
        "frozen": True,
        "label_hidden_during_inference": True,
        "independent_adjudication": False,
        "limitation": "No independent adjudicator; labels encode author-declared conformance.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    audit = audit_corpus(public_doc, labels_doc)
    if not audit["passed"]:
        raise RuntimeError("v0.4 consistency audit failed: " + "; ".join(audit["errors"]))
    (review / "internal-consistency-audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": VERSION,
        "generated_at": generated,
        "corpus_hash": digest,
        "case_count": 36,
        "base_case_count": 24,
        "variation_count": 12,
        "development_example_count": 6,
        "primary_statistical_unit": "case",
        "repeated_measurements": ["model", "seed"],
        "success_criteria": {
            "hybrid_false_accepts": 0,
            "feasible_case_majority_accepts_at_least": 12,
            "minimum_per_model_feasible_recall": 0.60,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("corpus/v0.4"))
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.output, args.author), indent=2))


if __name__ == "__main__":
    main()
