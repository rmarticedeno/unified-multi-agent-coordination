"""Generate and audit the frozen, author-labelled 36-case defense corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATA = (
    "direct_feasible",
    "decomposed_feasible",
    "runtime_failure",
    "missing_or_auxiliary_overreach",
    "schema_modality_graph_invalidity",
    "authority_constraint_registry_invalidity",
)
CORPUS_VERSION = "0.3"
LABELING_PROTOCOL_VERSION = "author-conformance-v1"
DEFAULT_AUTHOR = "Roberto Marti Cedeno"


def _case(stratum: str, index: int) -> dict[str, Any]:
    variant = (index - 1) // len(STRATA)
    capability_id = f"{stratum}-capability-{index}"
    expected_feasible = stratum in {
        "direct_feasible",
        "decomposed_feasible",
        "runtime_failure",
    }
    expected_status = (
        "failed"
        if stratum == "runtime_failure"
        else "completed"
        if expected_feasible
        else "infeasible"
    )
    required_trust = "admin" if stratum == "authority_constraint_registry_invalidity" else "standard"
    advertised_trust = "standard"
    registry: list[dict[str, Any]] = [] if stratum == "missing_or_auxiliary_overreach" else [
        {
            "agent_id": f"agent-{index}",
            "name": f"Agent {index}",
            "service_endpoint": f"https://agent-{index}.example/a2a",
            "trust_level": advertised_trust,
            "skills": [
                {
                    "name": f"Capability {index}",
                    "capability_id": capability_id,
                    "input_modes": ["text"],
                    "output_modes": ["json"],
                    "validation_contract": {
                        "json_schema": {"type": "object"}
                    },
                }
            ],
            "input_modes": ["text"],
            "output_modes": ["json"],
        }
    ]
    request: dict[str, Any] = {
        "user_goal": f"Complete defense corpus task {stratum} {index}.",
        "requirements": [
            {
                "name": f"Capability {index}",
                "requirement_id": capability_id,
                "capability_id": capability_id,
                "input_modes": ["table"]
                if stratum == "schema_modality_graph_invalidity"
                else ["text"],
                "output_modes": ["json"],
                "required_trust_level": required_trust,
                "validation_contract": {
                    "required_artifacts": [f"artifact-{index}"]
                },
            }
        ],
        "required_artifacts": [f"artifact-{index}"],
        "context": {"case_index": index, "allowed": True},
    }
    proposal: dict[str, Any] = {
        "tasks": [
            {
                "task_id": "t1",
                "requirement_name": f"Capability {index}",
                "requirement_id": capability_id,
                "capability_id": capability_id,
                "assigned_to": f"agent-{index}",
                "expected_artifacts": [f"artifact-{index}"],
                "validation_contract": {
                    "required_artifacts": [f"artifact-{index}"]
                },
            }
        ],
        "execution_order": ["t1"],
        "expected_artifacts": [f"artifact-{index}"],
        "completion_contract": {
            "required_task_states": ["completed"],
            "required_artifacts": [f"artifact-{index}"],
            "require_all_task_validators": True,
        },
    }
    if stratum == "decomposed_feasible":
        second_id = capability_id + "-finish"
        request["requirements"][0]["output_modes"] = ["intermediate"]
        request["requirements"].append(
            {
                "name": f"Finish Capability {index}",
                "requirement_id": second_id,
                "capability_id": second_id,
                "input_modes": ["intermediate"],
                "output_modes": ["json"],
                "validation_contract": {"required_artifacts": [f"artifact-{index}"]},
            }
        )
        registry[0]["skills"][0]["output_modes"] = ["intermediate"]
        registry[0]["output_modes"] = ["intermediate"]
        registry.append(
            {
                "agent_id": f"finisher-{index}",
                "name": f"Finisher {index}",
                "service_endpoint": f"https://finisher-{index}.example/a2a",
                "trust_level": "standard",
                "skills": [
                    {
                        "name": f"Finish Capability {index}",
                        "capability_id": second_id,
                        "input_modes": ["intermediate"],
                        "output_modes": ["json"],
                        "validation_contract": {"required_artifacts": [f"artifact-{index}"]},
                    }
                ],
                "input_modes": ["intermediate"],
                "output_modes": ["json"],
            }
        )
        proposal["tasks"][0]["expected_artifacts"] = [f"intermediate-{index}"]
        proposal["tasks"][0]["validation_contract"] = {
            "required_artifacts": [f"intermediate-{index}"]
        }
        proposal["tasks"].append(
            {
                "task_id": "t2",
                "requirement_name": f"Finish Capability {index}",
                "requirement_id": second_id,
                "capability_id": second_id,
                "assigned_to": f"finisher-{index}",
                "depends_on": ["t1"],
                "expected_artifacts": [f"artifact-{index}"],
                "validation_contract": {"required_artifacts": [f"artifact-{index}"]},
            }
        )
        proposal["execution_order"] = ["t1", "t2"]
    elif stratum == "schema_modality_graph_invalidity":
        if variant == 0:
            request["requirements"][0]["input_schema"] = {
                "type": "object",
                "required": ["nested"],
            }
            registry[0]["skills"][0]["input_schema"] = {"type": "string"}
            request["requirements"][0]["input_modes"] = ["text"]
        elif variant == 1:
            request["requirements"][0]["input_modes"] = ["table"]
        elif variant == 2:
            proposal["tasks"][0]["depends_on"] = ["t1"]
        else:
            proposal["execution_order"] = []
    elif stratum == "authority_constraint_registry_invalidity":
        if variant == 0:
            request["requirements"][0]["required_trust_level"] = "admin"
        elif variant == 1:
            request["requirements"][0]["required_trust_level"] = "standard"
            request["constraints"] = [
                {
                    "constraint_id": f"deny-{index}",
                    "source": "request_context",
                    "path": "/allowed",
                    "operator": "eq",
                    "expected": False,
                }
            ]
        elif variant == 2:
            request["requirements"][0]["required_trust_level"] = "standard"
            registry[0]["status"] = "unavailable"
        else:
            request["requirements"][0]["required_trust_level"] = "standard"
            proposal["tasks"][0]["assigned_to"] = "unknown-agent"
    return {
        "case_id": f"base-{index:02d}",
        "case_type": "base",
        "stratum": stratum,
        "request_text": request["user_goal"],
        "request": request,
        "registry_snapshot": registry,
        "payload": {"text": f"Input for case {index}", "force_failure": stratum == "runtime_failure"},
        "shared_raw_proposal": proposal,
        "reference": {
            "feasible": expected_feasible,
            "status": expected_status,
            "minimum_justification": f"Reference follows the declared {stratum} contract boundary.",
        },
    }


def build_corpus() -> list[dict[str, Any]]:
    bases = [_case(stratum, index) for index, stratum in enumerate(STRATA * 4, 1)]
    variations: list[dict[str, Any]] = []
    for offset in range(12):
        source = json.loads(json.dumps(bases[offset]))
        source["case_id"] = f"variation-{offset + 1:02d}"
        source["case_type"] = "variation"
        source["derived_from"] = bases[offset]["case_id"]
        if offset < 4:
            source["variation_type"] = "paraphrase"
            source["request_text"] = "Please " + source["request_text"].lower()
        elif offset < 8:
            source["variation_type"] = "availability_change"
            for agent in source["registry_snapshot"]:
                agent["status"] = "unavailable"
            source["reference"] = {
                "feasible": False,
                "status": "infeasible",
                "minimum_justification": "The only canonical capability provider is unavailable.",
            }
        else:
            source["variation_type"] = "capability_alias_perturbation"
            for agent in source["registry_snapshot"]:
                for skill in agent["skills"]:
                    skill["name"] = "Friendly alias that is not an identity"
                    skill["capability_id"] += "-mismatch"
            source["reference"] = {
                "feasible": False,
                "status": "infeasible",
                "minimum_justification": "Display-name similarity cannot replace canonical capability identity.",
            }
        variations.append(source)
    return bases + variations


def audit_corpus(public_document: dict[str, Any], labels_document: dict[str, Any]) -> dict[str, Any]:
    """Check internal consistency; this is not independent label adjudication."""
    cases = public_document.get("cases", [])
    labels = labels_document.get("labels", [])
    errors: list[str] = []
    case_ids = [item.get("case_id") for item in cases]
    label_ids = [item.get("case_id") for item in labels]
    if len(cases) != 36:
        errors.append(f"expected 36 public cases, found {len(cases)}")
    if len(set(case_ids)) != len(case_ids):
        errors.append("duplicate public case_id")
    if len(set(label_ids)) != len(label_ids):
        errors.append("duplicate reference-label case_id")
    if set(case_ids) != set(label_ids):
        errors.append("public cases and reference labels have different case_id sets")
    if public_document.get("corpus_hash") != labels_document.get("corpus_hash"):
        errors.append("public and hidden corpus hashes differ")
    allowed_statuses = {"completed", "failed", "infeasible"}
    for label in labels:
        if not isinstance(label.get("feasible"), bool):
            errors.append(f"{label.get('case_id')}: feasible must be boolean")
        if label.get("status") not in allowed_statuses:
            errors.append(f"{label.get('case_id')}: invalid status")
        if label.get("feasible") is False and label.get("status") != "infeasible":
            errors.append(f"{label.get('case_id')}: infeasible label has contradictory status")
        if not str(label.get("minimum_justification", "")).strip():
            errors.append(f"{label.get('case_id')}: missing minimum justification")
    return {
        "audit_type": "internal_consistency_only",
        "independent_adjudication": False,
        "case_count": len(cases),
        "label_count": len(labels),
        "checks": [
            "document shape and count",
            "unique scenario identities",
            "public/hidden identity alignment",
            "hash alignment",
            "label status consistency",
            "minimum justification presence",
        ],
        "passed": not errors,
        "errors": errors,
    }


def write_corpus(root: Path, *, author: str = DEFAULT_AUTHOR) -> dict[str, Any]:
    cases = build_corpus()
    public_cases = [
        {key: value for key, value in case.items() if key != "reference"}
        for case in cases
    ]
    labels = [
        {"case_id": case["case_id"], **case["reference"]} for case in cases
    ]
    canonical = json.dumps(
        {"cases": public_cases, "labels": labels}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    corpus_hash = hashlib.sha256(canonical).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat()
    public = root / "public"
    hidden = root / "hidden"
    review = root / "review"
    public.mkdir(parents=True, exist_ok=True)
    hidden.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    public_document = {"version": CORPUS_VERSION, "corpus_hash": corpus_hash, "cases": public_cases}
    labels_document = {"version": CORPUS_VERSION, "corpus_hash": corpus_hash, "labels": labels}
    provenance_path = root / "label-provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("frozen") is True:
            try:
                existing_public = json.loads((public / "cases.json").read_text(encoding="utf-8"))
                existing_labels = json.loads(
                    (hidden / "reference-labels.json").read_text(encoding="utf-8")
                )
                existing_canonical = json.dumps(
                    {"cases": existing_public["cases"], "labels": existing_labels["labels"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                existing_hash = hashlib.sha256(existing_canonical).hexdigest()
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError("Frozen corpus files are missing or invalid.") from exc
            if len({existing_hash, provenance.get("corpus_hash"), corpus_hash}) != 1:
                raise RuntimeError("Frozen corpus cannot be changed; create a new corpus version.")
            return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    (public / "cases.json").write_text(json.dumps(public_document, indent=2), encoding="utf-8")
    (hidden / "reference-labels.json").write_text(json.dumps(labels_document, indent=2), encoding="utf-8")
    (review / "label-review-packet.json").write_text(
        json.dumps(
            {
                "version": CORPUS_VERSION,
                "corpus_hash": corpus_hash,
                "instructions": "Author labels frozen before collection; model outputs are unavailable here.",
                "items": labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    provenance = {
        "annotation_type": "author_labeled",
        "author": author,
        "annotation_date": generated_at,
        "labeling_protocol_version": LABELING_PROTOCOL_VERSION,
        "corpus_version": CORPUS_VERSION,
        "corpus_hash": corpus_hash,
        "frozen": True,
        "independent_adjudication": False,
        "limitation": "No independent label adjudicator was available; labels encode the author's declared framework conformance criteria.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    audit = audit_corpus(public_document, labels_document)
    if not audit["passed"]:
        raise RuntimeError("Corpus internal-consistency audit failed: " + "; ".join(audit["errors"]))
    (review / "internal-consistency-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    manifest = {
        "version": CORPUS_VERSION,
        "generated_at": generated_at,
        "corpus_hash": corpus_hash,
        "case_count": len(cases),
        "base_case_count": 24,
        "variation_count": 12,
        "strata": list(STRATA),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("corpus/v0.3"))
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.output, author=args.author), indent=2))


if __name__ == "__main__":
    main()
