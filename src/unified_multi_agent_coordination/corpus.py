"""Generate the versioned, label-separated 36-case defense corpus."""

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


def write_corpus(root: Path) -> dict[str, Any]:
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
    (public / "cases.json").write_text(
        json.dumps({"version": "0.2", "corpus_hash": corpus_hash, "cases": public_cases}, indent=2),
        encoding="utf-8",
    )
    (hidden / "reference-labels.json").write_text(
        json.dumps({"version": "0.2", "corpus_hash": corpus_hash, "labels": labels}, indent=2),
        encoding="utf-8",
    )
    (review / "label-review-packet.json").write_text(
        json.dumps(
            {
                "version": "0.2",
                "corpus_hash": corpus_hash,
                "instructions": "Review labels and minimum justifications without model outputs.",
                "items": labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    signoff = root / "label-signoff.json"
    existing_signoff = (
        json.loads(signoff.read_text(encoding="utf-8")) if signoff.exists() else {}
    )
    if existing_signoff.get("approved") is True and existing_signoff.get("corpus_hash") != corpus_hash:
        raise RuntimeError("Refusing to replace a signed label record for a different corpus hash.")
    if not signoff.exists() or existing_signoff.get("approved") is not True:
        signoff.write_text(
            json.dumps(
                {
                    "corpus_hash": corpus_hash,
                    "approved": False,
                    "reviewer_role": "",
                    "reviewer_name": "",
                    "review_date": "",
                    "notes": "Independent qualified reviewer must complete this record.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    manifest = {
        "version": "0.2",
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
    parser.add_argument("--output", type=Path, default=Path("corpus/v0.2"))
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.output), indent=2))


if __name__ == "__main__":
    main()
