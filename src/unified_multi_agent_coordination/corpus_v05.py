"""Build the fresh matched-pair v0.5 defense corpus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import DEFAULT_AUTHOR, audit_corpus

VERSION = "0.5"
PROTOCOL = "author-conformance-dual-arm-v1"
DOMAINS = (
    "document processing",
    "travel logistics",
    "data analytics",
    "software maintenance",
    "research administration",
    "inventory management",
    "customer support",
    "iot inspection",
)
INVALIDITY_CATEGORIES = (
    "missing_capability",
    "unavailable_agent",
    "insufficient_trust",
    "incompatible_modality",
    "typed_constraint_violation",
    "invalid_artifact_contract",
    "dependency_order_error",
    "auxiliary_overreach",
)
OFFSETS = (0, 3, 5)


def _base_case(domain: str, pair_number: int, category: str) -> dict[str, Any]:
    stem = f"v05-pair-{pair_number:02d}"
    prepare_id = f"{stem}-prepare"
    finalize_id = f"{stem}-finalize"
    intermediate = f"{stem}-intermediate"
    result = f"{stem}-result"
    request: dict[str, Any] = {
        "user_goal": (
            f"Prepare and finalize the declared {domain} workflow while preserving "
            "authority, data-flow, and completion contracts."
        ),
        "requirements": [
            {
                "name": f"Prepare {domain}",
                "requirement_id": prepare_id,
                "capability_id": prepare_id,
                "input_modes": ["text"],
                "output_modes": ["intermediate"],
                "required_trust_level": "standard",
                "side_effect_class": "read_only",
                "validation_contract": {
                    "required_artifacts": [intermediate],
                    "json_schema": {"type": "object"},
                },
            },
            {
                "name": f"Finalize {domain}",
                "requirement_id": finalize_id,
                "capability_id": finalize_id,
                "input_modes": ["intermediate"],
                "output_modes": ["json"],
                "depends_on_requirement_ids": [prepare_id],
                "required_trust_level": "standard",
                "side_effect_class": "read_only",
                "validation_contract": {
                    "required_artifacts": [result],
                    "json_schema": {"type": "object"},
                },
            },
        ],
        "required_artifacts": [result],
        "context": {"approved": True, "pair_number": pair_number},
    }
    registry: list[dict[str, Any]] = [
        {
            "agent_id": f"{stem}-preparer",
            "name": f"{domain.title()} preparer",
            "service_endpoint": f"https://{stem}-preparer.example/a2a",
            "trust_level": "standard",
            "status": "available",
            "input_modes": ["text"],
            "output_modes": ["intermediate"],
            "skills": [copy.deepcopy(request["requirements"][0])],
        },
        {
            "agent_id": f"{stem}-finisher",
            "name": f"{domain.title()} finisher",
            "service_endpoint": f"https://{stem}-finisher.example/a2a",
            "trust_level": "standard",
            "status": "available",
            "input_modes": ["intermediate"],
            "output_modes": ["json"],
            "skills": [copy.deepcopy(request["requirements"][1])],
        },
    ]
    proposal: dict[str, Any] = {
        "tasks": [
            {
                "task_id": "t1",
                "requirement_name": request["requirements"][0]["name"],
                "requirement_id": prepare_id,
                "capability_id": prepare_id,
                "assigned_to": registry[0]["agent_id"],
                "expected_artifacts": [intermediate],
                "validation_contract": request["requirements"][0]["validation_contract"],
            },
            {
                "task_id": "t2",
                "requirement_name": request["requirements"][1]["name"],
                "requirement_id": finalize_id,
                "capability_id": finalize_id,
                "assigned_to": registry[1]["agent_id"],
                "depends_on": ["t1"],
                "expected_artifacts": [result],
                "validation_contract": request["requirements"][1]["validation_contract"],
            },
        ],
        "execution_order": ["t1", "t2"],
        "expected_artifacts": [result],
        "completion_contract": {
            "required_task_states": ["completed"],
            "required_artifacts": [result],
            "require_all_task_validators": True,
        },
    }
    return {
        "case_id": f"{stem}-feasible",
        "pair_id": stem,
        "case_type": "matched_feasible",
        "domain": domain,
        "invalidity_category": category,
        "request_text": request["user_goal"],
        "request": request,
        "registry_snapshot": registry,
        "payload": {"text": f"Input for {domain} pair {pair_number}"},
        "oracle_proposal": proposal,
        "reference": {
            "feasible": True,
            "status": "completed",
            "minimum_justification": "All admitted predicates and contracts are satisfied.",
        },
    }


def _invalid_counterpart(feasible: dict[str, Any], category: str) -> dict[str, Any]:
    case = copy.deepcopy(feasible)
    case["case_id"] = case["case_id"].replace("-feasible", "-infeasible")
    case["case_type"] = "matched_infeasible"
    request = case["request"]
    registry = case["registry_snapshot"]
    proposal = case["oracle_proposal"]
    final = request["requirements"][1]
    final_skill = registry[1]["skills"][0]
    justification = ""
    if category == "missing_capability":
        final["capability_id"] += "-missing"
        proposal["tasks"][1]["capability_id"] = final["capability_id"]
        justification = "No admitted agent advertises the exact required capability."
    elif category == "unavailable_agent":
        registry[1]["status"] = "unavailable"
        justification = "The only canonical provider is unavailable."
    elif category == "insufficient_trust":
        final["required_trust_level"] = "high"
        justification = "The available provider does not satisfy required trust."
    elif category == "incompatible_modality":
        final_skill["input_modes"] = ["text"]
        registry[1]["input_modes"] = ["text"]
        justification = "The provider cannot consume the required intermediate modality."
    elif category == "typed_constraint_violation":
        final["constraints"] = [{
            "constraint_id": f"{case['pair_id']}-approval",
            "source": "request_context",
            "path": "approved",
            "operator": "eq",
            "expected": True,
            "requirement_id": final["requirement_id"],
        }]
        request["context"]["approved"] = False
        justification = "A typed request-context constraint evaluates false."
    elif category == "invalid_artifact_contract":
        final["validation_contract"] = {}
        final_skill["validation_contract"] = {}
        proposal["tasks"][1]["validation_contract"] = {}
        justification = "The required terminal artifact has no enforceable validator contract."
    elif category == "dependency_order_error":
        request["requirements"][0]["depends_on_requirement_ids"] = [final["requirement_id"]]
        proposal["tasks"][0]["depends_on"] = ["t2"]
        justification = "The admitted requirement dependencies form a cycle."
    elif category == "auxiliary_overreach":
        final["capability_id"] += "-unadmitted-auxiliary"
        final["auxiliary_eligible"] = False
        proposal["tasks"][1]["capability_id"] = final["capability_id"]
        justification = "The missing capability is explicitly ineligible for auxiliary synthesis."
    else:  # pragma: no cover - guarded by constants and tests
        raise ValueError(category)
    case["reference"] = {
        "feasible": False,
        "status": "infeasible",
        "minimum_justification": justification,
    }
    return case


def build_corpus() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    pair = 0
    for domain_index, domain in enumerate(DOMAINS):
        for offset in OFFSETS:
            pair += 1
            category = INVALIDITY_CATEGORIES[(domain_index + offset) % len(INVALIDITY_CATEGORIES)]
            feasible = _base_case(domain, pair, category)
            cases.extend((feasible, _invalid_counterpart(feasible, category)))
    return cases


def _runtime_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    categories = (
        "dependency_failure",
        "timeout_unknown",
        "retry_recovery",
        "artifact_trace",
    )
    public: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    for index in range(12):
        category = categories[index // 3]
        case_id = f"runtime-{index + 1:02d}"
        public.append({
            "case_id": case_id,
            "category": category,
            "payload": {"text": f"Runtime validation {index + 1}"},
            "scored_as_planning_accuracy": False,
        })
        hidden.append({
            "case_id": case_id,
            "required_checks": ["authorization_before_dispatch", "trace_order_complete"],
        })
    return public, hidden


def _documents() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str]:
    cases = build_corpus()
    public_cases = [
        {
            key: value
            for key, value in case.items()
            if key not in {"reference", "oracle_proposal"}
        }
        for case in cases
    ]
    labels = [
        {
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "domain": case["domain"],
            "invalidity_category": case["invalidity_category"],
            "oracle_proposal": case["oracle_proposal"],
            **case["reference"],
        }
        for case in cases
    ]
    runtime_public, runtime_hidden = _runtime_cases()
    canonical = json.dumps(
        {"cases": public_cases, "labels": labels, "runtime": runtime_public,
         "runtime_labels": runtime_hidden},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return (
        {"version": VERSION, "corpus_hash": digest, "cases": public_cases},
        {"version": VERSION, "corpus_hash": digest, "labels": labels},
        {"version": VERSION, "corpus_hash": digest, "cases": runtime_public},
        {"version": VERSION, "corpus_hash": digest, "labels": runtime_hidden},
        digest,
    )


def write_corpus(root: Path, author: str = DEFAULT_AUTHOR) -> dict[str, Any]:
    public_doc, labels_doc, runtime_doc, runtime_labels, digest = _documents()
    provenance_path = root / "label-provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("frozen") and provenance.get("corpus_hash") != digest:
            raise RuntimeError("Frozen v0.5 corpus cannot be modified; create a new version.")
        return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative in ("public", "hidden", "development", "runtime", "review"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    (root / "public/cases.json").write_text(json.dumps(public_doc, indent=2), encoding="utf-8")
    (root / "hidden/reference-labels.json").write_text(
        json.dumps(labels_doc, indent=2), encoding="utf-8"
    )
    (root / "runtime/cases.json").write_text(json.dumps(runtime_doc, indent=2), encoding="utf-8")
    (root / "hidden/runtime-labels.json").write_text(
        json.dumps(runtime_labels, indent=2), encoding="utf-8"
    )
    development = [
        {key: value for key, value in _base_case(domain, 100 + index, "development").items()
         if key not in {"reference", "oracle_proposal"}}
        for index, domain in enumerate(DOMAINS, start=1)
    ]
    (root / "development/cases.json").write_text(
        json.dumps({"version": VERSION, "scored": False, "cases": development}, indent=2),
        encoding="utf-8",
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
        "pre_specified_before_collection": True,
        "label_hidden_during_inference": True,
        "independent_adjudication": False,
        "limitation": "Author labels measure declared-framework conformance, not neutral truth.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    audit = audit_corpus(public_doc, labels_doc, expected_count=48)
    if not audit["passed"]:
        raise RuntimeError("v0.5 consistency audit failed: " + "; ".join(audit["errors"]))
    (root / "review/internal-consistency-audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    manifest = {
        "version": VERSION,
        "generated_at": generated,
        "corpus_hash": digest,
        "case_count": 48,
        "matched_pair_count": 24,
        "feasible_case_count": 24,
        "infeasible_case_count": 24,
        "development_example_count": 8,
        "runtime_case_count": 12,
        "primary_statistical_unit": "matched case pair",
        "repeated_measurements": ["arm", "model", "requested_seed"],
        "primary_output_count": 1440,
        "success_criteria": {
            "hybrid_false_accepts": 0,
            "minimum_per_model_feasible_recall": 0.60,
            "paired_accuracy_difference_vs_direct_llm": "positive_with_reported_interval",
            "runtime_authorization_and_trace_compliance": 1.0,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("corpus/v0.5"))
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.output, args.author), indent=2))


if __name__ == "__main__":
    main()
