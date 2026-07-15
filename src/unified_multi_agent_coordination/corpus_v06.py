"""Generate the frozen raw-language matched-pair corpus for study v0.6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import DEFAULT_AUTHOR

VERSION = "0.6"
PROTOCOL = "raw-language-matched-pairs-v1"
CATEGORIES = (
    "paraphrase",
    "dependencies",
    "trust_policy",
    "artifact_schema",
    "ambiguity",
    "negation",
)
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


def _skill(
    name: str,
    requirement_id: str,
    capability_id: str,
    input_modes: list[str],
    output_modes: list[str],
    artifacts: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "requirement_id": requirement_id,
        "capability_id": capability_id,
        "input_modes": input_modes,
        "output_modes": output_modes,
        "required_trust_level": "standard",
        "side_effect_class": "read_only",
        "validation_contract": {
            "required_artifacts": artifacts,
            "json_schema": {"type": "object"},
        },
    }


def _environment(pair_number: int, domain: str) -> dict[str, Any]:
    stem = f"v06-pair-{pair_number:02d}"
    prepare = f"{stem}-prepare"
    verify = f"{stem}-verify"
    deliver = f"{stem}-deliver"
    archive = f"{stem}-archive"
    intermediate = f"{stem}-source-bundle"
    checked = f"{stem}-checked-bundle"
    result = f"{stem}-result-json"
    signed = f"{stem}-signed-pdf"
    catalog: list[dict[str, Any]] = [
        {
            "capability_id": prepare,
            "name": f"Compile {domain} source bundle",
            "description": "assemble and prepare the source material",
            "depends_on_capability_ids": [],
            "input_modes": ["text"],
            "output_modes": ["source_bundle"],
            "default_artifact": intermediate,
        },
        {
            "capability_id": verify,
            "name": f"Verify {domain} source bundle",
            "description": "check and validate the prepared material",
            "depends_on_capability_ids": [prepare],
            "input_modes": ["source_bundle"],
            "output_modes": ["checked_bundle"],
            "default_artifact": checked,
        },
        {
            "capability_id": deliver,
            "name": f"Release {domain} verified summary",
            "description": "issue, publish, or deliver the checked final summary",
            "depends_on_capability_ids": [verify],
            "input_modes": ["checked_bundle"],
            "output_modes": ["json"],
            "default_artifact": result,
        },
        {
            "capability_id": archive,
            "name": f"Archive {domain} verified summary",
            "description": "preserve the checked summary in a long-term archive",
            "depends_on_capability_ids": [verify],
            "input_modes": ["checked_bundle"],
            "output_modes": ["json"],
            "default_artifact": result,
        },
    ]
    agents: list[tuple[str, str, str, dict[str, Any]]] = [
        ("preparer", "Source compiler", "standard", catalog[0]),
        ("verifier", "Source verifier", "standard", catalog[1]),
        ("delivery-a", "Primary delivery", "standard", catalog[2]),
        ("delivery-b", "Backup delivery", "standard", catalog[2]),
    ]
    registry = []
    for suffix, name, trust, capability in agents:
        agent_id = f"{stem}-{suffix}"
        skill = _skill(
            capability["name"],
            capability["capability_id"],
            capability["capability_id"],
            capability["input_modes"],
            capability["output_modes"],
            [capability["default_artifact"]],
        )
        registry.append(
            {
                "agent_id": agent_id,
                "name": name,
                "service_endpoint": f"https://{agent_id}.example/a2a",
                "trust_level": trust,
                "status": "available",
                "input_modes": skill["input_modes"],
                "output_modes": skill["output_modes"],
                "skills": [skill],
            }
        )
    return {
        "stem": stem,
        "catalog": catalog,
        "registry": registry,
        "policy": {
            "trust_options": [
                {
                    "policy_id": f"{stem}-trust-standard",
                    "name": "ordinary assurance",
                    "required_trust_level": "standard",
                },
                {
                    "policy_id": f"{stem}-trust-high",
                    "name": "high assurance only",
                    "required_trust_level": "high",
                },
            ],
            "artifact_contracts": [
                {
                    "contract_id": f"{stem}-contract-json",
                    "name": "machine-readable JSON",
                    "output_modes": ["json"],
                    "required_artifacts": [result],
                    "json_schema": {"type": "object"},
                },
                {
                    "contract_id": f"{stem}-contract-signed-pdf",
                    "name": "signed PDF",
                    "output_modes": ["signed_pdf"],
                    "required_artifacts": [signed],
                    "json_schema": {"type": "object"},
                },
            ],
        },
        "ids": {
            "prepare": prepare,
            "verify": verify,
            "deliver": deliver,
            "archive": archive,
            "trust_standard": f"{stem}-trust-standard",
            "trust_high": f"{stem}-trust-high",
            "json": f"{stem}-contract-json",
            "signed_pdf": f"{stem}-contract-signed-pdf",
            "delivery_a": f"{stem}-delivery-a",
            "delivery_b": f"{stem}-delivery-b",
        },
    }


def _semantic(
    ids: dict[str, str],
    *,
    goals: list[str] | None = None,
    policy: str = "trust_standard",
    contract: str = "json",
    forbidden_capabilities: list[str] | None = None,
    forbidden_agents: list[str] | None = None,
    ambiguous: bool = False,
) -> dict[str, Any]:
    return {
        "interpretation_status": "ambiguous" if ambiguous else "resolved",
        "goal_capability_ids": goals or [ids["deliver"]],
        "required_policy_ids": [ids[policy]],
        "required_artifact_contract_ids": [ids[contract]],
        "forbidden_capability_ids": forbidden_capabilities or [],
        "forbidden_agent_ids": forbidden_agents or [],
    }


def _texts(category: str, domain: str) -> tuple[str, str]:
    normal = (
        f"For the {domain} material, compile the source bundle, verify it, and release the "
        "verified summary with ordinary assurance as machine-readable JSON."
    )
    if category == "paraphrase":
        return (
            f"For {domain}, assemble the source material, check it, then issue the checked "
            "summary under ordinary assurance in machine-readable JSON.",
            f"For {domain}, preserve the checked summary in the long-term archive under "
            "ordinary assurance and return machine-readable JSON.",
        )
    if category == "dependencies":
        return (
            normal,
            f"Release the verified {domain} summary with ordinary assurance as machine-readable "
            "JSON, but do not compile the source bundle.",
        )
    if category == "trust_policy":
        return (
            normal,
            f"Compile, verify, and release the {domain} summary as machine-readable JSON, using "
            "high assurance only.",
        )
    if category == "artifact_schema":
        return (
            normal,
            f"Compile, verify, and release the {domain} summary with ordinary assurance as a "
            "signed PDF.",
        )
    if category == "ambiguity":
        return (
            normal,
            f"For {domain}, compile and verify the material, then complete the checked-summary "
            "output with ordinary assurance as machine-readable JSON; it is unclear whether "
            "release or archival is intended.",
        )
    return (
        f"Compile, verify, and release the {domain} summary with ordinary assurance as "
        "machine-readable JSON. Do not use the backup delivery agent.",
        f"Compile, verify, and release the {domain} summary with ordinary assurance as "
        "machine-readable JSON. Do not use either the primary or backup delivery agent.",
    )


def _pair(pair_number: int, category: str, domain: str) -> list[dict[str, Any]]:
    env = _environment(pair_number, domain)
    ids = env["ids"]
    feasible_text, infeasible_text = _texts(category, domain)
    feasible_semantic = _semantic(ids)
    infeasible_semantic = _semantic(ids)
    reason = "The public registry and policy satisfy the resolved request."
    if category == "paraphrase":
        infeasible_semantic = _semantic(ids, goals=[ids["archive"]])
        reason = "No admitted agent provides the requested archival capability."
    elif category == "dependencies":
        infeasible_semantic = _semantic(ids, forbidden_capabilities=[ids["prepare"]])
        reason = "A required dependency is explicitly forbidden."
    elif category == "trust_policy":
        infeasible_semantic = _semantic(ids, policy="trust_high")
        reason = "No provider satisfies the requested high-assurance policy."
    elif category == "artifact_schema":
        infeasible_semantic = _semantic(ids, contract="signed_pdf")
        reason = "The selected providers cannot produce the required signed-PDF mode."
    elif category == "ambiguity":
        infeasible_semantic = _semantic(ids, goals=[ids["deliver"], ids["archive"]], ambiguous=True)
        reason = "The final operation is explicitly unresolved between two public capabilities."
    elif category == "negation":
        feasible_semantic = _semantic(ids, forbidden_agents=[ids["delivery_b"]])
        infeasible_semantic = _semantic(
            ids, forbidden_agents=[ids["delivery_a"], ids["delivery_b"]]
        )
        reason = "Every admitted provider of the goal capability is explicitly forbidden."
    common = {
        "pair_id": env["stem"],
        "domain": domain,
        "category": category,
        "capability_catalog": env["catalog"],
        "registry_snapshot": env["registry"],
        "policy": env["policy"],
        "payload": {"text": f"Input material for {domain} pair {pair_number}."},
    }
    return [
        {
            **common,
            "case_id": f"{env['stem']}-feasible",
            "case_type": "matched_feasible",
            "request_text": feasible_text,
            "reference": {
                "feasible": True,
                "minimum_justification": "The public registry and policy satisfy the request.",
                "typed_reference_request": feasible_semantic,
            },
        },
        {
            **common,
            "case_id": f"{env['stem']}-infeasible",
            "case_type": "matched_infeasible",
            "request_text": infeasible_text,
            "reference": {
                "feasible": False,
                "minimum_justification": reason,
                "typed_reference_request": infeasible_semantic,
            },
        },
    ]


def build_corpus() -> list[dict[str, Any]]:
    return [
        case
        for index in range(40)
        for case in _pair(
            index + 1,
            CATEGORIES[index % len(CATEGORIES)],
            DOMAINS[index % len(DOMAINS)],
        )
    ]


def _canonical_hash(public_cases: list[dict[str, Any]], labels: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        {"cases": public_cases, "labels": labels},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _review_csv(cases: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "case_id",
            "pair_id",
            "category",
            "request_text",
            "public_environment_json",
            "reviewer_feasible",
            "reviewer_justification",
        ),
    )
    writer.writeheader()
    for case in sorted(
        cases, key=lambda item: hashlib.sha256(item["case_id"].encode()).hexdigest()
    ):
        writer.writerow(
            {
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "category": case["category"],
                "request_text": case["request_text"],
                "public_environment_json": json.dumps(
                    {
                        "capability_catalog": case["capability_catalog"],
                        "registry_snapshot": case["registry_snapshot"],
                        "policy": case["policy"],
                        "payload": case["payload"],
                    },
                    separators=(",", ":"),
                ),
                "reviewer_feasible": "",
                "reviewer_justification": "",
            }
        )
    return buffer.getvalue()


def write_corpus(root: Path, author: str = DEFAULT_AUTHOR) -> dict[str, Any]:
    if root.exists():
        raise FileExistsError(f"v0.6 output already exists: {root}")
    cases = build_corpus()
    public_cases = [
        {key: value for key, value in case.items() if key != "reference"} for case in cases
    ]
    labels = [
        {
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "category": case["category"],
            **case["reference"],
        }
        for case in cases
    ]
    digest = _canonical_hash(public_cases, labels)
    generated = datetime.now(timezone.utc).isoformat()
    for relative in ("public", "hidden", "review"):
        (root / relative).mkdir(parents=True, exist_ok=False)
    (root / "public/cases.json").write_text(
        json.dumps({"version": VERSION, "corpus_hash": digest, "cases": public_cases}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (root / "hidden/reference-labels.json").write_text(
        json.dumps({"version": VERSION, "corpus_hash": digest, "labels": labels}, indent=2) + "\n",
        encoding="utf-8",
    )
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
        "limitation": "Author-only labels are not an independent benchmark.",
    }
    (root / "label-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    counts = Counter(case["category"] for case in public_cases)
    manifest = {
        "version": VERSION,
        "generated_at": generated,
        "corpus_hash": digest,
        "case_count": 80,
        "matched_pair_count": 40,
        "feasible_case_count": 40,
        "infeasible_case_count": 40,
        "category_case_counts": dict(sorted(counts.items())),
        "primary_statistical_unit": "matched case pair",
        "repeated_measurements": ["arm", "model", "requested_seed"],
        "expected_llm_outputs": 2400,
        "labels_exposed_during_collection": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "review/blind-label-review.csv").write_text(
        _review_csv(public_cases), encoding="utf-8", newline=""
    )
    (root / "review/reviewer-instructions.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "blind": True,
                "author_labels_included": False,
                "instructions": [
                    "Judge feasibility using only the request and public environment.",
                    "Enter exactly true or false in reviewer_feasible for every case.",
                    "Give a short predicate-level justification.",
                    "Return the completed CSV with reviewer identity and date separately.",
                ],
                "status": "awaiting_independent_reviewer",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("corpus/v0.6"))
    parser.add_argument("--author", default=DEFAULT_AUTHOR)
    args = parser.parse_args()
    print(json.dumps(write_corpus(args.output, args.author), indent=2))


if __name__ == "__main__":
    main()
