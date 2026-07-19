"""Small versionless live validation for the redesigned hybrid pipeline."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .feasibility import FeasibilityAnalyzer
from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    ProblemRequest,
    ValidationContract,
)
from .semantic_admission import (
    ArtifactContractOption,
    CapabilityCatalogEntry,
    OpenAICompatibleSemanticInterpreter,
    SemanticCatalog,
    SemanticIntentOutput,
    SemanticRequestAdmitter,
    TrustPolicyOption,
    semantic_intent_schema,
    semantic_prompt,
)
from .symbolic_plan_compiler import SymbolicPlanCompiler


MODELS = (
    "qwen/qwen3-1.7b",
    "google/gemma-4-e2b",
    "qwen/qwen3-8b",
)
FIXTURE_ROOT = Path("validation/hybrid_strategy")
PUBLIC_PATH = FIXTURE_ROOT / "public_cases.json"
EXPECTED_PATH = FIXTURE_ROOT / "expected.json"
FROZEN_CORPUS_ROOT = Path("corpus/v0.5")
FROZEN_PUBLIC_PATH = FROZEN_CORPUS_ROOT / "public/cases.json"
FROZEN_LABELS_PATH = FROZEN_CORPUS_ROOT / "hidden/reference-labels.json"
ACCEPTED_ANALYSIS_PATH = Path("e/v5/analysis-v0.5.0.json")
SENTINEL_CASES = {
    "paraphrase-feasible",
    "artifact-alternative-provider",
    "forbidden-dependency",
    "unresolved-ambiguity",
}


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _hash_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def strategy_catalog() -> SemanticCatalog:
    def capability(
        capability_id: str,
        name: str,
        description: str,
        aliases: list[str],
        dependencies: list[str],
        input_modes: list[str],
        output_modes: list[str],
        artifact: str,
    ) -> CapabilityCatalogEntry:
        return CapabilityCatalogEntry(
            capability_id=capability_id,
            name=name,
            description=description,
            aliases=aliases,
            depends_on_capability_ids=dependencies,
            input_schema={},
            output_schema={},
            input_modes=input_modes,
            output_modes=output_modes,
            side_effect_class="read_only",
            auxiliary_eligible=False,
            validation_contract=ValidationContract(
                json_schema={"type": "object"},
                required_artifacts=[artifact],
            ),
        )

    return SemanticCatalog(
        capabilities=[
            capability(
                "prepare",
                "Compile source bundle",
                "Assemble and prepare source material.",
                ["compile", "assemble", "prepare"],
                [],
                ["text"],
                ["source_bundle"],
                "source-bundle",
            ),
            capability(
                "verify",
                "Verify source bundle",
                "Check and validate the prepared material.",
                ["verify", "check", "validate"],
                ["prepare"],
                ["source_bundle"],
                ["checked_bundle"],
                "checked-bundle",
            ),
            capability(
                "deliver",
                "Release verified summary",
                "Issue, publish, deliver, or release the checked summary.",
                ["release", "issue", "publish", "deliver"],
                ["verify"],
                ["checked_bundle"],
                ["json", "signed_pdf"],
                "result-json",
            ),
            capability(
                "archive",
                "Archive verified summary",
                "Preserve the checked summary in a long-term archive.",
                ["archive", "preserve", "long-term"],
                ["verify"],
                ["checked_bundle"],
                ["json"],
                "archive-json",
            ),
        ],
        trust_policies=[
            TrustPolicyOption(
                policy_id="ordinary",
                name="ordinary assurance",
                description="Standard admitted trust is sufficient.",
                aliases=["ordinary", "standard"],
                required_trust_level="standard",
            ),
            TrustPolicyOption(
                policy_id="elevated",
                name="elevated assurance",
                description="Every selected provider must have elevated trust.",
                aliases=["elevated", "high assurance"],
                required_trust_level="elevated",
            ),
        ],
        artifact_contracts=[
            ArtifactContractOption(
                contract_id="json",
                name="machine-readable JSON",
                description="Return the final result as JSON.",
                aliases=["json", "machine-readable"],
                output_modes=["json"],
                required_artifacts=["result-json"],
                json_schema={"type": "object"},
            ),
            ArtifactContractOption(
                contract_id="signed-pdf",
                name="signed PDF",
                description="Return the final result as a signed PDF.",
                aliases=["signed pdf", "pdf"],
                output_modes=["signed_pdf"],
                required_artifacts=["signed-pdf"],
                json_schema={"type": "object"},
            ),
        ],
        default_trust_policy_id="ordinary",
        default_artifact_contract_id="json",
    )


def _skill(
    catalog: SemanticCatalog,
    capability_id: str,
    output_modes: list[str] | None = None,
) -> CapabilityRequirement:
    capability = next(
        item for item in catalog.capabilities if item.capability_id == capability_id
    )
    return CapabilityRequirement(
        name=capability.name,
        requirement_id=capability.capability_id,
        capability_id=capability.capability_id,
        input_modes=list(capability.input_modes),
        output_modes=output_modes or list(capability.output_modes),
        validation_contract=capability.validation_contract.model_copy(deep=True),
    )


def strategy_registry(variant: str, catalog: SemanticCatalog) -> list[AgentRegistryEntry]:
    elevated_upstream = variant == "trust-alternative"
    registry = [
        AgentRegistryEntry(
            agent_id="source-preparer",
            name="Source preparer",
            service_endpoint="https://source-preparer.example/a2a",
            trust_level="elevated" if elevated_upstream else "standard",
            skills=[_skill(catalog, "prepare")],
        ),
        AgentRegistryEntry(
            agent_id="source-verifier",
            name="Source verifier",
            service_endpoint="https://source-verifier.example/a2a",
            trust_level="elevated" if elevated_upstream else "standard",
            skills=[_skill(catalog, "verify")],
        ),
        AgentRegistryEntry(
            agent_id="primary-delivery",
            name="Primary delivery",
            service_endpoint="https://primary-delivery.example/a2a",
            trust_level="standard",
            skills=[_skill(catalog, "deliver", ["json"])],
        ),
        AgentRegistryEntry(
            agent_id="backup-delivery",
            name="Backup delivery",
            service_endpoint="https://backup-delivery.example/a2a",
            trust_level=(
                "elevated" if variant == "trust-alternative" else "standard"
            ),
            skills=[
                _skill(
                    catalog,
                    "deliver",
                    ["signed_pdf"]
                    if variant == "artifact-alternative"
                    else ["json"],
                )
            ],
        ),
    ]
    if variant == "archive-available":
        registry.append(AgentRegistryEntry(
            agent_id="archive-agent",
            name="Archive agent",
            service_endpoint="https://archive-agent.example/a2a",
            skills=[_skill(catalog, "archive")],
        ))
    return registry


def _public_cases(case_set: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    public = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    cases = list(public["cases"])
    if case_set == "sentinel":
        cases = [item for item in cases if item["case_id"] in SENTINEL_CASES]
    return public, cases


async def _collect_case(
    case: dict[str, Any],
    model: str,
    interpreter: OpenAICompatibleSemanticInterpreter,
) -> dict[str, Any]:
    catalog = strategy_catalog()
    registry = strategy_registry(case["variant"], catalog)
    interpretation = await interpreter.interpret(
        case["request_text"], catalog, registry
    )
    admission = None
    compilation = None
    if interpretation.intent is not None:
        admission = SemanticRequestAdmitter().admit(
            case["request_text"], catalog, interpretation.intent, registry
        )
        if admission.request is not None:
            compilation = SymbolicPlanCompiler(
                FeasibilityAnalyzer()
            ).compile(admission.request, registry)
    schema = semantic_intent_schema(catalog, registry)
    messages = semantic_prompt(case["request_text"], catalog, registry)
    return {
        "identity": {
            "case_id": case["case_id"],
            "model_id": model,
            "seed": 11,
        },
        "temperature": 0,
        "public_input": {
            "case": case,
            "catalog": catalog.model_dump(mode="json"),
            "registry": [item.model_dump(mode="json") for item in registry],
        },
        "public_input_hash": _canonical_hash({
            "case": case,
            "catalog": catalog.model_dump(mode="json"),
            "registry": [item.model_dump(mode="json") for item in registry],
        }),
        "prompt_hash": _canonical_hash(messages),
        "schema_hash": _canonical_hash(schema),
        "interpretation": interpretation.model_dump(mode="json"),
        "admission": admission.model_dump(mode="json") if admission else None,
        "compilation": compilation.model_dump(mode="json") if compilation else None,
        "accepted": bool(compilation and compilation.report.feasible),
        "expected_data_loaded_during_collection": False,
    }


async def collect(
    model: str,
    case_set: str,
    output_dir: Path,
    run: Path | None,
) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unsupported model {model!r}.")
    public, cases = _public_cases(case_set)
    public_hash = _canonical_hash(public)
    if run is None:
        run = output_dir / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + public_hash[:10]
        )
        run.mkdir(parents=True, exist_ok=False)
        provenance = {
            "fixture": public["fixture"],
            "public_fixture_hash": public_hash,
            "source_hash": _hash_bytes(Path(__file__).read_bytes()),
            "expected_data_loaded_during_collection": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (run / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        if provenance["public_fixture_hash"] != public_hash:
            raise RuntimeError("The validation fixture changed after collection began.")
    model_dir = run / model.replace("/", "__")
    model_dir.mkdir(parents=True, exist_ok=True)
    interpreter = OpenAICompatibleSemanticInterpreter(model)
    for case in cases:
        path = model_dir / f"{case['case_id']}.json"
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite preserved output {path}.")
        record = await _collect_case(case, model, interpreter)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return run


def collect_typed_corpus_replay(
    corpus_root: Path,
    output_dir: Path,
) -> Path:
    """Replay frozen typed public inputs without loading their hidden labels."""

    manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    public_path = corpus_root / "public/cases.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if public["corpus_hash"] != manifest["corpus_hash"]:
        raise RuntimeError("The public corpus hash differs from its frozen manifest.")
    public_hash = _canonical_hash(public)
    run = output_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-typed-corpus-"
        + public_hash[:10]
    )
    run.mkdir(parents=True, exist_ok=False)
    provenance = {
        "classification": "developmental_replay_of_frozen_typed_public_inputs",
        "source_corpus_version": public["version"],
        "source_corpus_hash": public["corpus_hash"],
        "public_input_hash": public_hash,
        "source_hash": _hash_bytes(Path(__file__).read_bytes()),
        "compiler_source_hash": _hash_bytes(
            Path(SymbolicPlanCompiler.__module__.replace(".", "/") + ".py").read_bytes()
            if Path(SymbolicPlanCompiler.__module__.replace(".", "/") + ".py").exists()
            else Path("src/unified_multi_agent_coordination/symbolic_plan_compiler.py").read_bytes()
        ),
        "expected_data_loaded_during_collection": False,
        "typed_request_llm_bypass": True,
        "model_calls": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    output_root = run / "typed"
    output_root.mkdir()
    for case in public["cases"]:
        request = ProblemRequest.model_validate(case["request"])
        registry = [
            AgentRegistryEntry.model_validate(item)
            for item in case["registry_snapshot"]
        ]
        started = time.perf_counter()
        compilation = SymbolicPlanCompiler(FeasibilityAnalyzer()).compile(
            request, registry
        )
        latency_ms = (time.perf_counter() - started) * 1000
        record = {
            "identity": {"case_id": case["case_id"]},
            "source_corpus_hash": public["corpus_hash"],
            "public_input_hash": _canonical_hash(case),
            "typed_request_llm_bypass": True,
            "semantic_call_count": 0,
            "compilation_latency_ms": latency_ms,
            "compilation": compilation.model_dump(mode="json"),
            "accepted": compilation.report.feasible,
            "expected_data_loaded_during_collection": False,
        }
        (output_root / f"{case['case_id']}.json").write_text(
            json.dumps(record, indent=2) + "\n",
            encoding="utf-8",
        )
    return run


def analyze_typed_corpus_replay(
    run: Path,
    labels_path: Path,
    accepted_analysis_path: Path,
) -> dict[str, Any]:
    """Score a completed replay only after every public-input decision exists."""

    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run / "typed").glob("*.json"))
    ]
    if len(records) != 48:
        raise RuntimeError(
            f"Scoring requires all 48 frozen cases; found {len(records)}."
        )
    if any(
        record["source_corpus_hash"] != provenance["source_corpus_hash"]
        for record in records
    ):
        raise RuntimeError("A replay record has the wrong frozen corpus hash.")

    labels_doc = json.loads(labels_path.read_text(encoding="utf-8"))
    if labels_doc["corpus_hash"] != provenance["source_corpus_hash"]:
        raise RuntimeError("The hidden labels do not match the replayed corpus.")
    labels = {item["case_id"]: item for item in labels_doc["labels"]}
    rows = []
    for record in records:
        case_id = record["identity"]["case_id"]
        expected = bool(labels[case_id]["feasible"])
        accepted = bool(record["accepted"])
        rows.append({
            "case_id": case_id,
            "invalidity_category": labels[case_id]["invalidity_category"],
            "accepted": accepted,
            "reference_feasible": expected,
            "correct": accepted == expected,
            "false_acceptance": accepted and not expected,
            "false_refusal": not accepted and expected,
            "compilation_latency_ms": record["compilation_latency_ms"],
        })
    feasible = [row for row in rows if row["reference_feasible"]]
    accepted_historical = json.loads(
        accepted_analysis_path.read_text(encoding="utf-8")
    )["metrics_repeated_observations"]["hybrid_repaired"]
    by_category = {}
    for category in sorted({row["invalidity_category"] for row in rows}):
        items = [row for row in rows if row["invalidity_category"] == category]
        by_category[category] = {
            "cases": len(items),
            "correct": sum(row["correct"] for row in items),
            "false_acceptances": sum(row["false_acceptance"] for row in items),
            "false_refusals": sum(row["false_refusal"] for row in items),
        }
    result = {
        "classification": "developmental_typed_request_architecture_replay",
        "claim_boundary": (
            "The same frozen author-designed corpus is reused descriptively. "
            "This is not a new study, independent adjudication, or evidence of "
            "general model or pipeline superiority."
        ),
        "source_corpus_hash": provenance["source_corpus_hash"],
        "typed_request_llm_bypass": True,
        "model_calls": 0,
        "metrics": {
            "cases": len(rows),
            "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "feasible_cases": len(feasible),
            "feasible_recall": (
                sum(row["accepted"] for row in feasible) / len(feasible)
            ),
            "false_acceptances": sum(row["false_acceptance"] for row in rows),
            "false_refusals": sum(row["false_refusal"] for row in rows),
            "compilation_latency_ms": sum(
                row["compilation_latency_ms"] for row in rows
            ),
        },
        "accepted_historical_hybrid_repaired": {
            "observations": accepted_historical["n"],
            "feasible_recall": accepted_historical["recall"],
            "false_acceptances": accepted_historical["false_acceptance"],
            "false_refusals": accepted_historical["false_refusal"],
        },
        "by_invalidity_category": by_category,
        "rows": rows,
    }
    (run / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _canonical_intent(
    value: dict[str, Any] | None,
    catalog: SemanticCatalog,
) -> dict[str, Any] | None:
    if value is None:
        return None
    intent = SemanticIntentOutput.model_validate(value)
    capability_by_id = {
        item.capability_id: item for item in catalog.capabilities
    }
    terminal = SemanticRequestAdmitter._terminal_goals(
        intent.goals, capability_by_id
    )
    goals = sorted(
        (
            item.capability_id,
            item.trust_policy_id or catalog.default_trust_policy_id,
            item.artifact_contract_id or catalog.default_artifact_contract_id,
        )
        for item in terminal
    )
    return {
        "interpretation_status": intent.interpretation_status,
        "goals": goals,
        "forbidden_capability_ids": sorted(intent.forbidden_capability_ids),
        "forbidden_agent_ids": sorted(intent.forbidden_agent_ids),
        "has_unresolved_terms": bool(intent.unresolved_terms),
    }


def analyze(run: Path) -> dict[str, Any]:
    public = json.loads(PUBLIC_PATH.read_text(encoding="utf-8"))
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))["expectations"]
    catalog = strategy_catalog()
    rows = []
    for path in sorted(run.glob("*__*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        case_id = record["identity"]["case_id"]
        expectation = expected[case_id]
        interpretation = record["interpretation"]
        observed_intent = (
            record["admission"]["canonical_intent"]
            if record["admission"] is not None
            else None
        )
        semantic_match = _canonical_intent(
            observed_intent, catalog
        ) == _canonical_intent(expectation["intent"], catalog)
        accepted = bool(record["accepted"])
        reference = bool(expectation["feasible"])
        rows.append({
            "case_id": case_id,
            "model_id": record["identity"]["model_id"],
            "schema_valid": observed_intent is not None,
            "repair_attempted": interpretation["repair_attempted"],
            "semantic_match": semantic_match,
            "accepted": accepted,
            "reference_feasible": reference,
            "correct": accepted == reference,
            "false_acceptance": accepted and not reference,
            "false_refusal": not accepted and reference,
            "provider_recovery": bool(
                record["compilation"]
                and record["compilation"]["diagnostics"][
                    "recovered_alternative_provider"
                ]
            ),
            "call_count": interpretation["call_count"],
            "latency_ms": interpretation["latency_ms"],
            "prompt_tokens": interpretation["prompt_tokens"],
            "completion_tokens": interpretation["completion_tokens"],
        })
    by_model: dict[str, Any] = {}
    for model in sorted({row["model_id"] for row in rows}):
        items = [row for row in rows if row["model_id"] == model]
        feasible = [row for row in items if row["reference_feasible"]]
        by_model[model] = {
            "observations": len(items),
            "schema_valid": sum(row["schema_valid"] for row in items),
            "repairs": sum(row["repair_attempted"] for row in items),
            "semantic_matches": sum(row["semantic_match"] for row in items),
            "correct": sum(row["correct"] for row in items),
            "feasible_recall": (
                sum(row["accepted"] for row in feasible) / len(feasible)
                if feasible
                else None
            ),
            "false_acceptances": sum(row["false_acceptance"] for row in items),
            "false_refusals": sum(row["false_refusal"] for row in items),
            "provider_recoveries": sum(row["provider_recovery"] for row in items),
            "calls": sum(row["call_count"] for row in items),
            "latency_ms": sum(row["latency_ms"] for row in items),
            "prompt_tokens": sum(row["prompt_tokens"] for row in items),
            "completion_tokens": sum(row["completion_tokens"] for row in items),
        }
    result = {
        "fixture": public["fixture"],
        "classification": "small_author_designed_developmental_validation",
        "claim_boundary": (
            "Descriptive engineering validation only; no model or pipeline "
            "superiority claim."
        ),
        "rows": rows,
        "by_model": by_model,
    }
    (run / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--model", required=True, choices=MODELS)
    collect_parser.add_argument(
        "--case-set", choices=("all", "sentinel"), default="all"
    )
    collect_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_runs/hybrid_strategy_validation"),
    )
    collect_parser.add_argument("--run", type=Path)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--run", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay-typed-corpus")
    replay_parser.add_argument(
        "--corpus-root", type=Path, default=FROZEN_CORPUS_ROOT
    )
    replay_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_runs/hybrid_strategy_validation"),
    )
    replay_analysis_parser = subparsers.add_parser("analyze-typed-corpus")
    replay_analysis_parser.add_argument("--run", type=Path, required=True)
    replay_analysis_parser.add_argument(
        "--labels", type=Path, default=FROZEN_LABELS_PATH
    )
    replay_analysis_parser.add_argument(
        "--accepted-analysis", type=Path, default=ACCEPTED_ANALYSIS_PATH
    )
    args = parser.parse_args()
    if args.command == "collect":
        run = asyncio.run(
            collect(args.model, args.case_set, args.output_dir, args.run)
        )
        print(run)
    elif args.command == "analyze":
        result = analyze(args.run)
        print(json.dumps(result["by_model"], indent=2))
    elif args.command == "replay-typed-corpus":
        print(collect_typed_corpus_replay(args.corpus_root, args.output_dir))
    else:
        result = analyze_typed_corpus_replay(
            args.run, args.labels, args.accepted_analysis
        )
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
