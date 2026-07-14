"""Post-collection scoring for the v0.4 constrained bridge study.

Schema v0.4.1 keeps deterministic baselines at the 36-case unit, treats model
and seed outputs as repeated observations, attributes latency to the work each
configuration actually performs, and never rewrites a completed analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

from .defense_study import MODELS, SEEDS, validate_frozen_labels
from .defense_study_v04 import LinguisticBridgeOutput
from .feasibility import FeasibilityAnalyzer
from .models import AgentRegistryEntry, ProblemRequest, SolutionProposal, ValidationContract
from .plan_hydration import PlanHydrator
from .study_analysis import _collapse_by_case, _holm_adjust, _mcnemar, _metrics

ANALYSIS_SCHEMA_VERSION = "0.4.1"
REPEATED_CONFIGURATIONS = (
    "hybrid_bridge",
    "llm_only_no_symbolic_gate",
    "ablation_typed_constraints",
    "ablation_authority",
    "ablation_contract_validation",
)
DETERMINISTIC_CONFIGURATIONS = (
    "natural_language_rule_only",
    "structured_oracle_upper_bound",
)
CONFIGURATIONS = (*REPEATED_CONFIGURATIONS, *DETERMINISTIC_CONFIGURATIONS)


def _decision(
    configuration: str, case: dict[str, Any], linguistic: LinguisticBridgeOutput
) -> tuple[bool, dict[str, Any], float]:
    request = ProblemRequest.model_validate(case["request"])
    registry = [AgentRegistryEntry.model_validate(item) for item in case["registry_snapshot"]]
    if configuration == "natural_language_rule_only":
        started = time.perf_counter()
        words = set(case["request_text"].lower().replace("-", " ").split())
        skill_words = {
            word for agent in registry for skill in agent.skills for word in skill.name.lower().split()
        }
        accepted = bool(words & skill_words) and bool(registry)
        return accepted, {"matcher_overlap": sorted(words & skill_words)}, _elapsed_ms(started)
    if configuration == "structured_oracle_upper_bound":
        proposal = SolutionProposal.model_validate(case["shared_raw_proposal"])
        started = time.perf_counter()
        report = FeasibilityAnalyzer().check(request, registry, proposal)
        return report.feasible, {"oracle": True}, _elapsed_ms(started)
    if configuration == "llm_only_no_symbolic_gate":
        return bool(linguistic.draft.selections), {
            "selection_count": len(linguistic.draft.selections)
        }, 0.0
    if configuration == "ablation_typed_constraints":
        request = request.model_copy(update={"constraints": []}, deep=True)
        request.requirements = [
            item.model_copy(update={"constraints": []}) for item in request.requirements
        ]
    elif configuration == "ablation_authority":
        request.requirements = [
            item.model_copy(update={"required_trust_level": "standard"})
            for item in request.requirements
        ]
    elif configuration == "ablation_contract_validation":
        request.requirements = [
            item.model_copy(
                update={"validation_contract": ValidationContract(json_schema={"type": "object"})}
            )
            for item in request.requirements
        ]
    started = time.perf_counter()
    hydration = PlanHydrator().hydrate(request, registry, linguistic.draft)
    if not hydration.complete or hydration.proposal is None:
        return False, {
            "hydration_issues": [item.model_dump(mode="json") for item in hydration.issues]
        }, _elapsed_ms(started)
    report = FeasibilityAnalyzer().check(request, registry, hydration.proposal)
    return report.feasible, {
        "hydration_issues": [],
        "predicates": [item.model_dump(mode="json") for item in report.evidence],
        "risks": report.risks,
    }, _elapsed_ms(started)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _initial_linguistic(record: dict[str, Any]) -> LinguisticBridgeOutput:
    raw = record["raw_response"]
    content = raw["choices"][0]["message"]["content"]
    return LinguisticBridgeOutput.model_validate_json(content)


def _row(
    *,
    case_id: str,
    configuration: str,
    accepted: bool,
    reference_feasible: bool,
    evidence: dict[str, Any],
    symbolic_ms: float,
    linguistic_ms: float,
    repair_ms: float,
    model_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "model_id": model_id,
        "seed": seed,
        "configuration": configuration,
        "accepted": accepted,
        "reference_feasible": reference_feasible,
        "correct": accepted == reference_feasible,
        "latency_ms": linguistic_ms + repair_ms + symbolic_ms,
        "linguistic_latency_ms": linguistic_ms,
        "repair_latency_ms": repair_ms,
        "symbolic_validation_latency_ms": symbolic_ms,
        "execution_latency_ms": None,
        "evidence": evidence,
    }


def _cross_seed_stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        content = record["raw_response"]["choices"][0]["message"]["content"]
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        grouped[(record["model_id"], record["case_id"])].add(digest)
    result: dict[str, Any] = {}
    for model in MODELS:
        counts = [len(grouped[(model, case_id)]) for _, case_id in grouped if _ == model]
        result[model] = {
            "cases": len(counts),
            "requested_seeds_per_case": len(SEEDS),
            "mean_unique_initial_outputs_per_case": sum(counts) / len(counts),
            "unique_output_ratio": sum(counts) / (len(counts) * len(SEEDS)),
            "seed_honored_by_backend": "not_verifiable_from_response",
        }
    return result


def analyze(run_root: Path, corpus_root: Path) -> dict[str, Any]:
    provenance = validate_frozen_labels(corpus_root)
    completion_path = run_root / "collection-complete.json"
    if not completion_path.exists() or not json.loads(completion_path.read_text())["complete"]:
        raise RuntimeError("Scoring is blocked until collection is complete.")
    public = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    hidden = json.loads(
        (corpus_root / "hidden/reference-labels.json").read_text(encoding="utf-8")
    )
    cases = {case["case_id"]: case for case in public["cases"]}
    labels = {label["case_id"]: label for label in hidden["labels"]}
    paths = [path for path in run_root.glob("*/seed-*/*.json") if path.name != "sentinel.json"]
    expected = set(product(MODELS, SEEDS, cases))
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    raw.sort(key=lambda item: (item["model_id"], item["seed"], item["case_id"]))
    identities = {(item["model_id"], item["seed"], item["case_id"]) for item in raw}
    if len(raw) != len(expected) or identities != expected:
        raise RuntimeError(f"Expected the exact {len(expected)}-output matrix; found {len(raw)} files.")

    calibrated_timings = run_root / "symbolic-validation-timings-v0.4.1.json"
    timings_path = (
        calibrated_timings
        if calibrated_timings.exists()
        else run_root / "symbolic-validation-timings.json"
    )
    timings = json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else {}
    new_timings: dict[str, float] = {}
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Deterministic controls are one observation per case, not repeated for every
    # model and requested seed.
    representative = LinguisticBridgeOutput.model_validate(raw[0]["parsed_object"])
    for case_id, case in sorted(cases.items()):
        reference = labels[case_id]
        for configuration in DETERMINISTIC_CONFIGURATIONS:
            accepted, evidence, measured_ms = _decision(configuration, case, representative)
            key = f"deterministic|{case_id}|{configuration}"
            symbolic_ms = float(timings.get(key, measured_ms))
            new_timings[key] = symbolic_ms
            rows[configuration].append(
                _row(
                    case_id=case_id,
                    configuration=configuration,
                    accepted=accepted,
                    reference_feasible=reference["feasible"],
                    evidence=evidence,
                    symbolic_ms=symbolic_ms,
                    linguistic_ms=0.0,
                    repair_ms=0.0,
                )
            )

    for item in raw:
        case = cases[item["case_id"]]
        reference = labels[item["case_id"]]
        final_linguistic = LinguisticBridgeOutput.model_validate(item["parsed_object"])
        initial_linguistic = _initial_linguistic(item)
        for configuration in REPEATED_CONFIGURATIONS:
            linguistic = (
                initial_linguistic
                if configuration == "llm_only_no_symbolic_gate"
                else final_linguistic
            )
            accepted, evidence, measured_ms = _decision(configuration, case, linguistic)
            timing_key = "|".join(
                (item["model_id"], str(item["seed"]), item["case_id"], configuration)
            )
            symbolic_ms = float(timings.get(timing_key, measured_ms))
            new_timings[timing_key] = symbolic_ms
            is_llm_only = configuration == "llm_only_no_symbolic_gate"
            rows[configuration].append(
                _row(
                    case_id=item["case_id"],
                    model_id=item["model_id"],
                    seed=item["seed"],
                    configuration=configuration,
                    accepted=accepted,
                    reference_feasible=reference["feasible"],
                    evidence=evidence,
                    symbolic_ms=0.0 if is_llm_only else symbolic_ms,
                    linguistic_ms=float(item["linguistic_latency_ms"]),
                    repair_ms=0.0 if is_llm_only else float(item.get("repair_latency_ms", 0.0)),
                )
            )

    metrics = {name: _metrics(items) for name, items in rows.items()}
    hybrid = rows["hybrid_bridge"]
    hybrid_cases = _collapse_by_case(hybrid)
    comparisons = {
        name: {
            **_mcnemar(hybrid_cases, _collapse_by_case(items)),
            "accuracy_difference_vs_hybrid": metrics[name]["accuracy"]
            - metrics["hybrid_bridge"]["accuracy"],
        }
        for name, items in rows.items()
        if name != "hybrid_bridge"
    }
    _holm_adjust(comparisons)
    repeated_metrics_by_model_seed = {
        name: {
            f"{model}|{seed}": _metrics(
                [row for row in items if row["model_id"] == model and row["seed"] == seed]
            )
            for model in MODELS
            for seed in SEEDS
        }
        for name, items in rows.items()
        if name in REPEATED_CONFIGURATIONS
    }
    paired_by_model_seed: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for seed in SEEDS:
            left = [
                row for row in hybrid if row["model_id"] == model and row["seed"] == seed
            ]
            for name, items in rows.items():
                if name == "hybrid_bridge" or name in DETERMINISTIC_CONFIGURATIONS:
                    continue
                right = [
                    row for row in items if row["model_id"] == model and row["seed"] == seed
                ]
                paired_by_model_seed[f"{model}|{seed}|{name}"] = _mcnemar(left, right)
    _holm_adjust(paired_by_model_seed)

    by_model = {
        model: _metrics([row for row in hybrid if row["model_id"] == model])
        for model in MODELS
    }
    feasible_cases = {case_id for case_id, label in labels.items() if label["feasible"]}
    case_majority = sum(
        sum(row["accepted"] for row in hybrid if row["case_id"] == case_id)
        >= (len(MODELS) * len(SEEDS) // 2 + 1)
        for case_id in feasible_cases
    )
    acceptance = {
        "zero_hybrid_false_accepts": metrics["hybrid_bridge"]["false_acceptance"] == 0,
        "feasible_case_majority_at_least_12": case_majority >= 12,
        "every_model_recall_at_least_0_60": all(
            item["recall"] >= 0.60 for item in by_model.values()
        ),
    }
    return {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_version": "0.4",
        "annotation_type": provenance["annotation_type"],
        "independent_adjudication": False,
        "primary_statistical_unit": "case",
        "observation_counts": {name: len(items) for name, items in rows.items()},
        "metrics": metrics,
        "hybrid_by_model": by_model,
        "metrics_by_model_seed": repeated_metrics_by_model_seed,
        "cross_seed_stability": _cross_seed_stability(raw),
        "feasible_cases_with_majority_acceptance": case_majority,
        "paired_case_level_comparisons": comparisons,
        "paired_comparisons_by_model_seed": paired_by_model_seed,
        "preregistered_acceptance": acceptance,
        "all_preregistered_criteria_met": all(acceptance.values()),
        "timings": new_timings,
        "rows": rows,
        "limitations": [
            "The corpus and reference labels were authored by the thesis author.",
            "Requested seeds are stability repetitions; backend seed compliance is not verifiable.",
            "The lexical rule-only control is a deliberately naive diagnostic baseline.",
            "Repeated model outputs do not increase the 36-case primary sample size.",
        ],
    }


def write_analysis(result: dict[str, Any], output: Path, timing_output: Path | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if timing_output is not None:
        timing_output.parent.mkdir(parents=True, exist_ok=True)
        with timing_output.open("x", encoding="utf-8") as handle:
            json.dump(result["timings"], handle, indent=2, sort_keys=True)
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.4"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / f"analysis-v{ANALYSIS_SCHEMA_VERSION}.json"
    result = analyze(args.run, args.corpus)
    write_analysis(result, output, args.timing_output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "timings"}}, indent=2))


if __name__ == "__main__":
    main()
