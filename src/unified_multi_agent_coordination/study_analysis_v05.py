"""Deterministic post-collection analysis for the frozen v0.5 dual-arm study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .defense_study import MODELS, SEEDS, validate_frozen_labels
from .defense_study_v04 import LinguisticBridgeOutput
from .defense_study_v05 import ARMS, DirectCoordinatorOutput, EXPECTED_OUTPUTS, validate_collection
from .feasibility import FeasibilityAnalyzer
from .models import (
    AgentRegistryEntry,
    DraftRequirementSelection,
    LinguisticPlanDraft,
    ProblemRequest,
    SolutionProposal,
)
from .plan_hydration import PlanHydrator
from .study_analysis import _holm_adjust, _mcnemar, _metrics

ANALYSIS_SCHEMA_VERSION = "0.5.0"
VIEWS = ("hybrid_initial", "hybrid_repaired", "direct_initial", "direct_repaired")
BASELINES = ("greedy_symbolic", "lexical_diagnostic", "structured_oracle_upper_bound")


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _hybrid_decision(
    case: dict[str, Any], output: LinguisticBridgeOutput
) -> tuple[bool, dict[str, Any], dict[str, float]]:
    request = ProblemRequest.model_validate(case["request"])
    registry = [AgentRegistryEntry.model_validate(item) for item in case["registry_snapshot"]]
    hydration_started = time.perf_counter()
    hydration = PlanHydrator().hydrate(request, registry, output.draft)
    hydration_ms = _elapsed(hydration_started)
    if not hydration.complete or hydration.proposal is None:
        return False, {
            "hydration_issues": [item.model_dump(mode="json") for item in hydration.issues]
        }, {"hydration_ms": hydration_ms, "symbolic_validation_ms": 0.0}
    symbolic_started = time.perf_counter()
    report = FeasibilityAnalyzer().check(request, registry, hydration.proposal)
    symbolic_ms = _elapsed(symbolic_started)
    return report.feasible, {
        "hydration_issues": [],
        "predicates": [item.model_dump(mode="json") for item in report.evidence],
        "risks": report.risks,
    }, {"hydration_ms": hydration_ms, "symbolic_validation_ms": symbolic_ms}


def _direct_decision(
    case: dict[str, Any], output: DirectCoordinatorOutput, *, fixed_decision: str | None = None
) -> tuple[bool, dict[str, Any], dict[str, float]]:
    request = ProblemRequest.model_validate(case["request"])
    registry = [AgentRegistryEntry.model_validate(item) for item in case["registry_snapshot"]]
    started = time.perf_counter()
    report = FeasibilityAnalyzer().check(request, registry, output.proposal)
    validation_ms = _elapsed(started)
    decision = fixed_decision or output.decision
    return decision == "accept", {
        "explicit_decision": decision,
        "reported_decision": output.decision,
        "post_hoc_proposal_feasible": report.feasible,
        "post_hoc_predicates": [item.model_dump(mode="json") for item in report.evidence],
    }, {"post_hoc_validation_ms": validation_ms}


def _greedy_draft(case: dict[str, Any]) -> LinguisticPlanDraft:
    request = ProblemRequest.model_validate(case["request"])
    return LinguisticPlanDraft(
        selections=[
            DraftRequirementSelection(
                requirement_id=requirement.requirement_id,
                capability_id=requirement.capability_id,
                depends_on_requirement_ids=list(requirement.depends_on_requirement_ids),
            )
            for requirement in request.requirements
        ],
        rationale="Stable admitted-order selection using public predicates only.",
    )


def _baseline_decision(
    name: str, case: dict[str, Any], label: dict[str, Any]
) -> tuple[bool, dict[str, Any], dict[str, float]]:
    if name == "greedy_symbolic":
        return _hybrid_decision(case, LinguisticBridgeOutput(draft=_greedy_draft(case)))
    if name == "lexical_diagnostic":
        request_words = set(case["request_text"].lower().replace("-", " ").split())
        skill_words = {
            word
            for agent in case["registry_snapshot"]
            for skill in agent["skills"]
            for word in skill["name"].lower().split()
        }
        overlap = sorted(request_words & skill_words)
        return bool(overlap), {"matcher_overlap": overlap}, {}
    request = ProblemRequest.model_validate(case["request"])
    registry = [AgentRegistryEntry.model_validate(item) for item in case["registry_snapshot"]]
    proposal = SolutionProposal.model_validate(label["oracle_proposal"])
    started = time.perf_counter()
    report = FeasibilityAnalyzer().check(request, registry, proposal)
    return report.feasible, {"oracle": True}, {"symbolic_validation_ms": _elapsed(started)}


def _row(
    *,
    case: dict[str, Any],
    label: dict[str, Any],
    view: str,
    accepted: bool,
    evidence: dict[str, Any],
    model: str | None = None,
    seed: int | None = None,
    repaired: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "domain": case["domain"],
        "invalidity_category": case["invalidity_category"],
        "model_id": model,
        "seed": seed,
        "view": view,
        "accepted": accepted,
        "reference_feasible": label["feasible"],
        "correct": accepted == label["feasible"],
        "repair_attempted": repaired,
        "latency_ms": 0.0,
        "evidence": evidence,
    }


def _majority_rows(rows: list[dict[str, Any]], *, group_model: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["model_id"] if group_model else None)].append(row)
    result = []
    for (case_id, model), items in sorted(grouped.items()):
        accepted = sum(int(item["accepted"]) for item in items) * 2 >= len(items)
        result.append(
            {
                "case_id": case_id,
                "model_id": model,
                "accepted": accepted,
                "reference_feasible": items[0]["reference_feasible"],
                "correct": accepted == items[0]["reference_feasible"],
                "latency_ms": 0.0,
            }
        )
    return result


def _clustered_difference_interval(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, samples: int = 5000
) -> list[float]:
    """Bootstrap paired accuracy differences by the pre-specified matched-pair cluster."""
    left_map = {(row["case_id"], row["model_id"], row["seed"]): row for row in left}
    right_map = {(row["case_id"], row["model_id"], row["seed"]): row for row in right}
    if left_map.keys() != right_map.keys():
        raise RuntimeError("Paired bootstrap inputs have different identities.")
    pairs: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key in sorted(left_map):
        pairs[left_map[key]["pair_id"]].append((left_map[key], right_map[key]))
    pair_ids = sorted(pairs)
    rng = random.Random(20260714)
    values: list[float] = []
    for _ in range(samples):
        selected = [pair_ids[rng.randrange(len(pair_ids))] for _ in pair_ids]
        observations = [observation for pair_id in selected for observation in pairs[pair_id]]
        values.append(
            sum(int(left_row["correct"]) - int(right_row["correct"]) for left_row, right_row in observations)
            / len(observations)
        )
    values.sort()
    return [values[int(samples * 0.025)], values[int(samples * 0.975) - 1]]


def _stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for record in records:
        identity = record["identity"]
        content = json.dumps(record["initial_parsed_object"], sort_keys=True, separators=(",", ":"))
        grouped[(identity["arm"], identity["model_id"], identity["case_id"])].add(
            hashlib.sha256(content.encode()).hexdigest()
        )
    result: dict[str, Any] = {}
    for arm in ARMS:
        for model in MODELS:
            counts = [len(value) for (a, m, _), value in grouped.items() if a == arm and m == model]
            result[f"{arm}|{model}"] = {
                "cases": len(counts),
                "requested_seeds_per_case": len(SEEDS),
                "mean_unique_initial_outputs_per_case": sum(counts) / len(counts),
                "requested_seeds_are_independent_samples": False,
            }
    return result


def _repair_summary(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    initial_by_view = {"hybrid_repaired": "hybrid_initial", "direct_repaired": "direct_initial"}
    result: dict[str, Any] = {}
    for final_view, initial_view in initial_by_view.items():
        initial = {
            (row["case_id"], row["model_id"], row["seed"]): row for row in rows[initial_view]
        }
        final = rows[final_view]
        attempted = [row for row in final if row["repair_attempted"]]
        successful = [
            row
            for row in attempted
            if not initial[(row["case_id"], row["model_id"], row["seed"])]["correct"]
            and row["correct"]
        ]
        dimensions: dict[str, Any] = {}
        for dimension in ("model_id", "domain", "invalidity_category"):
            dimensions[dimension] = {}
            for value in sorted({str(row[dimension]) for row in final}):
                selected = [row for row in final if str(row[dimension]) == value]
                selected_attempted = [row for row in selected if row["repair_attempted"]]
                selected_success = [row for row in successful if str(row[dimension]) == value]
                dimensions[dimension][value] = {
                    "n": len(selected),
                    "repair_rate": len(selected_attempted) / len(selected),
                    "repair_successes": len(selected_success),
                    "repair_success_rate_per_attempt": (
                        len(selected_success) / len(selected_attempted) if selected_attempted else 0.0
                    ),
                }
        result[final_view] = {
            "repair_rate": len(attempted) / len(final),
            "repair_successes": len(successful),
            "repair_success_rate_per_attempt": len(successful) / len(attempted) if attempted else 0.0,
            "by": dimensions,
        }
    return result


def analyze(run_root: Path, corpus_root: Path) -> dict[str, Any]:
    provenance = validate_frozen_labels(corpus_root)
    public = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    hidden = json.loads((corpus_root / "hidden/reference-labels.json").read_text(encoding="utf-8"))
    completion = json.loads((run_root / "collection-complete.json").read_text(encoding="utf-8"))
    if not completion.get("complete") or completion.get("expected_outputs") != EXPECTED_OUTPUTS:
        raise RuntimeError("Scoring is blocked until the exact v0.5 collection is complete.")
    cases = {item["case_id"]: item for item in public["cases"]}
    labels = {item["case_id"]: item for item in hidden["labels"]}
    validate_collection(run_root, list(cases.values()))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in run_root.glob("outputs/*/*/seed-*/*.json")
    ]
    records.sort(
        key=lambda item: (
            item["identity"]["arm"], item["identity"]["model_id"],
            item["identity"]["seed"], item["identity"]["case_id"],
        )
    )
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timings: list[dict[str, Any]] = []
    direct_decision_changes = 0
    for record in records:
        identity = record["identity"]
        case = cases[identity["case_id"]]
        label = labels[identity["case_id"]]
        if identity["arm"] == ARMS[0]:
            hybrid_initial = LinguisticBridgeOutput.model_validate(record["initial_parsed_object"])
            hybrid_repaired = LinguisticBridgeOutput.model_validate(record["repaired_parsed_object"])
            for view, hybrid_output in (
                ("hybrid_initial", hybrid_initial),
                ("hybrid_repaired", hybrid_repaired),
            ):
                accepted, evidence, measured = _hybrid_decision(case, hybrid_output)
                rows[view].append(
                    _row(
                        case=case, label=label, view=view, accepted=accepted, evidence=evidence,
                        model=identity["model_id"], seed=identity["seed"],
                        repaired=bool(record["repair_attempted"]),
                    )
                )
                repair_ms = (
                    float(record["repair_latency_ms"])
                    if view == "hybrid_repaired" and record["repair_attempted"]
                    else 0.0
                )
                initial_ms = float(record["initial_latency_ms"])
                timings.append(
                    {
                        "identity": identity,
                        "view": view,
                        "initial_call_ms": initial_ms,
                        "repair_call_ms": repair_ms,
                        **measured,
                        "total_ms": initial_ms + repair_ms + sum(measured.values()),
                    }
                )
        else:
            direct_initial = DirectCoordinatorOutput.model_validate(record["initial_parsed_object"])
            direct_repaired = DirectCoordinatorOutput.model_validate(record["repaired_parsed_object"])
            if direct_repaired.decision != direct_initial.decision:
                direct_decision_changes += 1
            for view, direct_output in (
                ("direct_initial", direct_initial),
                ("direct_repaired", direct_repaired),
            ):
                accepted, evidence, measured = _direct_decision(
                    case, direct_output, fixed_decision=direct_initial.decision
                )
                rows[view].append(
                    _row(
                        case=case, label=label, view=view, accepted=accepted, evidence=evidence,
                        model=identity["model_id"], seed=identity["seed"],
                        repaired=bool(record["repair_attempted"]),
                    )
                )
                repair_ms = (
                    float(record["repair_latency_ms"])
                    if view == "direct_repaired" and record["repair_attempted"]
                    else 0.0
                )
                initial_ms = float(record["initial_latency_ms"])
                timings.append(
                    {
                        "identity": identity,
                        "view": view,
                        "initial_call_ms": initial_ms,
                        "repair_call_ms": repair_ms,
                        **measured,
                        "total_ms": initial_ms + repair_ms + sum(measured.values()),
                    }
                )
    for case_id, case in sorted(cases.items()):
        for baseline in BASELINES:
            accepted, evidence, measured = _baseline_decision(baseline, case, labels[case_id])
            rows[baseline].append(
                _row(
                    case=case, label=labels[case_id], view=baseline,
                    accepted=accepted, evidence=evidence,
                )
            )
            timings.append({"case_id": case_id, "view": baseline, **measured})

    metrics = {name: _metrics(items) for name, items in rows.items()}
    for item in metrics.values():
        item.pop("latency_ms", None)
    majority_metrics = {
        name: _metrics(_majority_rows(items))
        for name, items in rows.items()
        if name in VIEWS
    }
    for item in majority_metrics.values():
        item.pop("latency_ms", None)
    by_model = {
        view: {
            model: _metrics([row for row in rows[view] if row["model_id"] == model])
            for model in MODELS
        }
        for view in VIEWS
    }
    for views in by_model.values():
        for item in views.values():
            item.pop("latency_ms", None)
    by_dimension: dict[str, Any] = {}
    for dimension in ("domain", "invalidity_category"):
        by_dimension[dimension] = {
            view: {
                value: _metrics([row for row in rows[view] if row[dimension] == value])
                for value in sorted({str(row[dimension]) for row in rows[view]})
            }
            for view in VIEWS
        }
        for views in by_dimension[dimension].values():
            for item in views.values():
                item.pop("latency_ms", None)

    comparisons: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        hybrid = _majority_rows(
            [row for row in rows["hybrid_repaired"] if row["model_id"] == model],
            group_model=True,
        )
        direct = _majority_rows(
            [row for row in rows["direct_repaired"] if row["model_id"] == model],
            group_model=True,
        )
        comparison = _mcnemar(hybrid, direct)
        comparison["case_majority_accuracy_difference_hybrid_minus_direct"] = (
            sum(int(row["correct"]) for row in hybrid) / len(hybrid)
            - sum(int(row["correct"]) for row in direct) / len(direct)
        )
        comparisons[model] = comparison
    _holm_adjust(comparisons)
    difference = metrics["hybrid_repaired"]["accuracy"] - metrics["direct_repaired"]["accuracy"]
    difference_ci = _clustered_difference_interval(
        rows["hybrid_repaired"], rows["direct_repaired"]
    )
    criteria = {
        "zero_measured_hybrid_false_acceptance": metrics["hybrid_repaired"]["false_acceptance"] == 0,
        "every_model_hybrid_feasible_recall_at_least_0_60": all(
            value["recall"] >= 0.60 for value in by_model["hybrid_repaired"].values()
        ),
        "positive_paired_accuracy_difference": difference > 0,
        "runtime_authorization_and_trace_compliance": "evaluated_in_separate_runtime_suite",
    }
    functional = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "study_version": "0.5",
        "corpus_hash": public["corpus_hash"],
        "annotation_type": provenance["annotation_type"],
        "independent_adjudication": False,
        "evidence_valid": True,
        "outcome": "passed" if all(value is True for value in list(criteria.values())[:3]) else "mixed",
        "claim_status": "supported" if all(value is True for value in list(criteria.values())[:3]) else "partially_supported",
        "primary_statistical_unit": "matched case pair",
        "requested_seeds_are_independent_samples": False,
        "observation_counts": {name: len(items) for name, items in rows.items()},
        "metrics_repeated_observations": metrics,
        "metrics_case_majority": majority_metrics,
        "metrics_by_model": by_model,
        "metrics_by_domain_and_invalidity": by_dimension,
        "repair_analysis": _repair_summary(rows),
        "initial_to_repaired_accuracy_change": {
            "hybrid": metrics["hybrid_repaired"]["accuracy"] - metrics["hybrid_initial"]["accuracy"],
            "direct": metrics["direct_repaired"]["accuracy"] - metrics["direct_initial"]["accuracy"],
        },
        "paired_comparisons_within_model": comparisons,
        "paired_accuracy_difference_hybrid_minus_direct": difference,
        "matched_pair_clustered_bootstrap_95_ci": difference_ci,
        "model_seed_stability": _stability(records),
        "direct_repair_decision_changes_ignored_for_scoring": direct_decision_changes,
        "pre_specified_criteria": criteria,
        "all_planning_criteria_met": all(value is True for value in list(criteria.values())[:3]),
        "rows": rows,
        "limitations": [
            "Reference labels were authored without independent adjudication.",
            "Requested seeds are repeated stability observations, not independent samples.",
            "The three local models and controlled cases limit external validity.",
            "Direct-arm proposals were scored post hoc and never dispatched.",
        ],
    }
    return {"functional": functional, "hardware_timings": timings}


def write_analysis(result: dict[str, Any], output: Path, timing_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    timing_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result["functional"], handle, indent=2, sort_keys=True)
        handle.write("\n")
    with timing_output.open("x", encoding="utf-8") as handle:
        json.dump(result["hardware_timings"], handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.5"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / f"analysis-v{ANALYSIS_SCHEMA_VERSION}.json"
    timing_output = args.timing_output or args.run / f"timings-v{ANALYSIS_SCHEMA_VERSION}.json"
    result = analyze(args.run, args.corpus)
    write_analysis(result, output, timing_output)
    printable = {key: value for key, value in result["functional"].items() if key != "rows"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
