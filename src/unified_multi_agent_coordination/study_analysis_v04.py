"""Post-collection scoring for the v0.4 constrained bridge study."""

from __future__ import annotations

import argparse
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

CONFIGURATIONS = (
    "hybrid_bridge",
    "natural_language_rule_only",
    "structured_oracle_upper_bound",
    "llm_only_no_symbolic_gate",
    "ablation_typed_constraints",
    "ablation_authority",
    "ablation_contract_validation",
)


def _decision(
    configuration: str, case: dict[str, Any], linguistic: LinguisticBridgeOutput
) -> tuple[bool, dict[str, Any], float]:
    request = ProblemRequest.model_validate(case["request"])
    registry = [AgentRegistryEntry.model_validate(item) for item in case["registry_snapshot"]]
    if configuration == "natural_language_rule_only":
        words = set(case["request_text"].lower().split())
        skill_words = {
            word for agent in registry for skill in agent.skills for word in skill.name.lower().split()
        }
        return bool(words & skill_words) and bool(registry), {}, 0.0
    if configuration == "structured_oracle_upper_bound":
        proposal = SolutionProposal.model_validate(case["shared_raw_proposal"])
        started = time.perf_counter()
        report = FeasibilityAnalyzer().check(request, registry, proposal)
        return report.feasible, {"oracle": True}, (time.perf_counter() - started) * 1000
    if configuration == "llm_only_no_symbolic_gate":
        return bool(linguistic.draft.selections), {"selection_count": len(linguistic.draft.selections)}, 0.0
    if configuration == "ablation_typed_constraints":
        request = request.model_copy(update={"constraints": []}, deep=True)
        request.requirements = [item.model_copy(update={"constraints": []}) for item in request.requirements]
    elif configuration == "ablation_authority":
        request.requirements = [
            item.model_copy(update={"required_trust_level": "standard"})
            for item in request.requirements
        ]
    elif configuration == "ablation_contract_validation":
        request.requirements = [item.model_copy(update={
            "validation_contract": ValidationContract(json_schema={"type": "object"})
        }) for item in request.requirements]
    started = time.perf_counter()
    hydration = PlanHydrator().hydrate(request, registry, linguistic.draft)
    if not hydration.complete or hydration.proposal is None:
        return False, {"hydration_issues": [i.model_dump(mode="json") for i in hydration.issues]}, (time.perf_counter() - started) * 1000
    report = FeasibilityAnalyzer().check(request, registry, hydration.proposal)
    return report.feasible, {
        "hydration_issues": [], "predicates": [p.model_dump(mode="json") for p in report.evidence],
        "risks": report.risks,
    }, (time.perf_counter() - started) * 1000


def analyze(run_root: Path, corpus_root: Path) -> dict[str, Any]:
    provenance = validate_frozen_labels(corpus_root)
    complete = run_root / "collection-complete.json"
    if not complete.exists() or not json.loads(complete.read_text())["complete"]:
        raise RuntimeError("Scoring is blocked until collection is complete.")
    public = json.loads((corpus_root / "public/cases.json").read_text())
    hidden = json.loads((corpus_root / "hidden/reference-labels.json").read_text())
    cases = {case["case_id"]: case for case in public["cases"]}
    labels = {label["case_id"]: label for label in hidden["labels"]}
    files = [p for p in run_root.glob("*/seed-*/*.json") if p.name != "sentinel.json"]
    expected = set(product(MODELS, SEEDS, cases))
    raw = [json.loads(path.read_text()) for path in files]
    raw.sort(key=lambda item: (item["model_id"], item["seed"], item["case_id"]))
    identities = {(item["model_id"], item["seed"], item["case_id"]) for item in raw}
    if len(raw) != 540 or identities != expected:
        raise RuntimeError(f"Expected the exact 540-output matrix; found {len(raw)} files.")
    timings_path = run_root / "symbolic-validation-timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw:
        case = cases[item["case_id"]]
        reference = labels[item["case_id"]]
        linguistic = LinguisticBridgeOutput.model_validate(item["parsed_object"])
        for configuration in CONFIGURATIONS:
            accepted, evidence, symbolic_ms = _decision(configuration, case, linguistic)
            timing_key = "|".join((item["model_id"], str(item["seed"]), item["case_id"], configuration))
            if timing_key in timings:
                symbolic_ms = timings[timing_key]
            else:
                timings[timing_key] = symbolic_ms
            rows[configuration].append({
                "case_id": item["case_id"], "model_id": item["model_id"], "seed": item["seed"],
                "configuration": configuration, "accepted": accepted,
                "reference_feasible": reference["feasible"],
                "correct": accepted == reference["feasible"],
                "latency_ms": item["linguistic_latency_ms"] + item.get("repair_latency_ms", 0) + symbolic_ms,
                "linguistic_latency_ms": item["linguistic_latency_ms"],
                "repair_latency_ms": item.get("repair_latency_ms", 0),
                "symbolic_validation_latency_ms": symbolic_ms,
                "execution_latency_ms": None,
                "evidence": evidence,
            })
    metrics = {name: _metrics(items) for name, items in rows.items()}
    hybrid = rows["hybrid_bridge"]
    comparisons = {
        name: _mcnemar(_collapse_by_case(hybrid), _collapse_by_case(items))
        for name, items in rows.items() if name != "hybrid_bridge"
    }
    _holm_adjust(comparisons)
    by_model = {
        model: _metrics([row for row in hybrid if row["model_id"] == model]) for model in MODELS
    }
    feasible_cases = {case_id for case_id, label in labels.items() if label["feasible"]}
    case_majority = sum(
        sum(row["accepted"] for row in hybrid if row["case_id"] == case_id) >= 8
        for case_id in feasible_cases
    )
    acceptance = {
        "zero_hybrid_false_accepts": metrics["hybrid_bridge"]["false_acceptance"] == 0,
        "feasible_case_majority_at_least_12": case_majority >= 12,
        "every_model_recall_at_least_0_60": all(item["recall"] >= 0.60 for item in by_model.values()),
    }
    result = {
        "study_version": "0.4", "annotation_type": provenance["annotation_type"],
        "independent_adjudication": False, "primary_statistical_unit": "case",
        "metrics": metrics, "hybrid_by_model": by_model,
        "feasible_cases_with_majority_acceptance": case_majority,
        "paired_case_level_comparisons": comparisons,
        "preregistered_acceptance": acceptance,
        "all_preregistered_criteria_met": all(acceptance.values()),
        "rows": rows,
    }
    if not timings_path.exists():
        timings_path.write_text(json.dumps(timings, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.4"))
    args = parser.parse_args()
    result = analyze(args.run, args.corpus)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
