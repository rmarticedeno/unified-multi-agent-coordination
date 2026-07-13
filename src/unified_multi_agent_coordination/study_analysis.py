"""Score frozen defense-study outputs and calculate paired uncertainty measures."""

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import product
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from .defense_study import MODELS, SEEDS, LinguisticBatchOutput, validate_frozen_labels
from .feasibility import FeasibilityAnalyzer
from .models import AgentRegistryEntry, ProblemRequest, SolutionProposal, ValidationContract


CONFIGURATIONS = (
    "hybrid",
    "natural_language_rule_only",
    "structured_oracle_upper_bound",
    "llm_only_no_symbolic_gate",
    "ablation_typed_constraints",
    "ablation_authority",
    "ablation_contract_validation",
)
RUNTIME_ONLY_ABLATIONS = ("dependency_gating", "bounded_auxiliary", "durable_trace")


def _decision(
    configuration: str,
    case: dict[str, Any],
    linguistic: LinguisticBatchOutput,
) -> tuple[bool, dict[str, Any]]:
    intended = ProblemRequest.model_validate(case["request"])
    registry = [AgentRegistryEntry.model_validate(item) for item in case["registry_snapshot"]]
    # Every configuration receives isolated objects. Ablations must not contaminate
    # later configurations through Pydantic's mutable nested model instances.
    proposal = linguistic.candidate_plans[0].model_copy(deep=True)
    request = linguistic.interpreted_request.model_copy(deep=True)
    if configuration == "structured_oracle_upper_bound":
        request = intended
        proposal = SolutionProposal.model_validate(case["shared_raw_proposal"])
    elif configuration == "natural_language_rule_only":
        words = set(case["request_text"].lower().replace("-", " ").split())
        skill_words = {
            word
            for agent in registry
            for skill in agent.skills
            for word in skill.name.lower().split()
        }
        accepted = bool(words & skill_words) and bool(registry)
        return accepted, {"matcher_overlap": sorted(words & skill_words)}
    elif configuration == "llm_only_no_symbolic_gate":
        return bool(proposal.tasks), {"task_count": len(proposal.tasks)}
    elif configuration == "ablation_typed_constraints":
        request = request.model_copy(update={"constraints": []})
        request.requirements = [item.model_copy(update={"constraints": []}) for item in request.requirements]
    elif configuration == "ablation_authority":
        request.requirements = [
            item.model_copy(update={"required_trust_level": "standard"})
            for item in request.requirements
        ]
    elif configuration == "ablation_contract_validation":
        request.requirements = [
            item.model_copy(update={"validation_contract": ValidationContract(json_schema={"type": "object"})})
            for item in request.requirements
        ]
        proposal.tasks = [
            item.model_copy(update={"validation_contract": ValidationContract(json_schema={"type": "object"})})
            for item in proposal.tasks
        ]
    # The final three mechanisms primarily affect runtime behavior. Their planning
    # decision remains the hybrid decision; compliance is reported from run traces.
    candidates = [item.model_copy(deep=True) for item in linguistic.candidate_plans]
    candidates[0] = proposal
    reports = [FeasibilityAnalyzer().check(request, registry, item) for item in candidates]
    report = next((item for item in reports if item.feasible), reports[0])
    return report.feasible, {
        "predicates": [item.model_dump(mode="json") for item in report.evidence],
        "capability_resolution": report.capability_resolution,
        "constraint_violations": report.constraint_violations,
        "validation_gaps": report.validation_gaps,
        "schema_violations": report.schema_violations,
        "risks": report.risks,
    }


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0, 0.0
    middle = median(ordered)
    half = len(ordered) // 2
    lower = ordered[:half]
    upper = ordered[-half:] if half else ordered
    return float(median(lower or ordered)), float(middle), float(median(upper or ordered))


def _bootstrap_accuracy(rows: list[dict[str, Any]], samples: int = 5000) -> list[float]:
    """Cluster bootstrap by case; model/seed observations are repeated measures."""
    rng = random.Random(20260712)
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[row["case_id"]].append(row)
    case_ids = sorted(clusters)
    scores = []
    for _ in range(samples):
        selected = [case_ids[rng.randrange(len(case_ids))] for _ in case_ids]
        draw = [row for case_id in selected for row in clusters[case_id]]
        scores.append(sum(int(item["correct"]) for item in draw) / len(draw))
    scores.sort()
    return [scores[int(samples * 0.025)], scores[int(samples * 0.975) - 1]]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(item["accepted"] and item["reference_feasible"] for item in rows)
    fp = sum(item["accepted"] and not item["reference_feasible"] for item in rows)
    fn = sum(not item["accepted"] and item["reference_feasible"] for item in rows)
    tn = len(rows) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    latencies = [float(item["latency_ms"]) for item in rows]
    q1, med, q3 = _quartiles(latencies)
    return {
        "n": len(rows),
        "accuracy": (tp + tn) / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_acceptance": fp,
        "false_refusal": fn,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "latency_ms": {"median": med, "iqr": [q1, q3]},
        "bootstrap_accuracy_95_ci": _bootstrap_accuracy(rows),
    }


def _mcnemar(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    paired = zip(left, right, strict=True)
    b = sum(a["correct"] and not c["correct"] for a, c in paired)
    paired = zip(left, right, strict=True)
    c = sum(not a["correct"] and d["correct"] for a, d in paired)
    n = b + c
    tail = sum(math.comb(n, index) for index in range(0, min(b, c) + 1)) / (2**n) if n else 1.0
    return {"b": b, "c": c, "exact_two_sided_p": min(1.0, 2 * tail)}


def _holm_adjust(comparisons: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(comparisons, key=lambda key: comparisons[key]["exact_two_sided_p"])
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, comparisons[key]["exact_two_sided_p"] * (count - rank))
        running = max(running, adjusted)
        comparisons[key]["holm_adjusted_p"] = running


def _collapse_by_case(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    return [
        {
            "case_id": case_id,
            "correct": sum(int(item["correct"]) for item in items) * 2 >= len(items),
        }
        for case_id, items in sorted(grouped.items())
    ]


def analyze(run_root: Path, corpus_root: Path) -> dict[str, Any]:
    provenance = validate_frozen_labels(corpus_root)
    completion_path = run_root / "collection-complete.json"
    if not completion_path.exists() or json.loads(completion_path.read_text(encoding="utf-8")).get("complete") is not True:
        raise RuntimeError("Scoring is blocked until raw-output collection is complete.")
    public = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    hidden = json.loads((corpus_root / "hidden/reference-labels.json").read_text(encoding="utf-8"))
    run_provenance = json.loads((run_root / "provenance.json").read_text(encoding="utf-8"))
    if len({public["corpus_hash"], hidden["corpus_hash"], provenance["corpus_hash"]}) != 1:
        raise RuntimeError("Corpus, labels, and provenance hashes differ.")
    cases = {item["case_id"]: item for item in public["cases"]}
    labels = {item["case_id"]: item for item in hidden["labels"]}
    rows_by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_outputs = len(MODELS) * len(SEEDS) * len(cases)
    files = list(run_root.glob("*/seed-*/*.json"))
    files = [item for item in files if item.name != "sentinel.json"]
    if len(files) != expected_outputs:
        raise RuntimeError(f"Expected {expected_outputs} immutable outputs, found {len(files)}.")
    if run_provenance.get("corpus_hash") != public["corpus_hash"]:
        raise RuntimeError("Run provenance corpus hash differs from the selected corpus.")
    raw_records = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    actual_identities = [
        (raw.get("model_id"), raw.get("seed"), raw.get("case_id")) for raw in raw_records
    ]
    expected_identities = set(product(MODELS, SEEDS, cases))
    if len(set(actual_identities)) != len(actual_identities):
        raise RuntimeError("Collection contains duplicate model/seed/case identities.")
    if set(actual_identities) != expected_identities:
        missing = expected_identities - set(actual_identities)
        extra = set(actual_identities) - expected_identities
        raise RuntimeError(f"Collection identity matrix differs: missing={missing}, extra={extra}.")
    for raw in raw_records:
        if raw.get("evaluation_status") != "raw_output_collected_not_scored":
            raise RuntimeError(f"Raw output {raw.get('case_id')} has an invalid collection status.")
        case = cases[raw["case_id"]]
        reference = labels[raw["case_id"]]
        linguistic = LinguisticBatchOutput.model_validate(raw["parsed_object"])
        for configuration in CONFIGURATIONS:
            accepted, evidence = _decision(configuration, case, linguistic)
            rows_by_config[configuration].append(
                {
                    "case_id": raw["case_id"],
                    "model_id": raw["model_id"],
                    "seed": raw["seed"],
                    "configuration": configuration,
                    "accepted": accepted,
                    "reference_feasible": reference["feasible"],
                    "correct": accepted == reference["feasible"],
                    "latency_ms": raw["latency_ms"],
                    "linguistic_latency_ms": raw["latency_ms"],
                    "symbolic_validation_latency_ms": None,
                    "execution_latency_ms": None,
                    "evidence": evidence,
                }
            )
    metrics = {name: _metrics(rows) for name, rows in rows_by_config.items()}
    hybrid = rows_by_config["hybrid"]
    comparisons = {
        name: {
            "accuracy_difference_vs_hybrid": metrics[name]["accuracy"] - metrics["hybrid"]["accuracy"],
        }
        for name, rows in rows_by_config.items()
        if name != "hybrid"
    }
    case_level_comparisons = {
        name: _mcnemar(_collapse_by_case(hybrid), _collapse_by_case(rows))
        for name, rows in rows_by_config.items()
        if name != "hybrid"
    }
    _holm_adjust(case_level_comparisons)
    stratified_comparisons: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for seed in SEEDS:
            stratum_comparisons: dict[str, dict[str, Any]] = {}
            for name, rows in rows_by_config.items():
                if name == "hybrid":
                    continue
                key = f"{name}|{model}|{seed}"
                left = sorted(
                    (r for r in hybrid if r["model_id"] == model and r["seed"] == seed),
                    key=lambda r: r["case_id"],
                )
                right = sorted(
                    (r for r in rows if r["model_id"] == model and r["seed"] == seed),
                    key=lambda r: r["case_id"],
                )
                stratum_comparisons[key] = _mcnemar(left, right)
            _holm_adjust(stratum_comparisons)
            stratified_comparisons.update(stratum_comparisons)
    return {
        "corpus_hash": public["corpus_hash"],
        "run_provenance": run_provenance,
        "collection_identity_matrix_complete": True,
        "label_provenance": provenance,
        "statistical_unit": "case (36 clusters); model and seed are repeated measurements",
        "metrics_by_model_seed": {
            name: {
                f"{model}|{seed}": _metrics([r for r in rows if r["model_id"] == model and r["seed"] == seed])
                for model in MODELS for seed in SEEDS
            }
            for name, rows in rows_by_config.items()
        },
        "metrics": metrics,
        "paired_comparisons": comparisons,
        "case_level_paired_comparisons": case_level_comparisons,
        "paired_comparisons_by_model_seed": stratified_comparisons,
        "rows": rows_by_config,
        "limitations": [
            "Planning decisions are evaluated on a designed 36-case corpus.",
            "Runtime-only ablations require separate execution-compliance reports.",
            "Symbolic-validation and execution latency require the separate controlled runtime runner; this analysis reports collected linguistic latency only.",
            "Reference labels were authored by the thesis author and were not independently adjudicated.",
            "Results measure conformance to the declared framework criteria, not neutral benchmark truth.",
            "No result establishes general superiority or semantic truth.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_root, args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
