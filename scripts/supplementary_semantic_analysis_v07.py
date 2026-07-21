"""Aggregate v0.7 semantic field accuracy and set-valued micro F1 post hoc."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
FIELDS = (
    "goals",
    "trust_policies",
    "artifact_contracts",
    "forbidden_capabilities",
    "forbidden_agents",
    "unresolved_terms",
)


def _load(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _sets(intent: JsonObject | None) -> dict[str, set[str]]:
    intent = intent or {}
    goals = intent.get("goals") or []
    return {
        "goals": {
            str(item.get("capability_id"))
            for item in goals
            if isinstance(item, dict) and item.get("capability_id")
        },
        "trust_policies": {
            str(item.get("trust_policy_id"))
            for item in goals
            if isinstance(item, dict) and item.get("trust_policy_id")
        },
        "artifact_contracts": {
            str(item.get("artifact_contract_id"))
            for item in goals
            if isinstance(item, dict) and item.get("artifact_contract_id")
        },
        "forbidden_capabilities": {
            str(item) for item in intent.get("forbidden_capability_ids") or []
        },
        "forbidden_agents": {
            str(item) for item in intent.get("forbidden_agent_ids") or []
        },
        "unresolved_terms": {
            normalized
            for item in intent.get("unresolved_terms") or []
            if (normalized := _norm(item))
        },
    }


def _intent(record: JsonObject) -> JsonObject | None:
    result = record.get("result") or {}
    if record["identity"]["arm"] == "production_hybrid_v07":
        value = (result.get("admission") or {}).get("canonical_intent")
    else:
        value = result.get("parsed")
    return value if isinstance(value, dict) else None


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, score


def _status_macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({value for pair in pairs for value in pair})
    scores = []
    for label in labels:
        tp = sum(reference == label and observed == label for reference, observed in pairs)
        fp = sum(reference != label and observed == label for reference, observed in pairs)
        fn = sum(reference == label and observed != label for reference, observed in pairs)
        scores.append(_f1(tp, fp, fn)[2])
    return sum(scores) / len(scores) if scores else 0.0


def analyze(run: Path, corpus: Path, *, seed: int) -> JsonObject:
    labels_document = _load(corpus / "hidden" / "reference-labels.json")
    labels = {item["case_id"]: item["intent"] for item in labels_document["labels"]}
    records = []
    for path in sorted(run.glob("o/*/m*/s*/*.json")):
        record = _load(path)
        if record["identity"]["seed"] == seed:
            records.append(record)
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for record in records:
        grouped[record["identity"]["arm"]].append(record)
    arms: JsonObject = {}
    for arm, observations in grouped.items():
        exact: Counter[str] = Counter()
        totals: Counter[str] = Counter()
        status_pairs: list[tuple[str, str]] = []
        for record in observations:
            case_id = record["identity"]["case_id"]
            reference = labels[case_id]
            observed = _intent(record)
            reference_sets = _sets(reference)
            observed_sets = _sets(observed)
            reference_status = str(reference.get("interpretation_status") or "invalid")
            observed_status = (
                str(observed.get("interpretation_status") or "invalid")
                if observed
                else "invalid"
            )
            status_pairs.append((reference_status, observed_status))
            for field in FIELDS:
                expected = reference_sets[field]
                predicted = observed_sets[field]
                exact[f"{field}:exact"] += expected == predicted
                exact[f"{field}:tp"] += len(expected & predicted)
                exact[f"{field}:fp"] += len(predicted - expected)
                exact[f"{field}:fn"] += len(expected - predicted)
                totals[field] += 1
        fields: JsonObject = {}
        for field in FIELDS:
            precision, recall, f1 = _f1(
                exact[f"{field}:tp"],
                exact[f"{field}:fp"],
                exact[f"{field}:fn"],
            )
            fields[field] = {
                "exact_accuracy": exact[f"{field}:exact"] / totals[field],
                "micro_precision": precision,
                "micro_recall": recall,
                "micro_f1": f1,
            }
        arms[arm] = {
            "observations": len(observations),
            "status_accuracy": sum(a == b for a, b in status_pairs) / len(status_pairs),
            "status_macro_f1": _status_macro_f1(status_pairs),
            "fields": fields,
        }
    return {
        "schema_version": "supplementary-semantic-v0.7.0",
        "classification": "post_freeze_descriptive_scoring_of_preserved_fields",
        "seed": seed,
        "arms": arms,
        "limitations": [
            "This scorer was added after protocol freeze and is descriptive.",
            "Set-valued F1 weights identifiers rather than matched pairs.",
            "Author-only reference labels remain a construct-validity limitation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.7"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.corpus, seed=args.seed)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["arms"], indent=2))


if __name__ == "__main__":
    main()
