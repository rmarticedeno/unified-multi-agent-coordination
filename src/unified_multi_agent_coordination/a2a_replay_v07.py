"""Replay eight v0.7 compiler-authorized cases through the fixture A2A boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .admission import AgentAdmissionPolicy
from .coordination_agent import CoordinationAgent
from .coordination_ledger import RetryPolicy
from .coordination_sdk import CoordinationSdk
from .defense_study_v07 import ARMS, _environment
from .semantic_admission import SemanticIntentOutput, SemanticRequestAdmitter
from .symbolic_plan_compiler import SymbolicPlanCompiler

JsonObject = dict[str, Any]
PRIMARY_SEED = 11


def _load_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _records(run_root: Path) -> list[JsonObject]:
    return [
        _load_object(path)
        for path in sorted(run_root.glob("o/*/m*/s*/*.json"))
    ]


def _observed_intents(
    analysis: JsonObject,
    records: list[JsonObject],
) -> dict[str, tuple[JsonObject, str]]:
    by_identity = {
        (
            record["identity"]["arm"],
            record["identity"]["model_id"],
            record["identity"]["seed"],
            record["identity"]["case_id"],
        ): record
        for record in records
    }
    candidates: dict[str, tuple[JsonObject, str]] = {}
    rows = analysis.get("rows") or []
    for row in rows:
        if (
            row.get("arm") != ARMS[0]
            or row.get("seed") != PRIMARY_SEED
            or not row.get("accepted")
            or not row.get("reference_feasible")
        ):
            continue
        identity = (
            row["arm"],
            row["model_id"],
            row["seed"],
            row["case_id"],
        )
        record = by_identity.get(identity)
        admission = ((record or {}).get("result") or {}).get("admission") or {}
        intent = admission.get("canonical_intent")
        if isinstance(intent, dict):
            candidates.setdefault(
                row["case_id"],
                (intent, f"hybrid_primary:{row['model_id']}"),
            )
    return candidates


def _select(
    cases: list[JsonObject],
    labels: dict[str, JsonObject],
    observed: dict[str, tuple[JsonObject, str]],
) -> list[tuple[JsonObject, JsonObject, str]]:
    case_by_id = {case["case_id"]: case for case in cases}
    selected: list[tuple[JsonObject, JsonObject, str]] = []
    selected_ids: set[str] = set()
    categories = sorted({case["category"] for case in cases})
    for category in categories:
        for case_id, (intent, source) in observed.items():
            case = case_by_id[case_id]
            if case["category"] == category and case_id not in selected_ids:
                selected.append((case, intent, source))
                selected_ids.add(case_id)
                break
    for case_id, (intent, source) in observed.items():
        if len(selected) >= 8:
            break
        if case_id not in selected_ids:
            selected.append((case_by_id[case_id], intent, source))
            selected_ids.add(case_id)
    for case in cases:
        if len(selected) >= 8:
            break
        label = labels[case["case_id"]]
        if case["case_id"] in selected_ids or not label.get("feasible"):
            continue
        fallback_intent = label.get("intent")
        if isinstance(fallback_intent, dict):
            selected.append((case, fallback_intent, "oracle_feasible_fallback"))
            selected_ids.add(case["case_id"])
    if len(selected) != 8:
        raise RuntimeError(
            f"Eight distinct authorized cases are required; selected {len(selected)}."
        )
    return selected


async def _replay_case(
    case: JsonObject,
    intent_document: JsonObject,
    selection_source: str,
) -> JsonObject:
    catalog, registry = _environment(case)
    intent = SemanticIntentOutput.model_validate(intent_document)
    admission = SemanticRequestAdmitter().admit(
        case["request_text"], catalog, intent, registry
    )
    if admission.request is None:
        raise RuntimeError(f"Selected case did not admit: {case['case_id']}")
    compilation = SymbolicPlanCompiler().compile(admission.request, registry)
    if not compilation.report.feasible:
        raise RuntimeError(f"Selected case did not compile: {case['case_id']}")
    artifacts_by_task = {
        task.task_id: [
            {
                "name": name,
                "kind": "data",
                "data": {"fixture": True, "task_id": task.task_id},
            }
            for name in task.expected_artifacts
        ]
        for task in compilation.proposal.tasks
    }
    dispatches: list[JsonObject] = []

    async def sender(agent_id: str, payload: JsonObject) -> JsonObject:
        metadata = payload.get("_coordination") or {}
        task_id = str(metadata.get("task_id") or "")
        artifacts = artifacts_by_task.get(task_id)
        if artifacts is None:
            raise RuntimeError(f"Unknown fixture task: {task_id}")
        dispatches.append(
            {
                "agent_id": agent_id,
                "task_id": task_id,
                "operation_key": metadata.get("operation_key"),
                "input_dependency_count": len(metadata.get("inputs_by_task") or {}),
            }
        )
        return {"artifacts": artifacts, "fixture_a2a": True}

    sdk = CoordinationSdk(
        task_sender=sender,
        admission_policy=AgentAdmissionPolicy(allow_insecure_development=True),
    )
    for entry in registry:
        endpoint = f"http://fixture.invalid/{entry.agent_id}"
        fixture_entry = entry.model_copy(
            update={
                "agent_kind": "remote_a2a",
                "service_endpoint": endpoint,
                "invocation_endpoint": endpoint,
            }
        )
        sdk._local_registry[fixture_entry.agent_id] = fixture_entry
    agent = CoordinationAgent(
        sdk,
        retry_policy=RetryPolicy(registry_retries=0, task_retries=0, backoff_s=0),
    )
    result = await agent.coordinate(
        admission.request,
        payload=dict(case.get("payload") or {}),
        timeout_s=10,
        session_id=f"v07-a2a-{case['case_id']}",
    )
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "selection_source": selection_source,
        "status": result.status,
        "compiler_feasible": compilation.report.feasible,
        "execution_order": compilation.proposal.execution_order,
        "dispatches": dispatches,
        "task_results": [
            item.model_dump(mode="json") for item in result.task_results
        ],
        "artifact_count": len(result.artifacts),
        "passed": result.status == "completed"
        and len(dispatches) == len(compilation.proposal.tasks),
    }


async def replay(
    run_root: Path,
    corpus_root: Path,
    analysis_path: Path,
) -> JsonObject:
    public = _load_object(corpus_root / "public" / "cases.json")
    hidden = _load_object(corpus_root / "hidden" / "reference-labels.json")
    cases = list(public["cases"])
    labels = {item["case_id"]: item for item in hidden["labels"]}
    analysis = _load_object(analysis_path)
    observed = _observed_intents(analysis, _records(run_root))
    selected = _select(cases, labels, observed)
    results = [
        await _replay_case(case, intent, source)
        for case, intent, source in selected
    ]
    return {
        "schema_version": "a2a-replay-v0.7.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": run_root.as_posix(),
        "analysis": analysis_path.as_posix(),
        "case_count": len(results),
        "hybrid_selected": sum(
            item["selection_source"].startswith("hybrid_primary")
            for item in results
        ),
        "oracle_fallback_selected": sum(
            item["selection_source"] == "oracle_feasible_fallback"
            for item in results
        ),
        "passed": len(results) == 8 and all(item["passed"] for item in results),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.7"))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(replay(args.run, args.corpus, args.analysis))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: report[key] for key in ("case_count", "passed")}, indent=2))


if __name__ == "__main__":
    main()
