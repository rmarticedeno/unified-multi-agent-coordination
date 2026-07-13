"""Paired baseline evaluation for the thesis scenarios.

The baselines reuse the deterministic thesis fixtures so that the hybrid
coordinator, a rule-only planner, and local-LLM-only planners face the same
requests, registries, payloads, and reference labels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .auxiliary import BoundedAuxiliaryCapabilityFactory
from .coordination_sdk import CoordinationSdk
from .end_to_end_scenarios import (
    ScenarioDefinition,
    ScenarioStatus,
    run_scenarios,
    scenario_definitions,
)
from .feasibility import FeasibilityAnalyzer
from .local_llm_reference import ALLOWED_MODELS
from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    FeasibilityReport,
    GeneratedNlpAgentSpec,
    ProblemRequest,
    SolutionProposal,
    TaskExecutionResult,
    TaskSpec,
)


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class CapabilityResolution:
    """Resolution from a model-supplied task label into a symbolic capability."""

    raw_name: str
    assigned_agent_id: str
    resolved: bool
    resolved_name: str = ""
    resolved_agent_id: str = ""
    method: str = "unresolved"


async def run_baselines(
    *,
    local_llm_root: Path = Path("demo_runs/local_llm"),
) -> JsonObject:
    """Run paired baselines and return a JSON-ready report."""
    scenarios = scenario_definitions()
    hybrid_report = await run_scenarios()
    hybrid_rows = [
        _hybrid_row(row)
        for row in hybrid_report.get("scenarios", [])
    ]
    rule_rows = [await _run_rule_only(scenario) for scenario in scenarios]
    llm_models: dict[str, JsonObject] = {}
    for model_id in ALLOWED_MODELS:
        report = _load_latest_llm_report(local_llm_root, model_id)
        if report is None:
            continue
        rows = [
            await _run_llm_only(scenario, _llm_scenario(report, scenario.scenario_id))
            for scenario in scenarios
        ]
        llm_models[model_id] = {
            "model_id": model_id,
            "prompt_version": report.get("prompt_version", ""),
            "run_id": report.get("run_id", ""),
            "rows": rows,
            "summary": _summarize_rows(rows),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_count": len(scenarios),
        "source_files": {
            "local_llm_root": str(local_llm_root),
        },
        "configurations": {
            "hybrid": {
                "rows": hybrid_rows,
                "summary": _summarize_rows(hybrid_rows),
            },
            "rule_only": {
                "rows": rule_rows,
                "summary": _summarize_rows(rule_rows),
            },
            "llm_only": {
                "models": llm_models,
            },
        },
    }


async def _run_rule_only(scenario: ScenarioDefinition) -> JsonObject:
    sdk = CoordinationSdk()
    await scenario.setup(sdk)
    registry = await sdk.registry_snapshot(refresh=False)
    started = time.perf_counter()
    proposal = build_rule_only_proposal(scenario.request, registry)
    report = FeasibilityAnalyzer().check(scenario.request, registry, proposal)
    task_results: list[TaskExecutionResult] = []
    if report.feasible:
        task_results = await _execute_authorized(
            sdk,
            report,
            proposal,
            scenario.payload,
            timeout_s=scenario.timeout_s,
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    status = _status_from_authorization_and_results(report.feasible, task_results)
    await _close_sdk_client(sdk)
    return _baseline_row(
        scenario=scenario,
        configuration="rule_only",
        observed_status=status,
        accepted=report.feasible,
        dispatch_attempts=len(task_results),
        latency_ms=latency_ms,
        symbolic_authorization_used=True,
        artifacts=[artifact for result in task_results for artifact in result.artifacts],
        proposal=proposal,
        feasibility_report=report,
        task_results=task_results,
    )


async def _run_llm_only(
    scenario: ScenarioDefinition,
    llm_row: JsonObject | None,
) -> JsonObject:
    sdk = CoordinationSdk()
    await scenario.setup(sdk)
    registry = await sdk.registry_snapshot(refresh=False)
    started = time.perf_counter()
    model_output = dict((llm_row or {}).get("model_output") or {})
    planning_verdict = str(model_output.get("planning_verdict") or "unknown")
    accepted = planning_verdict == "feasible"
    task_results: list[TaskExecutionResult] = []
    resolutions: list[CapabilityResolution] = []
    unresolved_dependencies = 0
    proposal = SolutionProposal()
    if accepted:
        proposal, resolutions, unresolved_dependencies = llm_output_to_proposal(
            model_output,
            registry,
        )
        for task in proposal.tasks:
            if not task.assigned_to:
                task_results.append(
                    TaskExecutionResult(
                        task_id=task.task_id,
                        agent_id="",
                        status="failed",
                        error="LLM-only proposal did not resolve an accountable executor.",
                    )
                )
                continue
            task_results.append(
                await sdk.invoke_agent(
                    task.assigned_to,
                    task,
                    _payload_for_task(scenario.payload, task, task_results),
                    timeout_s=scenario.timeout_s,
                )
            )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    status = _status_from_authorization_and_results(accepted, task_results)
    artifacts = [artifact for result in task_results for artifact in result.artifacts]
    await _close_sdk_client(sdk)
    row = _baseline_row(
        scenario=scenario,
        configuration="llm_only",
        observed_status=status,
        accepted=accepted,
        dispatch_attempts=len(task_results),
        latency_ms=latency_ms,
        symbolic_authorization_used=False,
        artifacts=artifacts,
        proposal=proposal,
        task_results=task_results,
        model_id=(llm_row or {}).get("model_id", ""),
        llm_parse_ok=bool((llm_row or {}).get("parse_ok")),
        llm_planning_verdict=planning_verdict,
        llm_prompt_latency_ms=(llm_row or {}).get("latency_ms", 0),
        llm_tokens=int((llm_row or {}).get("usage", {}).get("total_tokens") or 0),
    )
    row["requirement_resolution"] = _resolution_summary(
        resolutions,
        scenario.request.requirements,
    )
    row["unresolved_dependencies"] = unresolved_dependencies
    return row


def build_rule_only_proposal(
    request: ProblemRequest,
    registry: list[AgentRegistryEntry],
) -> SolutionProposal:
    """Build a non-linguistic exact-match proposal from a structured request."""
    factory = BoundedAuxiliaryCapabilityFactory()
    tasks: list[TaskSpec] = []
    selected_agents: dict[str, str] = {}
    generated: list[GeneratedNlpAgentSpec] = []
    for index, requirement in enumerate(request.requirements, start=1):
        task_id = f"t{index}"
        agent = _first_agent_with_skill(registry, requirement.name)
        auxiliary_spec_id = ""
        assigned_to = agent.agent_id if agent else None
        if assigned_to:
            selected_agents[task_id] = assigned_to
        elif requirement.auxiliary_eligible:
            spec = factory.specify(requirement, lifecycle="baseline:rule-only")
            if spec is not None:
                generated.append(spec)
                auxiliary_spec_id = spec.spec_id
        tasks.append(
            TaskSpec(
                task_id=task_id,
                requirement_name=requirement.name,
                assigned_to=assigned_to,
                auxiliary_spec_id=auxiliary_spec_id or None,
                validation_contract=requirement.validation_contract,
            )
        )

    req_by_task = {
        task.task_id: req
        for task, req in zip(tasks, request.requirements, strict=True)
    }
    for task in tasks:
        consumer = req_by_task[task.task_id]
        dependencies = [
            prior.task_id
            for prior in tasks[: tasks.index(task)]
            if _modes_overlap(req_by_task[prior.task_id].output_modes, consumer.input_modes)
        ]
        if dependencies:
            task.depends_on = dependencies

    return SolutionProposal(
        tasks=tasks,
        selected_agents=selected_agents,
        generated_nlp_agents=generated,
        execution_order=[task.task_id for task in tasks],
        expected_artifacts=list(request.required_artifacts),
        completion_criteria=["all explicit validators pass"],
    )


def resolve_capability_identity(
    *,
    raw_name: str,
    assigned_agent_id: str,
    registry: list[AgentRegistryEntry],
) -> CapabilityResolution:
    """Resolve an LLM label to a symbolic capability using deterministic rules."""
    agent = next((item for item in registry if item.agent_id == assigned_agent_id), None)
    if agent is not None and len(agent.skills) == 1:
        return CapabilityResolution(
            raw_name=raw_name,
            assigned_agent_id=assigned_agent_id,
            resolved=True,
            resolved_name=agent.skills[0].name,
            resolved_agent_id=agent.agent_id,
            method="agent_id_single_skill",
        )

    for candidate in _registry_skills(registry):
        if _norm(candidate.name) == _norm(raw_name):
            owner = _first_agent_with_skill(registry, candidate.name)
            return CapabilityResolution(
                raw_name=raw_name,
                assigned_agent_id=assigned_agent_id,
                resolved=True,
                resolved_name=candidate.name,
                resolved_agent_id=owner.agent_id if owner else assigned_agent_id,
                method="exact_normalized_skill_name",
            )

    return CapabilityResolution(
        raw_name=raw_name,
        assigned_agent_id=assigned_agent_id,
        resolved=False,
    )


def llm_output_to_proposal(
    model_output: JsonObject,
    registry: list[AgentRegistryEntry],
) -> tuple[SolutionProposal, list[CapabilityResolution], int]:
    """Convert a local-LLM candidate into executable task records."""
    tasks: list[TaskSpec] = []
    selected_agents: dict[str, str] = {}
    resolutions: list[CapabilityResolution] = []
    alias_to_task: dict[str, str] = {}
    unresolved_dependencies = 0

    candidate_tasks = list(model_output.get("candidate_tasks") or [])
    for index, item in enumerate(candidate_tasks, start=1):
        raw_name = str(item.get("requirement_name") or "")
        assigned_agent_id = str(item.get("assigned_agent_id") or "")
        resolution = resolve_capability_identity(
            raw_name=raw_name,
            assigned_agent_id=assigned_agent_id,
            registry=registry,
        )
        resolutions.append(resolution)
        task_id = f"t{index}"
        agent_id = resolution.resolved_agent_id or assigned_agent_id
        if not any(agent.agent_id == agent_id for agent in registry):
            agent_id = ""
        requirement_name = resolution.resolved_name or raw_name
        task = TaskSpec(
            task_id=task_id,
            requirement_name=requirement_name,
            assigned_to=agent_id or None,
        )
        if agent_id:
            selected_agents[task_id] = agent_id
        tasks.append(task)
        for alias in {raw_name, requirement_name, assigned_agent_id, task_id}:
            if alias:
                alias_to_task[_norm(alias)] = task_id

    for task, item in zip(tasks, candidate_tasks, strict=False):
        dependencies: list[str] = []
        for dependency in item.get("depends_on") or []:
            resolved_dependency = alias_to_task.get(_norm(str(dependency)))
            if resolved_dependency and resolved_dependency != task.task_id:
                dependencies.append(resolved_dependency)
            else:
                unresolved_dependencies += 1
        task.depends_on = list(dict.fromkeys(dependencies))

    return (
        SolutionProposal(
            tasks=tasks,
            selected_agents=selected_agents,
            execution_order=[task.task_id for task in tasks],
            expected_artifacts=list(model_output.get("required_artifacts") or []),
            completion_criteria=["LLM-only planning verdict accepted"],
        ),
        resolutions,
        unresolved_dependencies,
    )


async def _execute_authorized(
    sdk: CoordinationSdk,
    report: FeasibilityReport,
    proposal: SolutionProposal,
    payload: JsonObject,
    *,
    timeout_s: float,
) -> list[TaskExecutionResult]:
    results: list[TaskExecutionResult] = []
    task_by_id = {task.task_id: task for task in proposal.tasks}
    for task_id in proposal.execution_order or [task.task_id for task in proposal.tasks]:
        task = task_by_id[task_id]
        result = await sdk.send_task(
            report,
            task,
            _payload_for_task(payload, task, results),
            timeout_s=timeout_s,
        )
        results.append(result)
    return results


def _payload_for_task(
    payload: JsonObject,
    task: TaskSpec,
    prior_results: list[TaskExecutionResult],
) -> JsonObject:
    enriched = dict(payload)
    artifacts_by_task = {
        result.task_id: list(result.artifacts)
        for result in prior_results
    }
    enriched["_coordination"] = {
        **dict(enriched.get("_coordination") or {}),
        "inputs_by_task": {
            dependency: list(artifacts_by_task.get(dependency, []))
            for dependency in task.depends_on
        },
        "previous_artifacts": [
            artifact
            for artifacts in artifacts_by_task.values()
            for artifact in artifacts
        ],
    }
    return enriched


def _baseline_row(
    *,
    scenario: ScenarioDefinition,
    configuration: str,
    observed_status: ScenarioStatus,
    accepted: bool,
    dispatch_attempts: int,
    latency_ms: float,
    symbolic_authorization_used: bool,
    artifacts: list[JsonObject],
    proposal: SolutionProposal,
    feasibility_report: FeasibilityReport | None = None,
    task_results: list[TaskExecutionResult] | None = None,
    model_id: str = "",
    llm_parse_ok: bool | None = None,
    llm_planning_verdict: str = "",
    llm_prompt_latency_ms: float | int = 0,
    llm_tokens: int = 0,
) -> JsonObject:
    reference_planning = _reference_planning_verdict(scenario.reference_status)
    observed_planning = "feasible" if accepted else "infeasible"
    return {
        "configuration": configuration,
        "model_id": model_id,
        "id": scenario.scenario_id,
        "title": scenario.title,
        "reference_status": scenario.reference_status,
        "reference_planning_verdict": reference_planning,
        "observed_status": observed_status,
        "observed_planning_verdict": observed_planning,
        "status_matches_reference": observed_status == scenario.reference_status,
        "decision_matches_reference": observed_planning == reference_planning,
        "false_accept": accepted and reference_planning == "infeasible",
        "false_refuse": not accepted and reference_planning == "feasible",
        "accepted": accepted,
        "dispatch_attempts": dispatch_attempts,
        "dispatch_without_symbolic_authorization": (
            dispatch_attempts > 0 and not symbolic_authorization_used
        ),
        "symbolic_authorization_used": symbolic_authorization_used,
        "latency_ms": latency_ms,
        "produced_required_artifacts": _artifacts_cover(
            scenario.request.required_artifacts,
            artifacts,
        ),
        "promised_required_artifacts": _norm_set(scenario.request.required_artifacts)
        <= _norm_set(proposal.expected_artifacts),
        "task_count": len(proposal.tasks),
        "failed_predicates": [
            item.name
            for item in (feasibility_report.evidence if feasibility_report else [])
            if not item.passed
        ],
        "task_result_statuses": [
            result.status for result in task_results or []
        ],
        "llm_parse_ok": llm_parse_ok,
        "llm_planning_verdict": llm_planning_verdict,
        "llm_prompt_latency_ms": llm_prompt_latency_ms,
        "llm_tokens": llm_tokens,
    }


def _hybrid_row(row: JsonObject) -> JsonObject:
    accepted = bool(row.get("authorization", {}).get("feasible"))
    reference_status = str(row.get("reference_status") or "failed")
    reference_planning = _reference_planning_verdict(reference_status)
    observed_status = str(row.get("observed_status") or "failed")
    observed_planning = "feasible" if accepted else "infeasible"
    artifacts = list(row.get("artifacts") or [])
    required_artifacts = [
        str(artifact)
        for artifact in row.get("authorization", {}).get("required_artifacts", [])
    ]
    return {
        "configuration": "hybrid",
        "model_id": "",
        "id": row.get("id", ""),
        "title": row.get("title", ""),
        "reference_status": reference_status,
        "reference_planning_verdict": reference_planning,
        "observed_status": observed_status,
        "observed_planning_verdict": observed_planning,
        "status_matches_reference": bool(row.get("status_matches_reference")),
        "decision_matches_reference": observed_planning == reference_planning,
        "false_accept": accepted and reference_planning == "infeasible",
        "false_refuse": not accepted and reference_planning == "feasible",
        "accepted": accepted,
        "dispatch_attempts": int(row.get("dispatch_attempts") or 0),
        "dispatch_without_symbolic_authorization": False,
        "symbolic_authorization_used": True,
        "latency_ms": 0,
        "produced_required_artifacts": _artifacts_cover(required_artifacts, artifacts)
        if required_artifacts else None,
        "promised_required_artifacts": None,
        "task_count": len(row.get("task_results") or []),
        "failed_predicates": row.get("authorization", {}).get("failed_predicates", []),
        "task_result_statuses": [
            result.get("status", "")
            for result in row.get("task_results", [])
        ],
        "llm_parse_ok": None,
        "llm_planning_verdict": "",
        "llm_prompt_latency_ms": 0,
        "llm_tokens": 0,
    }


def _summarize_rows(rows: list[JsonObject]) -> JsonObject:
    latencies = [
        float(row.get("latency_ms", 0))
        for row in rows
        if isinstance(row.get("latency_ms"), int | float)
        and float(row.get("latency_ms", 0)) > 0
    ]
    resolution: list[JsonObject] = [
        value
        for row in rows
        if isinstance((value := row.get("requirement_resolution")), dict)
    ]
    return {
        "scenario_count": len(rows),
        "status_matches_reference": sum(
            1 for row in rows if row.get("status_matches_reference")
        ),
        "decision_matches_reference": sum(
            1 for row in rows if row.get("decision_matches_reference")
        ),
        "false_accepts": sum(1 for row in rows if row.get("false_accept")),
        "false_refusals": sum(1 for row in rows if row.get("false_refuse")),
        "accepted": sum(1 for row in rows if row.get("accepted")),
        "completed": sum(1 for row in rows if row.get("observed_status") == "completed"),
        "failed": sum(1 for row in rows if row.get("observed_status") == "failed"),
        "infeasible": sum(1 for row in rows if row.get("observed_status") == "infeasible"),
        "dispatch_attempts": sum(int(row.get("dispatch_attempts") or 0) for row in rows),
        "dispatch_without_symbolic_authorization": sum(
            1 for row in rows if row.get("dispatch_without_symbolic_authorization")
        ),
        "median_latency_ms": round(median(latencies), 2) if latencies else 0,
        "llm_tokens": sum(int(row.get("llm_tokens") or 0) for row in rows),
        "resolved_requirements": sum(
            int(item.get("resolved") or 0) for item in resolution
        ),
        "unresolved_requirements": sum(
            int(item.get("unresolved") or 0) for item in resolution
        ),
        "exact_requirement_name_matches": sum(
            int(item.get("exact_name_matches") or 0) for item in resolution
        ),
        "resolved_expected_names": sum(
            int(item.get("resolved_expected_names") or 0) for item in resolution
        ),
    }


def _resolution_summary(
    resolutions: list[CapabilityResolution],
    expected_requirements: list[CapabilityRequirement],
) -> JsonObject:
    exact_names = _norm_set([requirement.name for requirement in expected_requirements])
    resolved_names = _norm_set(
        [resolution.resolved_name for resolution in resolutions if resolution.resolved]
    )
    raw_names = _norm_set([resolution.raw_name for resolution in resolutions])
    return {
        "total": len(resolutions),
        "resolved": sum(1 for item in resolutions if item.resolved),
        "unresolved": sum(1 for item in resolutions if not item.resolved),
        "exact_name_matches": len(exact_names & raw_names),
        "resolved_expected_names": len(exact_names & resolved_names),
        "items": [
            {
                "raw_name": item.raw_name,
                "assigned_agent_id": item.assigned_agent_id,
                "resolved": item.resolved,
                "resolved_name": item.resolved_name,
                "resolved_agent_id": item.resolved_agent_id,
                "method": item.method,
            }
            for item in resolutions
        ],
    }


def _status_from_authorization_and_results(
    accepted: bool,
    task_results: list[TaskExecutionResult],
) -> ScenarioStatus:
    if not accepted:
        return "infeasible"
    if task_results and all(result.status == "completed" for result in task_results):
        return "completed"
    return "failed"


def _llm_scenario(report: JsonObject, scenario_id: str) -> JsonObject | None:
    for row in report.get("scenarios", []):
        if row.get("id") == scenario_id:
            return row
    return None


def _load_latest_llm_report(root: Path, model_id: str) -> JsonObject | None:
    path = root / _safe_model_dir(model_id) / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def _close_sdk_client(sdk: CoordinationSdk) -> None:
    close = getattr(sdk.http_client, "aclose", None)
    if close is not None:
        await close()


def _first_agent_with_skill(
    registry: list[AgentRegistryEntry],
    requirement_name: str,
) -> AgentRegistryEntry | None:
    wanted = _norm(requirement_name)
    for agent in registry:
        if agent.status != "available":
            continue
        if any(_norm(skill.name) == wanted for skill in agent.skills):
            return agent
    return None


def _registry_skills(registry: list[AgentRegistryEntry]) -> list[CapabilityRequirement]:
    return [skill for agent in registry for skill in agent.skills]


def _modes_overlap(outputs: list[str], inputs: list[str]) -> bool:
    if not outputs or not inputs:
        return False
    return bool(set(outputs) & set(inputs))


def _artifacts_cover(required_artifacts: list[str], artifacts: list[JsonObject]) -> bool:
    if not required_artifacts:
        return True
    available: set[str] = set()
    for artifact in artifacts:
        for key in ("artifact_id", "id", "name", "kind", "type"):
            value = artifact.get(key)
            if value:
                available.add(_norm(str(value)))
        data = artifact.get("data")
        if isinstance(data, dict):
            available.update(_norm(str(key)) for key in data)
    return _norm_set(required_artifacts) <= available


def _norm_set(values: list[str]) -> set[str]:
    return {_norm(value) for value in values if value}


def _reference_planning_verdict(reference_status: str) -> str:
    return "infeasible" if reference_status == "infeasible" else "feasible"


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run paired thesis baseline evaluations.",
    )
    parser.add_argument(
        "--local-llm-root",
        type=Path,
        default=Path("demo_runs/local_llm"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_runs/baselines/baseline_report.json"),
    )
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args(argv)

    report = asyncio.run(run_baselines(local_llm_root=args.local_llm_root))
    configs = report["configurations"]
    print(
        "hybrid: "
        f"{configs['hybrid']['summary']['status_matches_reference']}/"
        f"{configs['hybrid']['summary']['scenario_count']} status matches"
    )
    print(
        "rule_only: "
        f"{configs['rule_only']['summary']['status_matches_reference']}/"
        f"{configs['rule_only']['summary']['scenario_count']} status matches"
    )
    for model_id, model in configs["llm_only"]["models"].items():
        print(
            f"llm_only {model_id}: "
            f"{model['summary']['status_matches_reference']}/"
            f"{model['summary']['scenario_count']} status matches"
        )
    if not args.no_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote baseline report to {args.output}")


if __name__ == "__main__":
    main()
