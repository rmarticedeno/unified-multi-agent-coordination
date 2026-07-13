"""Generate thesis-ready summaries from preserved evaluation evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from .local_llm_reference import ALLOWED_MODELS


JsonObject = dict[str, Any]


def build_summary(
    *,
    scenarios_path: Path,
    docker_path: Path,
    distributed_path: Path,
    local_llm_root: Path,
    baselines_path: Path,
) -> JsonObject:
    """Read preserved reports and return one normalized thesis summary."""
    deterministic = _load_json(scenarios_path)
    docker = _load_optional_json(docker_path)
    distributed = _load_optional_json(distributed_path)
    baselines = _load_optional_json(baselines_path)
    llm_reports = {
        model_id: _load_latest_llm_report(local_llm_root, model_id)
        for model_id in ALLOWED_MODELS
    }
    return {
        "source_files": {
            "deterministic_scenarios": str(scenarios_path),
            "docker_system": str(docker_path) if docker_path.exists() else "",
            "distributed_system": str(distributed_path) if distributed_path.exists() else "",
            "local_llm_root": str(local_llm_root),
            "baselines": str(baselines_path) if baselines_path.exists() else "",
        },
        "deterministic": _summarize_deterministic(deterministic),
        "docker": _summarize_docker(docker) if docker else None,
        "distributed": _summarize_distributed(distributed) if distributed else None,
        "baselines": _summarize_baselines(baselines) if baselines else None,
        "local_llm": {
            model_id: _summarize_llm(report)
            for model_id, report in llm_reports.items()
            if report is not None
        },
    }


def write_outputs(summary: JsonObject, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tables.md").write_text(_markdown_tables(summary), encoding="utf-8")
    (output_dir / "tables.tex").write_text(_latex_tables(summary), encoding="utf-8")


def _load_json(path: Path) -> JsonObject:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> JsonObject | None:
    if not path.exists():
        return None
    return _load_json(path)


def _load_latest_llm_report(root: Path, model_id: str) -> JsonObject | None:
    path = root / _safe_model_dir(model_id) / "latest.json"
    if not path.exists():
        return None
    return _load_json(path)


def _summarize_deterministic(report: JsonObject) -> JsonObject:
    scenarios = list(report.get("scenarios", []))
    statuses = Counter(item.get("observed_status", "") for item in scenarios)
    return {
        "generated_at": report.get("generated_at", ""),
        "scenario_count": len(scenarios),
        "status_matches_reference": sum(
            1 for item in scenarios if item.get("status_matches_reference") is True
        ),
        "status_counts": dict(sorted(statuses.items())),
        "no_dispatch_before_authorization": sum(
            1 for item in scenarios if item.get("no_dispatch_before_authorization") is True
        ),
        "authorized_plans": sum(
            1 for item in scenarios if item.get("authorization", {}).get("feasible")
        ),
        "explicit_refusals": sum(
            1 for item in scenarios if not item.get("authorization", {}).get("feasible")
        ),
        "rows": [
            {
                "id": item.get("id", ""),
                "reference_status": item.get("reference_status", ""),
                "observed_status": item.get("observed_status", ""),
                "authorization": "feasible"
                if item.get("authorization", {}).get("feasible")
                else "infeasible",
                "dispatch_attempts": item.get("dispatch_attempts", 0),
                "status_matches_reference": bool(item.get("status_matches_reference")),
            }
            for item in scenarios
        ],
    }


def _summarize_docker(report: JsonObject) -> JsonObject:
    scenarios = list(report.get("scenarios", []))
    statuses = Counter(item.get("observed_status", "") for item in scenarios)
    latencies = [
        float(item.get("latency_ms", 0))
        for item in scenarios
        if isinstance(item.get("latency_ms"), int | float)
    ]
    boundary_relevant = [
        item
        for item in scenarios
        if item.get("id") not in {"health", "registry_discovery"}
    ]
    return {
        "scenario_count": int(report.get("scenario_count") or len(scenarios)),
        "passed": bool(report.get("passed")),
        "passed_rows": sum(1 for item in scenarios if item.get("passed") is True),
        "status_counts": dict(sorted(statuses.items())),
        "median_latency_ms": round(median(latencies), 2) if latencies else 0,
        "no_dispatch_before_authorization": sum(
            1
            for item in boundary_relevant
            if item.get("no_dispatch_before_authorization") is True
        ),
        "boundary_relevant_count": len(boundary_relevant),
        "rows": [
            {
                "id": item.get("id", ""),
                "reference_status": item.get("reference_status", ""),
                "observed_status": item.get("observed_status", ""),
                "passed": bool(item.get("passed")),
                "latency_ms": item.get("latency_ms", 0),
            }
            for item in scenarios
        ],
    }


def _summarize_llm(report: JsonObject) -> JsonObject:
    scenarios = list(report.get("scenarios", []))
    latencies = [
        float(item.get("latency_ms", 0))
        for item in scenarios
        if isinstance(item.get("latency_ms"), int | float)
    ]
    return {
        "generated_at": report.get("generated_at", ""),
        "endpoint": report.get("endpoint", ""),
        "model_id": report.get("model_id", ""),
        "run_id": report.get("run_id", ""),
        "prompt_version": report.get("prompt_version", ""),
        "scenario_count": len(scenarios),
        "parse_ok": sum(1 for item in scenarios if item.get("parse_ok")),
        "requirements_all_matched": sum(
            1
            for item in scenarios
            if item.get("checks", {}).get("requirements_all_matched")
        ),
        "artifacts_all_matched": sum(
            1
            for item in scenarios
            if item.get("checks", {}).get("artifacts_all_matched")
        ),
        "planning_verdict_matches": sum(
            1
            for item in scenarios
            if item.get("checks", {}).get("planning_verdict_matches")
        ),
        "uses_only_registered_agents": sum(
            1
            for item in scenarios
            if item.get("checks", {}).get("uses_only_registered_agents")
        ),
        "total_tokens": sum(
            int(item.get("usage", {}).get("total_tokens") or 0) for item in scenarios
        ),
        "median_latency_ms": round(median(latencies), 2) if latencies else 0,
        "rows": [
            {
                "id": item.get("id", ""),
                "parse_ok": bool(item.get("parse_ok")),
                "requirements_all_matched": bool(
                    item.get("checks", {}).get("requirements_all_matched")
                ),
                "artifacts_all_matched": bool(
                    item.get("checks", {}).get("artifacts_all_matched")
                ),
                "planning_verdict_matches": bool(
                    item.get("checks", {}).get("planning_verdict_matches")
                ),
                "uses_only_registered_agents": bool(
                    item.get("checks", {}).get("uses_only_registered_agents")
                ),
                "latency_ms": item.get("latency_ms", 0),
            }
            for item in scenarios
        ],
    }


def _summarize_distributed(report: JsonObject) -> JsonObject:
    scenarios = list(report.get("scenarios", []))
    metrics = dict(report.get("metrics") or {})
    latencies = [
        float(item.get("latency_ms", 0))
        for item in scenarios
        if isinstance(item.get("latency_ms"), int | float)
    ]
    return {
        "scenario_count": int(report.get("scenario_count") or len(scenarios)),
        "passed": bool(report.get("passed")),
        "passed_rows": sum(1 for item in scenarios if item.get("passed") is True),
        "median_latency_ms": round(median(latencies), 2) if latencies else 0,
        "lease_conflicts_observed": int(metrics.get("lease_conflicts_observed") or 0),
        "stale_write_rejections": int(metrics.get("stale_write_rejections") or 0),
        "duplicate_dispatch_count": int(metrics.get("duplicate_dispatch_count") or 0),
        "agent_side_duplicate_idempotency_key_requests": int(
            metrics.get("agent_side_duplicate_idempotency_key_requests") or 0
        ),
        "agent_side_repeated_session_task_effectful_executions": int(
            metrics.get("agent_side_repeated_session_task_effectful_executions") or 0
        ),
        "recovery_time_ms": metrics.get("recovery_time_ms"),
        "terminal_correctness": bool(metrics.get("terminal_correctness")),
        "rows": [
            {
                "id": item.get("scenario_id", item.get("id", "")),
                "passed": bool(item.get("passed")),
                "latency_ms": item.get("latency_ms", 0),
                "error": item.get("error", ""),
            }
            for item in scenarios
        ],
    }


def _summarize_baselines(report: JsonObject) -> JsonObject:
    rows: list[JsonObject] = []
    configurations = report.get("configurations", {})
    for key, label in (
        ("hybrid", "Hybrid CoordinationAgent"),
        ("rule_only", "Rule-only baseline"),
    ):
        config = configurations.get(key)
        if config:
            summary = dict(config.get("summary") or {})
            summary["configuration"] = label
            summary["model_id"] = ""
            rows.append(summary)

    for model_id, config in (
        configurations.get("llm_only", {}).get("models", {}) or {}
    ).items():
        summary = dict(config.get("summary") or {})
        summary["configuration"] = "LLM-only baseline"
        summary["model_id"] = model_id
        rows.append(summary)

    return {
        "generated_at": report.get("generated_at", ""),
        "scenario_count": int(report.get("scenario_count") or 0),
        "rows": rows,
    }


def _markdown_tables(summary: JsonObject) -> str:
    parts = [
        "# Thesis Evidence Summary",
        "",
        "## Deterministic Scenarios",
        "",
        "| Scenario | Reference | Observed | Authorization | Dispatches | Match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["deterministic"]["rows"]:
        parts.append(
            "| {id} | {reference_status} | {observed_status} | {authorization} | "
            "{dispatch_attempts} | {status_matches_reference} |".format(**row)
        )

    docker = summary.get("docker")
    if docker:
        parts.extend(
            [
                "",
                "## Docker A2A System Harness",
                "",
                "| Check | Reference | Observed | Passed | Latency ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in docker["rows"]:
            parts.append(
                "| {id} | {reference_status} | {observed_status} | {passed} | "
                "{latency_ms} |".format(**row)
            )

    distributed = summary.get("distributed")
    if distributed:
        parts.extend(
            [
                "",
                "## Distributed Coordinator Harness",
                "",
                "| Check | Passed | Latency ms | Error |",
                "|---|---:|---:|---|",
            ]
        )
        for row in distributed["rows"]:
            parts.append(
                "| {id} | {passed} | {latency_ms} | {error} |".format(**row)
            )
        parts.extend(
            [
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Lease conflicts observed | {distributed['lease_conflicts_observed']} |",
                f"| Stale write rejections | {distributed['stale_write_rejections']} |",
                f"| Duplicate dispatch count | {distributed['duplicate_dispatch_count']} |",
                "| Agent-side duplicate idempotency-key requests | "
                f"{distributed['agent_side_duplicate_idempotency_key_requests']} |",
                "| Agent-side repeated task effects | "
                f"{distributed['agent_side_repeated_session_task_effectful_executions']} |",
                f"| Recovery time ms | {distributed['recovery_time_ms']} |",
                f"| Terminal correctness | {distributed['terminal_correctness']} |",
            ]
        )

    baselines = summary.get("baselines")
    if baselines:
        parts.extend(
            [
                "",
                "## Paired Baseline Evaluation",
                "",
                "| Configuration | Model | Status match | Decision match | False accept | False refuse | Unauthorized dispatch rows | Resolved reqs | Unresolved reqs | Raw exact names | Resolved expected | Tokens |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in baselines["rows"]:
            parts.append(
                "| {configuration} | {model_id} | {status_matches_reference}/{scenario_count} | "
                "{decision_matches_reference}/{scenario_count} | {false_accepts} | "
                "{false_refusals} | {dispatch_without_symbolic_authorization} | "
                "{resolved_requirements} | {unresolved_requirements} | "
                "{exact_requirement_name_matches} | {resolved_expected_names} | "
                "{llm_tokens} |".format(
                    **row
                )
            )

    parts.extend(
        [
            "",
            "## Local LLM Reference Batches",
            "",
            "| Model | Parsed | Requirement names matched | Artifacts matched | Verdict matched | Registered agents only | Tokens | Median latency ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_id, row in summary.get("local_llm", {}).items():
        parts.append(
            "| {model_id} | {parse_ok}/{scenario_count} | "
            "{requirements_all_matched}/{scenario_count} | "
            "{artifacts_all_matched}/{scenario_count} | "
            "{planning_verdict_matches}/{scenario_count} | "
            "{uses_only_registered_agents}/{scenario_count} | "
            "{total_tokens} | {median_latency_ms} |".format(**row)
        )
    parts.append("")
    return "\n".join(parts)


def _latex_tables(summary: JsonObject) -> str:
    deterministic = summary["deterministic"]
    docker = summary.get("docker") or {}
    distributed = summary.get("distributed") or {}
    local_llm = summary.get("local_llm", {})
    lines = [
        "% Generated by unified-thesis-results.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Summary of preserved thesis evidence.}",
        "\\label{tab:preserved-evidence-summary}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Evidence source & Cases & Passed/matched & Median latency ms \\\\",
        "\\midrule",
        "Deterministic local scenarios & "
        f"{deterministic['scenario_count']} & "
        f"{deterministic['status_matches_reference']} & -- \\\\",
    ]
    if docker:
        lines.append(
            "Docker A2A system checks & "
            f"{docker['scenario_count']} & {docker['passed_rows']} & "
            f"{docker['median_latency_ms']} \\\\"
        )
    if distributed:
        lines.append(
            "Distributed coordinator checks & "
            f"{distributed['scenario_count']} & {distributed['passed_rows']} & "
            f"{distributed['median_latency_ms']} \\\\"
        )
    for model_id, row in local_llm.items():
        lines.append(
            f"{_latex_escape(model_id)} local LLM references & "
            f"{row['scenario_count']} & {row['parse_ok']} parsed & "
            f"{row['median_latency_ms']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    if local_llm:
        lines.extend(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Local LLM reference batch summary.}",
                "\\label{tab:local-llm-reference-summary}",
                "\\begingroup",
                "\\scriptsize",
                "\\setlength{\\tabcolsep}{3pt}",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{@{}lrrrrrrr@{}}",
                "\\toprule",
                "Model & Parsed & Req. names & Artifacts & Verdict & Agents & Tokens & Median ms \\\\",
                "\\midrule",
            ]
        )
        for model_id, row in local_llm.items():
            lines.append(
                f"{_latex_escape(model_id)} & "
                f"{row['parse_ok']}/{row['scenario_count']} & "
                f"{row['requirements_all_matched']}/{row['scenario_count']} & "
                f"{row['artifacts_all_matched']}/{row['scenario_count']} & "
                f"{row['planning_verdict_matches']}/{row['scenario_count']} & "
                f"{row['uses_only_registered_agents']}/{row['scenario_count']} & "
                f"{row['total_tokens']} & {row['median_latency_ms']} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}",
                "\\endgroup",
                "\\end{table}",
                "",
            ]
        )
    baselines = summary.get("baselines")
    if baselines:
        lines.extend(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                "\\caption{Paired baseline evaluation summary.}",
                "\\label{tab:baseline-summary}",
                "\\begingroup",
                "\\scriptsize",
                "\\setlength{\\tabcolsep}{3pt}",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{@{}llrrrrrrrrrr@{}}",
                "\\toprule",
                "Configuration & Model & Status & Decision & False accept & False refuse & Unauth. dispatch & Resolved & Unresolved & Raw exact & Resolved expected & Tokens \\\\",
                "\\midrule",
            ]
        )
        for row in baselines["rows"]:
            lines.append(
                f"{_latex_escape(str(row['configuration']))} & "
                f"{_latex_escape(str(row.get('model_id') or '--'))} & "
                f"{row['status_matches_reference']}/{row['scenario_count']} & "
                f"{row['decision_matches_reference']}/{row['scenario_count']} & "
                f"{row['false_accepts']} & {row['false_refusals']} & "
                f"{row['dispatch_without_symbolic_authorization']} & "
                f"{row['resolved_requirements']} & {row['unresolved_requirements']} & "
                f"{row['exact_requirement_name_matches']} & "
                f"{row['resolved_expected_names']} & {row['llm_tokens']} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}",
                "\\endgroup",
                "\\end{table}",
                "",
            ]
        )
    return "\n".join(lines)


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("&", "\\&")


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate thesis result summaries from preserved JSON evidence.",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("demo_runs/end_to_end_scenarios.json"),
    )
    parser.add_argument(
        "--docker",
        type=Path,
        default=Path("demo_runs/docker_system_report.json"),
    )
    parser.add_argument(
        "--distributed",
        type=Path,
        default=Path("demo_runs/distributed_system_report.json"),
    )
    parser.add_argument(
        "--local-llm-root",
        type=Path,
        default=Path("demo_runs/local_llm"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("demo_runs/baselines/baseline_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_runs/thesis_analysis"),
    )
    args = parser.parse_args(argv)

    summary = build_summary(
        scenarios_path=args.scenarios,
        docker_path=args.docker,
        distributed_path=args.distributed,
        local_llm_root=args.local_llm_root,
        baselines_path=args.baselines,
    )
    write_outputs(summary, args.output_dir)
    print(f"Wrote thesis analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
