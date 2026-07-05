"""Batched local-LLM reference checks for thesis evidence.

The checks intentionally use one model per process invocation. They ask a local
OpenAI-compatible model for a non-authoritative linguistic interpretation and
candidate decomposition, then compare that output with the deterministic thesis
scenario fixtures. Symbolic authorization remains outside the model.
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
from typing import Any

import httpx

from .coordination_sdk import CoordinationSdk
from .end_to_end_scenarios import ScenarioDefinition, scenario_definitions
from .models import AgentRegistryEntry


JsonObject = dict[str, Any]

DEFAULT_ENDPOINT = "http://127.0.0.1:1234"
ALLOWED_MODELS = ("qwen/qwen3-1.7b", "google/gemma-4-e2b")
PROMPT_VERSION = "local-llm-reference-v1"
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.0

REFERENCE_SCHEMA: JsonObject = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "matched_agent_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "matched_agent_id", "reason"],
            },
        },
        "required_artifacts": {"type": "array", "items": {"type": "string"}},
        "candidate_tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "requirement_name": {"type": "string"},
                    "assigned_agent_id": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["requirement_name", "assigned_agent_id", "depends_on"],
            },
        },
        "planning_verdict": {
            "type": "string",
            "enum": ["feasible", "infeasible", "unknown"],
        },
        "explanation": {"type": "string"},
    },
    "required": [
        "requirements",
        "required_artifacts",
        "candidate_tasks",
        "planning_verdict",
        "explanation",
    ],
}


@dataclass(frozen=True)
class ScenarioInput:
    scenario: ScenarioDefinition
    registry: list[AgentRegistryEntry]


async def _scenario_inputs() -> list[ScenarioInput]:
    inputs: list[ScenarioInput] = []
    for scenario in scenario_definitions():
        sdk = CoordinationSdk()
        await scenario.setup(sdk)
        registry = await sdk.registry_snapshot(refresh=False)
        await _close_sdk_client(sdk)
        inputs.append(ScenarioInput(scenario=scenario, registry=registry))
    return inputs


async def _close_sdk_client(sdk: CoordinationSdk) -> None:
    close = getattr(sdk.http_client, "aclose", None)
    if close is not None:
        await close()


def run_reference_batch(
    *,
    model_id: str,
    endpoint: str = DEFAULT_ENDPOINT,
    output_root: Path = Path("demo_runs/local_llm"),
    timeout_s: float = 120.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    run_id: str | None = None,
) -> JsonObject:
    """Run every local-LLM reference check for one model and write a report."""
    if model_id not in ALLOWED_MODELS:
        raise ValueError(
            f"Model {model_id!r} is not allowed. Allowed: {', '.join(ALLOWED_MODELS)}."
        )

    scenarios = asyncio.run(_scenario_inputs())
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_dir = output_root / _safe_model_dir(model_id) / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=endpoint, timeout=timeout_s) as client:
        catalog = _model_catalog(client)
        if model_id not in catalog:
            raise RuntimeError(
                f"Endpoint {endpoint} does not advertise required model {model_id!r}."
            )
        _sentinel_check(
            client=client,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        rows = [
            _run_one_scenario(
                client=client,
                endpoint=endpoint,
                model_id=model_id,
                item=item,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for item in scenarios
        ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model_id": model_id,
        "allowed_models": list(ALLOWED_MODELS),
        "run_id": run_id,
        "prompt_version": PROMPT_VERSION,
        "decoding": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": "json_schema",
        },
        "model_catalog": catalog,
        "scenario_count": len(rows),
        "summary": _summarize(rows),
        "scenarios": rows,
    }
    report_path = model_dir / "reference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest_path = output_root / _safe_model_dir(model_id) / "latest.json"
    latest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"{model_id}: wrote {len(rows)} local-LLM reference checks to {report_path}"
    )
    return report


def _model_catalog(client: httpx.Client) -> list[str]:
    response = client.get("/v1/models")
    response.raise_for_status()
    payload = response.json()
    return sorted(item.get("id", "") for item in payload.get("data", []))


def _sentinel_check(
    *,
    client: httpx.Client,
    model_id: str,
    max_tokens: int,
    temperature: float,
) -> None:
    response = _completion(
        client=client,
        model_id=model_id,
        messages=[
            {
                "role": "system",
                "content": "Return schema-valid JSON only.",
            },
            {
                "role": "user",
                "content": "Return ok=true for this model verification check.",
            },
        ],
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        max_tokens=min(max_tokens, 128),
        temperature=temperature,
    )
    _verify_response_model(response, model_id)
    content = _response_content(response)
    payload = json.loads(content)
    if payload.get("ok") is not True:
        raise RuntimeError(f"Sentinel check for {model_id} returned {content!r}.")


def _run_one_scenario(
    *,
    client: httpx.Client,
    endpoint: str,
    model_id: str,
    item: ScenarioInput,
    max_tokens: int,
    temperature: float,
) -> JsonObject:
    scenario = item.scenario
    messages = _scenario_messages(scenario, item.registry)
    started = time.perf_counter()
    error = ""
    raw_content = ""
    parsed: JsonObject | None = None
    usage: JsonObject = {}
    response_model = ""
    system_fingerprint = ""
    try:
        response = _completion(
            client=client,
            model_id=model_id,
            messages=messages,
            schema=REFERENCE_SCHEMA,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        _verify_response_model(response, model_id)
        response_model = response.get("model", "")
        system_fingerprint = response.get("system_fingerprint", "")
        usage = response.get("usage", {})
        raw_content = _response_content(response)
        parsed = json.loads(raw_content)
    except Exception as exc:  # pragma: no cover - intentionally serialized evidence
        error = str(exc)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    checks = _evaluate_model_output(parsed, scenario, item.registry)
    return {
        "id": scenario.scenario_id,
        "title": scenario.title,
        "showcase": scenario.showcase,
        "reference_status": scenario.reference_status,
        "reference_planning_verdict": _reference_planning_verdict(scenario),
        "model_id": model_id,
        "endpoint": endpoint,
        "response_model": response_model,
        "system_fingerprint": system_fingerprint,
        "latency_ms": latency_ms,
        "parse_ok": parsed is not None and not error,
        "error": error,
        "checks": checks,
        "usage": usage,
        "model_output": parsed,
        "raw_content": raw_content,
    }


def _completion(
    *,
    client: httpx.Client,
    model_id: str,
    messages: list[JsonObject],
    schema: JsonObject,
    max_tokens: int,
    temperature: float,
) -> JsonObject:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "local_llm_reference",
                    "strict": True,
                    "schema": schema,
                },
            },
        },
    )
    response.raise_for_status()
    return response.json()


def _verify_response_model(response: JsonObject, model_id: str) -> None:
    response_model = response.get("model")
    fingerprint = response.get("system_fingerprint")
    if response_model != model_id and fingerprint != model_id:
        raise RuntimeError(
            "Local LLM response came from an unexpected model: "
            f"model={response_model!r}, system_fingerprint={fingerprint!r}, "
            f"expected={model_id!r}."
        )


def _response_content(response: JsonObject) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Completion response did not include choices.")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Completion response did not include message content.")
    return content


def _scenario_messages(
    scenario: ScenarioDefinition,
    registry: list[AgentRegistryEntry],
) -> list[JsonObject]:
    registry_view = [
        {
            "agent_id": agent.agent_id,
            "agent_kind": agent.agent_kind,
            "status": agent.status,
            "trust_level": agent.trust_level,
            "skills": [
                {
                    "name": skill.name,
                    "input_modes": skill.input_modes,
                    "output_modes": skill.output_modes,
                    "auxiliary_eligible": skill.auxiliary_eligible,
                    "required_trust_level": skill.required_trust_level,
                    "validation_contract": skill.validation_contract,
                }
                for skill in agent.skills
            ],
        }
        for agent in registry
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a linguistic preprocessor for a thesis prototype. "
                "You may interpret the user request and propose a candidate "
                "decomposition, but you must not authorize execution. Use only "
                "the supplied registry. If a required capability is absent, set "
                "planning_verdict to infeasible. Return JSON that matches the "
                "schema exactly."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "scenario_id": scenario.scenario_id,
                    "user_request": scenario.user_request,
                    "registry": registry_view,
                },
                indent=2,
            ),
        },
    ]


def _evaluate_model_output(
    parsed: JsonObject | None,
    scenario: ScenarioDefinition,
    registry: list[AgentRegistryEntry],
) -> JsonObject:
    if parsed is None:
        return {
            "requirements_all_matched": False,
            "artifacts_all_matched": False,
            "planning_verdict_matches": False,
            "uses_only_registered_agents": False,
        }

    expected_requirements = {_norm(req.name) for req in scenario.request.requirements}
    actual_requirements = {
        _norm(item.get("name", "")) for item in parsed.get("requirements", [])
    }
    expected_artifacts = {_norm(item) for item in scenario.request.required_artifacts}
    actual_artifacts = {_norm(item) for item in parsed.get("required_artifacts", [])}
    registry_ids = {agent.agent_id for agent in registry}
    assigned_agents = [
        task.get("assigned_agent_id", "")
        for task in parsed.get("candidate_tasks", [])
        if task.get("assigned_agent_id", "")
    ]
    return {
        "requirements_all_matched": expected_requirements.issubset(
            actual_requirements
        ),
        "matched_requirement_count": len(expected_requirements & actual_requirements),
        "expected_requirement_count": len(expected_requirements),
        "artifacts_all_matched": expected_artifacts.issubset(actual_artifacts),
        "matched_artifact_count": len(expected_artifacts & actual_artifacts),
        "expected_artifact_count": len(expected_artifacts),
        "planning_verdict_matches": parsed.get("planning_verdict")
        == _reference_planning_verdict(scenario),
        "uses_only_registered_agents": all(agent in registry_ids for agent in assigned_agents),
    }


def _reference_planning_verdict(scenario: ScenarioDefinition) -> str:
    return "infeasible" if scenario.reference_status == "infeasible" else "feasible"


def _summarize(rows: list[JsonObject]) -> JsonObject:
    usage = [row.get("usage", {}) for row in rows]
    return {
        "parse_ok": sum(1 for row in rows if row.get("parse_ok")),
        "requirements_all_matched": sum(
            1 for row in rows if row.get("checks", {}).get("requirements_all_matched")
        ),
        "artifacts_all_matched": sum(
            1 for row in rows if row.get("checks", {}).get("artifacts_all_matched")
        ),
        "planning_verdict_matches": sum(
            1 for row in rows if row.get("checks", {}).get("planning_verdict_matches")
        ),
        "uses_only_registered_agents": sum(
            1 for row in rows if row.get("checks", {}).get("uses_only_registered_agents")
        ),
        "total_prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
        "total_completion_tokens": sum(
            int(item.get("completion_tokens") or 0) for item in usage
        ),
        "total_tokens": sum(int(item.get("total_tokens") or 0) for item in usage),
    }


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _safe_model_dir(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run thesis local-LLM reference checks for one model batch.",
    )
    parser.add_argument("--model", required=True, choices=ALLOWED_MODELS)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("demo_runs/local_llm"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args(argv)

    run_reference_batch(
        model_id=args.model,
        endpoint=args.endpoint,
        output_root=args.output_root,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
