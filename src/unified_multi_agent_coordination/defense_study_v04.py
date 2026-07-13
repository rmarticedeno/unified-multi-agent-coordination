"""Label-hidden v0.4 collection for the constrained linguistic bridge."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from .defense_study import (
    ENDPOINT,
    MODELS,
    SEEDS,
    _installed_models,
    _lm_studio_schema,
    _provenance,
    _run,
    validate_frozen_labels,
)
from .models import LinguisticPlanDraft
from .models import AgentRegistryEntry, ProblemRequest
from .plan_hydration import PlanHydrator

PROMPT_VERSION = "defense-v0.4-bridge-v1"


class LinguisticBridgeOutput(BaseModel):
    draft: LinguisticPlanDraft


def validate_prerequisites(corpus_root: Path, check_runtime: bool = True) -> dict[str, Any]:
    provenance = validate_frozen_labels(corpus_root)
    public = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    if public.get("version") != "0.4":
        raise RuntimeError("The bridge study requires corpus v0.4.")
    if not check_runtime:
        return {"case_count": 36, "corpus_hash": public["corpus_hash"]}
    missing = set(MODELS) - _installed_models()
    if missing:
        raise RuntimeError("Missing pinned models: " + ", ".join(sorted(missing)))
    httpx.get(f"{ENDPOINT}/models", timeout=5).raise_for_status()
    return {
        "case_count": 36,
        "corpus_hash": public["corpus_hash"],
        "annotation_type": provenance["annotation_type"],
        "models": list(MODELS),
    }


def _prompt(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": (
            "Return a non-executable LinguisticPlanDraft. Select every admitted "
            "requirement_id exactly once and only its declared capability_id. Dependencies "
            "may reference only admitted requirement IDs. Report ambiguous terms in "
            "unresolved_terms. Do not create task IDs, agents, trust, credentials, contracts, "
            "schemas, artifacts, or authority. Reference labels are unavailable."
        )},
        {"role": "user", "content": json.dumps({
            "request_text": case["request_text"],
            "admitted_request": case["request"],
            "registry_snapshot": case["registry_snapshot"],
        }, separators=(",", ":"))},
    ]


def _completion(
    client: httpx.Client, model: str, seed: int, messages: list[dict[str, str]]
) -> dict[str, Any]:
    schema = _lm_studio_schema(LinguisticBridgeOutput.model_json_schema())
    response = client.post(f"{ENDPOINT}/chat/completions", json={
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": seed,
        "max_tokens": 2048,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "linguistic_bridge_output", "strict": True, "schema": schema,
        }},
    })
    if response.is_error:
        raise RuntimeError(f"LM Studio completion failed ({response.status_code}): {response.text}")
    return response.json()


def _repair_prompt(
    messages: list[dict[str, str]], draft: LinguisticPlanDraft, issues: list[dict[str, Any]]
) -> list[dict[str, str]]:
    return [*messages, {"role": "assistant", "content": LinguisticBridgeOutput(draft=draft).model_dump_json()}, {
        "role": "user",
        "content": "One repair is allowed. Correct only these public hydration errors: " + json.dumps(issues),
    }]


def collect(corpus_root: Path, output_root: Path, resume_root: Path | None = None) -> Path:
    prerequisites = validate_prerequisites(corpus_root)
    corpus = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    if resume_root is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + prerequisites["corpus_hash"][:10]
        run_root = output_root / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        provenance = _provenance(prerequisites["corpus_hash"])
        provenance["prompt_version"] = PROMPT_VERSION
        (run_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    else:
        run_root = resume_root
        if (run_root / "collection-complete.json").exists():
            raise RuntimeError("Completed immutable runs cannot be resumed.")
        provenance = json.loads((run_root / "provenance.json").read_text(encoding="utf-8"))
        if provenance.get("corpus_hash") != prerequisites["corpus_hash"]:
            raise RuntimeError("Resume corpus hash mismatch.")
    with httpx.Client(timeout=300) as client:
        for model in MODELS:
            _run(["lms", "unload", "--all"], check=False)
            _run(["lms", "load", model, "--identifier", model, "--context-length", "16384", "--parallel", "1", "--yes"])
            for seed in SEEDS:
                batch = run_root / model.replace("/", "__") / f"seed-{seed}"
                batch.mkdir(parents=True, exist_ok=True)
                for case in corpus["cases"]:
                    path = batch / f"{case['case_id']}.json"
                    if path.exists():
                        continue
                    messages = _prompt(case)
                    started = time.perf_counter()
                    raw = _completion(client, model, seed, messages)
                    latency = (time.perf_counter() - started) * 1000
                    if str(raw.get("model") or "") != model:
                        raise RuntimeError("LM Studio returned a different model identity.")
                    parsed = LinguisticBridgeOutput.model_validate_json(
                        raw["choices"][0]["message"]["content"]
                    )
                    request = ProblemRequest.model_validate(case["request"])
                    registry = [AgentRegistryEntry.model_validate(item) for item in case["registry_snapshot"]]
                    hydration = PlanHydrator().hydrate(request, registry, parsed.draft)
                    repairable = {
                        "duplicate_requirement", "missing_requirement", "unknown_requirement",
                        "invalid_dependency", "dependency_cycle", "unresolved_term",
                    }
                    repair_raw = None
                    repair_latency = 0.0
                    repair_issues = [
                        issue.model_dump(mode="json") for issue in hydration.issues
                        if issue.code in repairable
                    ]
                    if repair_issues:
                        repair_messages = _repair_prompt(messages, parsed.draft, repair_issues)
                        repair_started = time.perf_counter()
                        repair_raw = _completion(client, model, seed, repair_messages)
                        repair_latency = (time.perf_counter() - repair_started) * 1000
                        parsed = LinguisticBridgeOutput.model_validate_json(
                            repair_raw["choices"][0]["message"]["content"]
                        )
                    record = {
                        "case_id": case["case_id"], "model_id": model, "seed": seed,
                        "temperature": 0, "prompt": messages, "raw_response": raw,
                        "parsed_object": parsed.model_dump(mode="json"),
                        "linguistic_latency_ms": latency,
                        "repair_attempted": repair_raw is not None,
                        "repair_latency_ms": repair_latency,
                        "repair_raw_response": repair_raw,
                        "evaluation_status": "raw_output_collected_not_scored",
                    }
                    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            _run(["lms", "unload", "--all"], check=False)
    (run_root / "collection-complete.json").write_text(json.dumps({
        "complete": True, "expected_outputs": len(MODELS) * len(SEEDS) * 36,
        "labels_loaded_during_collection": False,
    }, indent=2), encoding="utf-8")
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.4"))
    parser.add_argument("--output", type=Path, default=Path("demo_runs/v0.4"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.collect or args.resume:
        print(collect(args.corpus, args.output, args.resume))
    else:
        print(json.dumps(validate_prerequisites(args.corpus, not args.check), indent=2))


if __name__ == "__main__":
    main()
