"""Gated local-model collection protocol for the 36-case defense study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .models import ProblemRequest, SolutionProposal


MODELS = ("qwen/qwen3-1.7b", "google/gemma-4-e2b", "qwen/qwen3-8b")
SEEDS = (11, 22, 33, 44, 55)
ENDPOINT = "http://127.0.0.1:1234/v1"
PROMPT_VERSION = "defense-v0.2.0"


class LinguisticBatchOutput(BaseModel):
    interpreted_request: ProblemRequest
    candidate_plans: list[SolutionProposal] = Field(min_length=1, max_length=3)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def _installed_models() -> set[str]:
    result = _run(["lms", "ls", "--json"])
    return {str(item.get("modelKey")) for item in json.loads(result.stdout)}


def validate_prerequisites(corpus_root: Path, *, require_signoff: bool) -> dict[str, Any]:
    public = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    signoff = json.loads((corpus_root / "label-signoff.json").read_text(encoding="utf-8"))
    if len(public.get("cases", [])) != 36:
        raise RuntimeError("The public corpus must contain exactly 36 cases.")
    if signoff.get("corpus_hash") != public.get("corpus_hash"):
        raise RuntimeError("Label sign-off hash does not match the public corpus.")
    if require_signoff and not all(
        [
            signoff.get("approved") is True,
            signoff.get("reviewer_role"),
            signoff.get("review_date"),
        ]
    ):
        raise RuntimeError("Independent label approval is incomplete; model collection is blocked.")
    missing = set(MODELS) - _installed_models()
    if missing:
        raise RuntimeError("Required LM Studio models are missing: " + ", ".join(sorted(missing)))
    with httpx.Client(timeout=5.0) as client:
        response = client.get(f"{ENDPOINT}/models")
        response.raise_for_status()
    return {
        "case_count": 36,
        "corpus_hash": public["corpus_hash"],
        "signoff_approved": signoff.get("approved") is True,
        "models_installed": list(MODELS),
        "server_reachable": True,
    }


def _provenance(corpus_hash: str) -> dict[str, Any]:
    git_sha = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).stdout.strip())
    docker = _run(
        ["docker", "image", "inspect", "unified-multi-agent-coordination:local", "--format", "{{.Id}}"],
        check=False,
    )
    return {
        "git_sha": git_sha,
        "dirty_state": dirty,
        "corpus_hash": corpus_hash,
        "prompt_version": PROMPT_VERSION,
        "dependency_lock_hash": _sha256(Path("uv.lock")),
        "docker_image_digest": docker.stdout.strip() if docker.returncode == 0 else "unavailable",
        "python": platform.python_version(),
        "a2a_version": _package_version("a2a-sdk"),
        "postgresql_version": "16-alpine (distributed fixture)",
        "lm_studio_cli": _run(["lms", "--version"], check=False).stdout.strip(),
        "os": platform.platform(),
        "cpu": platform.processor() or os.getenv("PROCESSOR_IDENTIFIER", "unknown"),
        "ram_bytes": _windows_total_ram(),
        "accelerator": "recorded by LM Studio load/runtime logs",
    }


def _package_version(name: str) -> str:
    result = _run(["uv", "pip", "show", name], check=False)
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            return line.partition(":")[2].strip()
    return "unknown"


def _windows_total_ram() -> int | str:
    if platform.system() != "Windows":
        return "not-collected"
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.total_physical)
    except Exception:
        return "unknown"


def _prompt(case: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Interpret the request and return one to three typed candidate plans. "
                "Plans are non-authoritative. Use only exact capability_id and agent_id values "
                "from the registry. Do not infer trust or authorization."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "request": case["request_text"],
                    "registry_snapshot": case["registry_snapshot"],
                },
                separators=(",", ":"),
            ),
        },
    ]


def _completion(client: httpx.Client, model: str, seed: int, messages: list[dict[str, str]]) -> dict[str, Any]:
    response = client.post(
        f"{ENDPOINT}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "linguistic_batch_output",
                    "strict": True,
                    "schema": LinguisticBatchOutput.model_json_schema(),
                },
            },
        },
    )
    response.raise_for_status()
    return response.json()


def collect(corpus_root: Path, output_root: Path) -> Path:
    prerequisites = validate_prerequisites(corpus_root, require_signoff=True)
    corpus = json.loads((corpus_root / "public/cases.json").read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + prerequisites["corpus_hash"][:10]
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "provenance.json").write_text(
        json.dumps(_provenance(prerequisites["corpus_hash"]), indent=2), encoding="utf-8"
    )
    with httpx.Client(timeout=300.0) as client:
        for model in MODELS:
            _run(["lms", "unload", "--all"], check=False)
            _run(["lms", "load", model, "--identifier", model, "--context-length", "16384", "--parallel", "1", "--yes"])
            catalog = client.get(f"{ENDPOINT}/models").json()
            if not any(item.get("id") == model for item in catalog.get("data", [])):
                raise RuntimeError(f"Loaded model catalog does not expose exact ID {model}.")
            for seed in SEEDS:
                sentinel = _completion(
                    client,
                    model,
                    seed,
                    [{"role": "user", "content": "Return a typed no-op coordination request and plan."}],
                )
                batch_dir = run_root / model.replace("/", "__") / f"seed-{seed}"
                batch_dir.mkdir(parents=True)
                (batch_dir / "sentinel.json").write_text(json.dumps(sentinel, indent=2), encoding="utf-8")
                for case in corpus["cases"]:
                    messages = _prompt(case)
                    started = time.perf_counter()
                    raw = _completion(client, model, seed, messages)
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    returned_model = str(raw.get("model") or "")
                    if returned_model != model:
                        raise RuntimeError(f"Response model {returned_model!r} != requested {model!r}.")
                    content = raw["choices"][0]["message"]["content"]
                    parsed = LinguisticBatchOutput.model_validate_json(content)
                    record = {
                        "case_id": case["case_id"],
                        "model_id": model,
                        "seed": seed,
                        "temperature": 0,
                        "backend_seed_honored": "not_verifiable_from_response",
                        "prompt": messages,
                        "raw_response": raw,
                        "parsed_object": parsed.model_dump(mode="json"),
                        "latency_ms": elapsed_ms,
                        "evaluation_status": "raw_output_collected_not_scored",
                    }
                    (batch_dir / f"{case['case_id']}.json").write_text(
                        json.dumps(record, indent=2), encoding="utf-8"
                    )
            _run(["lms", "unload", "--all"], check=False)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.2"))
    parser.add_argument("--output", type=Path, default=Path("demo_runs/v0.2"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        print(collect(args.corpus, args.output))
    else:
        print(json.dumps(validate_prerequisites(args.corpus, require_signoff=False), indent=2))


if __name__ == "__main__":
    main()
