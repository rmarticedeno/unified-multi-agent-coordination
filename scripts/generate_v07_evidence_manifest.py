"""Generate a hash-complete v0.7 evidence manifest without rewriting history."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _inventory(repository: Path, paths: list[Path]) -> list[JsonObject]:
    files: set[Path] = set()
    for item in paths:
        resolved = item if item.is_absolute() else repository / item
        if resolved.is_file():
            files.add(resolved.resolve())
        elif resolved.is_dir():
            files.update(path.resolve() for path in resolved.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(resolved)
    return [
        {
            "path": path.relative_to(repository).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(files)
    ]


def build(args: argparse.Namespace) -> JsonObject:
    repository = args.repository.resolve()
    run = args.run.resolve()
    corpus = args.corpus.resolve()
    protocol_path = corpus / "protocol-lock.json"
    protocol = _load(protocol_path)
    provenance = _load(run / "provenance.json")
    analysis = _load(args.analysis.resolve())
    collections = [
        _load(path)
        for path in sorted(run.glob("collection-m*.json"))
    ]
    roots = [
        run,
        corpus,
        args.benchmark.resolve(),
        args.semantic.resolve(),
        args.consensus.resolve(),
        args.a2a.resolve(),
        args.coverage.resolve(),
        args.figures.resolve(),
    ]
    artifacts = _inventory(repository, roots)
    raw_outputs = len(list(run.glob("o/*/m*/s*/*.json")))
    criteria = analysis.get("predeclared_criteria")
    return {
        "schema_version": "0.7.0",
        "classification": "frozen_production_path_confirmatory_evidence",
        "generated_from_git_head": _git("rev-parse", "HEAD"),
        "workspace_dirty": bool(_git("status", "--porcelain")),
        "evaluated_source_commit": provenance.get("git_sha"),
        "protocol_lock": {
            "path": protocol_path.relative_to(repository).as_posix(),
            "sha256": _sha256(protocol_path),
            "protocol_hash": provenance.get("protocol_hash"),
            "corpus_hash": protocol.get("corpus_hash"),
            "prompt_version": protocol.get("prompt_version"),
            "source_sha256": protocol.get("source_sha256"),
        },
        "immutable_provenance": provenance,
        "known_metadata_inconsistencies": [
            {
                "field": "immutable_provenance.prompt_version",
                "recorded": provenance.get("prompt_version"),
                "authoritative_protocol_lock": protocol.get("prompt_version"),
                "impact": (
                    "label-only provenance defect; protocol hash, source hashes, "
                    "schemas, and raw prompts remain preserved"
                ),
            }
        ],
        "model_collections": collections,
        "raw_output_count": raw_outputs,
        "expected_output_count": protocol.get("expected_outputs"),
        "matrix_complete": raw_outputs == protocol.get("expected_outputs")
        and len(collections) == 3
        and all(item.get("complete") for item in collections),
        "analysis": {
            "path": args.analysis.resolve().relative_to(repository).as_posix(),
            "sha256": _sha256(args.analysis.resolve()),
            "criteria": criteria,
            "all_criteria_met": analysis.get("all_criteria_met"),
            "supplementary_semantic_path": args.semantic.resolve()
            .relative_to(repository)
            .as_posix(),
            "supplementary_semantic_sha256": _sha256(args.semantic.resolve()),
        },
        "deterministic_benchmark": {
            "path": args.benchmark.resolve().relative_to(repository).as_posix(),
            "sha256": _sha256(args.benchmark.resolve()),
        },
        "a2a_replay": {
            "path": args.a2a.resolve().relative_to(repository).as_posix(),
            "sha256": _sha256(args.a2a.resolve()),
        },
        "consensus_v4": {
            "path": args.consensus.resolve().relative_to(repository).as_posix(),
            "sha256": _sha256(args.consensus.resolve()),
        },
        "validation": {
            "test_count": args.test_count,
            "production_branch_coverage_percent": args.production_coverage,
            "coverage_json": args.coverage.resolve().relative_to(repository).as_posix(),
            "coverage_sha256": _sha256(args.coverage.resolve()),
        },
        "figure_provenance": {
            "generator": "scripts/generate_v07_figures.py",
            "input_analysis_sha256": _sha256(args.analysis.resolve()),
            "input_benchmark_sha256": _sha256(args.benchmark.resolve()),
            "input_consensus_sha256": _sha256(args.consensus.resolve()),
            "directory": args.figures.resolve().relative_to(repository).as_posix(),
        },
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/v0.7"))
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--a2a", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--production-coverage", type=float, required=True)
    parser.add_argument("--output", type=Path, default=Path("e/v7/evidence-manifest.json"))
    args = parser.parse_args()
    payload = build(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact_count": payload["artifact_count"],
                "matrix_complete": payload["matrix_complete"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
