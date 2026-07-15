"""Generate a deterministic SHA-256 inventory for packaged defense evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_ROOTS = (
    "e/v5",
    "demo_runs/consensus/20260715T014124Z-7a31b42-v1",
    "demo_runs/consensus/20260715T022727Z-3f8093a-v2",
    "demo_runs/deterministic/20260715T0255Z-3f8093a",
    "demo_runs/system/20260715T0212Z-7a31b42",
    "demo_runs/postgres/20260715T0214Z-7a31b42-failed",
    "demo_runs/postgres/20260715T0225Z-3f8093a-passed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(repository: Path, roots: tuple[str, ...]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for relative_root in roots:
        root = repository / relative_root
        if not root.is_dir():
            raise FileNotFoundError(f"missing evidence root: {relative_root}")
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            entries.append(
                {
                    "path": path.relative_to(repository).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return {
        "schema_version": "1.0.0",
        "evaluated_source_commit": "3f8093af1ecc7c49312ac34856bba825bd81f381",
        "roots": list(roots),
        "artifact_count": len(entries),
        "artifacts": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("e/artifact-manifest.json"))
    args = parser.parse_args()
    repository = args.repository.resolve()
    payload = build_manifest(repository, DEFAULT_ROOTS)
    output = repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
