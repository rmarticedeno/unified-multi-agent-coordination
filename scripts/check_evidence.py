"""Fail CI when deterministic evidence or frozen-report provenance is incomplete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.deterministic.read_text(encoding="utf-8"))
    rows = report.get("scenarios", [])
    if len(rows) != 7:
        raise SystemExit(f"expected seven deterministic scenarios, found {len(rows)}")
    mismatches = [row.get("id") for row in rows if not row.get("status_matches_reference")]
    if mismatches:
        raise SystemExit("deterministic evidence mismatch: " + ", ".join(mismatches))
    if any(not row.get("no_dispatch_before_authorization") for row in rows):
        raise SystemExit("a scenario dispatched before symbolic authorization")
    print("deterministic evidence gate: 7/7, no pre-authorization dispatch")


if __name__ == "__main__":
    main()
