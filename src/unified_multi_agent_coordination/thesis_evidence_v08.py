"""Generate versioned thesis macros and a compact figure from validated v0.8 analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _percent(value: float) -> str:
    return f"{100 * value:.2f}"


def generate(analysis_path: Path, output_dir: Path) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    phase = analysis["phase"]
    expected = 576 if phase == "development" else 768
    if analysis.get("observation_count") != expected:
        raise RuntimeError(
            f"Refusing incomplete v0.8 {phase} evidence: expected {expected} observations."
        )
    metrics = analysis["metrics_primary_seed"]
    hybrid = metrics["production_hybrid_v08"]
    direct = metrics["direct_llm_v08"]
    prefix = "VEightDevelopment" if phase == "development" else "VEightConfirmatory"
    macros = (
        "% Generated from validated v0.8 analysis; do not edit by hand.\n"
        f"\\newcommand{{\\{prefix}OutputCount}}{{{expected}}}\n"
        f"\\newcommand{{\\{prefix}HybridBalancedAccuracy}}{{{_percent(hybrid['balanced_accuracy'])}}}\n"
        f"\\newcommand{{\\{prefix}HybridRecall}}{{{_percent(hybrid['feasible_recall'])}}}\n"
        f"\\newcommand{{\\{prefix}HybridUnsafe}}{{{hybrid['unsafe_acceptances']}}}\n"
        f"\\newcommand{{\\{prefix}DirectBalancedAccuracy}}{{{_percent(direct['balanced_accuracy'])}}}\n"
        f"\\newcommand{{\\{prefix}DirectUnsafe}}{{{direct['unsafe_acceptances']}}}\n"
    )
    if phase == "development":
        status = analysis["qualification"]["status_macro_f1"]
        macros += f"\\newcommand{{\\{prefix}StatusMacroFOne}}{{{_percent(status)}}}\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    macro_path = output_dir / f"v08_{phase}_macros.tex"
    macro_path.write_text(macros, encoding="utf-8")
    figure_path = output_dir / f"v08-{phase}-safety-utility.tex"
    figure_path.write_text(
        "\\begin{tikzpicture}[x=0.025\\textwidth,y=0.035cm]\n"
        f"\\fill[black!70] (0,0) rectangle ({100 * hybrid['balanced_accuracy']:.2f},6);\n"
        f"\\fill[black!35] (0,10) rectangle ({100 * direct['balanced_accuracy']:.2f},16);\n"
        "\\node[anchor=east] at (0,3) {Hybrid};\n"
        "\\node[anchor=east] at (0,13) {Direct};\n"
        "\\node[anchor=west] at (102,8) {balanced accuracy (\\%)};\n"
        "\\end{tikzpicture}\n",
        encoding="utf-8",
    )
    manifest = {
        "version": "0.8.0",
        "phase": phase,
        "analysis": str(analysis_path),
        "analysis_sha256": hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "observation_count": expected,
        "macros": str(macro_path),
        "figure": str(figure_path),
    }
    (output_dir / f"v08-{phase}-evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.analysis, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
