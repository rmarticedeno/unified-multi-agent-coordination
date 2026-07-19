"""Generate thesis TikZ figures directly from v0.7 and consensus JSON evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def _load(path: Path) -> JsonObject:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _model_label(model: str) -> str:
    return {
        "qwen/qwen3-1.7b": "Qwen 1.7B",
        "google/gemma-4-e2b": "Gemma E2B",
        "qwen/qwen3-8b": "Qwen 8B",
    }.get(model, model)


def _safety_utility(analysis: JsonObject) -> str:
    by_model = analysis["metrics_by_model_primary_seed"]
    points: list[str] = []
    legend: list[str] = []
    markers = ("circle", "rectangle", "diamond")
    for index, (model, views) in enumerate(by_model.items()):
        label = _model_label(model)
        marker = markers[index % len(markers)]
        for arm, fill in (("hybrid", "black"), ("direct", "white")):
            metric = views[arm]
            recall = float(metric["feasible_recall"])
            unsafe = float(metric["unsafe_acceptances"]) / max(
                int(metric["observations"]), 1
            )
            points.append(
                rf"\node[draw,fill={fill},{marker},minimum size=5pt,inner sep=1pt] "
                rf"at ({recall:.4f},{unsafe:.4f}) {{}};"
            )
        legend.append(
            rf"\node[anchor=west] at (1.04,{0.90 - index * 0.09:.2f}) "
            rf"{{\scriptsize {_escape(label)}}};"
        )
    return "\n".join(
        [
            r"\begin{tikzpicture}[x=8.0cm,y=5.2cm]",
            r"\draw[->] (0,0) -- (1.02,0) node[right]{\scriptsize feasible recall};",
            r"\draw[->] (0,0) -- (0,1.02) node[above]{\scriptsize unsafe-acceptance rate};",
            r"\foreach \x in {0,0.25,0.5,0.75,1} {"
            r"\draw[black!25] (\x,0)--(\x,1);"
            r"\node[below] at (\x,0){\tiny \x};}",
            r"\foreach \y in {0,0.25,0.5,0.75,1} {"
            r"\draw[black!25] (0,\y)--(1,\y);"
            r"\node[left] at (0,\y){\tiny \y};}",
            *points,
            r"\node[draw,fill=black,circle,minimum size=5pt,inner sep=1pt] at (1.06,0.98) {};",
            r"\node[anchor=west] at (1.09,0.98){\scriptsize hybrid};",
            r"\node[draw,fill=white,circle,minimum size=5pt,inner sep=1pt] at (1.06,0.94) {};",
            r"\node[anchor=west] at (1.09,0.94){\scriptsize direct};",
            *legend,
            r"\end{tikzpicture}",
            "",
        ]
    )


def _heatmap(analysis: JsonObject) -> str:
    rows = [
        row
        for row in analysis["rows"]
        if row["arm"] == "production_hybrid_v07" and row["seed"] == 11
    ]
    models = list(analysis["models"])
    categories = sorted({row["category"] for row in rows})
    grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["category"])].append(bool(row["correct"]))
    lines = [r"\begin{tikzpicture}[x=1.05cm,y=0.72cm]"]
    for x, category in enumerate(categories):
        lines.append(
            rf"\node[rotate=45,anchor=west,font=\tiny] at ({x + 0.5},0.15) "
            rf"{{{_escape(category)}}};"
        )
    for y, model in enumerate(models, start=1):
        lines.append(
            rf"\node[anchor=east,font=\scriptsize] at (-0.10,{y + 0.5}) "
            rf"{{{_escape(_model_label(model))}}};"
        )
        for x, category in enumerate(categories):
            values = grouped[(model, category)]
            accuracy = sum(values) / len(values) if values else 0.0
            shade = round(accuracy * 85)
            color = "white" if shade >= 55 else "black"
            lines.append(
                rf"\draw[fill=black!{shade}] ({x},{y}) rectangle ({x + 1},{y + 1});"
            )
            lines.append(
                rf"\node[text={color},font=\tiny] at ({x + 0.5},{y + 0.5}) "
                rf"{{{accuracy:.2f}}};"
            )
    lines.extend(
        [
            rf"\node[anchor=west,font=\tiny] at (0,{len(models) + 1.35}) "
            r"{Cell value: primary-seed hybrid balanced case accuracy};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    return "\n".join(lines)


def _compiler_scaling(benchmark: JsonObject) -> str:
    rows = benchmark["rows"]
    providers = sorted({int(row["provider_count"]) for row in rows})
    tasks = sorted({int(row["task_count"]) for row in rows})
    maximum = max(float(row["latency_ms_p95"]) for row in rows) * 1.1
    x_by_task = {task: index / (len(tasks) - 1) for index, task in enumerate(tasks)}
    lines = [
        r"\begin{tikzpicture}[x=9cm,y=4.8cm]",
        r"\draw[->] (0,0)--(1.03,0) node[right]{\scriptsize tasks};",
        r"\draw[->] (0,0)--(0,1.03) node[above]{\scriptsize p95 latency};",
    ]
    for task, x in x_by_task.items():
        lines.append(rf"\draw ({x:.4f},0)--({x:.4f},-0.02) node[below]{{\tiny {task}}};")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        lines.append(
            rf"\draw[black!20] (0,{fraction:.2f})--(1,{fraction:.2f});"
            rf"\node[left,font=\tiny] at (0,{fraction:.2f}) "
            rf"{{{maximum * fraction:.2f} ms}};"
        )
    styles = ("solid", "dashed", "dotted", "dash dot")
    for index, provider in enumerate(providers):
        relevant = sorted(
            (row for row in rows if int(row["provider_count"]) == provider),
            key=lambda row: int(row["task_count"]),
        )
        coordinates = " -- ".join(
            rf"({x_by_task[int(row['task_count'])]:.4f},"
            rf"{float(row['latency_ms_p95']) / maximum:.4f})"
            for row in relevant
        )
        lines.append(rf"\draw[thick,{styles[index]}] {coordinates};")
        lines.append(
            rf"\node[anchor=west,font=\tiny] at (1.05,{0.92 - index * 0.08:.2f}) "
            rf"{{{provider} provider{'s' if provider != 1 else ''}}};"
        )
    lines.extend(
        [
            r"\node[anchor=west,font=\tiny,text width=36mm] at (1.05,0.52) "
            r"{The $6\times8$ space has 262,144 assignments; the compiler "
            r"refuses after the declared 4,096 bound.};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    return "\n".join(lines)


def _consensus_matrix(campaign: JsonObject) -> str:
    results = [item for item in campaign["results"] if item.get("primary", True)]
    conditions = sorted({item["scenario"] for item in results})
    by_key = {(item["scenario"], int(item["trial"])): item for item in results}
    colors = {
        "passed": ("black!15", "black"),
        "invariant_failed": ("black!80", "white"),
        "infrastructure_error": ("black!45", "black"),
    }
    lines = [r"\begin{tikzpicture}[x=0.72cm,y=0.42cm]"]
    for y, condition in enumerate(conditions):
        lines.append(
            rf"\node[anchor=east,font=\tiny] at (-0.15,{-y - 0.5}) "
            rf"{{{_escape(condition)}}};"
        )
        for trial in (1, 2, 3):
            item = by_key[(condition, trial)]
            status = str(item["status"])
            fill, text = colors.get(status, ("white", "black"))
            symbol = {"passed": "P", "invariant_failed": "I", "infrastructure_error": "E"}.get(
                status, "?"
            )
            lines.append(
                rf"\draw[fill={fill}] ({trial - 1},{-y - 1}) "
                rf"rectangle ({trial},{-y});"
                rf"\node[text={text},font=\tiny] at ({trial - 0.5},{-y - 0.5})"
                rf"{{{symbol}}};"
            )
    for trial in (1, 2, 3):
        lines.append(
            rf"\node[font=\scriptsize] at ({trial - 0.5},0.45) {{trial {trial}}};"
        )
    legend_y = -len(conditions) - 0.7
    lines.extend(
        [
            rf"\node[anchor=west,font=\tiny] at (0,{legend_y}) "
            r"{P: passed\quad I: invariant failed\quad E: infrastructure error};",
            r"\end{tikzpicture}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("thesis/figures/generated"))
    args = parser.parse_args()
    analysis = _load(args.analysis)
    benchmark = _load(args.benchmark)
    consensus = _load(args.consensus)
    outputs = {
        "v07-safety-utility.tex": _safety_utility(analysis),
        "v07-model-category-heatmap.tex": _heatmap(analysis),
        "v07-compiler-scaling.tex": _compiler_scaling(benchmark),
        "v07-consensus-matrix.tex": _consensus_matrix(consensus),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (args.output_dir / name).write_text(content, encoding="utf-8")
    print(json.dumps({"generated": sorted(outputs)}, indent=2))


if __name__ == "__main__":
    main()
