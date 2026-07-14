# Thesis

This directory contains the LaTeX source for the master's thesis associated with the project.

## Structure

- `main.tex`: root LaTeX document.
- `uhmasterthesis.cls`: University of Havana branded thesis class.
- `Graphics/uhlogo.pdf`: University of Havana cover mark supplied by the branding template.
- `chapters/`: chapter source files.
- `figures/`: figures and diagrams used by the thesis.
- `references/references.bib`: BibTeX bibliography database.
- `build/`: generated PDFs and LaTeX build output.

## Build

From this directory:

```powershell
latexmk -lualatex -bibtex -interaction=nonstopmode -halt-on-error main.tex
```

Generated build output is intentionally ignored by Git.

## Evidence outputs

The results chapter is supported by generated evidence under the repository
root's `demo_runs/` directory:

- `end_to_end_scenarios.json`: deterministic local coordination scenarios.
- `docker_system_report.json`: Docker A2A system harness report.
- `distributed_system_report.json`: PostgreSQL-backed replicated-coordinator
  harness report.
- `v0.4/<run-id>/analysis-v0.4.1-*.json`: corrected immutable comparison
  analyses; deterministic controls have 36 case rows and model configurations
  have 540 repeated observations.
- `consensus/<run-id>/`: immutable 3/5/7 consensus campaign reports. No
  accepted campaign exists until a clean-source full run passes.
- `local_llm/`: batched local LLM reference reports for `qwen/qwen3-1.7b`
  and `google/gemma-4-e2b`.
- `baselines/baseline_report.json`: paired hybrid, rule-only, and LLM-only
  baseline comparison report.
- `thesis_analysis/`: generated summaries and thesis-ready table drafts.

Regenerate the analysis from the repository root with:

```powershell
uv run unified-analyze-defense-study-v04 --run demo_runs/v0.4/<run-id> `
  --corpus corpus/v0.4 --output demo_runs/v0.4/<run-id>/<new-analysis>.json
uv run unified-evidence-preflight --corpus corpus/v0.4 --repository .
```

## Argument structure

The thesis follows this research sequence: introduction and problem formulation; critical state of the art; research methodology; proposed protocol-independent theoretical framework; protocol-adaptable system design; evaluation and results; discussion; and conclusions. The evaluation has two core tracks: the frozen linguistic conformance studies and a consensus-backed crash-fault campaign. They remain statistically separate. The current manuscript reports the 3/5/7 campaign as an open evidence gate and does not claim Byzantine, multi-host, or production validation.

The cover follows the University of Havana and Faculty of Mathematics and Computer Science branding supplied in the curated template. Its QR code is generated at compile time from the repository URL declared with `\repositoryurl{...}` in `main.tex`; no separately generated QR image is required.
