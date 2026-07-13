# Reproducing v0.3 and v0.4 evidence

The deterministic checks can be run without an LLM:

```text
uv sync --locked --group dev
uv run pytest
uv run unified-coordination-scenarios --output tmp/reproduction/deterministic.json
uv run python scripts/check_evidence.py --deterministic tmp/reproduction/deterministic.json
docker compose up --build --abort-on-container-exit --exit-code-from system-tests
docker compose -f docker-compose.distributed.yml up --build --abort-on-container-exit --exit-code-from distributed-system-tests
```

The 36-case local-model study is gated on hash-matched, frozen author provenance
in `corpus/v0.3/label-provenance.json`. No independent adjudication is claimed.
Reference labels are never placed in prompts and are loaded only after raw-output
collection is complete.
Use LM Studio at `http://127.0.0.1:1234/v1`, load exactly one specified model at
a time, and retain each run-ID directory. Do not treat files under legacy paths
as v0.3 evidence.

Resume an interrupted collection with
`uv run unified-defense-study --resume demo_runs/v0.3/<run-id>`. The runner
validates corpus and record identities, skips completed cases, and refuses to
resume a completed immutable run.

The accepted run is analyzed reproducibly with:

```text
uv run unified-analyze-defense-study demo_runs/v0.3/20260713T013624Z-b9fd8b6f83 --corpus corpus/v0.3 --output demo_runs/v0.3/20260713T013624Z-b9fd8b6f83/analysis.json
uv run unified-evidence-preflight
```

The accepted analysis SHA-256 is
`b70a169d8527b2f4b95a5ef1a7ac083e0eb81874c896f50412352b37b14b03a3`.

Version 0.3 is not regenerated or rewritten after the bridge redesign. The
v0.4 study has its own frozen author-labelled corpus and immutable run:

```text
uv run unified-generate-defense-corpus-v04 --output corpus/v0.4
uv run unified-defense-study-v04 --corpus corpus/v0.4 --check
uv run unified-defense-study-v04 --corpus corpus/v0.4 --collect
uv run unified-analyze-defense-study-v04 --run demo_runs/v0.4/<run-id> --corpus corpus/v0.4
uv run unified-runtime-ablation-study --output-dir demo_runs/runtime_ablation/<new-run-id>
```

The v0.4 prompt receives the complete admitted request and registry but only
returns a non-executable draft. Hydration and symbolic validation are measured
separately. Runtime ablations are deterministic negative controls and are never
combined with planning-accuracy rows.

The thesis is compiled with LuaLaTeX/BibTeX:

```text
cd thesis
latexmk -lualatex -bibtex -interaction=nonstopmode -halt-on-error main.tex
```
