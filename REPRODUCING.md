# Reproducing thesis evidence

The deterministic checks can be run without an LLM:

```text
uv sync --locked --group dev
uv run ruff check src tests scripts
uv run mypy src/unified_multi_agent_coordination
uv run pytest --cov --cov-branch --cov-report=json --cov-fail-under=0
uv run python scripts/check_coverage.py coverage.json
uv run unified-coordination-scenarios --output tmp/reproduction/deterministic.json
uv run python scripts/check_evidence.py --deterministic tmp/reproduction/deterministic.json
$env:EVIDENCE_RUN_DIR="demo_runs/system/<new-run-id>"
docker compose -f docker-compose.system.yml up --build --abort-on-container-exit --exit-code-from system-tests
$env:EVIDENCE_RUN_DIR="demo_runs/postgres/<new-run-id>"
docker compose -f docker-compose.distributed.yml up --build --abort-on-container-exit --exit-code-from distributed-system-tests
```

Every `EVIDENCE_RUN_DIR` must be new. Report and ledger writers use exclusive
creation and refuse to overwrite existing evidence.

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

The historical v0.3 run is analyzed reproducibly with:

```text
uv run unified-analyze-defense-study demo_runs/v0.3/20260713T013624Z-b9fd8b6f83 --corpus corpus/v0.3 --output demo_runs/v0.3/20260713T013624Z-b9fd8b6f83/analysis.json
uv run unified-evidence-preflight --corpus corpus/v0.5 --repository .
```

The historical analysis SHA-256 is
`b70a169d8527b2f4b95a5ef1a7ac083e0eb81874c896f50412352b37b14b03a3`.

Version 0.3 is not regenerated or rewritten after the bridge redesign. The
v0.4 study has its own frozen author-labelled corpus and immutable run:

```text
uv run unified-generate-defense-corpus-v04 --output corpus/v0.4
uv run unified-defense-study-v04 --corpus corpus/v0.4 --check
uv run unified-defense-study-v04 --corpus corpus/v0.4 --collect
uv run unified-analyze-defense-study-v04 --run demo_runs/v0.4/<run-id> --corpus corpus/v0.4 --output demo_runs/v0.4/<run-id>/<new-analysis>.json
uv run unified-runtime-ablation-study --output-dir demo_runs/runtime_ablation/<new-run-id>
```

The v0.4 prompt receives the complete admitted request and registry but only
returns a non-executable draft. Hydration and symbolic validation are measured
separately. Runtime ablations are deterministic negative controls and are never
combined with planning-accuracy rows.

Analysis schema 0.4.1 evaluates deterministic controls once per case, parses
the initial response for LLM-only, uses the repaired response only for the
hybrid, reports case-majority and model/seed strata, clusters bootstrap samples
by case, and attributes only configuration-specific latency. The historical
v0.4 raw run is dirty-source evidence. Do not promote it or suppress its failed
Qwen3 1.7B recall criterion.

## Clean v0.5 comparison and consensus campaign

Create and push a non-final evidence-candidate commit only after
`git status --porcelain` is empty. The v0.5 public cases, hidden author labels,
prompts, schemas, criteria, and analysis code must already be frozen in that
commit. Collect and analyze both arms from that exact revision:

```text
uv run unified-defense-study-v05 --corpus corpus/v0.5 --collect
uv run unified-analyze-defense-study-v05 --run demo_runs/v0.5/<run-id> --corpus corpus/v0.5
```

Completion requires 1,440 unique arm/model/seed/case outputs. Each arm has the
same one-initial-plus-one-repair maximum. Requested seeds are stability
repetitions rather than independent samples. Functional metrics are written to
a deterministic report and wall-clock measurements to a separate artifact.
Unsuccessful pre-specified outcomes are retained, not rerun away.

Run the consensus campaign from the same clean revision:

```text
uv run unified-consensus-matrix --output-dir demo_runs/consensus/<new-run-id> --trials 3 --promotion-candidate
uv run unified-evidence-preflight --require-release --corpus corpus/v0.5 --repository .
```

The campaign creates isolated Compose projects and fresh volumes for 3-, 5-,
and 7-voter formation, reconfiguration, leader and partition faults, quorum
restoration, restart/replacement, audit-projection failure, concurrency, and
three crash windows. Dirty, incomplete, overwritten, or hash-mismatched output
is never accepted. Containers share one host; this is crash-fault evidence, not
Byzantine or independent-failure-domain validation.

The third-party A2A test uses the vendored upstream-derived Hello World snapshot
pinned in `vendor/a2a-samples/UPSTREAM.json` and `a2a-sdk==1.1.0`. It must pass
from a clean checkout with no pre-existing vendor cache.

The thesis is compiled with LuaLaTeX/BibTeX:

```text
cd thesis
latexmk -lualatex -bibtex -interaction=nonstopmode -halt-on-error main.tex
```
