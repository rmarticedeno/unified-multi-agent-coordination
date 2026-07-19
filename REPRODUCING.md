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

## Frozen v0.7 Qwen-first study

Version 0.7 is the production-path comparison. Version 0.5 is retained as the
historical experiment that motivated the redesign, and the unexecuted
study-specific v0.6 protocol is marked superseded. The 48-case Qwen3-1.7B
development phase is qualification evidence only: prompt or implementation
changes were permitted and its accuracy is not used to answer a research
question.

After the development gates pass, freeze and validate the 64-case held-out
protocol before collecting any confirmatory result:

```text
uv run unified-generate-defense-corpus-v07 --output corpus/v0.7
uv run unified-symbolic-benchmark-v07 --output demo_runs/v0.7/deterministic-benchmark.json
uv run unified-defense-study-v07 --corpus corpus/v0.7 --phase developmental --model qwen/qwen3-1.7b
uv run unified-defense-study-v07 --corpus corpus/v0.7 --freeze-protocol
uv run unified-defense-study-v07 --corpus corpus/v0.7 --check
```

The frozen confirmatory sequence is deliberately model-serial. Load one model
at a time in LM Studio and reuse the run directory printed by the first
command:

```text
uv run unified-defense-study-v07 --corpus corpus/v0.7 --phase confirmatory --model qwen/qwen3-1.7b
uv run unified-defense-study-v07 --corpus corpus/v0.7 --phase confirmatory --model google/gemma-4-e2b --resume demo_runs/v0.7/<run-id>
uv run unified-defense-study-v07 --corpus corpus/v0.7 --phase confirmatory --model qwen/qwen3-8b --resume demo_runs/v0.7/<run-id>
uv run unified-analyze-defense-study-v07 --run demo_runs/v0.7/<run-id> --corpus corpus/v0.7 --phase confirmatory --benchmark demo_runs/v0.7/deterministic-benchmark.json
uv run python -m unified_multi_agent_coordination.a2a_replay_v07 --run demo_runs/v0.7/<run-id> --corpus corpus/v0.7 --analysis demo_runs/v0.7/<run-id>/analysis-confirmatory-v0.7.0.json --output demo_runs/v0.7/<run-id>/a2a-replay-v0.7.0.json
uv run unified-consensus-matrix --output-dir demo_runs/consensus/<new-v4-run-id> --trials 3
uv run python scripts/generate_v07_figures.py --analysis demo_runs/v0.7/<run-id>/analysis-confirmatory-v0.7.0.json --benchmark demo_runs/v0.7/deterministic-benchmark.json --consensus demo_runs/consensus/<new-v4-run-id>/campaign.json --output-dir thesis/figures/generated
```

Completion requires exactly 768 immutable observations: three models, two
requested seeds, 64 held-out cases, and two LLM arms. The frozen settings are
temperature 0.2, top-p 1, and an 800-token limit. Invalid schema is scored as a
failed observation without semantic repair; transport/server failures are
reported separately. Labels are author-only and are not independent
adjudication. After freeze, a behavioral change requires a new protocol
version and a complete three-model rerun.

After the final pytest/coverage gates, generate `e/v7/evidence-manifest.json`
with `scripts/generate_v07_evidence_manifest.py`, then merge its references
into the historical top-level manifest with
`scripts/merge_v07_evidence_manifest.py`. The merge preserves older evidence
families and does not promote a dirty consensus campaign as accepted evidence.

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

## Accepted v0.5 comparison and consensus campaign

The accepted collection was produced from clean evidence-candidate commit
`7a31b42069104c3a4d7f10d8441d82357a35fd8b`. The packaged run is `e/v5`.
To repeat the protocol, create and push a non-final evidence-candidate commit
only after `git status --porcelain` is empty. The public cases, hidden labels,
prompts, schemas, criteria, and analysis code must already be frozen:

```text
uv run unified-defense-study-v05 --corpus corpus/v0.5 --collect
uv run unified-analyze-defense-study-v05 --run <run-directory> --corpus corpus/v0.5
```

Completion requires 1,440 unique arm/model/seed/case outputs. Each arm has the
same one-initial-plus-one-repair maximum. Requested seeds are stability
repetitions rather than independent samples. Functional metrics are written to
a deterministic report and wall-clock measurements to a separate artifact.
Unsuccessful pre-specified outcomes are retained, not rerun away. In the
accepted run, the hybrid's zero-false-acceptance criterion passed, but its
per-model recall and positive paired-accuracy-difference criteria failed.

The primary consensus campaign was run from correction commit
`3f8093af1ecc7c49312ac34856bba825bd81f381` and is packaged under
`demo_runs/consensus/20260715T022727Z-3f8093a-v2`. It is complete and valid,
but failed with 20 passed, seven invariant-failed, and six infrastructure-error
trials. Repeat campaigns with:

```text
uv run unified-consensus-matrix --output-dir demo_runs/consensus/<new-run-id> --trials 3 --promotion-candidate
uv run unified-evidence-preflight --corpus corpus/v0.5 --repository .
```

The campaign creates isolated Compose projects and fresh volumes for 3-, 5-,
and 7-voter formation, reconfiguration, leader and partition faults, quorum
restoration, restart/replacement, audit-projection failure, concurrency, and
three crash windows. Dirty, incomplete, overwritten, or hash-mismatched output
is never accepted. Containers share one host; this is crash-fault evidence, not
Byzantine or independent-failure-domain validation.

The clean PostgreSQL v1 failure is preserved under
`demo_runs/postgres/20260715T0214Z-7a31b42-failed`. It exposed a lease
read-then-upsert race. The atomic conditional-upsert correction is commit
`3f8093af1ecc7c49312ac34856bba825bd81f381`; its v2 report under
`demo_runs/postgres/20260715T0225Z-3f8093a-passed` passes all five checks.

The third-party A2A test uses the vendored upstream-derived Hello World snapshot
pinned in `vendor/a2a-samples/UPSTREAM.json` and `a2a-sdk==1.1.0`. It must pass
from a clean checkout with no pre-existing vendor cache.

## Versionless developmental strategy checks

These checks exercise the later production architecture and do not alter,
replace, or extend the accepted v0.5 evidence. For the fixed raw-language
subset, load exactly one LM Studio model at a time with 16,384-token context and
parallelism one:

```text
uv run unified-hybrid-strategy-validation collect --model qwen/qwen3-1.7b --case-set all
uv run unified-hybrid-strategy-validation collect --model google/gemma-4-e2b --case-set sentinel --run <same-run>
uv run unified-hybrid-strategy-validation collect --model qwen/qwen3-8b --case-set sentinel --run <same-run>
uv run unified-hybrid-strategy-validation analyze --run <same-run>
```

Collection reads only `validation/hybrid_strategy/public_cases.json`; analysis
opens `expected.json` afterward. The preserved final run is
`demo_runs/hybrid_strategy_validation/20260717T031509Z-3a9e1f9123`.
Preliminary negative runs in the same parent directory are retained.

To replay the unchanged public corpus that produced the historical 1.39%
hybrid recall through the typed-request production path:

```text
uv run unified-hybrid-strategy-validation replay-typed-corpus --corpus-root corpus/v0.5
uv run unified-hybrid-strategy-validation analyze-typed-corpus --run <new-run>
```

The replay writes all 48 compiler decisions before the analyzer reads
`corpus/v0.5/hidden/reference-labels.json`. The preserved run
`demo_runs/hybrid_strategy_validation/20260717T035308Z-typed-corpus-f3d214432a`
reports 48/48 correct, 100% feasible recall, and zero false acceptances or
refusals. It makes zero model calls because typed `ProblemRequest` inputs
explicitly bypass linguistic admission. These author-designed checks are
developmental descriptive evidence only; do not use them to revise the frozen
RQ4 analysis or claim superiority.

The thesis is compiled with LuaLaTeX/BibTeX:

```text
cd thesis
latexmk -lualatex -bibtex -interaction=nonstopmode -halt-on-error main.tex
```
