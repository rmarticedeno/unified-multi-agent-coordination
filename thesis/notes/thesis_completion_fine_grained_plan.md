# Thesis Completion Fine-Grained Step-by-Step Plan

This plan converts the thesis-completion protocol into an execution checklist.
It is written for repeatable evidence generation, manuscript completion, and
final PDF verification.

## Source Scope

This plan covers completion of the current thesis using only local evidence:

- Deterministic coordination scenarios from `demo_runs/end_to_end_scenarios.json`.
- Docker A2A system evidence from `demo_runs/docker_system_report.json`.
- Batched local LLM reference evidence from `demo_runs/local_llm/`.
- Thesis-ready analysis output from `demo_runs/thesis_analysis/`.
- The LaTeX manuscript under `thesis/`.

## Must-Follow Rules

1. Use no cloud LLMs for thesis evidence.
2. Use only the local endpoint `http://127.0.0.1:1234` for local LLM reference
   checks.
3. Use only these local models:
   - `qwen/qwen3-1.7b`
   - `google/gemma-4-e2b`
4. Run one model batch completely before switching models.
5. Do not alternate models inside one scenario suite, retry loop, or analysis
   pass.
6. Abort a local LLM batch if the endpoint does not advertise the expected
   model.
7. Abort a local LLM batch if the completion response does not identify the
   requested model.
8. Preserve every evidence file used in the thesis.
9. Treat local LLM outputs as reference evidence for linguistic candidate
   behavior only.
10. Ground authorization, refusal, traceability, SDK-boundary, and Docker
    system claims in deterministic or Docker reports.
11. Do not claim general benchmark superiority, production safety, or
    distributed fault tolerance from prototype-scale evidence.
12. Every thesis claim in the abstract, methodology, results, discussion, and
    conclusion must trace to preserved evidence or be explicitly framed as
    limitation/future work.

## Step-by-Step Outline

### Step 0. Freeze The Completion Scope

Description:
Define exactly what evidence and manuscript sections count as part of the
completion effort before regenerating anything.

Must-follow steps:

1. Read `README.md` and `thesis/README.md` for the current commands and output
   paths.
2. List the current generated evidence under `demo_runs/`.
3. List the current thesis chapters under `thesis/chapters/`.
4. Record which files are generated outputs and which are authored manuscript
   sources.
5. Do not delete or overwrite preserved evidence unless a replacement run has
   completed successfully.

Deliverables:

- A confirmed evidence inventory.
- A confirmed manuscript-source inventory.
- A clear distinction between generated artifacts and editable thesis text.

Verification evidence:

- `git status --short`
- `rg --files demo_runs thesis`

### Step 1. Run The Non-Docker Test Suite

Description:
Confirm that the local Python package behavior is healthy before generating
thesis evidence.

Must-follow steps:

1. Run:

   ```powershell
   uv run pytest
   ```

2. Check that pytest collects only intended project tests.
3. Record the pass/fail result.
4. If tests fail, fix the implementation or test expectation before continuing.
5. Do not update thesis claims from evidence produced by a failing codebase.

Deliverables:

- Passing local test suite.
- Noted warnings, if any, with an assessment of whether they affect thesis
  claims.

Verification evidence:

- Pytest output showing the final pass/fail summary.

### Step 2. Regenerate Deterministic End-to-End Scenarios

Description:
Regenerate the local deterministic scenario report that proves the core
coordination behavior without remote services.

Must-follow steps:

1. Run:

   ```powershell
   uv run --with-editable . unified-coordination-scenarios
   ```

2. Confirm the report is written to:

   ```text
   demo_runs/end_to_end_scenarios.json
   ```

3. Inspect the report for scenario count, expected status, actual status, trace
   events, ledger events, task dispatches, and terminal outcomes.
4. Confirm that feasible, infeasible, runtime-failure, and auxiliary-behavior
   cases are represented.
5. If a scenario changes, update the results explanation only after confirming
   whether the change is intended.

Deliverables:

- Fresh `demo_runs/end_to_end_scenarios.json`.
- Confirmed deterministic scenario outcome summary.

Verification evidence:

- Scenario report exists.
- Every scenario has a clear expected status and actual status.
- Status mismatches are either zero or explicitly explained as intended changes.

### Step 3. Regenerate Docker A2A System Evidence

Description:
Regenerate the multi-container Docker report when Docker is available. This is
the evidence layer for remote A2A fixture agents, registry behavior, service
boundaries, and containerized integration.

Must-follow steps:

1. Check whether Docker is available.
2. If Docker is available, run:

   ```powershell
   docker compose -f docker-compose.system.yml up --build --abort-on-container-exit --exit-code-from system-tests
   ```

3. Confirm the report is written to:

   ```text
   demo_runs/docker_system_report.json
   ```

4. Inspect pass/fail counts and named checks.
5. Confirm the Docker report is deterministic and uses no external LLM or cloud
   service.
6. If Docker is not available, record that fact and do not substitute a narrower
   in-process test as Docker evidence.

Deliverables:

- Fresh `demo_runs/docker_system_report.json`, or a clear unavailable-Docker
  note if Docker cannot run.
- Docker system check summary.

Verification evidence:

- Docker process exit code.
- `demo_runs/docker_system_report.json` with named system checks.

### Step 4. Prepare The Local LLM Endpoint

Description:
Confirm that LM Studio or another OpenAI-compatible local server is reachable
before running model batches.

Must-follow steps:

1. Confirm the endpoint is reachable at:

   ```text
   http://127.0.0.1:1234
   ```

2. Confirm the endpoint exposes OpenAI-compatible model and chat-completion
   routes used by the runner.
3. Do not run the local LLM checks against a cloud endpoint.
4. Do not run the local LLM checks against a different local port unless the
   runner and thesis text are updated consistently.
5. Do not continue if the endpoint is unavailable and the plan requires fresh
   local LLM evidence.

Deliverables:

- Confirmed local endpoint availability.
- Confirmed no cloud LLM dependency.

Verification evidence:

- Endpoint health/model-list response.
- Runner preflight output.

### Step 5. Run The Full Qwen Batch

Description:
Run the complete local LLM reference suite for `qwen/qwen3-1.7b` before touching
the second model.

Must-follow steps:

1. Unload any active model:

   ```powershell
   lms unload --all
   ```

2. Load Qwen with the exact identifier:

   ```powershell
   lms load qwen/qwen3-1.7b --identifier qwen/qwen3-1.7b -y
   ```

3. Run the reference suite:

   ```powershell
   uv run --with-editable . unified-local-llm-reference --model qwen/qwen3-1.7b
   ```

4. Confirm the runner verifies the active advertised model before calling chat
   completions.
5. Confirm the runner verifies the response model before preserving outputs.
6. Do not switch to another model until the Qwen report is complete and saved.
7. Do not retry individual scenarios on a different model.

Deliverables:

- Model-specific Qwen report under:

  ```text
  demo_runs/local_llm/qwen__qwen3-1.7b/
  ```

- A preserved `reference_report.json` for the run.

Verification evidence:

- Report metadata includes endpoint, model ID, prompt/config version, decoding
  settings, timestamps, and run ID.
- Report scenario count matches the expected local LLM reference suite.
- No substituted model appears in the Qwen run.

### Step 6. Switch Once To Gemma

Description:
Perform the single expensive model switch only after the Qwen batch has fully
completed and its outputs have been preserved.

Must-follow steps:

1. Confirm the Qwen report exists and is readable.
2. Unload the Qwen model:

   ```powershell
   lms unload --all
   ```

3. Load Gemma with the exact identifier:

   ```powershell
   lms load google/gemma-4-e2b --identifier google/gemma-4-e2b -y
   ```

4. Confirm no Qwen-specific retry remains pending.
5. Do not run mixed-model batches.

Deliverables:

- Gemma loaded as the only expected active model.

Verification evidence:

- Model-list response or runner preflight identifying
  `google/gemma-4-e2b`.

### Step 7. Run The Full Gemma Batch

Description:
Run the same local LLM reference suite for `google/gemma-4-e2b` after the single
model switch.

Must-follow steps:

1. Run:

   ```powershell
   uv run --with-editable . unified-local-llm-reference --model google/gemma-4-e2b
   ```

2. Confirm the runner verifies the advertised active model.
3. Confirm the runner verifies each completion response model.
4. Preserve the run output separately from the Qwen output.
5. Do not substitute another Gemma variant or any other model if this model is
   unavailable.

Deliverables:

- Model-specific Gemma report under:

  ```text
  demo_runs/local_llm/google__gemma-4-e2b/
  ```

- A preserved `reference_report.json` for the run.

Verification evidence:

- Report metadata includes endpoint, model ID, prompt/config version, decoding
  settings, timestamps, and run ID.
- Report scenario count matches the expected local LLM reference suite.
- No substituted model appears in the Gemma run.

### Step 8. Run Thesis Evidence Analysis

Description:
Generate thesis-ready tables from the deterministic, Docker, and local LLM
reports.

Must-follow steps:

1. Confirm these inputs exist:
   - `demo_runs/end_to_end_scenarios.json`
   - `demo_runs/docker_system_report.json`
   - Qwen `reference_report.json`
   - Gemma `reference_report.json`
2. Run:

   ```powershell
   uv run --with-editable . unified-thesis-results
   ```

3. Confirm the analysis writes:
   - `demo_runs/thesis_analysis/summary.json`
   - `demo_runs/thesis_analysis/tables.md`
   - `demo_runs/thesis_analysis/tables.tex`
4. Inspect generated tables for traceability to the preserved JSON reports.
5. Do not hand-edit generated analysis tables unless the generator is also
   updated or the manual edit is explicitly documented.

Deliverables:

- Fresh thesis analysis summary.
- Fresh Markdown and LaTeX table drafts.

Verification evidence:

- Analysis output files exist.
- Summary input paths point to preserved reports.
- Tables include deterministic outcomes, Docker outcomes, local LLM outcomes,
  trace completeness, authorization-before-dispatch evidence, runtime failure
  categories, and latency summaries when available.

### Step 9. Update Methodology

Description:
Ensure the methodology chapter describes the actual evidence protocol, not a
future or hypothetical evaluation.

Must-follow steps:

1. State that no cloud LLMs are used.
2. State that local LLMs are run through `http://127.0.0.1:1234`.
3. State the allowed model IDs exactly:
   - `qwen/qwen3-1.7b`
   - `google/gemma-4-e2b`
4. Explain the batched local-model protocol.
5. Explain why model switching is minimized.
6. Explain that local LLM outputs are non-authoritative candidate/reference
   evidence.
7. Explain that symbolic authorization and refusal evidence come from
   deterministic and Docker reports.

Deliverables:

- Methodology text aligned with actual commands and outputs.

Verification evidence:

- Methodology chapter includes endpoint, allowed models, batching rule, and
  no-cloud statement.

### Step 10. Update Results

Description:
Rewrite results around measured evidence rather than pending protocol language.

Must-follow steps:

1. Report deterministic scenario outcomes from
   `demo_runs/end_to_end_scenarios.json`.
2. Report Docker system outcomes from `demo_runs/docker_system_report.json`.
3. Report local LLM reference outcomes separately by model.
4. Explain exact symbolic contract matching separately from semantic usefulness.
5. Include trace completeness and authorization-before-dispatch evidence.
6. Include runtime failure categories and latency summaries when available.
7. Add a short "what this proves" interpretation after each major result table.
8. Avoid saying local LLM results prove authorization correctness.

Deliverables:

- Results chapter that reports preserved measurements.
- Tables inserted or summarized from `demo_runs/thesis_analysis/`.

Verification evidence:

- Every numeric claim in results can be traced to generated JSON or analysis
  tables.
- Results distinguish feasibility refusal, runtime failure, and successful
  completion.

### Step 11. Update Discussion, Abstract, And Conclusion

Description:
Align high-level claims with the evidence now present in the results chapter.

Must-follow steps:

1. In the abstract, claim a reproducible prototype and evidence-backed
   coordination boundary.
2. In the discussion, separate achieved evidence from limitations.
3. In the conclusion, state the contribution as a precise architecture and
   reproducible pattern.
4. Do not claim production-grade reliability.
5. Do not claim broad empirical superiority.
6. Do not claim distributed fault tolerance unless distributed coordinator
   evidence has been implemented and tested.
7. Keep future-work wording separate from completed results.

Deliverables:

- Abstract, discussion, and conclusion aligned with actual evidence.

Verification evidence:

- Search for overbroad phrases and confirm they are removed or qualified.
- Chapter claims match the generated evidence summary.

### Step 12. Rebuild The Thesis PDF

Description:
Compile the LaTeX manuscript and confirm references, citations, tables, and
figures resolve.

Must-follow steps:

1. From `thesis/`, run:

   ```powershell
   New-Item -ItemType Directory -Force build | Out-Null
   pdflatex -output-directory=build main.tex
   bibtex build/main
   pdflatex -output-directory=build main.tex
   pdflatex -output-directory=build main.tex
   ```

2. Inspect the LaTeX log for undefined citations.
3. Inspect the LaTeX log for undefined references.
4. Inspect the LaTeX log for severe overfull boxes.
5. Confirm the final PDF exists.
6. If the built PDF is copied to `thesis/main.pdf`, confirm it matches the
   latest build.

Deliverables:

- Rebuilt thesis PDF.
- Clean citation/reference status.

Verification evidence:

- `thesis/build/main.pdf` or `thesis/main.pdf`.
- LaTeX log with no undefined citations or references.

### Step 13. Perform Final Claim Audit

Description:
Read the manuscript as an examiner would and ensure every strong claim is
supported.

Must-follow steps:

1. Audit the abstract for evidence-backed wording.
2. Audit the methodology for exact commands, models, and endpoint.
3. Audit the results for measured values only.
4. Audit the discussion for clear limitations.
5. Audit the conclusion for precise contribution language.
6. Confirm no section implies that local LLM outputs authorize execution.
7. Confirm no section implies distributed coordination has been implemented
   unless that work has actually been done.

Deliverables:

- Final claim-audit notes or clean-audit confirmation.

Verification evidence:

- Search results for risky terms such as "production", "distributed fault
  tolerance", "guarantees", "superior", "cloud", and model names.
- Manual chapter inspection.

## Final Completion Gate

The thesis-completion work is done only when all of the following are true:

1. `uv run pytest` passes.
2. Deterministic scenarios are regenerated and preserved.
3. Docker report is regenerated and preserved, or Docker unavailability is
   explicitly documented.
4. Qwen local LLM batch is complete and preserved.
5. Gemma local LLM batch is complete and preserved.
6. Model batches were not interleaved.
7. Thesis analysis outputs are regenerated from preserved JSON.
8. Methodology describes the exact evidence protocol.
9. Results report measured evidence.
10. Discussion, abstract, and conclusion match the evidence.
11. Thesis PDF builds without undefined citations or references.
12. Final claim audit finds no unsupported major claim.
