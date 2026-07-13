# Reproducing v0.2 evidence

The deterministic checks can be run without an LLM:

```text
uv sync --locked --group dev
uv run pytest
uv run unified-coordination-scenarios --output tmp/reproduction/deterministic.json
uv run python scripts/check_evidence.py --deterministic tmp/reproduction/deterministic.json
docker compose up --build --abort-on-container-exit --exit-code-from system-tests
docker compose -f docker-compose.distributed.yml up --build --abort-on-container-exit --exit-code-from distributed-system-tests
```

The 36-case local-model study is intentionally gated on an independently signed
`corpus/v0.2/label-signoff.json`. Reference labels are never placed in prompts.
Use LM Studio at `http://127.0.0.1:1234/v1`, load exactly one specified model at
a time, and retain each run-ID directory. Do not treat files under legacy paths
as v0.2 evidence.

The thesis is compiled with LuaLaTeX/BibTeX:

```text
cd thesis
latexmk -lualatex -bibtex -interaction=nonstopmode -halt-on-error main.tex
```
