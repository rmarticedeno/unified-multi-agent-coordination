# Contributing

Use Python 3.13 and `uv sync --locked --group dev`. Before proposing a change,
run `uv run ruff check src tests`, `uv run mypy src/unified_multi_agent_coordination`,
and `uv run pytest`. Changes to feasibility, execution, persistence, traces, or
evaluation schemas require a regression test and an update to the evidence
manifest. Never replace a frozen run directory; create a new immutable run ID.

Security-sensitive changes must preserve the authorization-before-dispatch
boundary, local trust assignment, fail-closed unknown constraints, and secret
redaction from reports.
