FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV COORDINATION_SERVICE_HOST=0.0.0.0
ENV COORDINATION_SERVICE_PORT=8000

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY main.py ./

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["/app/.venv/bin/unified-coordination-service"]
