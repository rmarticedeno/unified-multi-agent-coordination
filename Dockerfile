FROM gcr.io/etcd-development/etcd:v3.6.13 AS etcd

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV COORDINATION_SERVICE_HOST=0.0.0.0
ENV COORDINATION_SERVICE_PORT=8000

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY main.py ./
COPY --from=etcd /usr/local/bin/etcd /usr/local/bin/etcd
COPY --from=etcd /usr/local/bin/etcdctl /usr/local/bin/etcdctl

RUN uv sync --frozen --no-dev

EXPOSE 8000 2379 2380 7947/udp

CMD ["/app/.venv/bin/unified-coordination-service"]
