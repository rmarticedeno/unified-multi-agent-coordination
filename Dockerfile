FROM gcr.io/etcd-development/etcd:v3.6.13@sha256:24fc96ede8e787eb769e771d1245ddb2333301697a8b571e20415a0caccc375e AS etcd

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

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
