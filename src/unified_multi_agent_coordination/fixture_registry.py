"""Tiny card-url registry for Docker system tests."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the deterministic fixture registry app."""
    app = FastAPI(title="A2A Fixture Registry")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/agents")
    async def agents() -> dict[str, list[str]]:
        return {"card_urls": _card_urls()}

    return app


def _card_urls() -> list[str]:
    raw = os.getenv("A2A_CARD_URLS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    """Run the fixture registry service."""
    host = os.getenv("FIXTURE_REGISTRY_HOST", "0.0.0.0")
    port = int(os.getenv("FIXTURE_REGISTRY_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
