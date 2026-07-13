"""HTTP service wrapper for the coordination prototype."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .coordination_agent import CoordinationAgent
from .coordination_ledger import JsonlCoordinationLedger, RetryPolicy
from .coordination_sdk import CoordinationSdk, RemoteRegistryError
from .coordination_store import LeaseConflictError, StaleFenceError, store_from_url
from .models import (
    AgentRegistryEntry,
    ProblemRequest,
    SolutionProposal,
)


class PlanRequest(BaseModel):
    """Request body for plan-only endpoints."""

    user_request: str | None = None
    problem: ProblemRequest | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class CoordinateRequest(PlanRequest):
    """Request body for end-to-end coordination."""

    payload: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float = 30.0
    session_id: str | None = None


class FeasibilityRequest(BaseModel):
    """Request body for standalone symbolic feasibility checks."""

    request: ProblemRequest
    proposal: SolutionProposal
    registry_snapshot: list[AgentRegistryEntry] | None = None


def sdk_from_env() -> CoordinationSdk:
    """Create the SDK from deployment environment variables."""
    timeout = float(os.getenv("COORDINATION_REQUEST_TIMEOUT_S", "10.0"))
    return CoordinationSdk(
        remote_registry_url=os.getenv("COORDINATION_REGISTRY_URL") or None,
        self_agent_id=os.getenv("COORDINATION_SELF_AGENT_ID") or None,
        request_timeout_s=timeout,
    )


def agent_from_env(sdk: CoordinationSdk) -> CoordinationAgent:
    """Create the coordination agent from deployment environment variables."""
    ledger_path = os.getenv("COORDINATION_LEDGER_PATH") or None
    store_url = os.getenv("COORDINATION_STORE_URL") or None
    store = store_from_url(store_url, ledger_path)
    ledger = JsonlCoordinationLedger(ledger_path) if ledger_path and not store_url else None
    lease_ttl_s = float(os.getenv("COORDINATION_LEASE_TTL_S", "30.0"))
    lease_renew_interval_s = float(
        os.getenv(
            "COORDINATION_LEASE_RENEW_INTERVAL_S",
            str(max(lease_ttl_s / 2, 0.1)),
        )
    )
    return CoordinationAgent(
        sdk=sdk,
        ledger=ledger,
        store=store,
        retry_policy=RetryPolicy(
            registry_retries=int(os.getenv("COORDINATION_REGISTRY_RETRIES", "2")),
            task_retries=int(os.getenv("COORDINATION_TASK_RETRIES", "1")),
            backoff_s=float(os.getenv("COORDINATION_RETRY_BACKOFF_S", "0.05")),
        ),
        coordinator_id=os.getenv("COORDINATION_COORDINATOR_ID") or None,
        lease_ttl_s=lease_ttl_s,
        lease_renew_interval_s=lease_renew_interval_s,
    )


def create_app(
    sdk: CoordinationSdk | None = None,
    agent: CoordinationAgent | None = None,
) -> FastAPI:
    """Create the FastAPI app around an SDK/CoordinationAgent pair."""
    app = FastAPI(
        title="Unified Multi-Agent Coordination",
        version="0.1.0",
    )
    app.state.sdk = sdk or sdk_from_env()
    app.state.agent = agent or agent_from_env(app.state.sdk)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "unified-multi-agent-coordination"}

    @app.get("/registry")
    async def registry(refresh: bool = True) -> dict[str, list[AgentRegistryEntry]]:
        try:
            snapshot = await app.state.sdk.registry_snapshot(refresh=refresh)
        except RemoteRegistryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"agents": snapshot}

    @app.post("/plan")
    async def plan(request: PlanRequest):
        try:
            return await app.state.agent.build_solution_plan(
                _request_input(request),
                context=request.context,
            )
        except RemoteRegistryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/coordinate")
    async def coordinate(request: CoordinateRequest):
        try:
            return await app.state.agent.coordinate(
                _request_input(request),
                context=request.context,
                payload=request.payload,
                timeout_s=request.timeout_s,
                session_id=request.session_id,
            )
        except (LeaseConflictError, StaleFenceError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume")
    async def resume_session(session_id: str, request: CoordinateRequest | None = None):
        try:
            return await app.state.agent.resume_session(
                session_id,
                payload=request.payload if request is not None else None,
                timeout_s=request.timeout_s if request is not None else 30.0,
            )
        except (LeaseConflictError, StaleFenceError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/feasibility")
    async def feasibility(request: FeasibilityRequest):
        registry_snapshot = request.registry_snapshot
        if registry_snapshot is None:
            registry_snapshot = await app.state.sdk.registry_snapshot(refresh=False)
        return app.state.agent.feasibility_analyzer.check(
            request.request,
            registry_snapshot,
            request.proposal,
        )

    return app


def _request_input(request: PlanRequest) -> str | ProblemRequest:
    if request.problem is not None:
        return request.problem
    if request.user_request:
        return request.user_request
    raise HTTPException(
        status_code=422,
        detail="Either problem or user_request must be provided.",
    )


def main() -> None:
    """Run the HTTP service."""
    import uvicorn

    host = os.getenv("COORDINATION_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("COORDINATION_SERVICE_PORT", "8000"))
    uvicorn.run(
        "unified_multi_agent_coordination.service:create_app",
        host=host,
        port=port,
        factory=True,
    )


if __name__ == "__main__":
    main()
