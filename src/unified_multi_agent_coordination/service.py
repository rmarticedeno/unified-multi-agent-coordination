"""HTTP service wrapper for the coordination prototype."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .coordination_agent import CoordinationAgent
from .coordination_ledger import JsonlCoordinationLedger, RetryPolicy
from .coordination_sdk import CoordinationSdk, RemoteRegistryError
from .coordination_store import (
    AuditProjectingCoordinationStore,
    LeaseConflictError,
    StaleFenceError,
    store_from_url,
)
from .agent_registry import EtcdAgentRegistry, RegisteredAgent
from .cluster import (
    ClusterConfiguration,
    ConfigurationConflictError,
    CoordinatorNodeRecord,
    HmacAuthenticator,
    MembershipManager,
    SignedEnvelope,
)
from .cluster_discovery import MulticastDiscoveryResponder
from .etcd_client import EtcdClient, EtcdQuorumUnavailableError
from .etcd_store import etcd_endpoints_from_url
from .models import (
    AgentRegistryEntry,
    ProblemRequest,
    SolutionProposal,
)
from .feasibility import FeasibilityAnalyzer
from .semantic_admission import (
    OpenAICompatibleSemanticInterpreter,
    SemanticCatalog,
)


class PlanRequest(BaseModel):
    """Request body for plan-only endpoints."""

    user_request: str | None = None
    problem: ProblemRequest | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    semantic_catalog: SemanticCatalog | None = None


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
    distributed_registry = None
    store_url = os.getenv("COORDINATION_STORE_URL") or ""
    if store_url.startswith("etcd://"):
        distributed_registry = EtcdAgentRegistry(
            EtcdClient(etcd_endpoints_from_url(store_url), timeout_s=timeout),
            cluster_id=os.getenv("COORDINATION_CLUSTER_ID", "default"),
        )
    return CoordinationSdk(
        remote_registry_url=os.getenv("COORDINATION_REGISTRY_URL") or None,
        self_agent_id=os.getenv("COORDINATION_SELF_AGENT_ID") or None,
        request_timeout_s=timeout,
        distributed_registry=distributed_registry,
    )


def agent_from_env(sdk: CoordinationSdk) -> CoordinationAgent:
    """Create the coordination agent from deployment environment variables."""
    ledger_path = os.getenv("COORDINATION_LEDGER_PATH") or None
    store_url = os.getenv("COORDINATION_STORE_URL") or None
    store = store_from_url(store_url, ledger_path)
    audit_store_url = os.getenv("COORDINATION_AUDIT_STORE_URL") or None
    if audit_store_url:
        if not store_url or not store_url.startswith("etcd://"):
            raise ValueError("Audit projection requires etcd as the authoritative store.")
        if not audit_store_url.startswith(("postgres://", "postgresql://")):
            raise ValueError("COORDINATION_AUDIT_STORE_URL must be a PostgreSQL URL.")
        store = AuditProjectingCoordinationStore(
            store,
            store_from_url(audit_store_url),
        )
    ledger = JsonlCoordinationLedger(ledger_path) if ledger_path and not store_url else None
    lease_ttl_s = float(os.getenv("COORDINATION_LEASE_TTL_S", "30.0"))
    lease_renew_interval_s = float(
        os.getenv(
            "COORDINATION_LEASE_RENEW_INTERVAL_S",
            str(max(lease_ttl_s / 2, 0.1)),
        )
    )
    catalog_path = os.getenv("COORDINATION_SEMANTIC_CATALOG_PATH") or ""
    semantic_catalog = (
        SemanticCatalog.model_validate_json(
            Path(catalog_path).read_text(encoding="utf-8")
        )
        if catalog_path
        else None
    )
    semantic_model = os.getenv("COORDINATION_SEMANTIC_MODEL") or ""
    semantic_interpreter = (
        OpenAICompatibleSemanticInterpreter(
            semantic_model,
            endpoint=os.getenv(
                "COORDINATION_SEMANTIC_ENDPOINT",
                "http://127.0.0.1:1234/v1",
            ),
            seed=int(os.getenv("COORDINATION_SEMANTIC_SEED", "11")),
        )
        if semantic_model
        else None
    )
    return CoordinationAgent(
        sdk=sdk,
        feasibility_analyzer=FeasibilityAnalyzer(
            require_effect_fencing=bool(store_url and store_url.startswith("etcd://"))
        ),
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
        semantic_catalog=semantic_catalog,
        semantic_interpreter=semantic_interpreter,
        max_concurrent_dispatches=int(
            os.getenv("COORDINATION_MAX_CONCURRENT_DISPATCHES", "16")
        ),
    )


def create_app(
    sdk: CoordinationSdk | None = None,
    agent: CoordinationAgent | None = None,
) -> FastAPI:
    """Create the FastAPI app around an SDK/CoordinationAgent pair."""
    configured_sdk = sdk or sdk_from_env()
    configured_agent = agent or agent_from_env(configured_sdk)
    membership, authenticator, responder = _distributed_components_from_env()
    agent_authenticator = (
        HmacAuthenticator(
            os.getenv("COORDINATION_AGENT_REGISTRATION_SECRET", ""),
            allow_insecure=_env_bool("COORDINATION_ALLOW_INSECURE_CLUSTER"),
        )
        if membership is not None
        else None
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if membership is not None:
            await membership.start()
        if responder is not None:
            await responder.start()
        try:
            yield
        finally:
            if responder is not None:
                await responder.stop()
            if membership is not None:
                await membership.stop()
                await membership.client.close()
            await app.state.agent.store.close()
            registry = getattr(app.state.sdk, "distributed_registry", None)
            if registry is not None:
                await registry.close()

    app = FastAPI(
        title="Unified Multi-Agent Coordination",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.sdk = configured_sdk
    app.state.agent = configured_agent
    app.state.membership = membership
    app.state.authenticator = authenticator
    app.state.agent_authenticator = agent_authenticator
    app.state.metrics = {
        "joins": 0,
        "leaves": 0,
        "configuration_changes": 0,
        "agent_registrations": 0,
        "agent_heartbeats": 0,
        "agent_removals": 0,
        "hmac_rejections": 0,
        "quorum_unavailable": 0,
        "stale_fence_rejections": 0,
        "lease_conflicts": 0,
        "discovery_fallbacks": int(
            os.getenv("COORDINATION_DISCOVERY_METHOD") == "multicast"
        ),
    }

    @app.exception_handler(EtcdQuorumUnavailableError)
    async def quorum_unavailable_handler(
        _request: Request, exc: EtcdQuorumUnavailableError
    ) -> JSONResponse:
        _increment(app, "quorum_unavailable")
        return JSONResponse(
            status_code=503,
            content={
                "code": "quorum_unavailable",
                "retryable": True,
                "message": str(exc),
            },
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "unified-multi-agent-coordination"}

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "ok", "service": "unified-multi-agent-coordination"}

    @app.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        try:
            timeout_s = float(os.getenv("COORDINATION_READINESS_TIMEOUT_S", "8"))
            async with asyncio.timeout(timeout_s):
                ready = getattr(app.state.agent.store, "ready", None)
                if ready is not None:
                    await ready()
                cluster_status = (
                    await app.state.membership.status()
                    if app.state.membership is not None
                    else None
                )
            return {"status": "ready", "cluster": cluster_status}
        except TimeoutError as exc:
            raise EtcdQuorumUnavailableError(
                f"Authoritative readiness probe exceeded {timeout_s:g}s."
            ) from exc
        except Exception as exc:
            if isinstance(exc, EtcdQuorumUnavailableError):
                raise
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/metrics")
    async def metrics() -> dict[str, int]:
        return dict(app.state.metrics)

    @app.get("/cluster/status")
    async def cluster_status() -> dict[str, Any]:
        if app.state.membership is None:
            return {
                "cluster_id": "",
                "role": "standalone",
                "degraded": True,
                "degraded_reason": "distributed etcd mode is not configured",
            }
        try:
            return await app.state.membership.status()
        except EtcdQuorumUnavailableError:
            raise

    @app.post("/internal/cluster/join")
    async def cluster_join(request: SignedEnvelope) -> SignedEnvelope:
        _require_distributed(app)
        _verify_envelope(app, request, "cluster_join")
        try:
            assignment = await app.state.membership.join(request)
            _increment(app, "joins")
            return app.state.authenticator.sign(
                SignedEnvelope(
                    message_type="cluster_join_response",
                    cluster_id=app.state.membership.configuration.cluster_id,
                    node_id=app.state.membership.current_node.node_id,
                    payload=assignment,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/internal/cluster/join/{node_id}")
    async def cluster_join_status(
        node_id: str,
        timestamp: float,
        nonce: str,
        signature: str,
    ) -> SignedEnvelope:
        _require_distributed(app)
        request = SignedEnvelope(
            message_type="cluster_join_status",
            cluster_id=app.state.membership.configuration.cluster_id,
            node_id=node_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
        )
        _verify_envelope(app, request, "cluster_join_status")
        try:
            assignment = await app.state.membership.join_status(node_id)
            return app.state.authenticator.sign(
                SignedEnvelope(
                    message_type="cluster_join_status_response",
                    cluster_id=app.state.membership.configuration.cluster_id,
                    node_id=app.state.membership.current_node.node_id,
                    payload=assignment,
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/internal/cluster/leave")
    async def cluster_leave(request: SignedEnvelope) -> dict[str, str]:
        _require_distributed(app)
        _verify_envelope(app, request, "cluster_leave")
        await app.state.membership.leave(request.node_id)
        _increment(app, "leaves")
        return {"status": "removed"}

    @app.put("/internal/cluster/configuration")
    async def cluster_configuration(request: SignedEnvelope) -> ClusterConfiguration:
        _require_distributed(app)
        _verify_envelope(app, request, "cluster_configuration")
        try:
            raw_target = request.payload.get("voter_target")
            if raw_target is None:
                raise ValueError("voter_target is required.")
            raw_generation = request.payload.get("expected_generation")
            if raw_generation is None:
                raise ValueError("expected_generation is required.")
            target = int(raw_target)
            updated = await app.state.membership.update_voter_target(
                target,
                expected_generation=int(raw_generation),
                updated_by=request.node_id,
            )
            _increment(app, "configuration_changes")
            return updated
        except ConfigurationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/internal/agents/register")
    async def register_agent(request: SignedEnvelope) -> RegisteredAgent:
        _require_distributed(app)
        _verify_agent_envelope(app, request, "agent_register")
        registry = app.state.sdk.distributed_registry
        if registry is None:
            raise HTTPException(status_code=503, detail="Distributed registry unavailable.")
        try:
            record = RegisteredAgent.model_validate(request.payload.get("record"))
            ttl_s = float(request.payload.get("ttl_s") or 30.0)
            registered = await registry.register(record, ttl_s=ttl_s)
            _increment(app, "agent_registrations")
            return registered
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/internal/agents/{agent_id}/heartbeat")
    async def heartbeat_agent(agent_id: str, request: SignedEnvelope) -> dict[str, Any]:
        _require_distributed(app)
        _verify_agent_envelope(app, request, "agent_heartbeat")
        registry = app.state.sdk.distributed_registry
        if registry is None:
            raise HTTPException(status_code=503, detail="Distributed registry unavailable.")
        try:
            ttl = await registry.heartbeat(agent_id)
            _increment(app, "agent_heartbeats")
            return {"agent_id": agent_id, "ttl": ttl}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/internal/agents/{agent_id}")
    async def remove_agent(agent_id: str, request: SignedEnvelope) -> dict[str, str]:
        _require_distributed(app)
        _verify_agent_envelope(app, request, "agent_remove")
        registry = app.state.sdk.distributed_registry
        if registry is None:
            raise HTTPException(status_code=503, detail="Distributed registry unavailable.")
        await registry.remove(agent_id)
        _increment(app, "agent_removals")
        return {"status": "removed"}

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
                semantic_catalog=request.semantic_catalog,
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
                semantic_catalog=request.semantic_catalog,
            )
        except EtcdQuorumUnavailableError:
            raise
        except (LeaseConflictError, StaleFenceError) as exc:
            _increment(
                app,
                "stale_fence_rejections"
                if isinstance(exc, StaleFenceError)
                else "lease_conflicts",
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/resume")
    async def resume_session(session_id: str, request: CoordinateRequest | None = None):
        try:
            return await app.state.agent.resume_session(
                session_id,
                payload=request.payload if request is not None else None,
                timeout_s=request.timeout_s if request is not None else 30.0,
            )
        except EtcdQuorumUnavailableError:
            raise
        except (LeaseConflictError, StaleFenceError) as exc:
            _increment(
                app,
                "stale_fence_rejections"
                if isinstance(exc, StaleFenceError)
                else "lease_conflicts",
            )
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


def _distributed_components_from_env() -> tuple[
    MembershipManager | None,
    HmacAuthenticator | None,
    MulticastDiscoveryResponder | None,
]:
    store_url = os.getenv("COORDINATION_STORE_URL") or ""
    if not store_url.startswith("etcd://"):
        return None, None, None
    cluster_id = os.getenv("COORDINATION_CLUSTER_ID", "default")
    node_id = os.getenv("COORDINATION_COORDINATOR_ID", "coordinator-node")
    allow_insecure = _env_bool("COORDINATION_ALLOW_INSECURE_CLUSTER")
    authenticator = HmacAuthenticator(
        os.getenv("COORDINATION_CLUSTER_SECRET", ""),
        allow_insecure=allow_insecure,
    )
    configuration = ClusterConfiguration(
        cluster_id=cluster_id,
        voter_target=int(os.getenv("COORDINATION_VOTER_TARGET", "3")),
    )
    current_node = CoordinatorNodeRecord(
        node_id=node_id,
        api_url=os.getenv("COORDINATION_ADVERTISE_API_URL", "http://127.0.0.1:8000"),
        peer_url=os.getenv("COORDINATION_ADVERTISE_PEER_URL", ""),
        client_url=os.getenv("COORDINATION_ADVERTISE_CLIENT_URL", ""),
        role=cast(
            Literal["voter", "learner", "client", "draining"],
            os.getenv("COORDINATION_NODE_ROLE", "client"),
        ),
        member_id=int(os.getenv("COORDINATION_MEMBER_ID", "0")),
        voter_target=configuration.voter_target,
    )
    manager = MembershipManager(
        EtcdClient(etcd_endpoints_from_url(store_url)),
        configuration,
        current_node,
        reconcile_interval_s=float(
            os.getenv("COORDINATION_MEMBERSHIP_RECONCILE_INTERVAL_S", "2.0")
        ),
        registration_ttl_s=float(
            os.getenv("COORDINATION_NODE_REGISTRATION_TTL_S", "15.0")
        ),
        failed_voter_grace_s=float(
            os.getenv("COORDINATION_FAILED_VOTER_GRACE_S", "60.0")
        ),
    )
    def response_payload() -> dict[str, str]:
        return {
            "api_url": current_node.api_url,
            "client_url": current_node.client_url,
        }
    responder = MulticastDiscoveryResponder(
        cluster_id=cluster_id,
        node_id=node_id,
        authenticator=authenticator,
        response_payload=response_payload,
        multicast_group=os.getenv("COORDINATION_MULTICAST_GROUP", "239.255.42.99"),
        multicast_port=int(os.getenv("COORDINATION_MULTICAST_PORT", "7947")),
    )
    return manager, authenticator, responder


def _require_distributed(app: FastAPI) -> None:
    if app.state.membership is None or app.state.authenticator is None:
        raise HTTPException(status_code=503, detail="Distributed etcd mode is not configured.")


def _verify_envelope(app: FastAPI, request: SignedEnvelope, message_type: str) -> None:
    if request.message_type != message_type:
        raise HTTPException(status_code=422, detail=f"Expected message type {message_type}.")
    try:
        app.state.authenticator.verify(
            request,
            expected_cluster_id=app.state.membership.configuration.cluster_id,
        )
    except ValueError as exc:
        _increment(app, "hmac_rejections")
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _verify_agent_envelope(
    app: FastAPI,
    request: SignedEnvelope,
    message_type: str,
) -> None:
    if request.message_type != message_type:
        raise HTTPException(status_code=422, detail=f"Expected message type {message_type}.")
    if app.state.agent_authenticator is None:
        raise HTTPException(status_code=503, detail="Agent authentication is unavailable.")
    try:
        app.state.agent_authenticator.verify(
            request,
            expected_cluster_id=app.state.membership.configuration.cluster_id,
        )
    except ValueError as exc:
        _increment(app, "hmac_rejections")
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _increment(app: FastAPI, name: str) -> None:
    app.state.metrics[name] = int(app.state.metrics.get(name, 0)) + 1


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
