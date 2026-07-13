"""Local admission and credential policy for remote A2A agents."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse


class AgentAdmissionError(RuntimeError):
    """Raised when a remote card is outside the locally trusted boundary."""


class CredentialProvider(Protocol):
    """Provide ephemeral request headers; implementations must not serialize secrets."""

    def headers_for(self, agent_id: str, schemes: list[str]) -> Mapping[str, str]: ...

    def supports(self, schemes: list[str]) -> bool: ...


@dataclass(slots=True)
class StaticCredentialProvider:
    """Small runtime-only provider useful for deployments and integration tests."""

    headers: Mapping[str, str]
    supported_schemes: set[str] = field(default_factory=set)

    def headers_for(self, agent_id: str, schemes: list[str]) -> Mapping[str, str]:
        del agent_id
        return dict(self.headers) if self.supports(schemes) else {}

    def supports(self, schemes: list[str]) -> bool:
        return not schemes or set(schemes).issubset(self.supported_schemes)


@dataclass(slots=True)
class AgentAdmissionPolicy:
    """Fail-closed origin, transport, trust, and credential admission policy."""

    allowed_origins: set[str] = field(default_factory=set)
    require_https: bool = True
    allow_insecure_development: bool = field(
        default_factory=lambda: os.getenv("COORDINATION_ALLOW_INSECURE_A2A", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )
    default_trust_level: str = "standard"
    trust_by_origin: dict[str, str] = field(default_factory=dict)

    def admit_endpoint(self, endpoint: str) -> str:
        parsed = urlparse(endpoint)
        if not parsed.scheme or not parsed.netloc:
            raise AgentAdmissionError("Agent endpoint must be an absolute URL.")
        origin = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
        if self.allowed_origins and origin not in self.allowed_origins:
            raise AgentAdmissionError(f"Agent origin {origin} is not allowed.")
        if (
            self.require_https
            and parsed.scheme.lower() != "https"
            and not self.allow_insecure_development
        ):
            raise AgentAdmissionError("Plain HTTP A2A endpoints require an explicit development exception.")
        return self.trust_by_origin.get(origin, self.default_trust_level)

    def require_credentials(
        self,
        required_schemes: list[str],
        provider: CredentialProvider | None,
    ) -> None:
        if required_schemes and (provider is None or not provider.supports(required_schemes)):
            raise AgentAdmissionError(
                "No local credential provider supports required schemes: "
                + ", ".join(required_schemes)
            )
