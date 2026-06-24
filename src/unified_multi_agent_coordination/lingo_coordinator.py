"""Lingo-backed linguistic coordination surface."""

from __future__ import annotations

from lingo import Context, Engine, LLM, Message, State

from .models import (
    AgentRegistryEntry,
    FeasibilityReport,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TraceEvent,
)


class CoordinatorState(State):
    """Lingo session state for the coordinator's linguistic ledger."""

    def __init__(self, **kwargs) -> None:
        data = {
            "registry_snapshot": [],
            "interpreted_request": None,
            "candidate_plan": None,
            "feasibility_evidence": [],
            "trace": [],
        }
        data.update(kwargs)
        super().__init__(data=data)

    def record(self, event_type: str, message: str, **data) -> TraceEvent:
        event = TraceEvent(event_type=event_type, message=message, data=data)
        self.trace.append(event.model_dump(mode="json"))
        return event


class LingoLinguisticCoordinator:
    """Produce typed requests and candidate plans without authorizing them."""

    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm or LLM()
        self.engine = Engine(self.llm)
        self.state = CoordinatorState()

    async def interpret_request(
        self, user_text: str, registry: list[AgentRegistryEntry]
    ) -> ProblemRequest:
        """Use Lingo structured generation to admit a typed problem request."""
        ctx = Context([])
        ctx.append(
            Message.system(
                "Extract a ProblemRequest. Do not authorize execution. "
                "Only describe requirements, constraints, artifacts, and context."
            )
        )
        ctx.append(
            Message.system(
                "Available registry snapshot:\n"
                + "\n".join(agent.model_dump_json() for agent in registry)
            )
        )
        ctx.append(Message.user(user_text))
        request = await self.engine.create(ctx, ProblemRequest)
        self.state.registry_snapshot = [agent.model_dump(mode="json") for agent in registry]
        self.state.interpreted_request = request.model_dump(mode="json")
        self.state.record("request_interpreted", "ProblemRequest extracted.")
        return request

    async def propose_solution(
        self, request: ProblemRequest, registry: list[AgentRegistryEntry]
    ) -> SolutionProposal:
        """Use Lingo structured generation to produce a non-authoritative plan."""
        ctx = Context([])
        ctx.append(
            Message.system(
                "Create a candidate SolutionProposal. The proposal is not "
                "authorized until the symbolic FeasibilityAnalyzer accepts it."
            )
        )
        ctx.append(Message.system(request.model_dump_json()))
        ctx.append(
            Message.system(
                "Registry snapshot:\n"
                + "\n".join(agent.model_dump_json() for agent in registry)
            )
        )
        proposal = await self.engine.create(ctx, SolutionProposal)
        self.state.candidate_plan = proposal.model_dump(mode="json")
        self.state.record("solution_proposed", "Candidate SolutionProposal produced.")
        return proposal

    def record_feasibility(self, report: FeasibilityReport) -> None:
        self.state.feasibility_evidence = [
            evidence.model_dump(mode="json") for evidence in report.evidence
        ]
        self.state.record(
            "feasibility_recorded",
            report.explanation,
            feasible=report.feasible,
        )


def evidence_names(report: FeasibilityReport) -> list[str]:
    """Small helper used by examples and tests."""
    return [item.name for item in report.evidence if isinstance(item, PredicateEvidence)]
