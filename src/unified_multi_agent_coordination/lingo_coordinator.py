"""Lingo-backed linguistic coordination surface."""

from __future__ import annotations

from lingo import Context, Engine, LLM, Message, State

from .models import (
    AgentRegistryEntry,
    FeasibilityReport,
    HydrationIssue,
    LinguisticPlanDraft,
    PredicateEvidence,
    ProblemRequest,
    SolutionProposal,
    TraceEvent,
)


class CoordinatorState(State):
    """Lingo session state for the coordinator's linguistic ledger."""

    def __init__(self, **kwargs) -> None:
        data: dict[str, object] = {
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

    def __init__(
        self,
        llm: LLM | None = None,
        *,
        max_candidates: int = 1,
        model_id: str = "",
        endpoint: str = "",
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.llm = llm or LLM()
        self.engine = Engine(self.llm)
        self.state = CoordinatorState()
        self.max_candidates = max(1, min(max_candidates, 3))
        self.model_id = model_id
        self.endpoint = endpoint
        self.temperature = temperature
        self.seed = seed

    @classmethod
    def for_lm_studio(
        cls,
        model_id: str,
        *,
        endpoint: str = "http://127.0.0.1:1234/v1",
        temperature: float = 0.0,
        seed: int = 11,
        max_candidates: int = 3,
    ) -> "LingoLinguisticCoordinator":
        """Create the explicitly configured local linguistic boundary."""
        llm = LLM(
            model=model_id,
            api_key="lm-studio-local",
            base_url=endpoint,
            temperature=temperature,
            seed=seed,
        )
        return cls(
            llm,
            max_candidates=max_candidates,
            model_id=model_id,
            endpoint=endpoint,
            temperature=temperature,
            seed=seed,
        )

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

    async def propose_draft(
        self, request: ProblemRequest, registry: list[AgentRegistryEntry]
    ) -> LinguisticPlanDraft:
        """Select only identifiers declared by the admitted request and registry."""
        ctx = Context([])
        ctx.append(Message.system(
            "Create a LinguisticPlanDraft, not an executable plan. Select each admitted "
            "requirement_id exactly once and use only its declared capability_id. Express "
            "dependencies only between declared requirement IDs. Never create task IDs, "
            "agents, credentials, trust levels, contracts, schemas, artifacts, or authority."
        ))
        ctx.append(Message.system("Admitted request:\n" + request.model_dump_json()))
        ctx.append(Message.system(
            "Available identifiers (descriptive only):\n"
            + "\n".join(agent.model_dump_json() for agent in registry)
        ))
        draft = await self.engine.create(ctx, LinguisticPlanDraft)
        self.state.candidate_plan = draft.model_dump(mode="json")
        self.state.record("linguistic_draft_proposed", "Non-executable draft produced.")
        return draft

    async def repair_draft(
        self,
        request: ProblemRequest,
        registry: list[AgentRegistryEntry],
        draft: LinguisticPlanDraft,
        issues: list[HydrationIssue],
    ) -> LinguisticPlanDraft:
        """Perform one caller-bounded repair using public hydration diagnostics."""
        ctx = Context([])
        ctx.append(Message.system(
            "Repair the LinguisticPlanDraft using the hydration errors. The same identifier "
            "and authority restrictions apply. Return only a LinguisticPlanDraft."
        ))
        ctx.append(Message.system("Admitted request:\n" + request.model_dump_json()))
        ctx.append(Message.system(
            "Registry:\n" + "\n".join(agent.model_dump_json() for agent in registry)
        ))
        ctx.append(Message.system("Rejected draft:\n" + draft.model_dump_json()))
        ctx.append(Message.system(
            "Hydration errors:\n" + "\n".join(issue.model_dump_json() for issue in issues)
        ))
        repaired = await self.engine.create(ctx, LinguisticPlanDraft)
        self.state.record("linguistic_draft_repaired", "One bounded repair produced.")
        return repaired

    async def propose_solutions(
        self, request: ProblemRequest, registry: list[AgentRegistryEntry]
    ) -> list[SolutionProposal]:
        """Request a bounded set of typed, non-authoritative candidates."""
        proposals = [
            await self.propose_solution(request, registry)
            for _ in range(self.max_candidates)
        ]
        self.state.record(
            "candidate_set_proposed",
            "Bounded candidate set produced.",
            candidate_count=len(proposals),
            model_id=self.model_id,
            endpoint=self.endpoint,
            temperature=self.temperature,
            seed=self.seed,
        )
        return proposals

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
