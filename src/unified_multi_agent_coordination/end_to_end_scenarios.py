"""End-to-end thesis demonstration scenarios.

The scenarios in this module are deliberately small, deterministic, and local.
They exercise the thesis implementation as a coordination system: a natural
language request is interpreted into a structured problem, a candidate plan is
proposed, symbolic feasibility authorizes or refuses the plan, the SDK dispatches
authorized tasks, and the final report preserves artifacts plus trace evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .coordination_agent import CoordinationAgent
from .admission import AgentAdmissionPolicy
from .coordination_ledger import InMemoryCoordinationLedger, RetryPolicy
from .coordination_sdk import CoordinationSdk
from .models import (
    AgentRegistryEntry,
    CapabilityRequirement,
    ConstraintSpec,
    FeasibilityReport,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
    ValidationContract,
)


JsonObject = dict[str, Any]
ScenarioStatus = Literal["completed", "infeasible", "failed"]


@dataclass(frozen=True)
class ScenarioDefinition:
    """One executable local scenario for thesis demonstration."""

    scenario_id: str
    title: str
    user_request: str
    thesis_value: str
    request: ProblemRequest
    proposal: SolutionProposal
    payload: JsonObject
    setup: Callable[[CoordinationSdk], Awaitable[None]]
    reference_status: ScenarioStatus
    showcase: bool = False
    timeout_s: float = 10.0


@dataclass
class _DeterministicCoordinatorState:
    interpreted_request: JsonObject = field(default_factory=dict)
    candidate_plan: JsonObject = field(default_factory=dict)
    trace: list[JsonObject] = field(default_factory=list)


class DeterministicScenarioCoordinator:
    """Fixed linguistic coordinator used instead of an external LLM service."""

    def __init__(self, request: ProblemRequest, proposal: SolutionProposal) -> None:
        self.request = request
        self.proposal = proposal
        self.state = _DeterministicCoordinatorState()

    async def interpret_request(
        self,
        user_request: str,
        registry: list[AgentRegistryEntry],
    ) -> ProblemRequest:
        self.state.interpreted_request = self.request.model_dump(mode="json")
        self._record(
            "request_interpreted",
            "Scenario request interpreted from a fixed local fixture.",
            user_request=user_request,
            registry_agents=[agent.agent_id for agent in registry],
        )
        return self.request

    async def propose_solution(
        self,
        request: ProblemRequest,
        registry: list[AgentRegistryEntry],
    ) -> SolutionProposal:
        self.state.candidate_plan = self.proposal.model_dump(mode="json")
        self._record(
            "solution_proposed",
            "Scenario candidate plan proposed from a fixed local fixture.",
            requirement_count=len(request.requirements),
            task_count=len(self.proposal.tasks),
            registry_agents=[agent.agent_id for agent in registry],
        )
        return self.proposal

    def record_feasibility(self, report: FeasibilityReport) -> None:
        self._record(
            "feasibility_recorded",
            "Symbolic feasibility report recorded for the scenario proposal.",
            feasible=report.feasible,
            failed_predicates=[
                item.name for item in report.evidence if not item.passed
            ],
            missing_capabilities=report.missing_capabilities,
            risks=report.risks,
        )

    def _record(self, event_type: str, message: str, **data: Any) -> None:
        self.state.trace.append(
            {
                "event_type": event_type,
                "message": message,
                "data": data,
            }
        )


class ScenarioCoordinationAgent(CoordinationAgent):
    """Demo agent that always asks the deterministic coordinator for a plan."""

    def _direct_solution_plan(
        self,
        request: ProblemRequest,
        registry: list[AgentRegistryEntry],
    ) -> SolutionProposal | None:
        return None


async def run_scenarios(
    *,
    only: set[str] | None = None,
    showcase_only: bool = False,
) -> JsonObject:
    """Run selected end-to-end scenarios and return a JSON-ready report."""
    definitions = scenario_definitions()
    selected = [
        scenario
        for scenario in definitions
        if (only is None or scenario.scenario_id in only)
        and (not showcase_only or scenario.showcase)
    ]
    unknown = sorted((only or set()) - {scenario.scenario_id for scenario in definitions})
    if unknown:
        raise ValueError(f"Unknown scenario id(s): {', '.join(unknown)}")

    results = [await _run_scenario(scenario) for scenario in selected]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "no_external_services": True,
        "scenario_count": len(results),
        "scenarios": results,
    }


def scenario_definitions() -> list[ScenarioDefinition]:
    """Return the built-in deterministic end-to-end thesis scenarios."""
    return [
        _research_brief_scenario(),
        _financial_report_scenario(),
        _smart_building_scenario(),
        _warehouse_scenario(),
        _incident_response_scenario(),
        _privacy_redaction_scenario(),
        _travel_plan_scenario(),
    ]


async def _run_scenario(scenario: ScenarioDefinition) -> JsonObject:
    sdk = CoordinationSdk(
        admission_policy=AgentAdmissionPolicy(allow_insecure_development=True)
    )
    await scenario.setup(sdk)
    ledger = InMemoryCoordinationLedger()
    coordinator = DeterministicScenarioCoordinator(
        scenario.request,
        scenario.proposal,
    )
    agent = ScenarioCoordinationAgent(
        sdk=sdk,
        linguistic_coordinator=coordinator,  # type: ignore[arg-type]
        ledger=ledger,
        retry_policy=RetryPolicy(registry_retries=0, task_retries=0, backoff_s=0),
    )
    session_id = f"demo-{scenario.scenario_id}"
    result = await agent.coordinate(
        scenario.user_request,
        payload=scenario.payload,
        timeout_s=scenario.timeout_s,
        session_id=session_id,
    )
    ledger_events = ledger.events(session_id)
    report = result.plan_result.feasibility_report
    return {
        "id": scenario.scenario_id,
        "title": scenario.title,
        "showcase": scenario.showcase,
        "user_request": scenario.user_request,
        "thesis_value": scenario.thesis_value,
        "reference_status": scenario.reference_status,
        "observed_status": result.status,
        "status_matches_reference": result.status == scenario.reference_status,
        "authorization": {
            "feasible": report.feasible,
            "explanation": report.explanation,
            "failed_predicates": [
                item.name for item in report.evidence if not item.passed
            ],
            "passed_predicates": [
                item.name for item in report.evidence if item.passed
            ],
            "missing_capabilities": report.missing_capabilities,
            "risks": report.risks,
        },
        "registry": [
            {
                "agent_id": agent_record.agent_id,
                "agent_kind": agent_record.agent_kind,
                "trust_level": agent_record.trust_level,
                "skills": [skill.name for skill in agent_record.skills],
            }
            for agent_record in result.plan_result.registry_snapshot
        ],
        "task_results": [
            {
                "task_id": task_result.task_id,
                "agent_id": task_result.agent_id,
                "agent_kind": task_result.agent_kind,
                "status": task_result.status,
                "artifacts": task_result.artifacts,
                "error": task_result.error,
            }
            for task_result in result.task_results
        ],
        "artifacts": result.artifacts,
        "trace_event_types": [event.event_type for event in result.trace],
        "ledger_event_types": [event.event_type for event in ledger_events],
        "dispatch_attempts": sum(
            1 for event in ledger_events if event.event_type == "task_attempt_started"
        ),
        "no_dispatch_before_authorization": _no_dispatch_before_authorization(
            [event.event_type for event in ledger_events]
        ),
    }


def _no_dispatch_before_authorization(event_types: list[str]) -> bool:
    if "task_attempt_started" not in event_types:
        return True
    try:
        return event_types.index("plan_authorized") < event_types.index(
            "task_attempt_started"
        )
    except ValueError:
        return False


def _research_brief_scenario() -> ScenarioDefinition:
    extract = _capability(
        "extract key points",
        input_modes=["notes"],
        output_modes=["key_points"],
        validation_contract={"required_fields": ["key_points"]},
    )
    summarize = _capability(
        "summarize brief",
        input_modes=["key_points"],
        output_modes=["draft_text"],
        validation_contract={"required_fields": ["draft"]},
    )
    check = _capability(
        "check citations",
        input_modes=["draft_text"],
        output_modes=["citation_report"],
        validation_contract={"required_fields": ["citation_status"]},
    )
    format_brief = _capability(
        "format brief",
        input_modes=["draft_text", "citation_report"],
        output_modes=["markdown"],
        validation_contract={"required_artifacts": ["research_brief"]},
    )
    request = ProblemRequest(
        user_goal="Create a short research brief from these notes.",
        requirements=[extract, summarize, check, format_brief],
        required_artifacts=["research_brief"],
    )
    proposal = _proposal(
        [
            TaskSpec(
                task_id="t1",
                requirement_name=extract.name,
                assigned_to="note-extractor",
            ),
            TaskSpec(
                task_id="t2",
                requirement_name=summarize.name,
                assigned_to="brief-summarizer",
                depends_on=["t1"],
            ),
            TaskSpec(
                task_id="t3",
                requirement_name=check.name,
                assigned_to="citation-checker",
                depends_on=["t2"],
            ),
            TaskSpec(
                task_id="t4",
                requirement_name=format_brief.name,
                assigned_to="brief-formatter",
                depends_on=["t2", "t3"],
                validation_contract=ValidationContract(
                    required_artifacts=["research_brief"]
                ),
            ),
        ],
        expected_artifacts=["research_brief"],
    )

    async def setup(sdk: CoordinationSdk) -> None:
        sdk.register_local_agent(
            "Note Extractor",
            [extract],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "data",
                        "data": {
                            "key_points": [
                                "Hybrid coordination separates proposal from authorization.",
                                "Symbolic predicates make refusals auditable.",
                                "SDK dispatch normalizes heterogeneous agents.",
                            ]
                        },
                    }
                ]
            },
        )
        sdk.register_linguistic_agent(
            "Brief Summarizer",
            [summarize],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "text",
                        "draft": "The notes argue for a hybrid coordinator that lets language propose while symbolic checks authorize execution.",
                    }
                ]
            },
        )
        await _register_fake_a2a_agent(
            sdk,
            agent_id="citation-checker",
            name="Citation Checker",
            capability=check,
            response={
                "artifacts": [
                    {
                        "kind": "data",
                        "data": {
                            "citation_status": "all cited claims have local references"
                        },
                    }
                ]
            },
        )
        sdk.register_local_agent(
            "Brief Formatter",
            [format_brief],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "markdown",
                        "name": "research_brief",
                        "markdown": "# Research Brief\n\nHybrid proposal and symbolic authorization provide an auditable coordination boundary.",
                    }
                ]
            },
        )

    return ScenarioDefinition(
        scenario_id="research_brief_production",
        title="Research Brief Production",
        user_request="Create a short research brief from these notes.",
        thesis_value="Successful decomposed coordination across local, linguistic, and fake A2A agents with traceable artifact evidence.",
        request=request,
        proposal=proposal,
        payload={
            "notes": "LLMs propose flexible plans. Symbolic predicates authorize capability, dependency, and contract evidence.",
        },
        setup=setup,
        reference_status="completed",
        showcase=True,
    )


def _financial_report_scenario() -> ScenarioDefinition:
    extract = _capability(
        "extract revenue figures",
        input_modes=["csv"],
        output_modes=["revenue_data"],
        validation_contract={"required_fields": ["q1_revenue", "q2_revenue"]},
    )
    calculate = _capability(
        "calculate quarterly growth",
        input_modes=["revenue_data"],
        output_modes=["growth_data"],
        validation_contract={"required_fields": ["growth_rate"]},
    )
    summarize = _capability(
        "summarize financial result",
        input_modes=["growth_data"],
        output_modes=["draft_text"],
        validation_contract={"required_fields": ["summary"]},
    )
    compliance = _capability(
        "approve compliance safe summary",
        input_modes=["draft_text"],
        output_modes=["compliance_report"],
        validation_contract={"required_fields": ["approval_status"]},
    )
    request = ProblemRequest(
        user_goal="Extract revenue figures, calculate quarterly growth, and produce a compliance-safe summary.",
        requirements=[extract, calculate, summarize, compliance],
        required_artifacts=["compliance_report"],
    )
    proposal = _proposal(
        [
            TaskSpec(task_id="t1", requirement_name=extract.name, assigned_to="table-extractor"),
            TaskSpec(
                task_id="t2",
                requirement_name=calculate.name,
                assigned_to="growth-calculator",
                depends_on=["t1"],
            ),
            TaskSpec(
                task_id="t3",
                requirement_name=summarize.name,
                assigned_to="finance-summarizer",
                depends_on=["t2"],
            ),
            TaskSpec(
                task_id="t4",
                requirement_name=compliance.name,
                assigned_to="compliance-checker",
                depends_on=["t3"],
            ),
        ],
        expected_artifacts=["compliance_report"],
    )

    async def setup(sdk: CoordinationSdk) -> None:
        sdk.register_local_agent(
            "Table Extractor",
            [extract],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "data",
                        "data": {"q1_revenue": 100000, "q2_revenue": 125000},
                    }
                ]
            },
        )
        sdk.register_local_agent(
            "Growth Calculator",
            [calculate],
            lambda payload: {
                "artifacts": [
                    {"kind": "data", "data": {"growth_rate": 0.25}}
                ]
            },
        )
        sdk.register_linguistic_agent(
            "Finance Summarizer",
            [summarize],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "text",
                        "summary": "Revenue increased by 25 percent from Q1 to Q2.",
                    }
                ]
            },
        )
        sdk.register_local_agent(
            "Compliance Checker",
            [compliance],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "data",
                        "data": {"checked_by": "local-policy-fixture"},
                    }
                ]
            },
        )

    return ScenarioDefinition(
        scenario_id="financial_report_validation",
        title="Financial Report Validation",
        user_request="Extract revenue figures, calculate quarterly growth, and produce a compliance-safe summary.",
        thesis_value="A plan can be symbolically authorized but still fail runtime artifact validation when required completion evidence is absent.",
        request=request,
        proposal=proposal,
        payload={"csv": "quarter,revenue\nQ1,100000\nQ2,125000\n"},
        setup=setup,
        reference_status="failed",
        showcase=True,
    )


def _smart_building_scenario() -> ScenarioDefinition:
    parse = _capability(
        "parse sensor alert",
        input_modes=["json"],
        output_modes=["sensor_event"],
    )
    classify = _capability(
        "classify maintenance fault",
        input_modes=["sensor_event"],
        output_modes=["fault_label"],
    )
    work_order = _capability(
        "create maintenance work order",
        input_modes=["fault_label"],
        output_modes=["work_order"],
    )
    shutdown = _capability(
        "shut down hvac",
        input_modes=["fault_label"],
        output_modes=["control_action"],
        required_trust_level="admin",
    )
    request = ProblemRequest(
        user_goal="Diagnose this building sensor alert and create the next maintenance action.",
        requirements=[parse, classify, work_order, shutdown],
        required_artifacts=["work_order", "control_action"],
    )
    proposal = _proposal(
        [
            TaskSpec(task_id="t1", requirement_name=parse.name, assigned_to="sensor-parser"),
            TaskSpec(
                task_id="t2",
                requirement_name=classify.name,
                assigned_to="fault-classifier",
                depends_on=["t1"],
            ),
            TaskSpec(
                task_id="t3",
                requirement_name=work_order.name,
                assigned_to="work-order-generator",
                depends_on=["t2"],
            ),
            TaskSpec(
                task_id="t4",
                requirement_name=shutdown.name,
                assigned_to="hvac-dispatcher",
                depends_on=["t2"],
            ),
        ],
        expected_artifacts=["work_order", "control_action"],
    )

    async def setup(sdk: CoordinationSdk) -> None:
        sdk.register_local_agent("Sensor Parser", [parse], lambda payload: {})
        sdk.register_local_agent("Fault Classifier", [classify], lambda payload: {})
        sdk.register_local_agent("Work Order Generator", [work_order], lambda payload: {})
        sdk.register_local_agent(
            "HVAC Dispatcher",
            [shutdown],
            lambda payload: {},
            trust_level="standard",
        )

    return ScenarioDefinition(
        scenario_id="smart_building_maintenance_triage",
        title="Smart Building Maintenance Triage",
        user_request="Diagnose this building sensor alert and create the next maintenance action.",
        thesis_value="The coordinator refuses an otherwise plausible building-control plan because a requested actuator action requires unavailable authority.",
        request=request,
        proposal=proposal,
        payload={"sensor": {"zone": "4B", "temperature_c": 35.2, "fan_rpm": 0}},
        setup=setup,
        reference_status="infeasible",
    )


def _warehouse_scenario() -> ScenarioDefinition:
    inventory = _capability(
        "lookup inventory item",
        input_modes=["json"],
        output_modes=["item_location"],
    )
    route = _capability(
        "plan warehouse route",
        input_modes=["item_location"],
        output_modes=["route"],
    )
    safety = _capability(
        "validate route safety",
        input_modes=["route"],
        output_modes=["safety_report"],
        validation_contract={"required_fields": ["safe"]},
    )
    dispatch = _capability(
        "dispatch warehouse move",
        input_modes=["safety_report"],
        output_modes=["dispatch_ticket"],
        validation_contract={"required_artifacts": ["dispatch_ticket"]},
    )
    request = ProblemRequest(
        user_goal="Move item A to packing station B while avoiding blocked zones.",
        requirements=[inventory, route, safety, dispatch],
        required_artifacts=["dispatch_ticket"],
    )
    proposal = _proposal(
        [
            TaskSpec(task_id="t1", requirement_name=inventory.name, assigned_to="inventory-lookup"),
            TaskSpec(
                task_id="t2",
                requirement_name=route.name,
                assigned_to="route-planner",
                depends_on=["t1"],
            ),
            TaskSpec(
                task_id="t3",
                requirement_name=safety.name,
                assigned_to="safety-validator",
                depends_on=["t2"],
            ),
            TaskSpec(
                task_id="t4",
                requirement_name=dispatch.name,
                assigned_to="move-dispatcher",
                depends_on=["t3"],
                validation_contract=ValidationContract(
                    required_artifacts=["dispatch_ticket"]
                ),
            ),
        ],
        expected_artifacts=["dispatch_ticket"],
    )

    async def setup(sdk: CoordinationSdk) -> None:
        sdk.register_local_agent(
            "Inventory Lookup",
            [inventory],
            lambda payload: {
                "artifacts": [
                    {"kind": "data", "data": {"item": "A", "location": [0, 0]}}
                ]
            },
        )
        sdk.register_local_agent(
            "Route Planner",
            [route],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "data",
                        "data": {"route": [[0, 0], [0, 1], [1, 1], [2, 1]]},
                    }
                ]
            },
        )
        sdk.register_local_agent(
            "Safety Validator",
            [safety],
            lambda payload: {
                "artifacts": [{"kind": "data", "data": {"safe": True}}]
            },
        )
        sdk.register_local_agent(
            "Move Dispatcher",
            [dispatch],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "data",
                        "name": "dispatch_ticket",
                        "data": {"ticket_id": "WH-LOCAL-001"},
                    }
                ]
            },
        )

    return ScenarioDefinition(
        scenario_id="warehouse_task_coordination",
        title="Simulated Warehouse Task Coordination",
        user_request="Move item A to packing station B while avoiding blocked zones.",
        thesis_value="A complete local autonomy-style workflow demonstrates ordered dependencies, safety validation, dispatch, and artifact aggregation.",
        request=request,
        proposal=proposal,
        payload={
            "grid": {
                "blocked": [[1, 0]],
                "item": "A",
                "destination": "packing-station-B",
            }
        },
        setup=setup,
        reference_status="completed",
    )


def _incident_response_scenario() -> ScenarioDefinition:
    summarize = _capability(
        "summarize incident log",
        input_modes=["log_text"],
        output_modes=["incident_summary"],
    )
    severity = _capability(
        "classify incident severity",
        input_modes=["incident_summary"],
        output_modes=["severity"],
    )
    recommend = _capability(
        "recommend recovery plan",
        input_modes=["severity"],
        output_modes=["runbook"],
    )
    execute = _capability(
        "execute production remediation",
        input_modes=["runbook"],
        output_modes=["remediation_receipt"],
        required_trust_level="admin",
    )
    request = ProblemRequest(
        user_goal="Triage this incident report and recommend a recovery plan.",
        requirements=[summarize, severity, recommend, execute],
        required_artifacts=["runbook", "remediation_receipt"],
    )
    proposal = _proposal(
        [
            TaskSpec(task_id="t1", requirement_name=summarize.name, assigned_to="log-summarizer"),
            TaskSpec(
                task_id="t2",
                requirement_name=severity.name,
                assigned_to="severity-classifier",
                depends_on=["t1"],
            ),
            TaskSpec(
                task_id="t3",
                requirement_name=recommend.name,
                assigned_to="runbook-recommender",
                depends_on=["t2"],
            ),
            TaskSpec(
                task_id="t4",
                requirement_name=execute.name,
                assigned_to="remediation-executor",
                depends_on=["t3"],
            ),
        ],
        expected_artifacts=["runbook", "remediation_receipt"],
    )

    async def setup(sdk: CoordinationSdk) -> None:
        sdk.register_linguistic_agent("Log Summarizer", [summarize], lambda payload: {})
        sdk.register_local_agent("Severity Classifier", [severity], lambda payload: {})
        sdk.register_local_agent("Runbook Recommender", [recommend], lambda payload: {})
        sdk.register_local_agent(
            "Remediation Executor",
            [execute],
            lambda payload: {},
            trust_level="standard",
        )

    return ScenarioDefinition(
        scenario_id="incident_response_runbook",
        title="Incident Response Runbook",
        user_request="Triage this incident report and recommend a recovery plan.",
        thesis_value="A plausible linguistic overreach into production remediation is blocked by symbolic authority checks before any dispatch.",
        request=request,
        proposal=proposal,
        payload={
            "log": "03:12 elevated errors; 03:15 primary database latency; 03:17 operator asks for recovery advice."
        },
        setup=setup,
        reference_status="infeasible",
        showcase=True,
    )


def _privacy_redaction_scenario() -> ScenarioDefinition:
    extract_pii = _capability(
        "extract pii fields",
        description="extract email and phone from a transcript",
        input_modes=["text"],
        output_modes=["pii_entities"],
        output_schema={"required": ["email", "phone"]},
        auxiliary_eligible=True,
    )
    redact = _capability(
        "redact transcript",
        input_modes=["pii_entities", "text"],
        output_modes=["redacted_text"],
        validation_contract={"required_fields": ["redacted"]},
    )
    validate = _capability(
        "validate redaction quality",
        input_modes=["redacted_text"],
        output_modes=["quality_report"],
        validation_contract={"required_fields": ["quality"]},
    )
    summarize = _capability(
        "summarize redacted transcript",
        input_modes=["quality_report"],
        output_modes=["shareable_summary"],
        validation_contract={"required_artifacts": ["shareable_summary"]},
    )
    request = ProblemRequest(
        user_goal="Prepare this customer transcript for sharing.",
        requirements=[extract_pii, redact, validate, summarize],
        required_artifacts=["shareable_summary"],
    )
    proposal = _proposal(
        [
            TaskSpec(task_id="t1", requirement_name=extract_pii.name),
            TaskSpec(
                task_id="t2",
                requirement_name=redact.name,
                assigned_to="transcript-redactor",
                depends_on=["t1"],
            ),
            TaskSpec(
                task_id="t3",
                requirement_name=validate.name,
                assigned_to="redaction-validator",
                depends_on=["t2"],
            ),
            TaskSpec(
                task_id="t4",
                requirement_name=summarize.name,
                assigned_to="privacy-summary-generator",
                depends_on=["t3"],
                validation_contract=ValidationContract(
                    required_artifacts=["shareable_summary"]
                ),
            ),
        ],
        expected_artifacts=["shareable_summary"],
    )

    async def setup(sdk: CoordinationSdk) -> None:
        def redact_handler(payload):
            dependency_artifacts = (
                payload.get("_coordination", {}).get("previous_artifacts", [])
            )
            extracted = next(
                (
                    artifact.get("data", {})
                    for artifact in dependency_artifacts
                    if isinstance(artifact.get("data"), dict)
                ),
                {},
            )
            return {
                "artifacts": [
                    {
                        "kind": "text",
                        "redacted": payload["text"]
                        .replace(extracted["email"], "[email]")
                        .replace(extracted["phone"], "[phone]"),
                    }
                ]
            }

        sdk.register_local_agent(
            "Transcript Redactor",
            [redact],
            redact_handler,
        )
        sdk.register_local_agent(
            "Redaction Validator",
            [validate],
            lambda payload: {
                "artifacts": [{"kind": "data", "data": {"quality": "passed"}}]
            },
        )
        sdk.register_linguistic_agent(
            "Privacy Summary Generator",
            [summarize],
            lambda payload: {
                "artifacts": [
                    {
                        "kind": "text",
                        "name": "shareable_summary",
                        "text": "Customer requested account help; direct identifiers were removed.",
                    }
                ]
            },
        )

    return ScenarioDefinition(
        scenario_id="data_privacy_redaction_pipeline",
        title="Data Privacy Redaction Pipeline",
        user_request="Prepare this customer transcript for sharing.",
        thesis_value="A narrow missing extraction step is admitted as a bounded auxiliary capability, then ordinary agents complete the redaction workflow.",
        request=request,
        proposal=proposal,
        payload={
            "text": "Customer Ana wrote from ana@example.com and asked for a callback at 555-0100.",
        },
        setup=setup,
        reference_status="completed",
    )


def _travel_plan_scenario() -> ScenarioDefinition:
    match = _capability(
        "match attractions",
        input_modes=["dataset"],
        output_modes=["candidate_stops"],
    )
    budget = _capability(
        "calculate itinerary budget",
        input_modes=["candidate_stops"],
        output_modes=["budget_report"],
    )
    accessibility = _capability(
        "verify wheelchair accessibility",
        input_modes=["budget_report"],
        output_modes=["accessibility_report"],
    )
    format_plan = _capability(
        "format one day itinerary",
        input_modes=["accessibility_report"],
        output_modes=["itinerary"],
    )
    request = ProblemRequest(
        user_goal="Plan a one-day itinerary under budget and accessibility constraints.",
        requirements=[match, budget, accessibility, format_plan],
        constraints=[
            ConstraintSpec.from_legacy_text("budget <= 80"),
            ConstraintSpec.from_legacy_text("wheelchair accessible"),
        ],
        required_artifacts=["itinerary"],
    )
    proposal = _proposal(
        [
            TaskSpec(task_id="t1", requirement_name=match.name, assigned_to="attraction-matcher"),
            TaskSpec(
                task_id="t2",
                requirement_name=budget.name,
                assigned_to="budget-calculator",
                depends_on=["t1"],
            ),
            TaskSpec(
                task_id="t3",
                requirement_name=accessibility.name,
                assigned_to="accessibility-checker",
                depends_on=["t2"],
            ),
            TaskSpec(
                task_id="t4",
                requirement_name=format_plan.name,
                assigned_to="itinerary-formatter",
                depends_on=["t3"],
            ),
        ],
        expected_artifacts=["itinerary"],
    )

    async def setup(sdk: CoordinationSdk) -> None:
        sdk.register_local_agent("Attraction Matcher", [match], lambda payload: {})
        sdk.register_local_agent("Budget Calculator", [budget], lambda payload: {})
        sdk.register_local_agent("Itinerary Formatter", [format_plan], lambda payload: {})

    return ScenarioDefinition(
        scenario_id="travel_plan_local_fixtures",
        title="Travel Plan With Local Fixtures",
        user_request="Plan a one-day itinerary under budget and accessibility constraints.",
        thesis_value="The coordinator refuses to fill an accessibility constraint with a nonexistent capability, even when the rest of the itinerary pipeline is available.",
        request=request,
        proposal=proposal,
        payload={
            "attractions": [
                {"name": "Museum", "cost": 25, "accessible": True},
                {"name": "Rooftop Walk", "cost": 40, "accessible": False},
            ],
            "budget": 80,
        },
        setup=setup,
        reference_status="infeasible",
    )


def _capability(
    name: str,
    *,
    description: str = "",
    input_modes: list[str] | None = None,
    output_modes: list[str] | None = None,
    output_schema: JsonObject | None = None,
    validation_contract: JsonObject | None = None,
    auxiliary_eligible: bool = False,
    required_trust_level: str = "standard",
) -> CapabilityRequirement:
    return CapabilityRequirement(
        name=name,
        description=description,
        input_modes=list(input_modes or []),
        output_modes=list(output_modes or []),
        output_schema=dict(output_schema or {}),
        validation_contract=ValidationContract.model_validate(
            validation_contract or {"json_schema": {"type": "object"}}
        ),
        auxiliary_eligible=auxiliary_eligible,
        required_trust_level=required_trust_level,
    )


def _proposal(
    tasks: list[TaskSpec],
    *,
    expected_artifacts: list[str],
) -> SolutionProposal:
    return SolutionProposal(
        tasks=tasks,
        selected_agents={
            task.task_id: task.assigned_to
            for task in tasks
            if task.assigned_to is not None
        },
        execution_order=[task.task_id for task in tasks],
        expected_artifacts=expected_artifacts,
        completion_criteria=[
            f"{artifact} artifact exists" for artifact in expected_artifacts
        ],
    )


async def _register_fake_a2a_agent(
    sdk: CoordinationSdk,
    *,
    agent_id: str,
    name: str,
    capability: CapabilityRequirement,
    response: JsonObject,
) -> None:
    async def fetcher(url: str) -> JsonObject:
        return {
            "id": agent_id,
            "name": name,
            "description": f"Fake local A2A card for {name}",
            "url": url,
            "skills": [
                {
                    "id": capability.capability_id,
                    "name": capability.name,
                    "description": capability.description,
                    "inputModes": capability.input_modes,
                    "outputModes": capability.output_modes,
                }
            ],
        }

    async def sender(target_agent_id: str, payload: JsonObject) -> JsonObject:
        if target_agent_id != agent_id:
            raise RuntimeError(f"Unexpected fake A2A target {target_agent_id}.")
        return response

    sdk.a2a_adapter.card_fetcher = fetcher
    sdk.a2a_adapter.task_sender = sender
    await sdk.register_a2a_agent(f"fake-a2a://{agent_id}/card.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic end-to-end thesis coordination scenarios.",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Scenario id to run. Repeat to run multiple selected scenarios.",
    )
    parser.add_argument(
        "--showcase",
        action="store_true",
        help="Run only the three primary thesis showcase scenarios.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("demo_runs/end_to_end_scenarios.json"),
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print the summary without writing a JSON report.",
    )
    args = parser.parse_args(argv)

    report = asyncio.run(
        run_scenarios(
            only=set(args.only) if args.only else None,
            showcase_only=args.showcase,
        )
    )
    for item in report["scenarios"]:
        feasible = item["authorization"]["feasible"]
        dispatches = item["dispatch_attempts"]
        print(
            f"{item['id']}: {item['observed_status']} "
            f"(reference={item['reference_status']}, feasible={feasible}, "
            f"dispatches={dispatches})"
        )

    if not args.no_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote scenario report to {args.output}")


if __name__ == "__main__":
    main()
