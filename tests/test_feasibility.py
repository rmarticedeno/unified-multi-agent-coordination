from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    BoundedAuxiliaryCapabilityFactory,
    CapabilityRequirement,
    FeasibilityAnalyzer,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
    ValidationContract,
)


def _agent(
    agent_id: str,
    skill: CapabilityRequirement,
    *,
    status: str = "available",
    trust_level: str = "standard",
) -> AgentRegistryEntry:
    return AgentRegistryEntry(
        agent_id=agent_id,
        name=agent_id,
        service_endpoint=f"http://{agent_id}.example",
        skills=[skill],
        input_modes=skill.input_modes,
        output_modes=skill.output_modes,
        status=status,  # type: ignore[arg-type]
        trust_level=trust_level,
    )


def _proposal(*tasks: TaskSpec, artifacts: list[str] | None = None) -> SolutionProposal:
    tasks = tuple(
        task
        if task.validation_contract.enforceable()
        else task.model_copy(
            update={"validation_contract": ValidationContract(json_schema={"type": "object"})}
        )
        for task in tasks
    )
    return SolutionProposal(
        tasks=list(tasks),
        execution_order=[task.task_id for task in tasks],
        expected_artifacts=artifacts or ["result"],
        completion_criteria=["all task validators pass"],
    )


def test_direct_delegation_is_authorized():
    req = CapabilityRequirement(name="summarize", input_modes=["text"], output_modes=["text"])
    request = ProblemRequest(
        user_goal="Summarize the report.",
        requirements=[req],
        required_artifacts=["summary"],
    )
    proposal = _proposal(
        TaskSpec(task_id="t1", requirement_name="summarize", assigned_to="a1"),
        artifacts=["summary"],
    )

    report = FeasibilityAnalyzer().check(request, [_agent("a1", req)], proposal)

    assert report.feasible
    assert report.matched_agents == {"t1": "a1"}


def test_decomposed_plan_is_authorized_when_modes_match():
    extract = CapabilityRequirement(
        name="extract numbers",
        input_modes=["table"],
        output_modes=["numbers"],
    )
    calculate = CapabilityRequirement(
        name="calculate percentage",
        input_modes=["numbers"],
        output_modes=["number"],
    )
    request = ProblemRequest(
        user_goal="Extract revenue and calculate percentage change.",
        requirements=[extract, calculate],
        required_artifacts=["percentage"],
    )
    proposal = _proposal(
        TaskSpec(task_id="t1", requirement_name="extract numbers", assigned_to="extractor"),
        TaskSpec(
            task_id="t2",
            requirement_name="calculate percentage",
            assigned_to="calculator",
            depends_on=["t1"],
        ),
        artifacts=["percentage"],
    )

    report = FeasibilityAnalyzer().check(
        request,
        [_agent("extractor", extract), _agent("calculator", calculate)],
        proposal,
    )

    assert report.feasible


def test_approved_auxiliary_extractor_can_cover_gap():
    extract = CapabilityRequirement(
        name="extract revenue cells",
        description="extract Q1 and Q2 revenue values",
        input_modes=["table"],
        output_modes=["numbers"],
        output_schema={"required": ["q1_revenue", "q2_revenue"]},
        auxiliary_eligible=True,
    )
    factory = BoundedAuxiliaryCapabilityFactory()
    spec = factory.specify(extract, lifecycle="plan-1")
    assert spec is not None
    assert not spec.persists
    assert factory.validate_result(spec, {"q1_revenue": 10, "q2_revenue": 15})

    request = ProblemRequest(
        user_goal="Extract revenue cells.",
        requirements=[extract],
        required_artifacts=["revenue_pair"],
    )
    proposal = SolutionProposal(
        tasks=[
            TaskSpec(
                task_id="t1",
                requirement_name="extract revenue cells",
                auxiliary_spec_id=spec.spec_id,
            )
        ],
        generated_nlp_agents=[spec],
        execution_order=["t1"],
        expected_artifacts=["revenue_pair"],
        completion_criteria=["auxiliary validator passes"],
    )

    report = FeasibilityAnalyzer().check(request, [], proposal)

    assert report.feasible


def test_missing_capability_is_refused():
    request = ProblemRequest(
        user_goal="Forecast revenue.",
        requirements=[CapabilityRequirement(name="forecast revenue")],
        required_artifacts=["forecast"],
    )
    proposal = _proposal(
        TaskSpec(task_id="t1", requirement_name="forecast revenue", assigned_to="a1"),
        artifacts=["forecast"],
    )

    report = FeasibilityAnalyzer().check(request, [], proposal)

    assert not report.feasible
    assert report.missing_capabilities == ["forecast revenue"]


def test_incompatible_modalities_are_refused():
    summarize = CapabilityRequirement(
        name="summarize",
        input_modes=["text"],
        output_modes=["text"],
    )
    calculate = CapabilityRequirement(
        name="calculate",
        input_modes=["numbers"],
        output_modes=["number"],
    )
    request = ProblemRequest(
        user_goal="Summarize and calculate.",
        requirements=[summarize, calculate],
        required_artifacts=["number"],
    )
    proposal = _proposal(
        TaskSpec(task_id="t1", requirement_name="summarize", assigned_to="a1"),
        TaskSpec(task_id="t2", requirement_name="calculate", assigned_to="a2", depends_on=["t1"]),
        artifacts=["number"],
    )

    report = FeasibilityAnalyzer().check(
        request,
        [_agent("a1", summarize), _agent("a2", calculate)],
        proposal,
    )

    assert not report.feasible
    assert "incompatible" in " ".join(report.risks)


def test_insufficient_authority_is_refused():
    req = CapabilityRequirement(name="delete record", required_trust_level="admin")
    request = ProblemRequest(
        user_goal="Delete a protected record.",
        requirements=[req],
        required_artifacts=["deletion_receipt"],
    )
    proposal = _proposal(
        TaskSpec(task_id="t1", requirement_name="delete record", assigned_to="a1"),
        artifacts=["deletion_receipt"],
    )

    report = FeasibilityAnalyzer().check(
        request,
        [_agent("a1", req, trust_level="standard")],
        proposal,
    )

    assert not report.feasible
    assert any(item.name == "authorized" and not item.passed for item in report.evidence)


def test_unavailable_agent_is_refused():
    req = CapabilityRequirement(name="summarize")
    request = ProblemRequest(
        user_goal="Summarize.",
        requirements=[req],
        required_artifacts=["summary"],
    )
    proposal = _proposal(
        TaskSpec(task_id="t1", requirement_name="summarize", assigned_to="a1"),
        artifacts=["summary"],
    )

    report = FeasibilityAnalyzer().check(
        request,
        [_agent("a1", req, status="unavailable")],
        proposal,
    )

    assert not report.feasible
    assert "summarize" in report.missing_capabilities


def test_execution_order_must_respect_dependencies():
    extract = CapabilityRequirement(
        name="extract numbers",
        output_modes=["numbers"],
    )
    calculate = CapabilityRequirement(
        name="calculate percentage",
        input_modes=["numbers"],
        output_modes=["number"],
    )
    request = ProblemRequest(
        user_goal="Extract and calculate.",
        requirements=[extract, calculate],
        required_artifacts=["percentage"],
    )
    proposal = SolutionProposal(
        tasks=[
            TaskSpec(task_id="t1", requirement_name="extract numbers", assigned_to="extractor"),
            TaskSpec(
                task_id="t2",
                requirement_name="calculate percentage",
                assigned_to="calculator",
                depends_on=["t1"],
            ),
        ],
        execution_order=["t2", "t1"],
        expected_artifacts=["percentage"],
        completion_criteria=["percentage exists"],
    )

    report = FeasibilityAnalyzer().check(
        request,
        [_agent("extractor", extract), _agent("calculator", calculate)],
        proposal,
    )

    assert not report.feasible
    assert any(item.name == "ordered" and not item.passed for item in report.evidence)


def test_persistent_auxiliary_spec_is_refused():
    extract = CapabilityRequirement(
        name="extract revenue cells",
        description="extract Q1 and Q2 revenue values",
        output_schema={"required": ["q1_revenue", "q2_revenue"]},
        auxiliary_eligible=True,
    )
    spec = BoundedAuxiliaryCapabilityFactory().specify(extract, lifecycle="plan-1")
    assert spec is not None
    spec = spec.model_copy(update={"persists": True})
    request = ProblemRequest(
        user_goal="Extract revenue cells.",
        requirements=[extract],
        required_artifacts=["revenue_pair"],
    )
    proposal = SolutionProposal(
        tasks=[
            TaskSpec(
                task_id="t1",
                requirement_name="extract revenue cells",
                auxiliary_spec_id=spec.spec_id,
            )
        ],
        generated_nlp_agents=[spec],
        execution_order=["t1"],
        expected_artifacts=["revenue_pair"],
        completion_criteria=["auxiliary validator passes"],
    )

    report = FeasibilityAnalyzer().check(request, [], proposal)

    assert not report.feasible
    assert any("persists" in risk for risk in report.risks)
