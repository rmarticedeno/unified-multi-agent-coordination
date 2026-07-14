from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    BoundedAuxiliaryCapabilityFactory,
    CapabilityRequirement,
    ConstraintSpec,
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


def _constraint(operator, expected=None):
    return ConstraintSpec(
        constraint_id=f"constraint-{operator}",
        source="request_context",
        path="/value",
        operator=operator,
        expected=expected,
    )


def test_constraint_pointer_and_operator_helpers_cover_all_predicates():
    analyzer = FeasibilityAnalyzer()
    value = {"nested": {"a/b": [{"~key": 7}]}, "value": 3}
    assert analyzer._json_pointer(value, "") == (True, value)
    assert analyzer._json_pointer(value, "/") == (True, value)
    assert analyzer._json_pointer(value, "nested") == (False, None)
    assert analyzer._json_pointer(value, "/nested/a~1b/0/~0key") == (True, 7)
    assert analyzer._json_pointer(value, "/nested/a~1b/4") == (False, None)

    cases = [
        ("eq", 3, 3, True),
        ("ne", 4, 3, True),
        ("lt", 4, 3, True),
        ("lte", 3, 3, True),
        ("gt", 2, 3, True),
        ("gte", 3, 3, True),
        ("in", [2, 3], 3, True),
        ("not_in", [4, 5], 3, True),
        ("contains", 2, [1, 2], True),
        ("matches", r"a.+", "abc", True),
    ]
    for operator, expected, actual, result in cases:
        assert analyzer._compare_constraint(_constraint(operator, expected), True, actual) is result
    assert analyzer._compare_constraint(_constraint("exists", True), True, None) is True
    assert analyzer._compare_constraint(_constraint("exists", False), False, None) is True
    assert analyzer._compare_constraint(_constraint("lt", "invalid"), True, 3) is False
    assert analyzer._compare_constraint(_constraint("matches", "["), True, "x") is False
    assert analyzer._compare_constraint(_constraint("eq", 1), False, None) is False


def test_schema_trust_dependency_structure_order_and_cycle_helpers():
    analyzer = FeasibilityAnalyzer()
    assert analyzer._trust_satisfies("admin", "standard") is True
    assert analyzer._trust_satisfies("unknown", "standard") is False
    assert analyzer._dependency_modes_compatible(["json"], []) is True
    assert analyzer._dependency_modes_compatible(["json"], ["text"]) is False
    assert analyzer._schema_compatible({}, {}) is True
    assert analyzer._schema_compatible({}, {"type": "object"}) is False
    assert analyzer._schema_compatible({"type": "string"}, {"type": "object"}) is False
    assert analyzer._schema_compatible(
        {"type": "object", "required": [], "properties": {}},
        {"type": "object", "required": ["x"], "properties": {"x": {"type": "string"}}},
    ) is False
    assert analyzer._schema_compatible(
        {
            "type": "object",
            "required": ["x"],
            "properties": {"x": {"type": "number"}},
        },
        {
            "type": "object",
            "required": ["x"],
            "properties": {"x": {"type": "string"}},
        },
    ) is False

    requirement = CapabilityRequirement(name="work")
    missing_ids = TaskSpec(task_id="a", requirement_name="work")
    duplicate = missing_ids.model_copy(update={"task_id": "same"})
    risks = []
    assert analyzer._structure_ok([], risks) is False
    risks = []
    assert analyzer._structure_ok([duplicate, duplicate], risks) is False

    first = TaskSpec(
        task_id="first",
        requirement_name=requirement.name,
        requirement_id=requirement.requirement_id,
        capability_id=requirement.capability_id,
        assigned_to="agent",
    )
    second = first.model_copy(update={"task_id": "second", "depends_on": ["first"]})
    proposal = _proposal(first, second)
    proposal.execution_order = ["second", "first"]
    risks = []
    assert analyzer._execution_order_ok(proposal, risks) is False
    proposal.execution_order = ["first"]
    assert analyzer._execution_order_ok(proposal, []) is False
    assert analyzer._is_acyclic([first, second]) is True
    cycle_first = first.model_copy(update={"depends_on": ["second"]})
    assert analyzer._is_acyclic([cycle_first, second]) is False
    unknown = second.model_copy(update={"depends_on": ["missing"]})
    assert analyzer._is_acyclic([first, unknown]) is False


def test_constraint_context_resolves_task_and_requirement_sources():
    analyzer = FeasibilityAnalyzer()
    requirement = CapabilityRequirement(name="work")
    request = ProblemRequest(user_goal="work", requirements=[requirement], context={"x": 1})
    proposal = SolutionProposal(constraint_evidence={"proof": True})
    contexts = {
        "task": {
            "requirement": requirement,
            "task": {"task_id": "task"},
            "agent": {"agent_id": "agent"},
            "capability": {"name": "work"},
        }
    }
    request_constraint = ConstraintSpec(
        constraint_id="request", source="request_context", path="/x", operator="eq", expected=1
    )
    evidence_constraint = ConstraintSpec(
        constraint_id="evidence",
        source="proposal_evidence",
        path="/proof",
        operator="eq",
        expected=True,
    )
    task_constraint = ConstraintSpec(
        constraint_id="task",
        source="agent",
        path="/agent_id",
        operator="eq",
        expected="agent",
        requirement_id=requirement.requirement_id,
    )
    missing_constraint = task_constraint.model_copy(
        update={"constraint_id": "missing", "requirement_id": "missing"}
    )
    assert analyzer._constraint_context(request_constraint, request, proposal, contexts) == {"x": 1}
    assert analyzer._constraint_context(evidence_constraint, request, proposal, contexts) == {"proof": True}
    assert analyzer._constraint_context(task_constraint, request, proposal, contexts) == {"agent_id": "agent"}
    assert analyzer._constraint_context(missing_constraint, request, proposal, contexts) == {}
