from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    CapabilityRequirement,
    ConstraintSpec,
    FeasibilityAnalyzer,
    ProblemRequest,
    SymbolicPlanCompiler,
    ValidationContract,
)


def _requirement(
    *,
    trust: str = "standard",
    input_modes: list[str] | None = None,
    output_modes: list[str] | None = None,
    side_effect: str = "read_only",
) -> CapabilityRequirement:
    return CapabilityRequirement(
        name="deliver",
        requirement_id="deliver",
        capability_id="deliver",
        input_modes=input_modes or ["text"],
        output_modes=output_modes or ["json"],
        required_trust_level=trust,
        side_effect_class=side_effect,
        validation_contract=ValidationContract(
            json_schema={"type": "object"},
            required_artifacts=["result"],
        ),
    )


def _agent(
    agent_id: str,
    skill: CapabilityRequirement,
    *,
    trust: str = "standard",
    status: str = "available",
    supports_fencing: bool = False,
) -> AgentRegistryEntry:
    return AgentRegistryEntry(
        agent_id=agent_id,
        name=agent_id,
        service_endpoint=f"local://{agent_id}",
        skills=[skill],
        trust_level=trust,
        status=status,
        supports_fencing=supports_fencing,
    )


def _request(requirement: CapabilityRequirement, constraints=None) -> ProblemRequest:
    return ProblemRequest(
        user_goal="Deliver the result.",
        requirements=[requirement],
        constraints=constraints or [],
        required_artifacts=["result"],
    )


def test_compiler_recovers_from_incompatible_first_provider():
    required = _requirement(output_modes=["json"])
    invalid_skill = _requirement(output_modes=["text"])
    valid_skill = _requirement(output_modes=["json"])
    compiler = SymbolicPlanCompiler()

    result = compiler.compile(
        _request(required),
        [_agent("a-invalid", invalid_skill), _agent("z-valid", valid_skill)],
    )

    assert result.report.feasible
    assert result.proposal.tasks[0].assigned_to == "z-valid"
    assert result.diagnostics.assignments_considered == 2
    assert result.diagnostics.recovered_alternative_provider


def test_compiler_filters_trust_fencing_unavailability_and_agent_exclusions():
    required = _requirement(trust="elevated", side_effect="unsafe")
    exclusion = ConstraintSpec(
        constraint_id="exclude-backup",
        source="agent",
        path="/agent_id",
        operator="not_in",
        expected=["backup"],
        requirement_id="deliver",
    )
    compiler = SymbolicPlanCompiler(
        FeasibilityAnalyzer(require_effect_fencing=True)
    )
    result = compiler.compile(
        _request(required, [exclusion]),
        [
            _agent("low", required, trust="standard", supports_fencing=True),
            _agent("backup", required, trust="elevated", supports_fencing=True),
            _agent("down", required, trust="admin", status="unavailable", supports_fencing=True),
            _agent("no-fence", required, trust="elevated"),
            _agent("winner", required, trust="elevated", supports_fencing=True),
        ],
    )

    assert result.report.feasible
    assert result.proposal.tasks[0].assigned_to == "winner"


def test_compiler_is_stable_and_uses_least_sufficient_privilege():
    required = _requirement(trust="standard")
    standard = _agent("z-standard", required, trust="standard")
    admin = _agent("a-admin", required, trust="admin")
    compiler = SymbolicPlanCompiler()

    first = compiler.compile(_request(required), [admin, standard])
    second = compiler.compile(_request(required), [standard, admin])

    assert first.proposal == second.proposal
    assert first.proposal.tasks[0].assigned_to == "z-standard"
    assert not first.diagnostics.recovered_alternative_provider


def test_compiler_fails_closed_when_search_limit_is_exhausted():
    required = _requirement(output_modes=["json"])
    invalid = _agent("a-invalid", _requirement(output_modes=["text"]))
    valid = _agent("z-valid", required)
    result = SymbolicPlanCompiler(max_assignment_evaluations=1).compile(
        _request(required),
        [invalid, valid],
    )

    assert not result.report.feasible
    assert result.diagnostics.search_exhausted
    assert result.diagnostics.issues[0].code == "assignment_search_exhausted"


def test_compiler_rejects_unknown_dependencies_and_cycles():
    first = _requirement()
    first.requirement_id = "first"
    first.capability_id = "first"
    first.depends_on_requirement_ids = ["second"]
    request = ProblemRequest(user_goal="Invalid", requirements=[first])

    unknown = SymbolicPlanCompiler().compile(request, [])

    assert not unknown.report.feasible
    assert unknown.diagnostics.issues[0].code == "unknown_dependency"

    second = _requirement()
    second.requirement_id = "second"
    second.capability_id = "second"
    second.depends_on_requirement_ids = ["first"]
    cyclic = SymbolicPlanCompiler().compile(
        ProblemRequest(user_goal="Cycle", requirements=[first, second]),
        [],
    )

    assert not cyclic.report.feasible
    assert cyclic.diagnostics.issues[0].code == "dependency_cycle"


def test_compiler_topologically_orders_requirements_independent_of_input_order():
    first = _requirement(output_modes=["bundle"])
    first.requirement_id = "first"
    first.capability_id = "first"
    first.validation_contract.required_artifacts = ["first-result"]
    second = _requirement(input_modes=["bundle"])
    second.requirement_id = "second"
    second.capability_id = "second"
    second.depends_on_requirement_ids = ["first"]
    second.validation_contract.required_artifacts = ["second-result"]
    registry = [
        _agent("first-agent", first),
        _agent("second-agent", second),
    ]
    request = ProblemRequest(
        user_goal="Run both.",
        requirements=[second, first],
        required_artifacts=["second-result"],
    )

    result = SymbolicPlanCompiler().compile(request, list(reversed(registry)))

    assert result.report.feasible
    assert [task.requirement_id for task in result.proposal.tasks] == ["first", "second"]
    assert result.proposal.execution_order == ["t1", "t2"]
    assert result.proposal.tasks[1].depends_on == ["t1"]
