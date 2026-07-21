import pytest
from pydantic import ValidationError

from unified_multi_agent_coordination import (
    AgentRegistryEntry,
    ArtifactContractOption,
    CapabilityCatalogEntry,
    CapabilityRequirement,
    CoordinationAgent,
    CoordinationSdk,
    ProblemRequest,
    SemanticCatalog,
    SemanticGoalSelection,
    SemanticIntentOutput,
    SemanticInterpretationResult,
    SemanticRequestAdmitter,
    TrustPolicyOption,
    ValidationContract,
    semantic_intent_schema,
)


def _capability(
    capability_id: str,
    *,
    dependencies: list[str] | None = None,
    output_modes: list[str] | None = None,
) -> CapabilityCatalogEntry:
    return CapabilityCatalogEntry(
        capability_id=capability_id,
        name=capability_id.replace("-", " "),
        description=f"Perform {capability_id}",
        aliases=[capability_id],
        depends_on_capability_ids=dependencies or [],
        input_schema={},
        output_schema={},
        input_modes=["text"] if not dependencies else ["bundle"],
        output_modes=output_modes or ["bundle"],
        side_effect_class="read_only",
        auxiliary_eligible=False,
        validation_contract=ValidationContract(
            json_schema={"type": "object"},
            required_artifacts=[f"{capability_id}-artifact"],
        ),
    )


def _catalog() -> SemanticCatalog:
    return SemanticCatalog(
        capabilities=[
            _capability("prepare"),
            _capability("deliver", dependencies=["prepare"], output_modes=["json"]),
        ],
        trust_policies=[
            TrustPolicyOption(
                policy_id="ordinary",
                name="ordinary assurance",
                description="Standard trust is sufficient.",
                aliases=["ordinary"],
                required_trust_level="standard",
            ),
            TrustPolicyOption(
                policy_id="elevated",
                name="elevated assurance",
                description="Elevated trust is required.",
                aliases=["elevated"],
                required_trust_level="elevated",
            ),
        ],
        artifact_contracts=[
            ArtifactContractOption(
                contract_id="json",
                name="JSON",
                description="Machine-readable JSON",
                aliases=["json"],
                output_modes=["json"],
                required_artifacts=["result-json"],
                json_schema={"type": "object"},
            )
        ],
        default_trust_policy_id="ordinary",
        default_artifact_contract_id="json",
    )


def _intent(**updates) -> SemanticIntentOutput:
    values = {
        "interpretation_status": "resolved",
        "goals": [
            SemanticGoalSelection(
                capability_id="deliver",
                trust_policy_id=None,
                artifact_contract_id=None,
            )
        ],
        "forbidden_capability_ids": [],
        "forbidden_agent_ids": [],
        "unresolved_terms": [],
    }
    values.update(updates)
    return SemanticIntentOutput.model_validate(values)


def test_semantic_wire_model_requires_every_field_and_forbids_rationale():
    with pytest.raises(ValidationError):
        SemanticIntentOutput.model_validate({})
    with pytest.raises(ValidationError):
        SemanticIntentOutput.model_validate({
            **_intent().model_dump(),
            "rationale": "This prose cannot substitute for structured fields.",
        })
    with pytest.raises(ValidationError):
        _intent(
            goals=[
                SemanticGoalSelection(
                    capability_id="deliver",
                    trust_policy_id=None,
                    artifact_contract_id=None,
                ),
                SemanticGoalSelection(
                    capability_id="deliver",
                    trust_policy_id="ordinary",
                    artifact_contract_id="json",
                ),
            ]
        )


def test_catalog_rejects_unknown_dependencies_cycles_and_invalid_defaults():
    values = _catalog().model_dump()
    values["capabilities"][1]["depends_on_capability_ids"] = ["unknown"]
    with pytest.raises(ValidationError, match="unknown dependencies"):
        SemanticCatalog.model_validate(values)

    values = _catalog().model_dump()
    values["capabilities"][0]["depends_on_capability_ids"] = ["deliver"]
    with pytest.raises(ValidationError, match="dependency cycle"):
        SemanticCatalog.model_validate(values)

    values = _catalog().model_dump()
    values["default_trust_policy_id"] = "invented"
    with pytest.raises(ValidationError, match="default trust policy"):
        SemanticCatalog.model_validate(values)

    values = _catalog().model_dump()
    values["capabilities"][1]["aliases"] = ["prepare"]
    with pytest.raises(ValidationError, match="Colliding normalized capability"):
        SemanticCatalog.model_validate(values)


def test_request_specific_schema_uses_only_admitted_identifiers():
    registry = [
        AgentRegistryEntry(
            agent_id="worker",
            name="Worker",
            service_endpoint="local://worker",
        )
    ]
    schema = semantic_intent_schema(_catalog(), registry)
    goal = schema["properties"]["goals"]["items"]["properties"]

    assert goal["capability_id"]["enum"] == ["deliver", "prepare"]
    assert goal["trust_policy_id"]["anyOf"][0]["enum"] == ["ordinary", "elevated"]
    assert schema["properties"]["forbidden_agent_ids"]["items"]["enum"] == ["worker"]
    assert "rationale" not in schema["properties"]


def test_semantic_prompt_exposes_dependencies_and_orders_terminal_outcomes_first():
    from unified_multi_agent_coordination.semantic_admission import semantic_prompt

    prompt = semantic_prompt("Deliver.", _catalog(), [])
    user = prompt[1]["content"]

    assert user.index("- deliver:") < user.index("- prepare:")
    assert "depends_on=['prepare']" in user


def test_semantic_admission_copies_authoritative_dependencies_contracts_and_exclusions():
    result = SemanticRequestAdmitter().admit(
        "Prepare then deliver JSON without the backup.",
        _catalog(),
        _intent(forbidden_agent_ids=["backup"]),
    )

    assert result.admitted
    assert result.request is not None
    assert [item.requirement_id for item in result.request.requirements] == [
        "prepare",
        "deliver",
    ]
    assert result.request.requirements[1].depends_on_requirement_ids == ["prepare"]
    assert result.request.requirements[1].validation_contract.required_artifacts == [
        "result-json"
    ]
    assert result.request.required_artifacts == [
        "prepare-artifact",
        "result-json",
    ]
    assert len(result.request.constraints) == 2
    assert all(item.expected == ["backup"] for item in result.request.constraints)


def test_semantic_admission_canonicalizes_redundant_prerequisite_goals():
    goals = [
        SemanticGoalSelection(
            capability_id=capability_id,
            trust_policy_id="ordinary",
            artifact_contract_id="json",
        )
        for capability_id in ("prepare", "deliver")
    ]
    result = SemanticRequestAdmitter().admit(
        "Prepare and then deliver JSON.",
        _catalog(),
        _intent(goals=goals),
    )

    assert result.admitted
    assert result.request is not None
    assert result.request.requirements[0].validation_contract.required_artifacts == [
        "prepare-artifact"
    ]
    assert result.request.requirements[1].validation_contract.required_artifacts == [
        "result-json"
    ]


def test_catalog_grounding_preserves_model_ambiguity_and_unresolved_terms():
    weak_model_output = _intent(
        interpretation_status="ambiguous",
        goals=[SemanticGoalSelection(
            capability_id="prepare",
            trust_policy_id="ordinary",
            artifact_contract_id="json",
        )],
        unresolved_terms=["unresolved safe option"],
    )
    result = SemanticRequestAdmitter().admit(
        "Deliver the result as machine-readable JSON, but do not prepare it.",
        _catalog(),
        weak_model_output,
    )

    assert not result.admitted
    assert result.canonical_intent is not None
    assert result.canonical_intent.interpretation_status == "ambiguous"
    assert [item.capability_id for item in result.canonical_intent.goals] == [
        "deliver"
    ]
    assert result.canonical_intent.forbidden_capability_ids == ["prepare"]
    assert result.canonical_intent.unresolved_terms == ["unresolved safe option"]
    assert {item.code for item in result.issues} == {
        "ambiguous_interpretation",
        "unresolved_term",
    }


def test_catalog_grounding_clears_only_exact_uniquely_owned_unresolved_aliases():
    exact = SemanticRequestAdmitter().admit(
        "Deliver as machine-readable JSON.",
        _catalog(),
        _intent(unresolved_terms=["machine-readable JSON"]),
    )
    unknown = SemanticRequestAdmitter().admit(
        "Deliver using ghost exporter.",
        _catalog(),
        _intent(unresolved_terms=["ghost exporter"]),
    )

    assert exact.admitted
    assert exact.canonical_intent is not None
    assert exact.canonical_intent.unresolved_terms == []
    assert not unknown.admitted
    assert unknown.canonical_intent is not None
    assert unknown.canonical_intent.unresolved_terms == ["ghost exporter"]


def test_catalog_grounding_rejects_explicit_unknown_identifier_shaped_names():
    result = SemanticRequestAdmitter().admit(
        "Deliver the result and require ghost-exporter-3.",
        _catalog(),
        _intent(),
    )

    assert not result.admitted
    assert result.canonical_intent is not None
    assert result.canonical_intent.interpretation_status == "ambiguous"
    assert result.canonical_intent.unresolved_terms == ["ghost-exporter-3"]
    assert {item.code for item in result.issues} == {
        "ambiguous_interpretation",
        "unresolved_term",
    }


def test_catalog_grounding_applies_only_explicit_agent_exclusions():
    registry = [
        AgentRegistryEntry(
            agent_id="primary",
            name="Primary delivery",
            service_endpoint="local://primary",
        ),
        AgentRegistryEntry(
            agent_id="backup",
            name="Backup delivery",
            service_endpoint="local://backup",
        ),
    ]
    result = SemanticRequestAdmitter().admit(
        "Deliver JSON, but do not use Primary delivery.",
        _catalog(),
        _intent(forbidden_agent_ids=["backup"]),
        registry,
    )

    assert result.admitted
    assert result.canonical_intent is not None
    assert result.canonical_intent.forbidden_agent_ids == ["backup", "primary"]


def test_semantic_admission_rejects_unknown_identifiers_before_canonicalization():
    result = SemanticRequestAdmitter().admit(
        "Deliver.",
        _catalog(),
        _intent(goals=[
            SemanticGoalSelection(
                capability_id="invented",
                trust_policy_id=None,
                artifact_contract_id=None,
            )
        ]),
    )

    assert not result.admitted
    assert result.issues[0].code == "unknown_identifier"


def test_semantic_admission_rejects_directly_contradictory_intent():
    result = SemanticRequestAdmitter().admit(
        "Deliver but do not deliver.",
        _catalog(),
        _intent(forbidden_capability_ids=["deliver"]),
    )

    assert not result.admitted
    assert "contradictory_intent" in {item.code for item in result.issues}


def test_semantic_admission_propagates_trust_per_dependency_branch():
    catalog = _catalog()
    values = catalog.model_dump()
    values["capabilities"].extend([
        _capability("inspect").model_dump(),
        _capability("publish", dependencies=["inspect"]).model_dump(),
    ])
    catalog = SemanticCatalog.model_validate(values)
    result = SemanticRequestAdmitter().admit(
        "Deliver normally and publish with elevated assurance.",
        catalog,
        _intent(goals=[
            SemanticGoalSelection(
                capability_id="deliver",
                trust_policy_id="ordinary",
                artifact_contract_id="json",
            ),
            SemanticGoalSelection(
                capability_id="publish",
                trust_policy_id="elevated",
                artifact_contract_id="json",
            ),
        ]),
    )

    assert result.admitted
    assert result.request is not None
    trust = {
        item.requirement_id: item.required_trust_level
        for item in result.request.requirements
    }
    assert trust == {
        "prepare": "standard",
        "deliver": "standard",
        "inspect": "elevated",
        "publish": "elevated",
    }


@pytest.mark.parametrize(
    ("intent", "code"),
    [
        (_intent(interpretation_status="ambiguous"), "ambiguous_interpretation"),
        (_intent(unresolved_terms=["the safest option"]), "unresolved_term"),
        (_intent(forbidden_capability_ids=["prepare"]), "forbidden_dependency"),
    ],
)
def test_semantic_admission_fails_closed_on_ambiguity_unresolved_terms_and_conflicts(
    intent,
    code,
):
    result = SemanticRequestAdmitter().admit("Request", _catalog(), intent)

    assert not result.admitted
    assert code in {item.code for item in result.issues}


@pytest.mark.asyncio
async def test_coordinator_uses_semantic_admission_and_typed_requests_bypass_the_llm():
    intent = _intent()

    class FixedInterpreter:
        calls = 0

        async def interpret(self, user_text, catalog, registry):
            self.calls += 1
            return SemanticInterpretationResult(
                intent=intent,
                initial_content=intent.model_dump_json(),
                call_count=1,
                model_id="fixed",
            )

    interpreter = FixedInterpreter()
    sdk = CoordinationSdk()
    prepare = CapabilityRequirement(
        name="prepare",
        requirement_id="prepare",
        capability_id="prepare",
        input_modes=["text"],
        output_modes=["bundle"],
        validation_contract=ValidationContract(
            json_schema={"type": "object"},
            required_artifacts=["prepare-artifact"],
        ),
    )
    deliver = CapabilityRequirement(
        name="deliver",
        requirement_id="deliver",
        capability_id="deliver",
        input_modes=["bundle"],
        output_modes=["json"],
        validation_contract=ValidationContract(
            json_schema={"type": "object"},
            required_artifacts=["result-json"],
        ),
    )
    sdk.register_local_agent("Preparer", [prepare], lambda payload: payload)
    sdk.register_local_agent("Deliverer", [deliver], lambda payload: payload)
    agent = CoordinationAgent(
        sdk,
        semantic_catalog=_catalog(),
        semantic_interpreter=interpreter,
    )

    raw = await agent.build_solution_plan("Prepare and deliver JSON.")

    assert raw.feasibility_report.feasible
    assert [task.requirement_id for task in raw.proposal.tasks] == [
        "prepare",
        "deliver",
    ]
    assert interpreter.calls == 1

    typed = await agent.build_solution_plan(ProblemRequest(
        user_goal="Prepare.",
        requirements=[prepare],
        required_artifacts=["prepare-artifact"],
    ))

    assert typed.feasibility_report.feasible
    assert interpreter.calls == 1


def test_catalog_rejects_duplicate_ids_contract_defaults_and_cross_entry_aliases():
    values = _catalog().model_dump()
    values["trust_policies"].append(values["trust_policies"][0])
    with pytest.raises(ValidationError, match="Duplicate trust policy"):
        SemanticCatalog.model_validate(values)

    values = _catalog().model_dump()
    values["default_artifact_contract_id"] = "missing"
    with pytest.raises(ValidationError, match="default artifact contract"):
        SemanticCatalog.model_validate(values)

    values = _catalog().model_dump()
    values["trust_policies"][1]["aliases"] = ["ordinary assurance"]
    with pytest.raises(ValidationError, match="Colliding normalized trust policy"):
        SemanticCatalog.model_validate(values)

    values = _catalog().model_dump()
    values["artifact_contracts"].append({
        **values["artifact_contracts"][0],
        "contract_id": "json-copy",
        "name": "JSON copy",
    })
    with pytest.raises(ValidationError, match="Colliding normalized artifact contract"):
        SemanticCatalog.model_validate(values)


def test_admission_rejects_missing_goal_defaults_and_unknown_custom_trust_rank():
    values = _catalog().model_dump()
    values["default_trust_policy_id"] = None
    catalog = SemanticCatalog.model_validate(values)
    missing = SemanticRequestAdmitter().admit("Deliver.", catalog, _intent())

    assert not missing.admitted
    assert {issue.code for issue in missing.issues} == {"missing_default"}
    assert SemanticRequestAdmitter()._trust_rank("custom") == 3


def test_schema_error_reporting_handles_syntax_non_object_and_schema_failures():
    from unified_multi_agent_coordination.semantic_admission import _schema_errors

    schema = {"type": "object", "required": ["value"]}
    parsed, errors = _schema_errors(schema, "{")
    assert parsed is None
    assert errors[0].startswith("json_syntax:")

    parsed, errors = _schema_errors(schema, "[]")
    assert parsed is None
    assert errors

    parsed, errors = _schema_errors(schema, "{}")
    assert parsed == {}
    assert errors == ["root: 'value' is a required property"]


@pytest.mark.asyncio
async def test_openai_interpreter_repairs_schema_and_accumulates_usage(monkeypatch):
    from unified_multi_agent_coordination.semantic_admission import (
        OpenAICompatibleSemanticInterpreter,
    )

    outputs = iter([
        (
            {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            "{",
        ),
        (
            {"usage": {"prompt_tokens": 12, "completion_tokens": 5}},
            _intent().model_dump_json(),
        ),
    ])
    interpreter = OpenAICompatibleSemanticInterpreter("fixture-model")

    async def complete(client, messages, schema):
        return next(outputs)

    monkeypatch.setattr(interpreter, "_complete", complete)
    result = await interpreter.interpret("Deliver.", _catalog(), [])

    assert result.intent == _intent()
    assert result.repair_attempted
    assert result.call_count == 2
    assert result.prompt_tokens == 22
    assert result.completion_tokens == 7


@pytest.mark.asyncio
async def test_openai_interpreter_preserves_invalid_schema_without_repair(monkeypatch):
    from unified_multi_agent_coordination.semantic_admission import (
        OpenAICompatibleSemanticInterpreter,
    )

    interpreter = OpenAICompatibleSemanticInterpreter(
        "fixture-model",
        allow_schema_repair=False,
    )

    async def complete(client, messages, schema):
        return {"usage": {}}, "{}"

    monkeypatch.setattr(interpreter, "_complete", complete)
    result = await interpreter.interpret("Deliver.", _catalog(), [])

    assert result.intent is None
    assert result.issues
    assert not result.repair_attempted
