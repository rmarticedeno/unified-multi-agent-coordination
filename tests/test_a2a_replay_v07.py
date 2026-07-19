from unified_multi_agent_coordination.a2a_replay_v07 import _select


def test_replay_selection_prefers_observed_categories_then_labels_fallback() -> None:
    cases = [
        {"case_id": f"case-{index}", "category": f"category-{index % 8}"}
        for index in range(10)
    ]
    labels = {
        case["case_id"]: {
            "case_id": case["case_id"],
            "feasible": True,
            "intent": {"goals": [{"capability_id": case["case_id"]}]},
        }
        for case in cases
    }
    observed = {
        "case-0": ({"goals": [{"capability_id": "observed-0"}]}, "hybrid_primary:qwen"),
        "case-1": ({"goals": [{"capability_id": "observed-1"}]}, "hybrid_primary:qwen"),
    }

    selected = _select(cases, labels, observed)

    assert len(selected) == 8
    assert [item[0]["case_id"] for item in selected[:2]] == ["case-0", "case-1"]
    assert sum(item[2].startswith("hybrid_primary") for item in selected) == 2
    assert sum(item[2] == "oracle_feasible_fallback" for item in selected) == 6
    assert len({item[0]["case_id"] for item in selected}) == 8
