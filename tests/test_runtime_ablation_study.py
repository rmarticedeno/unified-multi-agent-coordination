import json

from unified_multi_agent_coordination.runtime_ablation_study import run_study


def test_runtime_ablation_study_is_separate_complete_and_exclusive(tmp_path):
    output = run_study(tmp_path / "runtime-v1")
    analysis = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "runtime-observations.json").read_text(encoding="utf-8"))

    assert analysis["study_type"] == "runtime_only_ablation"
    assert analysis["planning_comparison"] is False
    assert analysis["observation_count"] == len(rows) == 350
    assert analysis["by_configuration"]["secure_default"]["safety_violations"] == 0
    assert analysis["by_configuration"]["without_dependency_gating"]["safety_violations"] == 40
    assert analysis["by_configuration"]["without_auxiliary_bounds"]["safety_violations"] == 40
    assert analysis["by_configuration"]["without_durable_trace_evidence"]["safety_violations"] == 50

    try:
        run_study(output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("official runtime evidence must not be overwritten")
