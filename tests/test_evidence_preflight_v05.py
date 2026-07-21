import hashlib
import json
import subprocess

import pytest

from unified_multi_agent_coordination.consensus_campaign import (
    TrialResult,
    _expected_checks,
    expected_scenarios,
)
from unified_multi_agent_coordination.evidence_preflight import (
    _json,
    _validate_consensus,
    _validate_release_metadata,
    _validate_v05,
    _validate_vendor,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_primary_evidence_sections_fail_closed_without_file_reads(tmp_path):
    assert _validate_v05(tmp_path, {}) == {
        "present": False,
        "evidence_valid": False,
        "outcome": "not_collected",
    }
    assert _validate_consensus(tmp_path, {}) == {
        "present": False,
        "accepted": False,
    }


def test_v05_preflight_accepts_complete_clean_matrix_independent_of_outcome(tmp_path):
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    (run / "collection-complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "expected_outputs": 1440,
                "observed_outputs": 1440,
                "labels_loaded_during_collection": False,
            }
        )
    )
    commit = "a" * 40
    (run / "provenance.json").write_text(json.dumps({"dirty_state": False, "git_sha": commit}))
    for index in range(1440):
        path = run / "outputs" / "arm" / "model" / "seed-1" / f"case-{index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "identity": {
                        "arm": "arm",
                        "case_id": f"case-{index}",
                        "model_id": "model",
                        "seed": 1,
                    },
                    "call_count": 1 + index % 2,
                }
            )
        )
    analysis = run / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "analysis_schema_version": "0.5.0",
                "outcome": "mixed",
                "claim_status": "partially_supported",
                "all_planning_criteria_met": False,
            }
        )
    )
    manifest = {
        "v05_study_evidence": {
            "run_root": "run",
            "analysis_file": "analysis.json",
            "analysis_sha256": _digest(analysis),
            "evaluated_source_commit": commit,
            "evidence_valid": True,
        }
    }

    result = _validate_v05(tmp_path, manifest)
    assert result["evidence_valid"] is True
    assert result["outcome"] == "mixed"
    assert result["planning_criteria_met"] is False

    first = next((run / "outputs").glob("*/*/seed-*/*.json"))
    record = json.loads(first.read_text())
    record["call_count"] = 3
    first.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="two-call budget"):
        _validate_v05(tmp_path, manifest)


def test_consensus_preflight_accepts_valid_failed_outcome_and_rejects_wrong_image(tmp_path):
    scenarios = []
    for trial in range(1, 4):
        scenarios.extend((f"formation-{topology}", trial) for topology in (3, 5, 7))
        scenarios.extend(
            (
                ("reconfigure-3-5-3", trial),
                ("reconfigure-5-7-5", trial),
                ("leader-partition-quorum-concurrency", trial),
                ("audit-sink-unavailable", trial),
                ("failed-voter-replacement", trial),
            )
        )
        scenarios.extend(
            (f"crash-{fault_point}", trial)
            for fault_point in (
                "after_task_attempt_start",
                "after_external_dispatch",
                "during_aggregation",
            )
        )
    results = []
    for scenario, trial in scenarios:
        checks = ["check"]
        results.append(
            {
                "scenario": scenario,
                "trial": trial,
                "expected_checks": checks,
                "executed_checks": checks,
                "unexecuted_checks": [],
            }
        )
    campaign_path = tmp_path / "campaign.json"
    campaign = {
        "schema_version": "consensus-campaign-v2",
        "provenance": {
            "dirty_state": False,
            "image": {"image_id": "sha256:image"},
        },
        "evidence_valid": True,
        "outcome": "failed",
        "claim_status": "unsupported",
        "trial_count": 33,
        "safety_checks_expected": 33,
        "safety_checks_executed": 33,
        "safety_violations": 1,
        "results": results,
    }
    campaign_path.write_text(json.dumps(campaign))
    manifest = {
        "consensus_campaign_evidence": {
            "campaign_file": "campaign.json",
            "sha256": _digest(campaign_path),
            "evidence_valid": True,
            "accepted": True,
        }
    }

    result = _validate_consensus(tmp_path, manifest)
    assert result["evidence_valid"] is True
    assert result["accepted"] is True
    assert result["outcome"] == "failed"

    campaign["provenance"]["image"]["image_id"] = "mutable"
    campaign_path.write_text(json.dumps(campaign))
    manifest["consensus_campaign_evidence"]["sha256"] = _digest(campaign_path)
    with pytest.raises(RuntimeError, match="wrong-image"):
        _validate_consensus(tmp_path, manifest)


@pytest.mark.parametrize(
    "schema_version",
    ("consensus-campaign-v3", "consensus-campaign-v4"),
)
def test_consensus_preflight_validates_v3_and_v4_condition_summaries(
    tmp_path, schema_version
):
    results = []
    conditions = {}
    supplementary_name = "leader-partition-quorum-concurrency"
    for scenario, trial in expected_scenarios(3, smoke=False):
        checks = _expected_checks(scenario)
        primary = scenario != supplementary_name
        result = TrialResult(
            scenario=scenario,
            trial=trial,
            topology=3,
            passed=True,
            duration_s=0,
            status="passed",
            expected_checks=checks,
            executed_checks=checks,
            checks={check: True for check in checks},
            primary=primary,
        )
        results.append(result.__dict__)
        if primary:
            summary = conditions.setdefault(
                scenario,
                {
                    "trial_count": 0,
                    "passed_trials": 0,
                    "invariant_failed_trials": 0,
                    "infrastructure_failed_trials": 0,
                    "checks_expected": 0,
                    "checks_executed": 0,
                    "violations": 0,
                    "supported": True,
                },
            )
            summary["trial_count"] += 1
            summary["passed_trials"] += 1
            summary["checks_expected"] += len(checks)
            summary["checks_executed"] += len(checks)
    campaign_path = tmp_path / "campaign-v3.json"
    campaign = {
        "schema_version": schema_version,
        "provenance": {
            "dirty_state": False,
            "image": {"image_id": "sha256:image"},
        },
        "evidence_valid": True,
        "outcome": "passed",
        "claim_status": "supported",
        "trial_count": 45,
        "primary_trial_count": 42,
        "supplementary_trial_count": 3,
        "condition_results": conditions,
        "results": results,
    }
    campaign_path.write_text(json.dumps(campaign))
    manifest = {
        "consensus_campaign_evidence": {
            "campaign_file": "campaign-v3.json",
            "sha256": _digest(campaign_path),
            "evidence_valid": True,
            "accepted": True,
        }
    }

    assert _validate_consensus(tmp_path, manifest)["evidence_valid"] is True
    campaign["condition_results"]["leader-termination"]["trial_count"] = 2
    campaign_path.write_text(json.dumps(campaign))
    manifest["consensus_campaign_evidence"]["sha256"] = _digest(campaign_path)
    with pytest.raises(RuntimeError, match="incomplete"):
        _validate_consensus(tmp_path, manifest)


def test_release_metadata_and_json_shape_are_fail_closed(tmp_path):
    thesis = tmp_path / "thesis"
    thesis.mkdir()
    (thesis / "main.tex").write_text("\\advisor{A}\\tribunal{B}")
    commits = {"evaluated_source_commit": "a" * 40, "packaging_commit": "b" * 40}
    assert _validate_release_metadata(tmp_path, commits)["passed"] is True

    (thesis / "main.tex").write_text("[Tutor por confirmar] [Tribunal por confirmar]")
    blocked = _validate_release_metadata(tmp_path, {})
    assert blocked["passed"] is False
    assert len(blocked["open_gates"]) == 4

    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="missing"):
        _json(missing)
    array = tmp_path / "array.json"
    array.write_text("[]")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        _json(array)


def _vendor_repo(root):
    vendor = root / "vendor/a2a-samples/helloworld"
    vendor.mkdir(parents=True)
    names = ("__main__.py", "agent_executor.py", "requirements.txt", "README.md")
    for name in names:
        (vendor / name).write_text(name)
    (root / "LICENSE").write_text("license")
    provenance = {
        "commit": "a" * 40,
        "sdk_version": "1.1.0",
        "license_file": "../../../LICENSE",
        "license_sha256": _digest(root / "LICENSE"),
        "local_sha256": {name: _digest(vendor / name) for name in names},
    }
    (vendor / "UPSTREAM.json").write_text(json.dumps(provenance))
    (root / "pyproject.toml").write_text('dependencies = ["a2a-sdk==1.1.0"]')
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    return vendor, provenance


def test_vendor_preflight_verifies_hashes_license_dependency_and_git_tracking(tmp_path):
    vendor, provenance = _vendor_repo(tmp_path)
    result = _validate_vendor(tmp_path)
    assert result["evidence_valid"] is True
    assert result["hashes_verified"] is True

    (vendor / "README.md").write_text("changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _validate_vendor(tmp_path)
    (vendor / "README.md").write_text("README.md")

    (vendor / "requirements.txt").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        _validate_vendor(tmp_path)
    (vendor / "requirements.txt").write_text("requirements.txt")

    provenance["sdk_version"] = "wrong"
    (vendor / "UPSTREAM.json").write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="not pinned"):
        _validate_vendor(tmp_path)


def test_v05_and_consensus_manifest_mismatches_fail_closed(tmp_path):
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    (run / "collection-complete.json").write_text("{}")
    (run / "provenance.json").write_text(json.dumps({"dirty_state": True, "git_sha": "a" * 40}))
    analysis = run / "analysis.json"
    analysis.write_text(json.dumps({"analysis_schema_version": "wrong"}))
    manifest = {
        "v05_study_evidence": {
            "run_root": "run",
            "analysis_file": "analysis.json",
            "analysis_sha256": "bad",
            "evaluated_source_commit": "a" * 40,
            "evidence_valid": True,
        }
    }
    with pytest.raises(RuntimeError, match="digest"):
        _validate_v05(tmp_path, manifest)
    manifest["v05_study_evidence"]["analysis_sha256"] = _digest(analysis)
    with pytest.raises(RuntimeError, match="unsupported schema"):
        _validate_v05(tmp_path, manifest)
    analysis.write_text(json.dumps({"analysis_schema_version": "0.5.0"}))
    manifest["v05_study_evidence"]["analysis_sha256"] = _digest(analysis)
    with pytest.raises(RuntimeError, match="incomplete or dirty"):
        _validate_v05(tmp_path, manifest)

    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"schema_version": "wrong"}))
    consensus = {
        "consensus_campaign_evidence": {
            "campaign_file": "campaign.json",
            "sha256": "bad",
            "evidence_valid": True,
        }
    }
    with pytest.raises(RuntimeError, match="digest"):
        _validate_consensus(tmp_path, consensus)
    consensus["consensus_campaign_evidence"]["sha256"] = _digest(campaign)
    with pytest.raises(RuntimeError, match="unsupported schema"):
        _validate_consensus(tmp_path, consensus)
