import asyncio
import json
from types import SimpleNamespace

import httpx

from unified_multi_agent_coordination import consensus_campaign
from unified_multi_agent_coordination.consensus_campaign import (
    ComposeController,
    TrialResult,
    _with_cluster,
    expected_scenarios,
    run_campaign,
)


def test_full_consensus_campaign_constructs_exact_v3_trial_matrix():
    matrix = expected_scenarios(3, smoke=False)
    assert len(matrix) == 45
    assert len(set(matrix)) == 45
    assert sum(scenario.startswith("formation-") for scenario, _ in matrix) == 9
    assert sum(scenario.startswith("crash-") for scenario, _ in matrix) == 9
    assert sum(scenario == "leader-partition-quorum-concurrency" for scenario, _ in matrix) == 3
    assert all(
        sum(candidate == scenario for candidate, _ in matrix) == 3
        for scenario in (
            "leader-termination",
            "minority-partition",
            "majority-loss-restoration",
            "concurrent-ownership",
        )
    )


def test_compose_trials_always_use_prebuilt_image_and_no_build(tmp_path, monkeypatch):
    commands = []

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def run(command, **_kwargs):
        commands.append(command)
        return Completed()

    monkeypatch.setattr(consensus_campaign.subprocess, "run", run)
    controller = ComposeController(
        project="test", evidence_dir=tmp_path, voter_target=3, image="image@sha256:abc"
    )
    controller.up(["coordination-a"])

    assert commands[0][-4:] == ["up", "-d", "--no-build", "coordination-a"]
    assert controller.env["COORDINATION_IMAGE"] == "image@sha256:abc"


def test_infrastructure_failure_never_counts_unexecuted_checks_as_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ComposeController,
        "up",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    monkeypatch.setattr(ComposeController, "down", lambda *_args, **_kwargs: None)

    async def action(*_args):
        raise AssertionError("unreachable")

    result = asyncio.run(
        _with_cluster(
            tmp_path,
            scenario="formation-3",
            topology=3,
            trial=1,
            initial_target=3,
            active_nodes=3,
            action=action,
            image="sha256:test",
        )
    )

    assert result.status == "infrastructure_error"
    assert result.passed is False
    assert result.executed_checks == []
    assert result.unexecuted_checks == result.expected_checks


def test_evidence_acceptance_is_independent_of_favorable_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(
        consensus_campaign,
        "_prepare_image",
        lambda *_args, **_kwargs: {
            "reference": "image",
            "image_id": "sha256:abc",
            "repo_digests": [],
        },
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_provenance",
        lambda image: {"dirty_state": False, "image": image},
    )

    async def failed(root, scenario, topology, trial):
        trial_dir = root / f"{scenario}-trial-{trial}"
        trial_dir.mkdir()
        (trial_dir / "compose-commands.jsonl").write_text("{}\n")
        checks = consensus_campaign._expected_checks(scenario)
        return TrialResult(
            scenario=scenario,
            trial=trial,
            topology=topology,
            passed=False,
            duration_s=0,
            status="invariant_failed",
            expected_checks=checks,
            executed_checks=checks,
            violated_checks=[checks[0]],
            checks={name: name != checks[0] for name in checks},
        )

    async def formation(root, topology, trial, _image):
        return await failed(root, f"formation-{topology}", topology, trial)

    async def atomic(root, trial, _image, scenario):
        return await failed(root, scenario, 3, trial)

    async def audit(root, trial, _image):
        return await failed(root, "audit-sink-unavailable", 3, trial)

    monkeypatch.setattr(consensus_campaign, "_formation_trial", formation)
    monkeypatch.setattr(
        consensus_campaign,
        "_leader_termination_trial",
        lambda root, trial, image: atomic(root, trial, image, "leader-termination"),
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_minority_partition_trial",
        lambda root, trial, image: atomic(root, trial, image, "minority-partition"),
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_majority_loss_trial",
        lambda root, trial, image: atomic(root, trial, image, "majority-loss-restoration"),
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_concurrent_ownership_trial",
        lambda root, trial, image: atomic(root, trial, image, "concurrent-ownership"),
    )
    monkeypatch.setattr(consensus_campaign, "_audit_failure_trial", audit)
    report = asyncio.run(
        run_campaign(tmp_path / "campaign", 1, smoke=True, promotion_candidate=False)
    )

    assert report["evidence_valid"] is True
    assert report["outcome"] == "failed"
    assert report["claim_status"] == "unsupported"
    assert report["accepted"] is False  # smoke data can never be promoted


def test_image_preparation_failure_is_preserved_as_campaign_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        consensus_campaign,
        "_prepare_image",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_provenance",
        lambda image: {"dirty_state": False, "image": image},
    )
    output = tmp_path / "failed"
    report = asyncio.run(run_campaign(output, 3))

    assert report["evidence_valid"] is False
    assert report["image_preparation_error"].endswith("build failed")
    assert json.loads((output / "campaign.json").read_text())["outcome"] == "failed"


class _ScenarioController:
    def __init__(self):
        self.events = []

    def stop(self, *names):
        self.events.append(("stop", names))

    def start(self, *names):
        self.events.append(("start", names))

    def disconnect(self, name):
        self.events.append(("disconnect", name))

    def reconnect(self, name):
        self.events.append(("reconnect", name))


def _response(status=200, body=None):
    return httpx.Response(
        status,
        json=body or {"status": "completed"},
        request=httpx.Request("GET", "http://test"),
    )


def test_all_consensus_scenario_actions_execute_with_controlled_observations(tmp_path, monkeypatch):
    controllers = []

    async def with_cluster(
        _root,
        *,
        scenario,
        topology,
        trial,
        initial_target,
        active_nodes,
        action,
        **_kwargs,
    ):
        controller = _ScenarioController()
        controllers.append(controller)
        names = list(consensus_campaign.COORDINATORS)[:active_nodes]
        checks, observations = await action(controller, names, initial_target)
        return TrialResult(
            scenario=scenario,
            trial=trial,
            topology=topology,
            passed=all(checks.values()),
            duration_s=0,
            checks=checks,
            observations=observations,
        )

    monkeypatch.setattr(consensus_campaign, "_with_cluster", with_cluster)

    status_calls = 0

    async def steady(names, target):
        nonlocal status_calls
        status_calls += 1
        return {
            name: {
                "steady_state": True,
                "configuration_generation": status_calls,
                "pending_membership_changes": 0,
                "leader": 1,
                "member_id": index + 1,
                "role": "voter" if name != "coordination-d" else "voter",
            }
            for index, name in enumerate(names)
        }

    async def coordinate(_url, _session):
        return _response()

    async def update(_url, _target, _generation):
        return _response()

    monkeypatch.setattr(consensus_campaign, "_wait_steady", steady)
    monkeypatch.setattr(consensus_campaign, "_wait_progress", lambda _names: steady(_names, 3))
    monkeypatch.setattr(
        consensus_campaign,
        "_wait_replacement",
        lambda names, target, _failed, _replacement: steady(names, target),
    )
    monkeypatch.setattr(consensus_campaign, "_coordinate", coordinate)
    monkeypatch.setattr(consensus_campaign, "_update_target", update)
    monkeypatch.setattr(consensus_campaign, "_ready", lambda _url: _return(_response(503)))
    monkeypatch.setattr(
        consensus_campaign,
        "_wait_ready_status",
        lambda _url, status, **_kwargs: asyncio.sleep(
            0, result=_response(status, {"code": "quorum_unavailable"})
        ),
    )

    formation = asyncio.run(consensus_campaign._formation_trial(tmp_path, 3, 1, "image"))
    reconfigure = asyncio.run(consensus_campaign._reconfiguration_trial(tmp_path, 3, 5, 1, "image"))
    fault = asyncio.run(consensus_campaign._fault_trial(tmp_path, 1, "image"))
    leader = asyncio.run(consensus_campaign._leader_termination_trial(tmp_path, 1, "image"))
    minority = asyncio.run(consensus_campaign._minority_partition_trial(tmp_path, 1, "image"))
    majority = asyncio.run(consensus_campaign._majority_loss_trial(tmp_path, 1, "image"))
    concurrent = asyncio.run(consensus_campaign._concurrent_ownership_trial(tmp_path, 1, "image"))
    audit = asyncio.run(consensus_campaign._audit_failure_trial(tmp_path, 1, "image"))
    replacement = asyncio.run(
        consensus_campaign._failed_voter_replacement_trial(tmp_path, 1, "image")
    )

    assert formation.checks["coordinate_completed"]
    assert all(reconfigure.checks.values())
    assert all(fault.checks.values())
    assert all(leader.checks.values())
    assert all(minority.checks.values())
    assert all(majority.checks.values())
    assert all(concurrent.checks.values())
    assert all(audit.checks.values())
    assert all(replacement.checks.values())
    assert any(
        event[0] == "disconnect" for controller in controllers for event in controller.events
    )


def test_crash_window_records_receiver_operation_and_fencing_evidence(tmp_path, monkeypatch):
    async def with_cluster(
        _root,
        *,
        scenario,
        topology,
        trial,
        initial_target,
        active_nodes,
        action,
        **_kwargs,
    ):
        checks, observations = await action(
            _ScenarioController(),
            list(consensus_campaign.COORDINATORS)[:active_nodes],
            initial_target,
        )
        return TrialResult(
            scenario=scenario,
            trial=trial,
            topology=topology,
            passed=all(checks.values()),
            duration_s=0,
            checks=checks,
            observations=observations,
        )

    async def crash_coordinate(_url, _session):
        raise httpx.ReadError("connection terminated")

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def post(self, *_args, **_kwargs):
            return _response(200, {"status": "completed"})

        async def get(self, *_args, **_kwargs):
            return _response(
                200,
                {
                    "repeated_session_task_effectful_executions": 0,
                    "highest_fence_by_operation": {"operation": 2},
                },
            )

    monkeypatch.setattr(consensus_campaign, "_with_cluster", with_cluster)
    monkeypatch.setattr(consensus_campaign, "_wait_steady", lambda *_args: _return({}))
    monkeypatch.setattr(
        consensus_campaign,
        "_wait_registered_agent",
        lambda *_args: _return({"agents": [{"agent_id": "summarizer"}]}),
    )
    monkeypatch.setattr(
        consensus_campaign, "_wait_ready_status", lambda *_args, **_kwargs: _return(_response())
    )
    monkeypatch.setattr(consensus_campaign, "_coordinate", crash_coordinate)
    monkeypatch.setattr(consensus_campaign.asyncio, "sleep", lambda *_args: _immediate())
    monkeypatch.setattr(consensus_campaign.httpx, "AsyncClient", Client)

    result = asyncio.run(
        consensus_campaign._crash_window_trial(tmp_path, 1, "after_external_dispatch", "image")
    )

    assert result.passed
    assert result.checks["receiver_operation_keys_observed"]
    assert result.checks["receiver_fencing_tokens_positive"]


def test_coordinate_http_failure_is_preserved_as_liveness_observation(monkeypatch):
    async def fail(_url, _session):
        raise httpx.ReadTimeout("bounded timeout")

    monkeypatch.setattr(consensus_campaign, "_coordinate", fail)
    response, error = asyncio.run(
        consensus_campaign._coordinate_observed("http://coordinator", "session")
    )

    assert response is None
    assert error == "ReadTimeout: bounded timeout"


async def _immediate():
    return None


async def _return(value):
    return value


def test_campaign_http_helpers_and_image_metadata(monkeypatch):
    async def helpers():
        calls = 0

        async def statuses(_names):
            nonlocal calls
            calls += 1
            return {"n": {"ready": calls > 1}}

        monkeypatch.setattr(consensus_campaign, "_statuses", statuses)
        monkeypatch.setattr(consensus_campaign.asyncio, "sleep", lambda *_args: _immediate())
        found = await consensus_campaign._wait_for(
            ["n"], lambda value: value["n"]["ready"], timeout_s=1
        )
        assert found["n"]["ready"]

        responses = [_response(503), _response(200)]
        monkeypatch.setattr(
            consensus_campaign,
            "_ready",
            lambda _url: _return(responses.pop(0)),
        )
        assert (
            await consensus_campaign._wait_ready_status("url", 200, timeout_s=1)
        ).status_code == 200

    asyncio.run(helpers())

    inspected = SimpleNamespace(
        returncode=0,
        stdout=json.dumps([{"Id": "sha256:abc", "RepoDigests": ["repo@sha256:def"]}]),
        stderr="",
    )
    monkeypatch.setattr(consensus_campaign.subprocess, "run", lambda *_args, **_kwargs: inspected)
    assert consensus_campaign._image_metadata("image")["image_id"] == "sha256:abc"


def test_compose_controller_lifecycle_and_inspection_helpers(tmp_path, monkeypatch):
    controller = ComposeController(
        project="project", evidence_dir=tmp_path, voter_target=3, image="image"
    )
    calls = []

    def command(*args, **kwargs):
        calls.append((args, kwargs))
        if args[:2] == ("config", "--services"):
            return "coordination-a\ncoordination-b"
        if args[:2] == ("ps", "-q"):
            return f"container-{args[2]}"
        return ""

    controller.command = command
    inspected = SimpleNamespace(returncode=0, stdout="sha256:image\n", stderr="")
    monkeypatch.setattr(consensus_campaign.subprocess, "run", lambda *_args, **_kwargs: inspected)
    controller.stop("coordination-a")
    controller.start("coordination-a")
    controller.remove("coordination-a")
    controller.disconnect("coordination-a")
    controller.reconnect("coordination-a")
    assert controller.services() == ["coordination-a", "coordination-b"]
    assert controller.container_id("coordination-a") == "container-coordination-a"
    assert controller.image_ids() == {
        "coordination-a": "sha256:image",
        "coordination-b": "sha256:image",
    }
    controller.down()
    assert any(call[0][0] == "down" for call in calls)


def test_full_campaign_report_preserves_failed_but_valid_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        consensus_campaign,
        "_prepare_image",
        lambda *_args, **_kwargs: {
            "reference": "image",
            "image_id": "sha256:abc",
            "repo_digests": [],
        },
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_provenance",
        lambda image: {"dirty_state": False, "image": image},
    )

    async def result(root, scenario, topology, trial):
        path = root / f"{scenario}-trial-{trial}"
        path.mkdir()
        (path / "compose-commands.jsonl").write_text("{}\n")
        checks = consensus_campaign._expected_checks(scenario)
        return TrialResult(
            scenario=scenario,
            trial=trial,
            topology=topology,
            passed=True,
            duration_s=0,
            status="passed",
            expected_checks=checks,
            executed_checks=checks,
            checks={name: True for name in checks},
        )

    async def formation(root, topology, trial, _image):
        return await result(root, f"formation-{topology}", topology, trial)

    async def reconfigure(root, initial, expanded, trial, _image):
        return await result(root, f"reconfigure-{initial}-{expanded}-{initial}", expanded, trial)

    async def fault(root, trial, _image):
        outcome = await result(root, "leader-partition-quorum-concurrency", 3, trial)
        outcome.primary = False
        return outcome

    async def atomic(root, trial, _image, scenario):
        return await result(root, scenario, 3, trial)

    async def audit(root, trial, _image):
        return await result(root, "audit-sink-unavailable", 3, trial)

    async def replacement(root, trial, _image):
        return await result(root, "failed-voter-replacement", 3, trial)

    async def crash(root, trial, fault_point, _image):
        return await result(root, f"crash-{fault_point}", 3, trial)

    monkeypatch.setattr(consensus_campaign, "_formation_trial", formation)
    monkeypatch.setattr(consensus_campaign, "_reconfiguration_trial", reconfigure)
    monkeypatch.setattr(
        consensus_campaign,
        "_leader_termination_trial",
        lambda root, trial, image: atomic(root, trial, image, "leader-termination"),
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_minority_partition_trial",
        lambda root, trial, image: atomic(root, trial, image, "minority-partition"),
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_majority_loss_trial",
        lambda root, trial, image: atomic(root, trial, image, "majority-loss-restoration"),
    )
    monkeypatch.setattr(
        consensus_campaign,
        "_concurrent_ownership_trial",
        lambda root, trial, image: atomic(root, trial, image, "concurrent-ownership"),
    )
    monkeypatch.setattr(consensus_campaign, "_fault_trial", fault)
    monkeypatch.setattr(consensus_campaign, "_audit_failure_trial", audit)
    monkeypatch.setattr(consensus_campaign, "_failed_voter_replacement_trial", replacement)
    monkeypatch.setattr(consensus_campaign, "_crash_window_trial", crash)
    report = asyncio.run(
        run_campaign(tmp_path / "full", 3, promotion_candidate=True, build_image=False)
    )

    assert report["schema_version"] == "consensus-campaign-v3"
    assert report["trial_count"] == 45
    assert report["primary_trial_count"] == 42
    assert report["supplementary_trial_count"] == 3
    assert all(item["supported"] for item in report["condition_results"].values())
    assert report["safety_checks_executed"] == report["safety_checks_expected"]
    assert report["evidence_valid"] is True
    assert report["accepted"] is True


def test_with_cluster_success_records_complete_checks_and_image(tmp_path, monkeypatch):
    monkeypatch.setattr(ComposeController, "up", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ComposeController, "down", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ComposeController, "image_ids", lambda *_args: {"a": "sha256:image"})

    async def action(_controller, _names, _target):
        return {"steady_state": True, "coordinate_completed": True}, {"detail": "ok"}

    result = asyncio.run(
        _with_cluster(
            tmp_path,
            scenario="formation-3",
            topology=3,
            trial=1,
            initial_target=3,
            active_nodes=3,
            action=action,
            image="image",
        )
    )
    assert result.status == "passed"
    assert set(result.executed_checks) == set(result.expected_checks)
    assert result.observations["image_ids"] == {"a": "sha256:image"}


def test_image_build_is_single_and_metadata_is_written(tmp_path, monkeypatch):
    completed = SimpleNamespace(returncode=0, stdout="built", stderr="")
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return completed

    monkeypatch.setattr(consensus_campaign.subprocess, "run", run)
    monkeypatch.setattr(
        consensus_campaign,
        "_image_metadata",
        lambda image: {"reference": image, "image_id": "sha256:abc", "repo_digests": []},
    )
    metadata = consensus_campaign._prepare_image(tmp_path, "campaign", build=True)
    assert metadata["image_id"] == "sha256:abc"
    assert calls == [["docker", "build", "--tag", "campaign", "."]]
    assert json.loads((tmp_path / "image-build.json").read_text())["returncode"] == 0
    assert json.loads((tmp_path / "image.json").read_text())["image_id"] == "sha256:abc"


def test_compose_command_records_process_errors_and_nonzero_status(tmp_path, monkeypatch):
    controller = ComposeController(
        project="project", evidence_dir=tmp_path, voter_target=3, image="image", profiles=("five",)
    )
    monkeypatch.setattr(
        consensus_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=2, stdout="", stderr="failure"),
    )
    try:
        controller.command("ps")
    except RuntimeError as exc:
        assert "failed (2)" in str(exc)
    monkeypatch.setattr(
        consensus_campaign.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("docker missing")),
    )
    try:
        controller.command("ps")
    except RuntimeError as exc:
        assert "docker missing" in str(exc)
    records = [
        json.loads(line) for line in (tmp_path / "compose-commands.jsonl").read_text().splitlines()
    ]
    assert records[-1]["returncode"] is None
