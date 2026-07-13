import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from unified_multi_agent_coordination import (
    AgentAdmissionPolicy,
    CapabilityRequirement,
    CoordinationSdk,
    FeasibilityAnalyzer,
    ProblemRequest,
    SolutionProposal,
    TaskSpec,
)


@pytest.mark.asyncio
async def test_official_a2a_hello_world_sample_interoperability():
    sample = Path("vendor/a2a-samples/helloworld").resolve()
    process = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=sample,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        async with httpx.AsyncClient() as client:
            for _ in range(50):
                try:
                    response = await client.get(
                        "http://127.0.0.1:9999/.well-known/agent-card.json",
                        timeout=0.2,
                    )
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.1)
            else:
                stderr = process.stderr.read() if process.stderr else ""
                pytest.fail("Official sample did not start: " + stderr[-1000:])

        sdk = CoordinationSdk(
            admission_policy=AgentAdmissionPolicy(allow_insecure_development=True)
        )
        entry = await sdk.register_a2a_agent(
            "http://127.0.0.1:9999/.well-known/agent-card.json"
        )
        requirement = CapabilityRequirement(
            name="Echo Bot",
            capability_id="echo_bot",
            input_modes=["text/plain"],
            output_modes=["text/plain"],
            validation_contract={"json_schema": {"type": "object"}},
        )
        request = ProblemRequest(user_goal="Say hello.", requirements=[requirement])
        task = TaskSpec(
            task_id="t1",
            requirement_name=requirement.name,
            capability_id="echo_bot",
            assigned_to=entry.agent_id,
            validation_contract={"json_schema": {"type": "object"}},
        )
        proposal = SolutionProposal(tasks=[task], execution_order=["t1"])
        report = FeasibilityAnalyzer().check(request, [entry], proposal)

        result = await sdk.send_task(report, task, {"text": "interoperability"})

        assert report.feasible
        assert result.status == "completed"
        assert "Hello, World!" in str(result.output)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
