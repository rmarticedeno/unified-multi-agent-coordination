"""Single-container launcher for a coordinator and optional local etcd member."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from .cluster import CoordinatorNodeRecord, HmacAuthenticator, SignedEnvelope
from .cluster_discovery import ClusterDiscovery
from .etcd_client import EtcdClient


class CoordinatorNodeLauncher:
    def __init__(self) -> None:
        self.cluster_id = os.getenv("COORDINATION_CLUSTER_ID", "default")
        self.node_id = os.getenv(
            "COORDINATION_NODE_ID",
            os.getenv("COORDINATION_COORDINATOR_ID", "coordinator-node"),
        )
        self.bootstrap = _env_bool("COORDINATION_BOOTSTRAP")
        self.allow_insecure = _env_bool("COORDINATION_ALLOW_INSECURE_CLUSTER")
        self.authenticator = HmacAuthenticator(
            os.getenv("COORDINATION_CLUSTER_SECRET", ""),
            allow_insecure=self.allow_insecure,
        )
        self.api_url = os.getenv(
            "COORDINATION_ADVERTISE_API_URL", f"http://{self.node_id}:8000"
        ).rstrip("/")
        self.peer_url = os.getenv(
            "COORDINATION_ADVERTISE_PEER_URL", f"http://{self.node_id}:2380"
        )
        self.client_url = os.getenv(
            "COORDINATION_ADVERTISE_CLIENT_URL", f"http://{self.node_id}:2379"
        )
        self.data_dir = Path(
            os.getenv("COORDINATION_ETCD_DATA_DIR", "/var/lib/unified-coordination/etcd")
        )
        self.assignment_path = self.data_dir / "coordinator-assignment.json"
        self.persisted_assignment = self._load_persisted_assignment()
        self.voter_target = int(self.persisted_assignment.get("voter_target") or 0) or int(
            os.getenv("COORDINATION_VOTER_TARGET", "3")
        )
        if self.voter_target not in {1, 3, 5, 7}:
            raise ValueError("COORDINATION_VOTER_TARGET must be 1, 3, 5, or 7.")
        self.etcd_binary = os.getenv("COORDINATION_ETCD_BINARY", "/usr/local/bin/etcd")
        self.discovery = ClusterDiscovery(
            cluster_id=self.cluster_id,
            node_id=self.node_id,
            authenticator=self.authenticator,
            seeds=_csv_env("COORDINATION_DISCOVERY_SEEDS"),
            multicast_group=os.getenv("COORDINATION_MULTICAST_GROUP", "239.255.42.99"),
            multicast_port=int(os.getenv("COORDINATION_MULTICAST_PORT", "7947")),
            multicast_hops=int(os.getenv("COORDINATION_MULTICAST_HOPS", "1")),
            timeout_s=float(os.getenv("COORDINATION_DISCOVERY_TIMEOUT_S", "3.0")),
        )
        self.etcd_process: asyncio.subprocess.Process | None = None
        self.service_process: asyncio.subprocess.Process | None = None
        self.seed_url: str | None = None
        self.etcd_endpoints: list[str] = []
        self._assignment_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def run(self) -> int:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        persisted_role = str(self.persisted_assignment.get("role") or "")
        member_data_exists = (self.data_dir / "member").is_dir()
        if persisted_role in {"voter", "learner", "client"} and (
            persisted_role == "client" or member_data_exists
        ):
            role = persisted_role
            member_id = int(self.persisted_assignment.get("member_id") or 0)
            endpoints = list(self.persisted_assignment.get("etcd_endpoints") or [])
            self.seed_url = str(self.persisted_assignment.get("seed_url") or "") or None
            os.environ["COORDINATION_DISCOVERY_METHOD"] = "persisted"
            if role in {"voter", "learner"}:
                initial_cluster = str(
                    self.persisted_assignment.get("initial_cluster") or ""
                )
                if initial_cluster:
                    await self._start_joined_member(initial_cluster, wait_for_health=False)
                else:
                    await self._start_bootstrap_member(wait_for_health=False)
                if "http://127.0.0.1:2379" not in endpoints:
                    endpoints.insert(0, "http://127.0.0.1:2379")
        else:
            seed_url, discovery_method = await self.discovery.discover()
            os.environ["COORDINATION_DISCOVERY_METHOD"] = discovery_method
            if seed_url is None:
                if not self.bootstrap:
                    seed_url = await self._wait_for_cluster()
                else:
                    await self._start_bootstrap_member()
                    self.seed_url = self.api_url
                    role = "voter"
                    member_id = 0
                    endpoints = ["http://127.0.0.1:2379"]
                    self._persist_assignment(
                        {
                            "role": role,
                            "member_id": member_id,
                            "initial_cluster": f"{self.node_id}={self.peer_url}",
                            "etcd_endpoints": endpoints,
                            "voter_target": self.voter_target,
                            "seed_url": self.api_url,
                        }
                    )
            if seed_url is not None and self.etcd_process is None:
                self.seed_url = seed_url
                assignment = await self._join(seed_url)
                role = str(assignment["role"])
                member_id = int(assignment.get("member_id") or 0)
                self.voter_target = int(
                    assignment.get("voter_target") or self.voter_target
                )
                endpoints = list(assignment.get("etcd_endpoints") or [])
                if role == "learner":
                    await self._start_joined_member(str(assignment["initial_cluster"]))
                    endpoints.insert(0, "http://127.0.0.1:2379")
                self._persist_assignment(
                    {
                        **assignment,
                        "etcd_endpoints": endpoints,
                        "seed_url": seed_url,
                    }
                )
        if not endpoints:
            raise RuntimeError("The cluster returned no usable etcd client endpoints.")
        os.environ["COORDINATION_COORDINATOR_ID"] = self.node_id
        self.etcd_endpoints = endpoints
        os.environ["COORDINATION_NODE_ROLE"] = role
        os.environ["COORDINATION_MEMBER_ID"] = str(member_id)
        os.environ["COORDINATION_STORE_URL"] = "etcd://" + ",".join(
            endpoint.removeprefix("http://").removeprefix("https://")
            for endpoint in endpoints
        )
        os.environ["COORDINATION_ADVERTISE_API_URL"] = self.api_url
        os.environ["COORDINATION_ADVERTISE_PEER_URL"] = self.peer_url
        os.environ["COORDINATION_ADVERTISE_CLIENT_URL"] = self.client_url
        self.service_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "unified_multi_agent_coordination.service",
            env=os.environ.copy(),
        )
        self._assignment_task = asyncio.create_task(self._assignment_monitor())
        self._install_signal_handlers()
        return await self._supervise()

    async def shutdown(self) -> None:
        if self._stopping.is_set():
            return
        self._stopping.set()
        if self._assignment_task is not None:
            self._assignment_task.cancel()
        if (
            _env_bool("COORDINATION_LEAVE_ON_SHUTDOWN", default=True)
            and self.seed_url
            and self.seed_url != self.api_url
        ):
            with suppress(Exception):
                await self._leave(self.seed_url)
        for process in (self.service_process, self.etcd_process):
            if process is not None and process.returncode is None:
                process.terminate()
        for process in (self.service_process, self.etcd_process):
            if process is not None and process.returncode is None:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=10)
                if process.returncode is None:
                    process.kill()
        await self.discovery.close()

    async def _wait_for_cluster(self) -> str:
        while not self._stopping.is_set():
            seed, method = await self.discovery.discover()
            if seed:
                os.environ["COORDINATION_DISCOVERY_METHOD"] = method
                return seed
            await asyncio.sleep(2.0)
        raise RuntimeError("Coordinator node stopped before discovering a cluster.")

    async def _start_bootstrap_member(self, *, wait_for_health: bool = True) -> None:
        await self._start_etcd(
            [
                "--name",
                self.node_id,
                "--initial-cluster",
                f"{self.node_id}={self.peer_url}",
                "--initial-cluster-state",
                "new",
            ],
            wait_for_health=wait_for_health,
        )

    async def _start_joined_member(
        self, initial_cluster: str, *, wait_for_health: bool = True
    ) -> None:
        await self._start_etcd(
            [
                "--name",
                self.node_id,
                "--initial-cluster",
                initial_cluster,
                "--initial-cluster-state",
                "existing",
            ],
            wait_for_health=wait_for_health,
        )

    async def _start_etcd(
        self,
        membership_args: list[str],
        *,
        wait_for_health: bool = True,
    ) -> None:
        args = [
            self.etcd_binary,
            "--data-dir",
            str(self.data_dir),
            "--listen-client-urls",
            "http://0.0.0.0:2379",
            "--advertise-client-urls",
            self.client_url,
            "--listen-peer-urls",
            "http://0.0.0.0:2380",
            "--initial-advertise-peer-urls",
            self.peer_url,
            "--initial-cluster-token",
            self.cluster_id,
            "--strict-reconfig-check=true",
            *membership_args,
        ]
        self.etcd_process = await asyncio.create_subprocess_exec(*args)
        if not wait_for_health:
            await asyncio.sleep(0.25)
            if self.etcd_process.returncode is not None:
                raise RuntimeError(
                    f"Local etcd exited with code {self.etcd_process.returncode}."
                )
            return
        client = EtcdClient(["http://127.0.0.1:2379"], timeout_s=1.0)
        try:
            for _ in range(60):
                if self.etcd_process.returncode is not None:
                    raise RuntimeError(
                        f"Local etcd exited with code {self.etcd_process.returncode}."
                    )
                try:
                    await client.status()
                    return
                except Exception:
                    await asyncio.sleep(0.25)
            raise RuntimeError("Local etcd did not become healthy.")
        finally:
            await client.close()

    async def _join(self, seed_url: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=15.0) as client:
            last_error: Exception | None = None
            for _ in range(120):
                request = self.authenticator.sign(
                    SignedEnvelope(
                        message_type="cluster_join",
                        cluster_id=self.cluster_id,
                        node_id=self.node_id,
                        payload={
                            "api_url": self.api_url,
                            "peer_url": self.peer_url,
                            "client_url": self.client_url,
                            "voter_target": self.voter_target,
                        },
                    )
                )
                try:
                    response = await client.post(
                        f"{seed_url}/internal/cluster/join",
                        json=request.model_dump(mode="json"),
                    )
                    if response.status_code >= 500 or response.status_code == 409:
                        await asyncio.sleep(1.0)
                        continue
                    response.raise_for_status()
                    envelope = SignedEnvelope.model_validate(response.json())
                    if envelope.message_type != "cluster_join_response":
                        raise RuntimeError("Coordinator returned an invalid join response.")
                    self.authenticator.verify(
                        envelope,
                        expected_cluster_id=self.cluster_id,
                    )
                    result = envelope.payload
                    break
                except httpx.HTTPError as exc:
                    last_error = exc
                    await asyncio.sleep(1.0)
            else:
                raise RuntimeError(f"Coordinator join did not converge: {last_error}")
        if not isinstance(result, dict):
            raise RuntimeError("Coordinator join returned an invalid assignment.")
        return result

    async def _leave(self, seed_url: str) -> None:
        request = self.authenticator.sign(
            SignedEnvelope(
                message_type="cluster_leave",
                cluster_id=self.cluster_id,
                node_id=self.node_id,
            )
        )
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{seed_url}/internal/cluster/leave",
                json=request.model_dump(mode="json"),
            )

    async def _supervise(self) -> int:
        if self.service_process is None:
            raise RuntimeError("Coordinator service was not started.")
        waiters = [asyncio.create_task(self.service_process.wait())]
        stop_waiter = asyncio.create_task(self._stopping.wait())
        done, pending = await asyncio.wait(
            [*waiters, stop_waiter], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        exit_code = 0
        for task in done:
            if task is not stop_waiter:
                exit_code = int(task.result())
        await self.shutdown()
        return exit_code

    async def _assignment_monitor(self) -> None:
        client = EtcdClient(self.etcd_endpoints, timeout_s=2.0)
        key = f"/umac/{self.cluster_id}/membership/intents/{self.node_id}".encode()
        try:
            while not self._stopping.is_set():
                try:
                    await client.sync_endpoints()
                    result = await client.range(key)
                    if result.values:
                        assignment = CoordinatorNodeRecord.model_validate_json(
                            result.values[0].value
                        )
                        self._persist_assignment(
                            {
                                **assignment.model_dump(mode="json"),
                                "etcd_endpoints": self.etcd_endpoints,
                                "seed_url": self.seed_url or "",
                            }
                        )
                        member_role = assignment.role in {"learner", "voter"}
                        member_stopped = (
                            self.etcd_process is None
                            or self.etcd_process.returncode is not None
                        )
                        if member_role and member_stopped:
                            if assignment.initial_cluster:
                                await self._start_joined_member(
                                    assignment.initial_cluster
                                )
                            elif self.bootstrap:
                                await self._start_bootstrap_member()
                except Exception:
                    # Quorum loss is surfaced by readiness.  Membership is not
                    # guessed or repaired without an authoritative assignment.
                    pass
                await asyncio.sleep(1.0)
        finally:
            await client.close()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(signum, lambda: asyncio.create_task(self.shutdown()))

    def _load_persisted_assignment(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.assignment_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("cluster_id") == self.cluster_id:
                return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        return {}

    def _persist_assignment(self, assignment: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.assignment_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"cluster_id": self.cluster_id, **assignment},
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.assignment_path)


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str) -> list[str]:
    return [value.strip() for value in os.getenv(name, "").split(",") if value.strip()]


def main() -> None:
    raise SystemExit(asyncio.run(CoordinatorNodeLauncher().run()))


if __name__ == "__main__":
    main()
