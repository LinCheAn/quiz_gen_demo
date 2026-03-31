from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from shutil import which
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from utils.config import AppConfig


@dataclass
class ManagedServerSpec:
    name: str
    base_url: str
    expected_model: str
    conda_env: str
    command: list[str]
    log_path: Path


class ModelServerManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logs_dir = self.config.artifacts_dir / "server_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._started_processes: dict[str, subprocess.Popen[Any]] = {}
        self._log_handles: dict[str, TextIOWrapper] = {}
        self._registered_atexit = False

    def ensure_servers_ready(self) -> None:
        if not self.config.auto_start_model_servers:
            print("AUTO_START_MODEL_SERVERS=0, skip starting summary/quiz servers.")
            return

        if which("conda") is None:
            raise RuntimeError("`conda` was not found in PATH, cannot auto-start model servers.")

        for server_name in ("summary", "quiz"):
            self.ensure_server_ready(server_name)

    def stop_started_servers(self) -> None:
        for server_name in list(self._started_processes):
            self.release_server(server_name)

    def prepare_for_target(self, server_name: str) -> None:
        if not self.config.auto_start_model_servers:
            return
        if self.config.model_server_start_strategy == "sequential":
            for other_name in list(self._started_processes):
                if other_name != server_name:
                    self.release_server(other_name)
        self.ensure_server_ready(server_name)

    def ensure_server_ready(self, server_name: str) -> None:
        if not self.config.auto_start_model_servers:
            return
        if which("conda") is None:
            raise RuntimeError("`conda` was not found in PATH, cannot auto-start model servers.")
        self._register_atexit_if_needed()
        spec = self._get_spec(server_name)
        self._ensure_single_server_ready(spec)

    def release_server(self, server_name: str) -> None:
        process = self._started_processes.pop(server_name, None)
        log_handle = self._log_handles.pop(server_name, None)
        if process is None:
            if log_handle is not None:
                log_handle.close()
            return

        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    process.kill()
        if log_handle is not None:
            log_handle.close()

    def _ensure_single_server_ready(self, spec: ManagedServerSpec) -> None:
        existing_process = self._started_processes.get(spec.name)
        if existing_process is not None and existing_process.poll() is not None:
            self.release_server(spec.name)

        ready, details = self._probe_endpoint(spec.base_url, spec.expected_model)
        if ready:
            print(f"[{spec.name}] Reusing existing endpoint {spec.base_url} with model {spec.expected_model}.")
            return
        if details.get("reachable"):
            available = ", ".join(details.get("models", [])) or "none"
            raise RuntimeError(
                f"[{spec.name}] Endpoint {spec.base_url} is already reachable but does not expose the expected "
                f"model `{spec.expected_model}`. Available models: {available}. "
                "Change the configured port/model or stop the conflicting server."
            )

        self._assert_cuda_available(spec.conda_env, spec.name)
        print(f"[{spec.name}] Starting model server on {spec.base_url}")
        print(f"[{spec.name}] Command: {' '.join(spec.command)}")
        log_file = spec.log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            spec.command,
            cwd=self.config.project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        self._started_processes[spec.name] = process
        self._log_handles[spec.name] = log_file

        timeout_at = time.time() + self.config.model_server_startup_timeout_s
        while time.time() < timeout_at:
            if process.poll() is not None:
                raise RuntimeError(
                    f"[{spec.name}] Server process exited early with code {process.returncode}. "
                    f"Check log: {spec.log_path}\nLast log lines:\n{self._tail_log(spec.log_path)}"
                )

            ready, details = self._probe_endpoint(spec.base_url, spec.expected_model)
            if ready:
                print(f"[{spec.name}] Ready at {spec.base_url}")
                return
            time.sleep(self.config.model_server_probe_interval_s)

        raise RuntimeError(
            f"[{spec.name}] Timed out waiting for endpoint {spec.base_url}. "
            f"Check log: {spec.log_path}\nLast log lines:\n{self._tail_log(spec.log_path)}"
        )

    def _get_spec(self, server_name: str) -> ManagedServerSpec:
        if server_name == "summary":
            return self._build_summary_spec()
        if server_name == "quiz":
            return self._build_quiz_spec()
        raise ValueError(f"Unknown server name: {server_name}")

    def _build_summary_spec(self) -> ManagedServerSpec:
        host, port = self._parse_host_port(self.config.summary_base_url)
        log_path = self.logs_dir / f"summary_server_{port}.log"
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.config.summary_server_conda_env,
            "vllm",
            "serve",
            self.config.summary_server_model,
            "--host",
            host,
            "--port",
            str(port),
            "--served-model-name",
            self.config.summary_model_name,
            "--tensor-parallel-size",
            str(self.config.summary_server_tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.config.summary_server_gpu_memory_utilization),
            "--max-model-len",
            str(self.config.summary_server_max_model_len),
            "--dtype",
            self.config.summary_server_dtype,
        ]
        return ManagedServerSpec(
            name="summary",
            base_url=self.config.summary_base_url,
            expected_model=self.config.summary_model_name,
            conda_env=self.config.summary_server_conda_env,
            command=command,
            log_path=log_path,
        )

    def _build_quiz_spec(self) -> ManagedServerSpec:
        host, port = self._parse_host_port(self.config.quiz_base_url)
        log_path = self.logs_dir / f"quiz_server_{port}.log"
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.config.quiz_server_conda_env,
            "vllm",
            "serve",
            self.config.quiz_server_model,
            "--enable-lora",
            "--lora-modules",
            f"{self.config.quiz_model_name}={self.config.quiz_model_path}",
            "--host",
            host,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(self.config.quiz_server_tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.config.quiz_server_gpu_memory_utilization),
            "--max-model-len",
            str(self.config.quiz_server_max_model_len),
            "--dtype",
            self.config.quiz_server_dtype,
        ]
        return ManagedServerSpec(
            name="quiz",
            base_url=self.config.quiz_base_url,
            expected_model=self.config.quiz_model_name,
            conda_env=self.config.quiz_server_conda_env,
            command=command,
            log_path=log_path,
        )

    def _assert_cuda_available(self, conda_env: str, server_name: str) -> None:
        check = subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                conda_env,
                "python",
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
            ],
            cwd=self.config.project_root,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            raise RuntimeError(
                f"[{server_name}] No CUDA device is visible inside conda env `{conda_env}`. "
                "Auto-start currently assumes GPU-backed vLLM serving."
            )

    def _probe_endpoint(self, base_url: str, expected_model: str) -> tuple[bool, dict[str, Any]]:
        models_url = self._join_url(base_url, "models")
        try:
            with urlopen(models_url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, ConnectionError, TimeoutError):
            return False, {"reachable": False, "models": []}
        except HTTPError as exc:
            return False, {"reachable": False, "models": [], "error": str(exc)}
        except Exception as exc:
            return False, {"reachable": False, "models": [], "error": str(exc)}

        models = [item.get("id", "") for item in payload.get("data", []) if isinstance(item, dict)]
        return expected_model in models, {"reachable": True, "models": models}

    @staticmethod
    def _parse_host_port(base_url: str) -> tuple[str, int]:
        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        if port is None:
            raise ValueError(f"base_url must include an explicit port: {base_url}")
        return host, port

    @staticmethod
    def _join_url(base_url: str, suffix: str) -> str:
        trimmed = base_url.rstrip("/")
        return f"{trimmed}/{suffix.lstrip('/')}"

    @staticmethod
    def _tail_log(log_path: Path, n_lines: int = 20) -> str:
        if not log_path.exists():
            return "(log file not created yet)"
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:]) if lines else "(empty log file)"

    def _register_atexit_if_needed(self) -> None:
        if self._registered_atexit:
            return
        atexit.register(self.stop_started_servers)
        self._registered_atexit = True
