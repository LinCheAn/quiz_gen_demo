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


ROLE_ORDER = {"summary": 0, "quiz": 1}


@dataclass(frozen=True)
class RoleServerSpec:
    name: str
    base_url: str
    expected_model: str
    conda_env: str
    server_model: str
    lora_path: str | None
    gpu_memory_utilization: float
    max_model_len: int
    tensor_parallel_size: int
    dtype: str | None
    quantization: str | None


@dataclass
class ManagedServerSpec:
    key: str
    name: str
    base_url: str
    expected_models: tuple[str, ...]
    conda_env: str
    command: list[str]
    log_path: Path
    roles: frozenset[str]


class ModelServerManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logs_dir = self.config.artifacts_dir / "server_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._started_processes: dict[str, subprocess.Popen[Any]] = {}
        self._log_handles: dict[str, TextIOWrapper] = {}
        self._registered_atexit = False
        self._role_specs = {
            "summary": self._build_role_spec("summary"),
            "quiz": self._build_role_spec("quiz"),
        }
        self._process_specs, self._role_to_process_key = self._plan_process_specs()

    def ensure_servers_ready(self) -> None:
        if not self.config.auto_start_model_servers:
            print("AUTO_START_MODEL_SERVERS=0, skip starting summary/quiz servers.")
            return

        if which("conda") is None:
            raise RuntimeError("`conda` was not found in PATH, cannot auto-start model servers.")

        for process_key in dict.fromkeys(self._role_to_process_key[name] for name in ("summary", "quiz")):
            self._ensure_process_ready(process_key)

    def stop_started_servers(self) -> None:
        for process_key in list(self._started_processes):
            self._release_process(process_key)

    def prepare_for_target(self, server_name: str) -> None:
        if not self.config.auto_start_model_servers:
            return
        self._validate_server_name(server_name)
        target_key = self._role_to_process_key[server_name]
        if self.config.model_server_start_strategy == "sequential":
            for other_key in list(self._started_processes):
                if other_key != target_key:
                    self._release_process(other_key)
        self._ensure_process_ready(target_key)

    def ensure_server_ready(self, server_name: str) -> None:
        if not self.config.auto_start_model_servers:
            return
        self._validate_server_name(server_name)
        self._ensure_process_ready(self._role_to_process_key[server_name])

    def release_server(self, server_name: str) -> None:
        self._validate_server_name(server_name)
        for process_key in self._releasable_process_keys_for_role(server_name):
            self._release_process(process_key)

    def _ensure_process_ready(self, process_key: str) -> None:
        if not self.config.auto_start_model_servers:
            return
        if which("conda") is None:
            raise RuntimeError("`conda` was not found in PATH, cannot auto-start model servers.")
        self._register_atexit_if_needed()
        spec = self._process_specs[process_key]
        self._ensure_single_server_ready(spec)

    def _release_process(self, process_key: str) -> None:
        process = self._started_processes.pop(process_key, None)
        log_handle = self._log_handles.pop(process_key, None)
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
        existing_process = self._started_processes.get(spec.key)
        if existing_process is not None and existing_process.poll() is not None:
            self._release_process(spec.key)

        ready, details = self._probe_endpoint(spec.base_url, spec.expected_models)
        expected_models = ", ".join(spec.expected_models)
        if ready:
            print(f"[{spec.name}] Reusing existing endpoint {spec.base_url} with models {expected_models}.")
            return
        if details.get("reachable"):
            available = ", ".join(details.get("models", [])) or "none"
            raise RuntimeError(
                f"[{spec.name}] Endpoint {spec.base_url} is already reachable but does not expose the expected "
                f"models `{expected_models}`. Available models: {available}. "
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
        self._started_processes[spec.key] = process
        self._log_handles[spec.key] = log_file

        timeout_at = time.time() + self.config.model_server_startup_timeout_s
        while time.time() < timeout_at:
            if process.poll() is not None:
                raise RuntimeError(
                    f"[{spec.name}] Server process exited early with code {process.returncode}. "
                    f"Check log: {spec.log_path}\nLast log lines:\n{self._tail_log(spec.log_path)}"
                )

            ready, details = self._probe_endpoint(spec.base_url, spec.expected_models)
            if ready:
                print(f"[{spec.name}] Ready at {spec.base_url}")
                return
            time.sleep(self.config.model_server_probe_interval_s)

        raise RuntimeError(
            f"[{spec.name}] Timed out waiting for endpoint {spec.base_url}. "
            f"Check log: {spec.log_path}\nLast log lines:\n{self._tail_log(spec.log_path)}"
        )

    def _plan_process_specs(self) -> tuple[dict[str, ManagedServerSpec], dict[str, str]]:
        summary_spec = self._role_specs["summary"]
        quiz_spec = self._role_specs["quiz"]

        if self._can_share_process(summary_spec, quiz_spec):
            process_spec = self._build_process_spec([summary_spec, quiz_spec])
            return (
                {process_spec.key: process_spec},
                {"summary": process_spec.key, "quiz": process_spec.key},
            )

        summary_process = self._build_process_spec([summary_spec])
        quiz_process = self._build_process_spec([quiz_spec])
        return (
            {
                summary_process.key: summary_process,
                quiz_process.key: quiz_process,
            },
            {
                "summary": summary_process.key,
                "quiz": quiz_process.key,
            },
        )

    def _build_process_spec(self, role_specs: list[RoleServerSpec]) -> ManagedServerSpec:
        primary = role_specs[0]
        host, port = self._parse_host_port(primary.base_url)
        role_names = frozenset(spec.name for spec in role_specs)
        base_aliases = sorted({spec.expected_model for spec in role_specs if spec.lora_path is None})
        lora_modules = self._build_lora_modules(role_specs)
        served_model_name = base_aliases[0] if base_aliases else None
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            primary.conda_env,
            "vllm",
            "serve",
            primary.server_model,
        ]
        if lora_modules:
            command.append("--enable-lora")
        if primary.quantization:
            command.extend(["--quantization", primary.quantization])
        command.extend(
            [
                "--host",
                host,
                "--port",
                str(port),
            ]
        )
        if served_model_name:
            command.extend(["--served-model-name", served_model_name])
        if lora_modules:
            command.extend(
                [
                    "--lora-modules",
                    *[f"{alias}={path}" for alias, path in sorted(lora_modules.items())],
                ]
            )
        command.extend(
            [
                "--tensor-parallel-size",
                str(primary.tensor_parallel_size),
                "--gpu-memory-utilization",
                str(primary.gpu_memory_utilization),
                "--max-model-len",
                str(primary.max_model_len),
            ]
        )
        if primary.dtype:
            command.extend(["--dtype", primary.dtype])

        process_name = "+".join(sorted(role_names))
        log_path = self.logs_dir / f"{process_name}_server_{port}.log"
        key = self._make_process_key(primary, served_model_name, lora_modules)
        return ManagedServerSpec(
            key=key,
            name=process_name,
            base_url=primary.base_url,
            expected_models=tuple(sorted({spec.expected_model for spec in role_specs})),
            conda_env=primary.conda_env,
            command=command,
            log_path=log_path,
            roles=role_names,
        )

    def _build_role_spec(self, server_name: str) -> RoleServerSpec:
        if server_name == "summary":
            return RoleServerSpec(
                name="summary",
                base_url=self.config.summary_base_url,
                expected_model=self.config.summary_model_name,
                conda_env=self.config.summary_server_conda_env,
                server_model=self.config.summary_server_model,
                lora_path=self.config.summary_model_path,
                gpu_memory_utilization=self.config.summary_server_gpu_memory_utilization,
                max_model_len=self.config.summary_server_max_model_len,
                tensor_parallel_size=self.config.summary_server_tensor_parallel_size,
                dtype=self.config.summary_server_dtype,
                quantization=self.config.summary_server_quantization,
            )
        if server_name == "quiz":
            return RoleServerSpec(
                name="quiz",
                base_url=self.config.quiz_base_url,
                expected_model=self.config.quiz_model_name,
                conda_env=self.config.quiz_server_conda_env,
                server_model=self.config.quiz_server_model,
                lora_path=self.config.quiz_model_path,
                gpu_memory_utilization=self.config.quiz_server_gpu_memory_utilization,
                max_model_len=self.config.quiz_server_max_model_len,
                tensor_parallel_size=self.config.quiz_server_tensor_parallel_size,
                dtype=self.config.quiz_server_dtype,
                quantization=self.config.quiz_server_quantization,
            )
        raise ValueError(f"Unknown server name: {server_name}")

    def _can_share_process(self, left: RoleServerSpec, right: RoleServerSpec) -> bool:
        comparable_fields = (
            "base_url",
            "conda_env",
            "server_model",
            "gpu_memory_utilization",
            "max_model_len",
            "tensor_parallel_size",
            "dtype",
            "quantization",
        )
        if any(getattr(left, field) != getattr(right, field) for field in comparable_fields):
            return False

        base_aliases = {spec.expected_model for spec in (left, right) if spec.lora_path is None}
        if len(base_aliases) > 1:
            return False

        lora_modules = self._build_lora_modules([left, right], strict=False)
        if lora_modules is None:
            return False
        if base_aliases and base_aliases.intersection(lora_modules):
            return False
        return True

    @staticmethod
    def _build_lora_modules(
        role_specs: list[RoleServerSpec],
        *,
        strict: bool = True,
    ) -> dict[str, str] | None:
        modules: dict[str, str] = {}
        for spec in role_specs:
            if not spec.lora_path:
                continue
            existing_path = modules.get(spec.expected_model)
            if existing_path is not None and existing_path != spec.lora_path:
                if strict:
                    raise ValueError(
                        f"Conflicting LoRA paths configured for model alias {spec.expected_model}: "
                        f"{existing_path} vs {spec.lora_path}"
                    )
                return None
            modules[spec.expected_model] = spec.lora_path
        return modules

    def _releasable_process_keys_for_role(self, server_name: str) -> set[str]:
        process_key = self._role_to_process_key[server_name]
        roles = self._process_specs[process_key].roles
        current_order = ROLE_ORDER[server_name]
        if any(ROLE_ORDER[role] > current_order for role in roles):
            return set()
        return {process_key}

    @staticmethod
    def _make_process_key(
        spec: RoleServerSpec,
        served_model_name: str | None,
        lora_modules: dict[str, str],
    ) -> str:
        lora_key = tuple(sorted(lora_modules.items()))
        return "|".join(
            [
                spec.base_url,
                spec.conda_env,
                spec.server_model,
                served_model_name or "",
                json.dumps(lora_key),
                str(spec.tensor_parallel_size),
                str(spec.gpu_memory_utilization),
                str(spec.max_model_len),
                spec.dtype or "",
                spec.quantization or "",
            ]
        )

    def _probe_endpoint(self, base_url: str, expected_models: tuple[str, ...]) -> tuple[bool, dict[str, Any]]:
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
        return set(expected_models).issubset(models), {"reachable": True, "models": models}

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

    def _build_summary_spec(self) -> ManagedServerSpec:
        return self._process_specs[self._role_to_process_key["summary"]]

    def _build_quiz_spec(self) -> ManagedServerSpec:
        return self._process_specs[self._role_to_process_key["quiz"]]

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

    @staticmethod
    def _validate_server_name(server_name: str) -> None:
        if server_name not in ROLE_ORDER:
            raise ValueError(f"Unknown server name: {server_name}")

    def _register_atexit_if_needed(self) -> None:
        if self._registered_atexit:
            return
        atexit.register(self.stop_started_servers)
        self._registered_atexit = True
