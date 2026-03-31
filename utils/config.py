from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class AppConfig:
    project_root: Path = PROJECT_ROOT
    uploads_dir: Path | None = None
    runs_dir: Path | None = None
    artifacts_dir: Path | None = None
    model_info_path: Path | None = None
    default_mode: str = os.getenv("DEMO_DEFAULT_MODE", "live")
    asr_model_name: str = os.getenv("ASR_MODEL_NAME", "MediaTek-Research/Breeze-ASR-25")
    asr_chunk_length_s: int = _env_int("ASR_CHUNK_LENGTH_S", 30)
    asr_conda_env: str = os.getenv("ASR_CONDA_ENV", "inference")
    # summary_model_name: str = os.getenv("SUMMARY_MODEL_NAME", "Qwen/Qwen3.5-4B")
    summary_model_name: str = os.getenv("SUMMARY_MODEL_NAME", "Qwen/Qwen3.5-4B")
    summary_base_url: str = os.getenv("SUMMARY_BASE_URL", "http://127.0.0.1:8001/v1")
    summary_api_key: str = os.getenv("SUMMARY_API_KEY", os.getenv("OPENAI_API_KEY", "0"))
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_conda_env: str = os.getenv("EMBEDDING_CONDA_ENV", "inference")
    embedding_use_fp16: bool = _env_bool("EMBEDDING_USE_FP16", True)
    quiz_model_name: str = os.getenv("QUIZ_MODEL_NAME", "grpo_v4.2")
    quiz_base_url: str = os.getenv("QUIZ_BASE_URL", "http://127.0.0.1:8000/v1")
    quiz_api_key: str = os.getenv("QUIZ_API_KEY", "0")
    quiz_model_path: str = os.getenv("QUIZ_MODEL_PATH", "/home/r13922145/rl_model/grpo_v4.2")
    quiz_base_model_name: str = os.getenv("QUIZ_BASE_MODEL_NAME", "unsloth/Llama-3.1-8B-Instruct")
    quiz_question_count: int = _env_int("QUIZ_QUESTION_COUNT", 3)
    quiz_temperature: float = _env_float("QUIZ_TEMPERATURE", 0.7)
    summary_temperature: float = _env_float("SUMMARY_TEMPERATURE", 0.3)
    auto_start_model_servers: bool = _env_bool("AUTO_START_MODEL_SERVERS", True)
    model_server_start_strategy: str = os.getenv("MODEL_SERVER_START_STRATEGY", "sequential")
    keep_model_servers_warm: bool = _env_bool("KEEP_MODEL_SERVERS_WARM", True)
    summary_server_conda_env: str = os.getenv("SUMMARY_SERVER_CONDA_ENV", "vllm")
    quiz_server_conda_env: str = os.getenv("QUIZ_SERVER_CONDA_ENV", "vllm")
    summary_server_model: str = os.getenv(
        "SUMMARY_SERVER_MODEL",
        # "Qwen/Qwen3.5-4B",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
    )
    quiz_server_model: str = os.getenv(
        "QUIZ_SERVER_MODEL",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
    )
    summary_server_gpu_memory_utilization: float = _env_float(
        "SUMMARY_SERVER_GPU_MEMORY_UTILIZATION",
        0.9,
    )
    quiz_server_gpu_memory_utilization: float = _env_float(
        "QUIZ_SERVER_GPU_MEMORY_UTILIZATION",
        0.8,
    )
    summary_server_max_model_len: int = _env_int("SUMMARY_SERVER_MAX_MODEL_LEN", 32768)
    quiz_server_max_model_len: int = _env_int("QUIZ_SERVER_MAX_MODEL_LEN", 8192)
    summary_server_dtype: str = os.getenv("SUMMARY_SERVER_DTYPE", "bfloat16")
    summary_server_quantization: str | None = os.getenv("SUMMARY_SERVER_QUANTIZATION")
    quiz_server_dtype: str = os.getenv("QUIZ_SERVER_DTYPE", "bfloat16")
    summary_server_tensor_parallel_size: int = _env_int("SUMMARY_SERVER_TP_SIZE", 1)
    quiz_server_tensor_parallel_size: int = _env_int("QUIZ_SERVER_TP_SIZE", 1)
    model_server_startup_timeout_s: int = _env_int("MODEL_SERVER_STARTUP_TIMEOUT_S", 300)
    model_server_probe_interval_s: float = _env_float("MODEL_SERVER_PROBE_INTERVAL_S", 2.0)
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = _env_int("APP_PORT", 7860)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.model_info_path = Path(self.model_info_path or self.project_root / "model_info.json")
        self.uploads_dir = Path(self.uploads_dir or self.project_root / "data" / "uploads")
        self.artifacts_dir = Path(self.artifacts_dir or self.project_root / "artifacts")
        self.runs_dir = Path(self.runs_dir or self.artifacts_dir / "runs")
        self.default_mode = "live"
        self.model_server_start_strategy = self.model_server_start_strategy.strip().lower()
        if self.model_server_start_strategy not in {"preload", "sequential"}:
            self.model_server_start_strategy = "sequential"

    def ensure_directories(self) -> None:
        for path in (self.uploads_dir, self.artifacts_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def copy_with_overrides(self, **overrides: object) -> "AppConfig":
        config = replace(self, **overrides)
        config.ensure_directories()
        return config


def load_config() -> AppConfig:
    config = AppConfig()
    config.ensure_directories()
    return config
