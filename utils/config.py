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


def describe_runtime_target(env_name: str | None) -> str:
    normalized = (env_name or "").strip()
    if normalized:
        return f"conda env `{normalized}`"
    return "current runtime"


@dataclass(frozen=True)
class ASRPreset:
    id: str
    label: str
    backend: str
    model_name: str


ASR_PRESETS: tuple[ASRPreset, ...] = (
    ASRPreset(
        id="legacy-transformers-breeze-asr-25",
        label="Legacy Transformers Breeze ASR 25",
        backend="transformers",
        model_name="MediaTek-Research/Breeze-ASR-25",
    ),
    ASRPreset(
        id="faster-whisper-breeze-asr-25",
        label="Faster-Whisper Breeze ASR 25",
        backend="faster_whisper",
        model_name="SoybeanMilk/faster-whisper-Breeze-ASR-25",
    ),
)
ASR_PRESETS_BY_ID = {preset.id: preset for preset in ASR_PRESETS}


@dataclass
class AppConfig:
    project_root: Path = PROJECT_ROOT
    uploads_dir: Path | None = None
    runs_dir: Path | None = None
    artifacts_dir: Path | None = None
    models_dir: Path | None = None
    hf_home: Path | None = None
    model_info_path: Path | None = None
    default_mode: str = os.getenv("DEMO_DEFAULT_MODE", "live")
    asr_preset_id: str = os.getenv("ASR_PRESET_ID", "faster-whisper-breeze-asr-25")
    asr_backend: str = os.getenv("ASR_BACKEND", "")
    asr_model_name: str = os.getenv("ASR_MODEL_NAME", "")
    asr_chunk_length_s: int = _env_int("ASR_CHUNK_LENGTH_S", 30)
    asr_conda_env: str = os.getenv("ASR_CONDA_ENV", "")
    summary_model_name: str = os.getenv("SUMMARY_MODEL_NAME", "Qwen/Qwen3.5-4B")
    summary_base_url: str = os.getenv("SUMMARY_BASE_URL", "http://127.0.0.1:8001/v1")
    summary_api_key: str = os.getenv("SUMMARY_API_KEY", os.getenv("OPENAI_API_KEY", "0"))
    summary_model_path: str | None = os.getenv("SUMMARY_MODEL_PATH")
    summary_base_model_name: str = os.getenv("SUMMARY_BASE_MODEL_NAME", "Qwen/Qwen3.5-4B")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    embedding_conda_env: str = os.getenv("EMBEDDING_CONDA_ENV", "")
    embedding_use_fp16: bool = _env_bool("EMBEDDING_USE_FP16", True)
    quiz_model_name: str = os.getenv("QUIZ_MODEL_NAME", "grpo_v4.2")
    quiz_base_url: str = os.getenv("QUIZ_BASE_URL", "http://127.0.0.1:8000/v1")
    quiz_api_key: str = os.getenv("QUIZ_API_KEY", "0")
    quiz_model_path: str | None = os.getenv("QUIZ_MODEL_PATH")
    quiz_base_model_name: str = os.getenv("QUIZ_BASE_MODEL_NAME", "unsloth/Llama-3.1-8B-Instruct")
    quiz_question_count: int = _env_int("QUIZ_QUESTION_COUNT", 3)
    quiz_temperature: float = _env_float("QUIZ_TEMPERATURE", 0.7)
    summary_temperature: float = _env_float("SUMMARY_TEMPERATURE", 0.3)
    auto_start_model_servers: bool = _env_bool("AUTO_START_MODEL_SERVERS", True)
    model_server_start_strategy: str = os.getenv("MODEL_SERVER_START_STRATEGY", "sequential")
    keep_model_servers_warm: bool = _env_bool("KEEP_MODEL_SERVERS_WARM", True)
    summary_server_conda_env: str = os.getenv("SUMMARY_SERVER_CONDA_ENV", "")
    quiz_server_conda_env: str = os.getenv("QUIZ_SERVER_CONDA_ENV", "")
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
    quiz_server_quantization: str | None = os.getenv("QUIZ_SERVER_QUANTIZATION")
    summary_server_tensor_parallel_size: int = _env_int("SUMMARY_SERVER_TP_SIZE", 1)
    quiz_server_tensor_parallel_size: int = _env_int("QUIZ_SERVER_TP_SIZE", 1)
    model_server_startup_timeout_s: int = _env_int("MODEL_SERVER_STARTUP_TIMEOUT_S", 300)
    model_server_probe_interval_s: float = _env_float("MODEL_SERVER_PROBE_INTERVAL_S", 2.0)
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = _env_int("APP_PORT", 7860)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.models_dir = Path(self.models_dir or self.project_root / "models")
        self.hf_home = Path(self.hf_home or self.models_dir / "cache" / "huggingface")
        self.model_info_path = Path(self.model_info_path or self.project_root / "model_info.json")
        self.uploads_dir = Path(self.uploads_dir or self.project_root / "data" / "uploads")
        self.artifacts_dir = Path(self.artifacts_dir or self.project_root / "artifacts")
        self.runs_dir = Path(self.runs_dir or self.artifacts_dir / "runs")
        self.default_mode = "live"
        self.asr_preset_id = self.asr_preset_id.strip()
        self.asr_backend = self.asr_backend.strip()
        self.asr_model_name = self.asr_model_name.strip()
        self.asr_conda_env = self.asr_conda_env.strip()
        self.embedding_conda_env = self.embedding_conda_env.strip()
        self.summary_server_conda_env = self.summary_server_conda_env.strip()
        self.quiz_server_conda_env = self.quiz_server_conda_env.strip()
        self.model_server_start_strategy = self.model_server_start_strategy.strip().lower()
        preset = self.resolve_asr_preset(self.asr_preset_id)
        if not self.asr_backend:
            self.asr_backend = preset.backend
        if not self.asr_model_name:
            self.asr_model_name = preset.model_name
        if self.model_server_start_strategy not in {"preload", "sequential"}:
            self.model_server_start_strategy = "sequential"

    def ensure_directories(self) -> None:
        for path in (
            self.models_dir,
            self.hf_home,
            self.uploads_dir,
            self.artifacts_dir,
            self.runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def apply_environment_defaults(self) -> None:
        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HF_HUB_CACHE", str(self.hf_home / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.hf_home / "transformers"))

    def copy_with_overrides(self, **overrides: object) -> "AppConfig":
        config = replace(self, **overrides)
        config.ensure_directories()
        config.apply_environment_defaults()
        return config

    def resolve_asr_preset(self, preset_id: str | None = None) -> ASRPreset:
        resolved_id = (preset_id or self.asr_preset_id).strip()
        try:
            return ASR_PRESETS_BY_ID[resolved_id]
        except KeyError as exc:
            raise ValueError(f"Unknown ASR preset id: {resolved_id}") from exc

    @staticmethod
    def asr_choices() -> list[tuple[str, str]]:
        return [(preset.label, preset.id) for preset in ASR_PRESETS]


def load_config() -> AppConfig:
    config = AppConfig()
    config.ensure_directories()
    config.apply_environment_defaults()
    return config
