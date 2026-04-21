from __future__ import annotations

import gc
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from utils.config import AppConfig


SUPPORTED_TRANSFORMERS_QUANTIZATION = {"bitsandbytes-8bit", "bitsandbytes-4bit"}


@dataclass(frozen=True)
class TransformersModelSpec:
    model_name: str
    base_model_name: str
    adapter_path: str | None
    dtype: str | None
    quantization: str | None

    @property
    def cache_key(self) -> tuple[str, str | None, str | None, str | None]:
        return (
            self.base_model_name,
            self.adapter_path,
            self.quantization,
            (self.dtype or "").lower() or None,
        )


@dataclass
class LoadedTransformersModel:
    spec: TransformersModelSpec
    tokenizer: Any
    model: Any


@dataclass(frozen=True)
class ModelSourceResolution:
    path: str
    local_files_only: bool
    warnings: tuple[str, ...] = ()
    explicit_local_path_error: str | None = None


class TransformersModelCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: LoadedTransformersModel | None = None

    def load(self, spec: TransformersModelSpec, config: AppConfig) -> LoadedTransformersModel:
        with self._lock:
            if self._active is not None and self._active.spec.cache_key == spec.cache_key:
                return self._active
            self._unload_locked()
            self._active = self._load_locked(spec, config)
            return self._active

    def clear(self) -> None:
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self._active is None:
            return
        loaded = self._active
        self._active = None
        del loaded.model
        del loaded.tokenizer
        gc.collect()
        try:
            import torch
        except Exception:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_locked(self, spec: TransformersModelSpec, config: AppConfig) -> LoadedTransformersModel:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError(
                "transformers backend requires torch and transformers in the current runtime"
            ) from exc

        model_kwargs: dict[str, Any] = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        torch_dtype = _resolve_torch_dtype(spec.dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if spec.quantization:
            if spec.quantization not in SUPPORTED_TRANSFORMERS_QUANTIZATION:
                raise RuntimeError(
                    f"Unsupported transformers quantization `{spec.quantization}`. "
                    "Supported values: bitsandbytes-8bit, bitsandbytes-4bit."
                )
            if spec.quantization == "bitsandbytes-8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
                )

        model_resolution = resolve_model_source_path(
            spec.base_model_name,
            config,
            purpose="model",
        )
        tokenizer_resolution = resolve_model_source_path(
            spec.base_model_name,
            config,
            purpose="tokenizer",
        )
        explicit_errors = [
            message
            for message in (
                model_resolution.explicit_local_path_error,
                tokenizer_resolution.explicit_local_path_error,
            )
            if message
        ]
        if explicit_errors:
            raise RuntimeError(" ".join(dict.fromkeys(explicit_errors)))

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_resolution.path,
                local_files_only=tokenizer_resolution.local_files_only,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_resolution.path,
                local_files_only=model_resolution.local_files_only,
                **model_kwargs,
            )
        except Exception as exc:
            diagnostics = _format_resolution_warnings(
                [
                    *model_resolution.warnings,
                    *tokenizer_resolution.warnings,
                ]
            )
            raise RuntimeError(
                f"Failed to load transformers model `{spec.base_model_name}` "
                f"(resolved source `{model_resolution.path}`). Original error: {exc}"
                f"{diagnostics}"
            ) from exc

        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if spec.adapter_path:
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError(
                    "transformers backend with LoRA adapters requires the `peft` package"
                ) from exc
            try:
                model = PeftModel.from_pretrained(model, spec.adapter_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load LoRA adapter `{spec.adapter_path}` for model "
                    f"`{spec.base_model_name}`. Original error: {exc}"
                ) from exc

        model.eval()
        return LoadedTransformersModel(spec=spec, tokenizer=tokenizer, model=model)


TRANSFORMERS_MODEL_CACHE = TransformersModelCache()


class TransformersTextGenerationBackend:
    def __init__(self, config: AppConfig, *, role: str) -> None:
        self.config = config
        self.role = role
        self._spec = self._build_spec(role)

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        assistant_prefix: str | None = None,
        stop_strings: list[str] | None = None,
    ) -> str:
        loaded = TRANSFORMERS_MODEL_CACHE.load(self._spec, self.config)
        tokenizer = loaded.tokenizer
        model = loaded.model
        prompt = self._build_prompt(tokenizer, messages, assistant_prefix)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_device = getattr(model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        output_ids = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _truncate_at_stop_strings(text, stop_strings)

    def _build_spec(self, role: str) -> TransformersModelSpec:
        if role == "summary":
            return TransformersModelSpec(
                model_name=self.config.summary_model_name,
                base_model_name=self.config.summary_server_model,
                adapter_path=self.config.summary_model_path,
                dtype=self.config.summary_server_dtype,
                quantization=self.config.summary_transformers_quantization,
            )
        if role == "quiz":
            return TransformersModelSpec(
                model_name=self.config.quiz_model_name,
                base_model_name=self.config.quiz_server_model,
                adapter_path=self.config.quiz_model_path,
                dtype=self.config.quiz_server_dtype,
                quantization=self.config.quiz_transformers_quantization,
            )
        raise ValueError(f"Unknown transformers generation role: {role}")

    @staticmethod
    def _build_prompt(
        tokenizer: Any,
        messages: list[dict[str, str]],
        assistant_prefix: str | None,
    ) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = _fallback_prompt(messages)
        else:
            prompt = _fallback_prompt(messages)
        if assistant_prefix:
            prompt = f"{prompt}{assistant_prefix}"
        return prompt


def _resolve_torch_dtype(dtype_name: str | None):
    if not dtype_name:
        return None
    try:
        import torch
    except Exception:
        return None

    normalized = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(normalized)


def resolve_model_source_path(
    model_name: str,
    config: AppConfig,
    *,
    purpose: Literal["model", "tokenizer"] = "model",
) -> ModelSourceResolution:
    candidate = Path(model_name)
    if candidate.exists():
        missing_items = _missing_local_model_requirements(candidate, purpose=purpose)
        if not missing_items:
            return ModelSourceResolution(path=str(candidate), local_files_only=True)
        return ModelSourceResolution(
            path=str(candidate),
            local_files_only=True,
            explicit_local_path_error=(
                f"Explicit local {purpose} path `{candidate}` is incomplete; "
                f"missing {', '.join(missing_items)}."
            ),
        )

    warnings: list[str] = []
    for cache_root in _candidate_hub_cache_roots(config):
        snapshot_path, cache_warnings = _find_hub_snapshot(cache_root, model_name, purpose=purpose)
        warnings.extend(cache_warnings)
        if snapshot_path is not None:
            return ModelSourceResolution(
                path=str(snapshot_path),
                local_files_only=True,
                warnings=tuple(dict.fromkeys(warnings)),
            )
    return ModelSourceResolution(
        path=model_name,
        local_files_only=False,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _candidate_hub_cache_roots(config: AppConfig) -> list[Path]:
    roots: list[Path] = []
    explicit_roots = [
        config.hf_home / "hub",
        _env_path("HF_HUB_CACHE"),
        _env_path("HF_HOME", suffix="hub"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in explicit_roots:
        if root is None:
            continue
        normalized = root.resolve()
        if normalized not in roots:
            roots.append(normalized)
    return roots


def _env_path(name: str, *, suffix: str | None = None) -> Path | None:
    raw_value = os.getenv(name)
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    return path / suffix if suffix else path


def _find_hub_snapshot(
    cache_root: Path,
    model_name: str,
    *,
    purpose: Literal["model", "tokenizer"],
) -> tuple[Path | None, list[str]]:
    repo_cache_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    if not repo_cache_dir.exists():
        return None, []

    refs_dir = repo_cache_dir / "refs"
    snapshots_dir = repo_cache_dir / "snapshots"
    ref_candidates = []
    for ref_name in ("main", "master"):
        ref_path = refs_dir / ref_name
        if ref_path.exists():
            ref_candidates.append(ref_path.read_text(encoding="utf-8").strip())
    for snapshot_dir in sorted(snapshots_dir.iterdir(), reverse=True) if snapshots_dir.exists() else []:
        ref_candidates.append(snapshot_dir.name)

    warnings: list[str] = []
    for snapshot_name in ref_candidates:
        if not snapshot_name:
            continue
        snapshot_path = snapshots_dir / snapshot_name
        if not snapshot_path.exists():
            continue
        missing_items = _missing_local_model_requirements(snapshot_path, purpose=purpose)
        if not missing_items:
            return snapshot_path, warnings
        warnings.append(
            f"Ignored incomplete local {purpose} snapshot `{snapshot_path}`; "
            f"missing {', '.join(missing_items)}."
        )
    return None, warnings


def _is_complete_local_model_dir(path: Path, *, purpose: Literal["model", "tokenizer"]) -> bool:
    return not _missing_local_model_requirements(path, purpose=purpose)


def _missing_local_model_requirements(path: Path, *, purpose: Literal["model", "tokenizer"]) -> list[str]:
    if not path.is_dir():
        return ["directory"]
    if purpose == "model":
        missing_items: list[str] = []
        if not (path / "config.json").exists():
            missing_items.append("config.json")
        if not any(
            (path / filename).exists()
            for filename in (
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            )
        ):
            missing_items.append("model weights")
        return missing_items

    if any(
        (path / filename).exists()
        for filename in (
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
    ):
        return []
    return ["tokenizer files"]


def _format_resolution_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    return " Diagnostics: " + " ".join(dict.fromkeys(warnings))


def _fallback_prompt(messages: list[dict[str, str]]) -> str:
    prompt_lines = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        prompt_lines.append(f"{role}:\n{content}")
    prompt_lines.append("ASSISTANT:\n")
    return "\n\n".join(prompt_lines)


def _truncate_at_stop_strings(text: str, stop_strings: list[str] | None) -> str:
    if not stop_strings:
        return text.strip()
    cut_index = len(text)
    for stop_string in stop_strings:
        if not stop_string:
            continue
        index = text.find(stop_string)
        if index >= 0:
            cut_index = min(cut_index, index)
    return text[:cut_index].strip()
