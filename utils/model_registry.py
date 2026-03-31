from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from utils.config import AppConfig
from utils.schemas import ModelSelectionSnapshot, QuizModelPreset, SummaryModelPreset


class ModelRegistryDefaults(BaseModel):
    summary_model_id: str
    quiz_model_id: str


class ModelRegistryFile(BaseModel):
    defaults: ModelRegistryDefaults
    summary_models: list[SummaryModelPreset]
    quiz_models: list[QuizModelPreset]


class ModelRegistry:
    def __init__(self, path: Path, payload: ModelRegistryFile) -> None:
        self.path = path
        self.defaults = payload.defaults
        self.summary_models = payload.summary_models
        self.quiz_models = payload.quiz_models
        self._summary_by_id = {item.id: item for item in self.summary_models}
        self._quiz_by_id = {item.id: item for item in self.quiz_models}
        self._validate()

    @classmethod
    def load(cls, path: str | Path) -> "ModelRegistry":
        registry_path = Path(path)
        payload = ModelRegistryFile.model_validate(
            json.loads(registry_path.read_text(encoding="utf-8"))
        )
        return cls(registry_path, payload)

    def _validate(self) -> None:
        if len(self._summary_by_id) != len(self.summary_models):
            raise ValueError(f"Duplicate summary model ids found in {self.path}")
        if len(self._quiz_by_id) != len(self.quiz_models):
            raise ValueError(f"Duplicate quiz model ids found in {self.path}")
        if self.defaults.summary_model_id not in self._summary_by_id:
            raise ValueError(
                f"Default summary model `{self.defaults.summary_model_id}` not found in {self.path}"
            )
        if self.defaults.quiz_model_id not in self._quiz_by_id:
            raise ValueError(
                f"Default quiz model `{self.defaults.quiz_model_id}` not found in {self.path}"
            )

    def summary_choices(self) -> list[tuple[str, str]]:
        return [(item.label, item.id) for item in self.summary_models]

    def quiz_choices(self) -> list[tuple[str, str]]:
        return [(item.label, item.id) for item in self.quiz_models]

    def resolve_selection(
        self,
        summary_model_id: str | None = None,
        quiz_model_id: str | None = None,
    ) -> ModelSelectionSnapshot:
        resolved_summary_id = summary_model_id or self.defaults.summary_model_id
        resolved_quiz_id = quiz_model_id or self.defaults.quiz_model_id
        try:
            summary = self._summary_by_id[resolved_summary_id]
        except KeyError as exc:
            raise ValueError(f"Unknown summary model id: {resolved_summary_id}") from exc
        try:
            quiz = self._quiz_by_id[resolved_quiz_id]
        except KeyError as exc:
            raise ValueError(f"Unknown quiz model id: {resolved_quiz_id}") from exc
        return ModelSelectionSnapshot(summary=summary, quiz=quiz)


def build_runtime_config(base_config: AppConfig, selection: ModelSelectionSnapshot) -> AppConfig:
    return base_config.copy_with_overrides(
        summary_model_name=selection.summary.model_name,
        summary_base_url=selection.summary.base_url,
        summary_server_conda_env=selection.summary.server_conda_env,
        summary_server_model=selection.summary.server_model,
        summary_server_gpu_memory_utilization=selection.summary.gpu_memory_utilization,
        summary_server_max_model_len=selection.summary.max_model_len,
        summary_server_tensor_parallel_size=selection.summary.tensor_parallel_size,
        summary_server_dtype=selection.summary.dtype or base_config.summary_server_dtype,
        summary_server_quantization=selection.summary.quantization,
        quiz_model_name=selection.quiz.model_name,
        quiz_base_url=selection.quiz.base_url,
        quiz_model_path=selection.quiz.lora_path,
        quiz_base_model_name=selection.quiz.server_model,
        quiz_server_conda_env=selection.quiz.server_conda_env,
        quiz_server_model=selection.quiz.server_model,
        quiz_server_gpu_memory_utilization=selection.quiz.gpu_memory_utilization,
        quiz_server_max_model_len=selection.quiz.max_model_len,
        quiz_server_tensor_parallel_size=selection.quiz.tensor_parallel_size,
        quiz_server_dtype=selection.quiz.dtype or base_config.quiz_server_dtype,
    )
