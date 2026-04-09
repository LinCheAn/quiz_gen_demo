from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


StepState = Literal["pending", "running", "completed", "failed", "skipped"]
ServiceMode = Literal["live"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StepStatus(BaseModel):
    key: str
    label: str
    status: StepState = "pending"
    message: str = ""
    artifact_path: str | None = None
    error: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class PipelineParameters(BaseModel):
    n_keywords: int = 5
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64
    summary_model_id: str | None = None
    quiz_model_id: str | None = None
    quiz_question_count: int = 3
    quiz_variant_count: int = 1


class ModelPreset(BaseModel):
    id: str
    label: str
    model_name: str
    base_url: str
    server_conda_env: str
    server_model: str
    lora_path: str | None = None
    gpu_memory_utilization: float
    max_model_len: int
    tensor_parallel_size: int
    dtype: str | None = None
    quantization: str | None = None


class ModelSelectionSnapshot(BaseModel):
    summary: ModelPreset
    quiz: ModelPreset


SummaryModelPreset = ModelPreset
QuizModelPreset = ModelPreset


class TranscriptResult(BaseModel):
    transcript: str
    source: str
    language: str = "zh"
    device: str | None = None
    artifact_path: str | None = None


class KeywordResult(BaseModel):
    keywords: list[str]
    model: str
    warning: str | None = None
    artifact_path: str | None = None


class TextChunk(BaseModel):
    chunk_id: str
    text: str
    start_char: int
    end_char: int


class ChunkResult(BaseModel):
    chunks: list[TextChunk]
    strategy: str
    chunk_size: int
    overlap: int
    artifact_path: str | None = None


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: str
    text: str
    score: float
    matched_keywords: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    results: list[RetrievedChunk]
    artifact_path: str | None = None


class QuizQuestion(BaseModel):
    question: str
    options: dict[str, str]
    answer: str
    explanation: str | None = None


class QuizResult(BaseModel):
    questions: list[QuizQuestion]
    model: str
    generation_mode: Literal["full", "options_only"]
    raw_response: Any | None = None
    artifact_path: str | None = None


class PipelineRunState(BaseModel):
    run_id: str
    mode: ServiceMode
    overview: str
    input_source: str = ""
    input_filename: str | None = None
    parameters: PipelineParameters
    selected_models: ModelSelectionSnapshot | None = None
    steps: dict[str, StepStatus]
    transcript: str = ""
    summary_warning: str | None = None
    keywords: list[str] = Field(default_factory=list)
    chunks: list[TextChunk] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    quiz_result: QuizResult | None = None
    quiz_results: list[QuizResult] = Field(default_factory=list)
    quiz_generation_count: int = 0
    errors: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
