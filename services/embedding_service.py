from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

from services.summary_service import extract_json_fragment
from utils.config import AppConfig
from utils.schemas import RetrievalResult, RetrievedChunk, TextChunk


ProgressCallback = Callable[[float, str], None]


class EmbeddingService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def retrieve(
        self,
        chunks: list[TextChunk],
        keywords: list[str],
        top_k: int,
        progress_callback: ProgressCallback | None = None,
    ) -> RetrievalResult:
        results = self._retrieve_live(chunks, keywords, top_k, progress_callback)
        return RetrievalResult(query=", ".join(keywords), top_k=top_k, results=results)

    def _retrieve_live(
        self,
        chunks: list[TextChunk],
        keywords: list[str],
        top_k: int,
        progress_callback: ProgressCallback | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        if progress_callback:
            progress_callback(
                0.1,
                f"Running embedding retrieval in conda env `{self.config.embedding_conda_env or 'current'}`",
            )

        payload = {
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            "keywords": keywords,
            "top_k": top_k,
            "model_name": self.config.embedding_model_name,
            "use_fp16": self.config.embedding_use_fp16,
        }
        completed = self._run_worker(payload)

        if progress_callback:
            progress_callback(0.85, "Parsing embedding retrieval output")

        parsed = extract_json_fragment(completed.stdout.strip())
        if parsed is None or "results" not in parsed or not isinstance(parsed["results"], list):
            raise RuntimeError(
                "Embedding worker did not return parsable JSON results. "
                f"Stdout: {completed.stdout.strip() or '<empty>'}"
            )
        return [RetrievedChunk.model_validate(item) for item in parsed["results"]]

    def _run_worker(self, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
        command = self._build_worker_command()
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                check=False,
                cwd=self.config.project_root,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("conda is required to launch the embedding worker environment") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or "<empty>"
            stdout = completed.stdout.strip() or "<empty>"
            raise RuntimeError(
                "Embedding retrieval failed while running "
                f"`{' '.join(command)}`. "
                f"Configured env: {self.config.embedding_conda_env or 'current'}. "
                f"STDERR: {stderr}\nSTDOUT: {stdout}"
            )
        return completed

    def _build_worker_command(self) -> list[str]:
        worker_path = Path(__file__).with_name("embedding_worker.py")
        if self.config.embedding_conda_env.strip():
            return [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.config.embedding_conda_env,
                "python",
                str(worker_path),
            ]
        return [sys.executable, str(worker_path)]
