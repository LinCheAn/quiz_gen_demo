from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

from services.summary_service import extract_json_fragment
from utils.config import AppConfig, describe_runtime_target
from utils.schemas import TranscriptResult


ProgressCallback = Callable[[float, str], None]


class ASRService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def transcribe(
        self,
        video_path: str,
        audio_output_dir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptResult:
        audio_dir = Path(audio_output_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{Path(video_path).stem}.wav"

        if progress_callback:
            progress_callback(0.05, "Extracting audio with ffmpeg")

        self._extract_audio(video_path, audio_path)

        if progress_callback:
            progress_callback(
                0.15,
                f"Running ASR in {describe_runtime_target(self.config.asr_conda_env)}",
            )

        payload = {
            "audio_path": str(audio_path),
            "model_name": self.config.asr_model_name,
            "chunk_length_s": self.config.asr_chunk_length_s,
        }
        completed = self._run_worker(payload)

        if progress_callback:
            progress_callback(0.9, "Parsing ASR output")

        parsed = extract_json_fragment(completed.stdout.strip())
        if parsed is None or "transcript" not in parsed:
            raise RuntimeError(
                "ASR worker did not return parsable JSON results. "
                f"Stdout: {completed.stdout.strip() or '<empty>'}"
            )

        transcript = str(parsed["transcript"]).strip()
        if not transcript:
            raise RuntimeError("ASR returned empty transcript")

        device = str(parsed.get("device", "")).strip() or None
        print(
            f"[asr] Completed with model {self.config.asr_model_name} on "
            f"{device or 'unknown'} via {describe_runtime_target(self.config.asr_conda_env)}"
        )

        if progress_callback and device:
            progress_callback(0.95, f"ASR completed on {device}")

        return TranscriptResult(
            transcript=transcript,
            source=str(parsed.get("source", self.config.asr_model_name)),
            language=str(parsed.get("language", "zh")),
            device=device,
        )

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
            if self.config.asr_conda_env.strip():
                raise RuntimeError("conda is required to launch the ASR worker environment") from exc
            raise RuntimeError("Python is required to launch the ASR worker") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip() or "<empty>"
            stdout = completed.stdout.strip() or "<empty>"
            raise RuntimeError(
                "ASR failed while running "
                f"`{' '.join(command)}`. "
                f"Configured runtime target: {describe_runtime_target(self.config.asr_conda_env)}. "
                f"STDERR: {stderr}\nSTDOUT: {stdout}"
            )
        return completed

    def _build_worker_command(self) -> list[str]:
        worker_path = Path(__file__).with_name("asr_worker.py")
        if self.config.asr_conda_env.strip():
            return [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                self.config.asr_conda_env,
                "python",
                str(worker_path),
            ]
        return [sys.executable, str(worker_path)]

    @staticmethod
    def _extract_audio(video_path: str, audio_path: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required but was not found in PATH") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "ffmpeg command failed"
            raise RuntimeError(stderr) from exc

        if completed.returncode != 0:
            raise RuntimeError("ffmpeg failed to extract audio")

        if not audio_path.exists() or os.path.getsize(audio_path) == 0:
            raise RuntimeError("Extracted audio file is missing or empty")
