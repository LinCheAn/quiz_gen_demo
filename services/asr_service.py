from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

from utils.config import AppConfig
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
        return self._transcribe_live(video_path, audio_output_dir, progress_callback)

    def _transcribe_live(
        self,
        video_path: str,
        audio_output_dir: str,
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptResult:
        try:
            import librosa
            import numpy as np
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "librosa, numpy, torch, and transformers are required for live ASR mode"
            ) from exc

        audio_dir = Path(audio_output_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{Path(video_path).stem}.wav"

        if progress_callback:
            progress_callback(0.05, "Extracting audio with ffmpeg")

        self._extract_audio(video_path, audio_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if progress_callback:
            progress_callback(0.15, f"Loading ASR model on {device}")

        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.config.asr_model_name,
            device=device,
        )

        if progress_callback:
            progress_callback(0.25, "Loading extracted audio")

        waveform, sample_rate = librosa.load(str(audio_path), sr=16000)
        duration_s = librosa.get_duration(y=waveform, sr=sample_rate)
        chunk_length = self.config.asr_chunk_length_s
        total_chunks = max(1, int(np.ceil(duration_s / chunk_length)))
        transcript_parts: list[str] = []

        for index in range(total_chunks):
            start = int(index * chunk_length * sample_rate)
            end = min(int((index + 1) * chunk_length * sample_rate), len(waveform))
            chunk = waveform[start:end]
            result = pipe(chunk, generate_kwargs={"language": "zh", "task": "transcribe"})
            transcript_parts.append(result["text"])
            if progress_callback:
                progress_callback(
                    0.25 + (0.75 * ((index + 1) / total_chunks)),
                    f"Transcribing chunk {index + 1}/{total_chunks}",
                )

        transcript = "".join(part.strip() for part in transcript_parts if part)
        if not transcript:
            raise RuntimeError("ASR returned empty transcript")

        return TranscriptResult(transcript=transcript, source=self.config.asr_model_name)

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
