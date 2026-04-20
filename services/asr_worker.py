from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from typing import Any


def load_audio(audio_path: Path) -> tuple["np.ndarray[Any, Any]", int]:  # type: ignore[name-defined]
    import numpy as np

    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        raw_frames = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError(f"ASR worker expects 16-bit PCM wav audio, got sample width {sample_width}")
    if channels != 1:
        raise RuntimeError(f"ASR worker expects mono wav audio, got {channels} channels")

    waveform = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32) / 32768.0
    return waveform, sample_rate


def resolve_device() -> tuple[str, str]:
    import torch

    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def transcribe_with_transformers(
    model_name: str,
    waveform: "np.ndarray[Any, Any]",  # type: ignore[name-defined]
    sample_rate: int,
    chunk_length: int,
) -> dict[str, str]:
    import numpy as np
    from transformers import pipeline

    device, _ = resolve_device()
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )

    duration_s = len(waveform) / sample_rate if sample_rate else 0
    total_chunks = max(1, int(np.ceil(duration_s / chunk_length)))
    transcript_parts: list[str] = []

    for index in range(total_chunks):
        start = int(index * chunk_length * sample_rate)
        end = min(int((index + 1) * chunk_length * sample_rate), len(waveform))
        chunk = waveform[start:end]
        result = pipe(chunk, generate_kwargs={"language": "zh", "task": "transcribe"})
        transcript_parts.append(str(result.get("text", "")))

    transcript = "".join(part.strip() for part in transcript_parts if part)
    if not transcript:
        raise RuntimeError("ASR returned empty transcript")

    return {
        "transcript": transcript,
        "source": model_name,
        "language": "zh",
        "device": device,
    }


def transcribe_with_faster_whisper(model_name: str, audio_path: Path) -> dict[str, str]:
    from faster_whisper import WhisperModel

    device, compute_type = resolve_device()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(str(audio_path), language="zh", task="transcribe")
    transcript = "".join(segment.text.strip() for segment in segments if segment.text).strip()
    if not transcript:
        raise RuntimeError("ASR returned empty transcript")

    return {
        "transcript": transcript,
        "source": model_name,
        "language": str(getattr(info, "language", "zh") or "zh"),
        "device": device,
    }


def main() -> int:
    payload = json.load(sys.stdin)
    audio_path = Path(str(payload.get("audio_path", "")).strip())
    backend = str(payload.get("backend", "transformers")).strip()
    model_name = str(payload.get("model_name", "")).strip()
    chunk_length = int(payload.get("chunk_length_s", 30))

    if not model_name:
        raise RuntimeError("ASR worker did not receive a model_name")
    if not audio_path.exists():
        raise RuntimeError(f"ASR worker could not find audio file: {audio_path}")

    if backend == "transformers":
        try:
            import numpy as np  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "numpy, torch, and transformers are required in the ASR worker environment"
            ) from exc
        waveform, sample_rate = load_audio(audio_path)
        result = transcribe_with_transformers(model_name, waveform, sample_rate, chunk_length)
    elif backend == "faster_whisper":
        try:
            import faster_whisper  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper and torch are required in the ASR worker environment"
            ) from exc
        result = transcribe_with_faster_whisper(model_name, audio_path)
    else:
        raise RuntimeError(f"Unsupported ASR backend: {backend}")

    json.dump(result, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
