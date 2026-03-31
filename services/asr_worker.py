from __future__ import annotations

import json
import sys
import wave
from pathlib import Path


def main() -> int:
    try:
        import numpy as np
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "numpy, torch, and transformers are required in the ASR worker environment"
        ) from exc

    payload = json.load(sys.stdin)
    audio_path = Path(str(payload.get("audio_path", "")).strip())
    model_name = str(payload.get("model_name", "")).strip()
    chunk_length = int(payload.get("chunk_length_s", 30))

    if not model_name:
        raise RuntimeError("ASR worker did not receive a model_name")
    if not audio_path.exists():
        raise RuntimeError(f"ASR worker could not find audio file: {audio_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )

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

    json.dump(
        {
            "transcript": transcript,
            "source": model_name,
            "language": "zh",
            "device": device,
        },
        sys.stdout,
        ensure_ascii=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
