from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import services.asr_worker as asr_worker


class ASRWorkerTest(unittest.TestCase):
    def test_transcribe_with_transformers_chunks_audio(self) -> None:
        pipeline_calls: list[object] = []

        def fake_pipeline(*args, **kwargs):
            self.assertEqual(kwargs["model"], "demo-model")
            self.assertEqual(kwargs["device"], "cuda")

            def run(chunk, generate_kwargs=None):
                pipeline_calls.append(chunk)
                return {"text": "測試 "}

            return run

        fake_transformers = types.SimpleNamespace(pipeline=fake_pipeline)
        fake_numpy = types.SimpleNamespace(ceil=lambda value: 2)

        with patch.object(asr_worker, "resolve_device", return_value=("cuda", "float16")):
            with patch.dict(
                sys.modules,
                {
                    "transformers": fake_transformers,
                    "numpy": fake_numpy,
                },
            ):
                result = asr_worker.transcribe_with_transformers(
                    "demo-model",
                    [0.0] * 64000,
                    16000,
                    2,
                )

        self.assertEqual(len(pipeline_calls), 2)
        self.assertEqual(result["transcript"], "測試測試")
        self.assertEqual(result["device"], "cuda")

    def test_transcribe_with_faster_whisper_uses_language_from_info(self) -> None:
        created_models: list[tuple[str, str, str]] = []

        class FakeSegment:
            def __init__(self, text: str) -> None:
                self.text = text

        class FakeWhisperModel:
            def __init__(self, model_name: str, device: str, compute_type: str) -> None:
                created_models.append((model_name, device, compute_type))

            def transcribe(self, audio_path: str, language: str, task: str):
                if language != "zh" or task != "transcribe":
                    raise AssertionError("Unexpected transcription options")
                return iter([FakeSegment("哈囉 "), FakeSegment("世界")]), SimpleNamespace(language="zh")

        with patch.object(asr_worker, "resolve_device", return_value=("cuda", "float16")):
            with patch.dict(
                sys.modules,
                {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisperModel)},
            ):
                result = asr_worker.transcribe_with_faster_whisper("fw-model", Path("/tmp/demo.wav"))

        self.assertEqual(created_models, [("fw-model", "cuda", "float16")])
        self.assertEqual(result["transcript"], "哈囉世界")
        self.assertEqual(result["language"], "zh")

    def test_transcribe_with_faster_whisper_raises_on_empty_transcript(self) -> None:
        class FakeWhisperModel:
            def __init__(self, model_name: str, device: str, compute_type: str) -> None:
                self.model_name = model_name
                self.device = device
                self.compute_type = compute_type

            def transcribe(self, audio_path: str, language: str, task: str):
                return iter([SimpleNamespace(text="  ")]), SimpleNamespace(language="zh")

        with patch.object(asr_worker, "resolve_device", return_value=("cpu", "int8")):
            with patch.dict(
                sys.modules,
                {"faster_whisper": types.SimpleNamespace(WhisperModel=FakeWhisperModel)},
            ):
                with self.assertRaisesRegex(RuntimeError, "empty transcript"):
                    asr_worker.transcribe_with_faster_whisper("fw-model", Path("/tmp/demo.wav"))

    def test_main_rejects_unknown_backend(self) -> None:
        with TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "demo.wav"
            audio_path.write_bytes(b"wav")

            with patch(
                "json.load",
                return_value={
                    "audio_path": str(audio_path),
                    "backend": "unknown",
                    "model_name": "demo-model",
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "Unsupported ASR backend"):
                    asr_worker.main()


if __name__ == "__main__":
    unittest.main()
