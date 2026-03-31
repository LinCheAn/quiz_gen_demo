from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services.asr_service import ASRService
from utils.config import AppConfig


class ASRServiceTest(unittest.TestCase):
    def test_transcribe_uses_configured_conda_env_worker(self) -> None:
        config = AppConfig()
        config.asr_conda_env = "inference"
        service = ASRService(config)
        worker_output = {
            "transcript": "測試逐字稿",
            "source": config.asr_model_name,
            "language": "zh",
            "device": "cuda",
        }
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(worker_output, ensure_ascii=False),
            stderr="",
        )

        with TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "demo.mp4"
            video_path.write_bytes(b"video")

            with patch.object(service, "_extract_audio") as extract_audio_mock:
                with patch("services.asr_service.subprocess.run", return_value=completed) as run_mock:
                    with patch("builtins.print") as print_mock:
                        result = service.transcribe(str(video_path), str(Path(tmpdir) / "audio"))

        extract_audio_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:5], ["conda", "run", "--no-capture-output", "-n", "inference"])
        self.assertEqual(command[5], "python")
        self.assertEqual(command[6], str(Path("services/asr_worker.py").resolve()))
        self.assertEqual(result.transcript, "測試逐字稿")
        self.assertEqual(result.language, "zh")
        self.assertEqual(result.device, "cuda")
        print_mock.assert_called_once()

    def test_transcribe_uses_current_python_when_asr_conda_env_empty(self) -> None:
        config = AppConfig()
        config.asr_conda_env = ""
        service = ASRService(config)
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"transcript": "ok", "source": "demo", "language": "zh"}),
            stderr="",
        )

        with TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "demo.mp4"
            video_path.write_bytes(b"video")

            with patch.object(service, "_extract_audio"):
                with patch("services.asr_service.subprocess.run", return_value=completed) as run_mock:
                    service.transcribe(str(video_path), str(Path(tmpdir) / "audio"))

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(Path("services/asr_worker.py").resolve()))


if __name__ == "__main__":
    unittest.main()
