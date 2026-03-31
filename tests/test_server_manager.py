from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.config import AppConfig
from utils.server_manager import ModelServerManager


class ServerManagerTest(unittest.TestCase):
    def test_build_specs_use_expected_ports_and_models(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            manager = ModelServerManager(config)

            summary = manager._build_summary_spec()
            quiz = manager._build_quiz_spec()

            self.assertEqual(summary.base_url, "http://127.0.0.1:8001/v1")
            self.assertEqual(quiz.base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(config.model_server_start_strategy, "sequential")
            self.assertIn(config.summary_model_name, summary.command)
            quiz_command = " ".join(quiz.command)
            self.assertIn(config.quiz_model_name, quiz_command)
            self.assertIn(config.quiz_model_path, quiz_command)


if __name__ == "__main__":
    unittest.main()
