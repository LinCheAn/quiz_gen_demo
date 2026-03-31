from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.config import AppConfig
from utils.model_registry import ModelRegistry, build_runtime_config
from utils.server_manager import ModelServerManager


class ServerManagerTest(unittest.TestCase):
    def test_build_specs_use_selected_model_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            registry = ModelRegistry.load(Path(__file__).resolve().parents[1] / "model_info.json")
            selection = registry.resolve_selection()
            runtime_config = build_runtime_config(config, selection)
            manager = ModelServerManager(runtime_config)

            summary = manager._build_summary_spec()
            quiz = manager._build_quiz_spec()

            self.assertEqual(summary.base_url, "http://127.0.0.1:8001/v1")
            self.assertEqual(quiz.base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(runtime_config.model_server_start_strategy, "sequential")
            self.assertIn(selection.summary.model_name, summary.command)
            quiz_command = " ".join(quiz.command)
            self.assertIn(selection.quiz.model_name, quiz_command)
            self.assertIn(selection.quiz.lora_path, quiz_command)


if __name__ == "__main__":
    unittest.main()
