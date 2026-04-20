from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.config import AppConfig
from utils.model_registry import (
    ModelRegistry,
    build_runtime_config,
    validate_model_selection_assets,
)


class ModelRegistryTest(unittest.TestCase):
    def test_load_registry_and_build_runtime_config(self) -> None:
        registry = ModelRegistry.load(Path(__file__).resolve().parents[1] / "model_info.json")
        selection = registry.resolve_selection()
        config = AppConfig(project_root=Path(tempfile.mkdtemp()))
        config.ensure_directories()

        runtime_config = build_runtime_config(config, selection)

        self.assertEqual(runtime_config.summary_model_name, selection.summary.model_name)
        self.assertEqual(runtime_config.summary_server_model, selection.summary.server_model)
        if selection.summary.lora_path is None:
            self.assertIsNone(runtime_config.summary_model_path)
        else:
            self.assertEqual(
                runtime_config.summary_model_path,
                str((config.project_root / selection.summary.lora_path).resolve()),
            )
        self.assertEqual(runtime_config.quiz_model_name, selection.quiz.model_name)
        if selection.quiz.lora_path is None:
            self.assertIsNone(runtime_config.quiz_model_path)
        else:
            self.assertEqual(
                runtime_config.quiz_model_path,
                str((config.project_root / selection.quiz.lora_path).resolve()),
            )

    def test_registry_allows_model_without_lora_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_info_path = Path(tempdir) / "model_info.json"
            model_info_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "summary_model_id": "summary-a",
                            "quiz_model_id": "quiz-base",
                        },
                        "models": [
                            {
                                "id": "summary-a",
                                "label": "Summary A",
                                "model_name": "summary-a",
                                "base_url": "http://127.0.0.1:8001/v1",
                                "server_conda_env": "",
                                "server_model": "summary-server",
                                "gpu_memory_utilization": 0.8,
                                "max_model_len": 4096,
                                "tensor_parallel_size": 1,
                            },
                            {
                                "id": "quiz-base",
                                "label": "Quiz Base",
                                "model_name": "quiz-base",
                                "base_url": "http://127.0.0.1:8000/v1",
                                "server_conda_env": "",
                                "server_model": "quiz-server",
                                "gpu_memory_utilization": 0.7,
                                "max_model_len": 4096,
                                "tensor_parallel_size": 1,
                                "dtype": "bfloat16",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            registry = ModelRegistry.load(model_info_path)
            selection = registry.resolve_selection()
            runtime_config = build_runtime_config(AppConfig(project_root=Path(tempdir)), selection)

            self.assertIsNone(selection.quiz.lora_path)
            self.assertIsNone(runtime_config.quiz_model_path)

    def test_missing_default_model_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_info_path = Path(tempdir) / "model_info.json"
            model_info_path.write_text(
                json.dumps(
                    {
                        "defaults": {
                            "summary_model_id": "missing-summary",
                            "quiz_model_id": "quiz-a",
                        },
                        "models": [
                            {
                                "id": "quiz-a",
                                "label": "Quiz A",
                                "model_name": "quiz-a",
                                "base_url": "http://127.0.0.1:8000/v1",
                                "server_conda_env": "",
                                "server_model": "quiz-server",
                                "lora_path": "models/adapters/quiz-a",
                                "gpu_memory_utilization": 0.7,
                                "max_model_len": 4096,
                                "tensor_parallel_size": 1,
                                "dtype": "bfloat16"
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Default summary model"):
                ModelRegistry.load(model_info_path)

    def test_summary_and_quiz_choices_share_same_model_pool(self) -> None:
        registry = ModelRegistry.load(Path(__file__).resolve().parents[1] / "model_info.json")

        self.assertEqual(registry.summary_choices(), registry.quiz_choices())
        self.assertEqual(registry.summary_choices(), registry.model_choices())

    def test_validate_model_selection_assets_raises_for_missing_relative_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            project_root = Path(tempdir)
            config = AppConfig(project_root=project_root)
            config.ensure_directories()
            registry = ModelRegistry.load(Path(__file__).resolve().parents[1] / "model_info.json")
            selection = registry.resolve_selection(quiz_model_id="grpo-v4.2")

            with self.assertRaisesRegex(ValueError, "Missing model assets"):
                validate_model_selection_assets(config, selection)

    def test_validate_model_selection_assets_accepts_existing_relative_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            project_root = Path(tempdir)
            adapter_dir = project_root / "models" / "adapters" / "grpo_v4.2"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            summary_adapter_dir = project_root / "models" / "adapters" / "dpo_v9.3_ocw"
            summary_adapter_dir.mkdir(parents=True, exist_ok=True)
            config = AppConfig(project_root=project_root)
            config.ensure_directories()
            registry = ModelRegistry.load(Path(__file__).resolve().parents[1] / "model_info.json")
            selection = registry.resolve_selection(quiz_model_id="grpo-v4.2")

            validate_model_selection_assets(config, selection)


if __name__ == "__main__":
    unittest.main()
