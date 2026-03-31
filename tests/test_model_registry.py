from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.config import AppConfig
from utils.model_registry import ModelRegistry, build_runtime_config


class ModelRegistryTest(unittest.TestCase):
    def test_load_registry_and_build_runtime_config(self) -> None:
        registry = ModelRegistry.load(Path(__file__).resolve().parents[1] / "model_info.json")
        selection = registry.resolve_selection()
        config = AppConfig(project_root=Path(tempfile.mkdtemp()))
        config.ensure_directories()

        runtime_config = build_runtime_config(config, selection)

        self.assertEqual(runtime_config.summary_model_name, selection.summary.model_name)
        self.assertEqual(runtime_config.summary_server_model, selection.summary.server_model)
        self.assertEqual(runtime_config.quiz_model_name, selection.quiz.model_name)
        self.assertEqual(runtime_config.quiz_model_path, selection.quiz.lora_path)

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
                        "summary_models": [],
                        "quiz_models": [
                            {
                                "id": "quiz-a",
                                "label": "Quiz A",
                                "model_name": "quiz-a",
                                "base_url": "http://127.0.0.1:8000/v1",
                                "server_conda_env": "vllm",
                                "server_model": "quiz-server",
                                "lora_path": "/tmp/adapter",
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


if __name__ == "__main__":
    unittest.main()
