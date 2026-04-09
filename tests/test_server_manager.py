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

            self.assertNotEqual(summary.key, quiz.key)
            self.assertEqual(summary.base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(quiz.base_url, "http://127.0.0.1:8000/v1")
            self.assertEqual(runtime_config.model_server_start_strategy, "sequential")
            self.assertIn(selection.summary.model_name, summary.command)
            quiz_command = " ".join(quiz.command)
            self.assertIn(selection.quiz.model_name, quiz_command)
            self.assertIn(selection.quiz.lora_path, quiz_command)
            self.assertIn("--enable-lora", quiz.command)

    def test_same_model_reuses_single_process_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(
                project_root=Path(tempdir),
                summary_model_name="llama-3.1-8b-instruct",
                summary_base_url="http://127.0.0.1:8000/v1",
                summary_server_conda_env="vllm",
                summary_server_model="unsloth/Meta-Llama-3.1-8B-Instruct",
                summary_model_path=None,
                summary_server_gpu_memory_utilization=0.8,
                summary_server_max_model_len=8192,
                summary_server_tensor_parallel_size=1,
                summary_server_dtype="bfloat16",
                summary_server_quantization=None,
                quiz_model_name="llama-3.1-8b-instruct",
                quiz_base_url="http://127.0.0.1:8000/v1",
                quiz_server_conda_env="vllm",
                quiz_server_model="unsloth/Meta-Llama-3.1-8B-Instruct",
                quiz_model_path=None,
                quiz_server_gpu_memory_utilization=0.8,
                quiz_server_max_model_len=8192,
                quiz_server_tensor_parallel_size=1,
                quiz_server_dtype="bfloat16",
                quiz_server_quantization=None,
            )
            config.ensure_directories()
            manager = ModelServerManager(config)

            summary = manager._build_summary_spec()
            quiz = manager._build_quiz_spec()

            self.assertEqual(summary.key, quiz.key)
            self.assertEqual(summary.roles, frozenset({"summary", "quiz"}))
            self.assertIn("--served-model-name", summary.command)
            self.assertNotIn("--enable-lora", summary.command)
            self.assertEqual(manager._releasable_process_keys_for_role("summary"), set())
            self.assertEqual(manager._releasable_process_keys_for_role("quiz"), {summary.key})

    def test_summary_base_and_quiz_lora_share_single_process(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(
                project_root=Path(tempdir),
                summary_model_name="llama-3.1-8b-instruct",
                summary_base_url="http://127.0.0.1:8000/v1",
                summary_server_conda_env="vllm",
                summary_server_model="unsloth/Meta-Llama-3.1-8B-Instruct",
                summary_model_path=None,
                summary_server_gpu_memory_utilization=0.8,
                summary_server_max_model_len=8192,
                summary_server_tensor_parallel_size=1,
                summary_server_dtype="bfloat16",
                summary_server_quantization=None,
                quiz_model_name="grpo_v4.2",
                quiz_base_url="http://127.0.0.1:8000/v1",
                quiz_server_conda_env="vllm",
                quiz_server_model="unsloth/Meta-Llama-3.1-8B-Instruct",
                quiz_model_path="/tmp/grpo",
                quiz_server_gpu_memory_utilization=0.8,
                quiz_server_max_model_len=8192,
                quiz_server_tensor_parallel_size=1,
                quiz_server_dtype="bfloat16",
                quiz_server_quantization=None,
            )
            config.ensure_directories()
            manager = ModelServerManager(config)

            summary = manager._build_summary_spec()
            quiz = manager._build_quiz_spec()

            self.assertEqual(summary.key, quiz.key)
            self.assertEqual(summary.expected_models, ("grpo_v4.2", "llama-3.1-8b-instruct"))
            self.assertIn("--enable-lora", summary.command)
            self.assertIn("--served-model-name", summary.command)
            command_text = " ".join(summary.command)
            self.assertIn("grpo_v4.2=/tmp/grpo", command_text)
            self.assertIn("llama-3.1-8b-instruct", command_text)

    def test_build_quiz_spec_without_lora_uses_base_model_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(
                project_root=Path(tempdir),
                quiz_model_name="llama-3.1-8b-instruct",
                quiz_server_model="unsloth/Meta-Llama-3.1-8B-Instruct",
                quiz_model_path=None,
            )
            config.ensure_directories()
            manager = ModelServerManager(config)

            quiz = manager._build_quiz_spec()

            self.assertEqual(quiz.expected_models, ("llama-3.1-8b-instruct",))
            self.assertNotIn("--enable-lora", quiz.command)
            self.assertIn("--served-model-name", quiz.command)
            self.assertIn("llama-3.1-8b-instruct", quiz.command)


if __name__ == "__main__":
    unittest.main()
