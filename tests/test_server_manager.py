from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_conflicting_managed_endpoint_is_reloaded(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            manager = ModelServerManager(config)
            spec = manager._build_summary_spec()

            class _FakeProcess:
                pid = 43210
                returncode = None

                @staticmethod
                def poll():
                    return None

                @staticmethod
                def wait(timeout=None):
                    return 0

                @staticmethod
                def terminate():
                    return None

                @staticmethod
                def kill():
                    return None

            with patch.object(
                manager,
                "_probe_endpoint",
                side_effect=[
                    (False, {"reachable": True, "models": ["old-model"]}),
                    (False, {"reachable": False, "models": []}),
                    (True, {"reachable": True, "models": list(spec.expected_models)}),
                ],
            ), patch.object(
                manager,
                "_try_release_managed_conflicting_server",
                return_value=True,
            ) as release_conflict, patch.object(
                manager,
                "_assert_cuda_available",
            ), patch(
                "utils.server_manager.subprocess.Popen",
                return_value=_FakeProcess(),
            ), patch(
                "utils.server_manager.time.sleep",
            ):
                manager._ensure_single_server_ready(spec)

            release_conflict.assert_called_once_with(spec)
            self.assertIn(spec.key, manager._started_processes)
            manager.stop_started_servers()

    def test_unmanaged_conflicting_endpoint_still_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            manager = ModelServerManager(config)
            spec = manager._build_summary_spec()

            with patch.object(
                manager,
                "_probe_endpoint",
                return_value=(False, {"reachable": True, "models": ["old-model"]}),
            ), patch.object(
                manager,
                "_try_release_managed_conflicting_server",
                return_value=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not expose the expected models"):
                    manager._ensure_single_server_ready(spec)

    def test_release_process_removes_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            manager = ModelServerManager(config)
            spec = manager._build_summary_spec()
            metadata_path = manager._metadata_path_for_process_key(spec.key)

            class _FinishedProcess:
                pid = 9876

                @staticmethod
                def poll():
                    return 0

            manager._started_processes[spec.key] = _FinishedProcess()
            manager._write_managed_process_metadata(spec, _FinishedProcess.pid)

            self.assertTrue(metadata_path.exists())

            manager._release_process(spec.key)

            self.assertFalse(metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
