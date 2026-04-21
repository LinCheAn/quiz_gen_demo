from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.transformers_backend import (
    LoadedTransformersModel,
    ModelSourceResolution,
    TransformersModelCache,
    TransformersModelSpec,
    resolve_model_source_path,
)
from utils.config import AppConfig


class TransformersBackendCacheTest(unittest.TestCase):
    def test_cache_reuses_same_model_spec(self) -> None:
        cache = TransformersModelCache()
        config = AppConfig()
        spec = TransformersModelSpec(
            model_name="alias",
            base_model_name="base-model",
            adapter_path=None,
            dtype="bfloat16",
            quantization="bitsandbytes-8bit",
        )
        load_calls: list[TransformersModelSpec] = []

        def fake_load(target_spec: TransformersModelSpec, target_config: AppConfig) -> LoadedTransformersModel:
            del target_config
            load_calls.append(target_spec)
            return LoadedTransformersModel(
                spec=target_spec,
                tokenizer=object(),
                model=object(),
            )

        cache._load_locked = fake_load  # type: ignore[method-assign]

        first = cache.load(spec, config)
        second = cache.load(spec, config)

        self.assertIs(first, second)
        self.assertEqual(load_calls, [spec])

    def test_cache_reloads_when_model_spec_changes(self) -> None:
        cache = TransformersModelCache()
        config = AppConfig()
        spec_a = TransformersModelSpec(
            model_name="alias-a",
            base_model_name="base-model",
            adapter_path=None,
            dtype="bfloat16",
            quantization="bitsandbytes-8bit",
        )
        spec_b = TransformersModelSpec(
            model_name="alias-b",
            base_model_name="base-model",
            adapter_path="/tmp/adapter",
            dtype="bfloat16",
            quantization="bitsandbytes-8bit",
        )
        load_calls: list[TransformersModelSpec] = []
        unload_calls: list[str] = []

        def fake_load(target_spec: TransformersModelSpec, target_config: AppConfig) -> LoadedTransformersModel:
            del target_config
            load_calls.append(target_spec)
            return LoadedTransformersModel(
                spec=target_spec,
                tokenizer=object(),
                model=object(),
            )

        def fake_unload() -> None:
            unload_calls.append("unloaded")
            cache._active = None

        cache._load_locked = fake_load  # type: ignore[method-assign]
        cache._unload_locked = fake_unload  # type: ignore[method-assign]

        cache.load(spec_a, config)
        cache.load(spec_b, config)

        self.assertEqual(load_calls, [spec_a, spec_b])
        self.assertEqual(unload_calls, ["unloaded", "unloaded"])

    def test_resolve_model_source_path_prefers_existing_snapshot_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            project_root = Path(tempdir)
            config = AppConfig(project_root=project_root)
            config.ensure_directories()
            snapshot_dir = (
                config.hf_home
                / "hub"
                / "models--unsloth--Meta-Llama-3.1-8B-Instruct"
                / "snapshots"
                / "abc123"
            )
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")
            (snapshot_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            (snapshot_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            refs_dir = snapshot_dir.parents[1] / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123", encoding="utf-8")

            with patch(
                "services.transformers_backend._candidate_hub_cache_roots",
                return_value=[config.hf_home / "hub"],
            ):
                resolution = resolve_model_source_path(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    config,
                    purpose="model",
                )

            self.assertEqual(resolution.path, str(snapshot_dir))
            self.assertTrue(resolution.local_files_only)
            self.assertEqual(resolution.warnings, ())

    def test_resolve_model_source_path_falls_back_when_snapshot_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            snapshot_dir = (
                config.hf_home
                / "hub"
                / "models--unsloth--Meta-Llama-3.1-8B-Instruct"
                / "snapshots"
                / "abc123"
            )
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (snapshot_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            (snapshot_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            refs_dir = snapshot_dir.parents[1] / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            (refs_dir / "main").write_text("abc123", encoding="utf-8")

            with patch(
                "services.transformers_backend._candidate_hub_cache_roots",
                return_value=[config.hf_home / "hub"],
            ):
                resolution = resolve_model_source_path(
                    "unsloth/Meta-Llama-3.1-8B-Instruct",
                    config,
                    purpose="model",
                )

            self.assertEqual(resolution.path, "unsloth/Meta-Llama-3.1-8B-Instruct")
            self.assertFalse(resolution.local_files_only)
            self.assertTrue(resolution.warnings)
            self.assertIn("Ignored incomplete local model snapshot", resolution.warnings[0])

    def test_resolve_model_source_path_errors_for_incomplete_explicit_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            model_dir = Path(tempdir) / "local-model"
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()

            resolution = resolve_model_source_path(str(model_dir), config, purpose="model")

            self.assertEqual(resolution.path, str(model_dir))
            self.assertTrue(resolution.local_files_only)
            self.assertIn("is incomplete", str(resolution.explicit_local_path_error))
            self.assertIn("config.json", str(resolution.explicit_local_path_error))

    def test_load_uses_resolution_warnings_in_final_error(self) -> None:
        cache = TransformersModelCache()
        config = AppConfig(project_root=Path(tempfile.mkdtemp()))
        config.ensure_directories()
        spec = TransformersModelSpec(
            model_name="alias",
            base_model_name="base-model",
            adapter_path=None,
            dtype="bfloat16",
            quantization=None,
        )

        with patch(
            "services.transformers_backend.resolve_model_source_path",
            side_effect=[
                ModelSourceResolution(
                    path="base-model",
                    local_files_only=False,
                    warnings=("Ignored incomplete local model snapshot `/tmp/bad`; missing config.json.",),
                ),
                ModelSourceResolution(
                    path="base-model",
                    local_files_only=False,
                    warnings=("Ignored incomplete local tokenizer snapshot `/tmp/bad`; missing tokenizer files.",),
                ),
            ],
        ), patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("download failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Diagnostics:"):
                cache._load_locked(spec, config)


if __name__ == "__main__":
    unittest.main()
