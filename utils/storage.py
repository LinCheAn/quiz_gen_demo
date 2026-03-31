from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from uuid import uuid4

from utils.config import AppConfig
from utils.schemas import PipelineRunState


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.strip())
    return cleaned or "input"


def make_run_id() -> str:
    return uuid4().hex[:12]


class RunArtifactManager:
    def __init__(self, config: AppConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.run_dir = self.config.runs_dir / run_id
        self.inputs_dir = self.run_dir / "inputs"
        self.audio_dir = self.run_dir / "audio"
        self.outputs_dir = self.run_dir / "outputs"
        self.logs_dir = self.run_dir / "logs"
        self.ensure_directories()

    def ensure_directories(self) -> None:
        for path in (
            self.run_dir,
            self.inputs_dir,
            self.audio_dir,
            self.outputs_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def copy_input_file(self, source_path: str | Path, target_name: str | None = None) -> str:
        source = Path(source_path)
        filename = sanitize_filename(target_name or source.name)
        target = self.inputs_dir / filename
        shutil.copy2(source, target)
        return str(target)

    def save_text(self, relative_path: str, content: str) -> str:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def save_json(self, relative_path: str, data: object) -> str:
        path = self.run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return str(path)

    def load_text(self, absolute_or_relative_path: str | Path) -> str:
        path = Path(absolute_or_relative_path)
        if not path.is_absolute():
            path = self.run_dir / path
        return path.read_text(encoding="utf-8")

    def save_state(self, state: PipelineRunState) -> str:
        return self.save_json("outputs/state.json", state.model_dump(mode="json"))
