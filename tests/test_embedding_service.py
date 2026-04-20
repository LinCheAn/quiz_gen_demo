from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from services.embedding_service import EmbeddingService
from utils.config import AppConfig
from utils.schemas import TextChunk


class EmbeddingServiceTest(unittest.TestCase):
    def test_retrieval_uses_configured_conda_env_worker(self) -> None:
        config = AppConfig()
        config.embedding_conda_env = "inference"
        service = EmbeddingService(config)
        chunks = [
            TextChunk(chunk_id="chunk_001", text="graph algorithm review", start_char=0, end_char=22),
            TextChunk(chunk_id="chunk_002", text="binary tree search tree balancing", start_char=23, end_char=57),
            TextChunk(chunk_id="chunk_003", text="queue stack heap", start_char=58, end_char=74),
        ]
        worker_output = {
            "results": [
                {
                    "rank": 1,
                    "chunk_id": "chunk_002",
                    "text": "binary tree search tree balancing",
                    "score": 0.9321,
                    "matched_keywords": ["tree", "search"],
                },
                {
                    "rank": 2,
                    "chunk_id": "chunk_001",
                    "text": "graph algorithm review",
                    "score": 0.1023,
                    "matched_keywords": [],
                },
            ]
        }
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(worker_output),
            stderr="",
        )

        with patch("services.embedding_service.subprocess.run", return_value=completed) as run_mock:
            result = service.retrieve(chunks=chunks, keywords=["tree", "search"], top_k=2)

        command = run_mock.call_args.args[0]
        self.assertEqual(command[:5], ["conda", "run", "--no-capture-output", "-n", "inference"])
        self.assertEqual(command[5], "python")
        self.assertEqual(command[6], str(Path("services/embedding_worker.py").resolve()))
        self.assertEqual(result.results[0].chunk_id, "chunk_002")
        self.assertEqual(len(result.results), 2)

    def test_retrieval_uses_current_python_when_embedding_conda_env_empty(self) -> None:
        config = AppConfig()
        config.embedding_conda_env = ""
        service = EmbeddingService(config)
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"results": []}),
            stderr="",
        )

        with patch("services.embedding_service.subprocess.run", return_value=completed) as run_mock:
            service.retrieve(chunks=[], keywords=["tree"], top_k=2)

        run_mock.assert_not_called()

        chunks = [TextChunk(chunk_id="chunk_001", text="tree", start_char=0, end_char=4)]
        with patch("services.embedding_service.subprocess.run", return_value=completed) as run_mock:
            service.retrieve(chunks=chunks, keywords=["tree"], top_k=1)

        command = run_mock.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], str(Path("services/embedding_worker.py").resolve()))


if __name__ == "__main__":
    unittest.main()
