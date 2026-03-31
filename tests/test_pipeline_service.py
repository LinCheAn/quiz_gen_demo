from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.pipeline_service import PipelineService
from utils.config import AppConfig
from utils.errors import ModelResponseFormatError
from utils.schemas import (
    ChunkResult,
    KeywordResult,
    PipelineParameters,
    QuizQuestion,
    QuizResult,
    RetrievalResult,
    RetrievedChunk,
    TextChunk,
)


class FakeServerManager:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def prepare_for_target(self, name: str) -> None:
        self.events.append(("prepare", name))

    def release_server(self, name: str) -> None:
        self.events.append(("release", name))

    def stop_started_servers(self) -> None:
        self.events.append(("cleanup", "all"))


class FakeSummaryService:
    def extract_keywords(self, text: str, n_keywords: int, progress_callback=None) -> KeywordResult:
        return KeywordResult(keywords=["tree", "search"], model="fake-summary")


class FakeChunkService:
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> ChunkResult:
        return ChunkResult(
            chunks=[TextChunk(chunk_id="chunk_001", text="tree search concept", start_char=0, end_char=18)],
            strategy="fixed_window_char",
            chunk_size=chunk_size,
            overlap=overlap,
        )


class FakeEmbeddingService:
    def retrieve(self, chunks, keywords, top_k: int, progress_callback=None) -> RetrievalResult:
        return RetrievalResult(
            query=", ".join(keywords),
            top_k=top_k,
            results=[
                RetrievedChunk(
                    rank=1,
                    chunk_id="chunk_001",
                    text="tree search concept",
                    score=0.9,
                    matched_keywords=["tree", "search"],
                )
            ],
        )


class FakeQuizService:
    def generate_quiz(self, context_chunks, variant: int = 0, progress_callback=None) -> QuizResult:
        return QuizResult(
            questions=[
                QuizQuestion(
                    question="What is a tree search concept?",
                    options={"A": "A", "B": "B", "C": "C", "D": "D"},
                    answer="A",
                    explanation="Because.",
                )
            ],
            model="fake-quiz",
            generation_mode="full",
        )

    def regenerate_options_only(self, existing_questions, context_chunks, variant: int = 1, progress_callback=None) -> QuizResult:
        return self.generate_quiz(context_chunks, variant=variant, progress_callback=progress_callback)

    def regenerate_full(self, context_chunks, variant: int = 1, progress_callback=None) -> QuizResult:
        return self.generate_quiz(context_chunks, variant=variant, progress_callback=progress_callback)


class SequentialPipelineService(PipelineService):
    def _build_services(self, mode: str):
        return {
            "asr": None,
            "summary": FakeSummaryService(),
            "chunk": FakeChunkService(),
            "embedding": FakeEmbeddingService(),
            "quiz": FakeQuizService(),
        }


class FailingSummaryService:
    def extract_keywords(self, text: str, n_keywords: int, progress_callback=None) -> KeywordResult:
        raise ModelResponseFormatError(
            step="summary",
            message="Summary model did not return parsable keywords JSON",
            raw_response='{"oops": "not keywords"}',
            model_name="fake-summary",
        )


class FailingSummaryPipelineService(PipelineService):
    def _build_services(self, mode: str):
        return {
            "asr": None,
            "summary": FailingSummaryService(),
            "chunk": FakeChunkService(),
            "embedding": FakeEmbeddingService(),
            "quiz": FakeQuizService(),
        }


class PipelineServiceTest(unittest.TestCase):
    def test_live_pipeline_and_options_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = SequentialPipelineService(config)
            parameters = PipelineParameters(n_keywords=4, top_k=3, chunk_size=80, chunk_overlap=10)

            states = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text=(
                        "Binary search trees support efficient search and insertion. "
                        "Balancing affects time complexity. Traversal order changes the output sequence."
                    ),
                    subtitle_path=None,
                )
            )
            final_state = states[-1]

            self.assertEqual(final_state.steps["input"].status, "completed")
            self.assertEqual(final_state.steps["asr"].status, "skipped")
            self.assertEqual(final_state.steps["quiz"].status, "completed")
            self.assertTrue(final_state.quiz_result)
            self.assertEqual(final_state.quiz_generation_count, 1)

            transcript_path = Path(config.runs_dir) / final_state.run_id / "outputs" / "transcript.txt"
            self.assertTrue(transcript_path.exists())

            regenerated_states = list(
                service.stream_regenerate_quiz(
                    run_state_payload=final_state.model_dump(mode="json"),
                    options_only=True,
                )
            )
            regenerated_state = regenerated_states[-1]

            original_questions = [question.question for question in final_state.quiz_result.questions]
            regenerated_questions = [question.question for question in regenerated_state.quiz_result.questions]
            self.assertEqual(original_questions, regenerated_questions)
            self.assertEqual(regenerated_state.quiz_generation_count, 2)

    def test_mock_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = PipelineService(config)
            parameters = PipelineParameters(n_keywords=2, top_k=1, chunk_size=80, chunk_overlap=10)

            with self.assertRaisesRegex(ValueError, "Only live mode is supported"):
                list(
                    service.stream_pipeline(
                        mode="mock",
                        parameters=parameters,
                        video_path=None,
                        transcript_text="Binary search tree example.",
                        subtitle_path=None,
                    )
                )

    def test_live_sequential_strategy_prepares_and_releases_servers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.auto_start_model_servers = True
            config.model_server_start_strategy = "sequential"
            config.ensure_directories()
            fake_manager = FakeServerManager()
            service = SequentialPipelineService(config, fake_manager)
            parameters = PipelineParameters(n_keywords=2, top_k=1, chunk_size=80, chunk_overlap=10)

            states = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Binary search tree example.",
                    subtitle_path=None,
                )
            )

            self.assertEqual(states[-1].steps["quiz"].status, "completed")
            self.assertEqual(
                fake_manager.events,
                [
                    ("prepare", "summary"),
                    ("release", "summary"),
                    ("prepare", "quiz"),
                ],
            )

    def test_live_sequential_strategy_can_release_servers_when_warm_keep_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.auto_start_model_servers = True
            config.model_server_start_strategy = "sequential"
            config.keep_model_servers_warm = False
            config.ensure_directories()
            fake_manager = FakeServerManager()
            service = SequentialPipelineService(config, fake_manager)
            parameters = PipelineParameters(n_keywords=2, top_k=1, chunk_size=80, chunk_overlap=10)

            states = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Binary search tree example.",
                    subtitle_path=None,
                )
            )

            self.assertEqual(states[-1].steps["quiz"].status, "completed")
            self.assertEqual(
                fake_manager.events,
                [
                    ("prepare", "summary"),
                    ("release", "summary"),
                    ("prepare", "quiz"),
                    ("release", "quiz"),
                    ("cleanup", "all"),
                ],
            )

    def test_parse_failure_logs_raw_response(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = FailingSummaryPipelineService(config)
            parameters = PipelineParameters(n_keywords=2, top_k=1, chunk_size=80, chunk_overlap=10)

            states = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Binary search tree example.",
                    subtitle_path=None,
                )
            )
            final_state = states[-1]
            run_dir = Path(config.runs_dir) / final_state.run_id
            error_path = run_dir / "outputs" / "error.json"

            self.assertEqual(final_state.steps["summary"].status, "failed")
            self.assertIn("Raw model response logged to", final_state.steps["summary"].error)

            error_payload = json.loads(error_path.read_text(encoding="utf-8"))
            self.assertEqual(error_payload["step"], "summary")
            self.assertEqual(error_payload["model_name"], "fake-summary")

            raw_log_path = Path(error_payload["raw_response_log"])
            self.assertTrue(raw_log_path.exists())
            self.assertEqual(raw_log_path.read_text(encoding="utf-8"), '{"oops": "not keywords"}')


if __name__ == "__main__":
    unittest.main()
