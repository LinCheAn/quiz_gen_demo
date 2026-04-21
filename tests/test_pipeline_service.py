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
    ModelSelectionSnapshot,
    PipelineParameters,
    QuizQuestion,
    QuizResult,
    QuizModelPreset,
    RetrievalResult,
    RetrievedChunk,
    SummaryModelPreset,
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
    def __init__(self) -> None:
        self.call_count = 0
        self.keywords = ["tree", "search"]

    def extract_keywords(self, text: str, n_keywords: int, progress_callback=None) -> KeywordResult:
        self.call_count += 1
        return KeywordResult(keywords=list(self.keywords), model="fake-summary")


class FakeSummaryServiceWithWarning:
    def extract_keywords(self, text: str, n_keywords: int, progress_callback=None) -> KeywordResult:
        return KeywordResult(
            keywords=["tree", "search"],
            model="fake-summary",
            warning="摘要輸入內容超過 summary model 的 max_model_len，已自動 cutoff。",
        )


class FakeChunkService:
    def __init__(self) -> None:
        self.call_count = 0

    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> ChunkResult:
        self.call_count += 1
        return ChunkResult(
            chunks=[TextChunk(chunk_id="chunk_001", text="tree search concept", start_char=0, end_char=18)],
            strategy="fixed_window_char",
            chunk_size=chunk_size,
            overlap=overlap,
        )


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.last_keywords = None

    def retrieve(self, chunks, keywords, top_k: int, progress_callback=None) -> RetrievalResult:
        self.last_keywords = list(keywords)
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
    def __init__(self) -> None:
        self.last_existing_questions = None
        self.last_question_stems = None
        self.last_question_count = None
        self.generate_call_count = 0

    def generate_quiz(self, context_chunks, variant: int = 0, question_count: int | None = None, progress_callback=None) -> QuizResult:
        self.generate_call_count += 1
        self.last_question_count = question_count
        total_questions = question_count or 1
        return QuizResult(
            questions=[
                QuizQuestion(
                    question=f"What is tree search concept variant {variant} question {index}?",
                    options={"A": "A", "B": "B", "C": "C", "D": "D"},
                    answer="A",
                    explanation="Because.",
                )
                for index in range(1, total_questions + 1)
            ],
            model="fake-quiz",
            generation_mode="full",
        )

    def regenerate_options_only(
        self,
        existing_questions=None,
        context_chunks=None,
        question_stems=None,
        variant: int = 1,
        progress_callback=None,
    ) -> QuizResult:
        self.last_existing_questions = existing_questions
        self.last_question_stems = question_stems
        stems = question_stems or [
            question.question
            for question in (existing_questions or [])
        ]
        if stems:
            return QuizResult(
                questions=[
                    QuizQuestion(
                        question=stem,
                        options={"A": "A", "B": "B", "C": "C", "D": "D"},
                        answer="A",
                        explanation="Because.",
                    )
                    for stem in stems
                ],
                model="fake-quiz",
                generation_mode="options_only",
            )
        return self.generate_quiz(context_chunks, variant=variant, progress_callback=progress_callback)

    def regenerate_full(self, context_chunks, variant: int = 1, question_count: int | None = None, progress_callback=None) -> QuizResult:
        return self.generate_quiz(
            context_chunks,
            variant=variant,
            question_count=question_count,
            progress_callback=progress_callback,
        )


class SequentialPipelineService(PipelineService):
    def __init__(self, config, server_manager=None, selected_models=None) -> None:
        super().__init__(config, server_manager, selected_models)
        self.fake_summary_service = FakeSummaryService()
        self.fake_chunk_service = FakeChunkService()
        self.fake_embedding_service = FakeEmbeddingService()
        self.fake_quiz_service = FakeQuizService()

    def _build_services(self, mode: str):
        return {
            "asr": None,
            "summary": self.fake_summary_service,
            "chunk": self.fake_chunk_service,
            "embedding": self.fake_embedding_service,
            "quiz": self.fake_quiz_service,
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


class WarningSummaryPipelineService(PipelineService):
    def _build_services(self, mode: str):
        return {
            "asr": None,
            "summary": FakeSummaryServiceWithWarning(),
            "chunk": FakeChunkService(),
            "embedding": FakeEmbeddingService(),
            "quiz": FakeQuizService(),
        }


class PipelineServiceTest(unittest.TestCase):
    def _make_selection(self) -> ModelSelectionSnapshot:
        return ModelSelectionSnapshot(
            summary=SummaryModelPreset(
                id="summary-test",
                label="Summary Test",
                model_name="summary-model",
                base_url="http://127.0.0.1:9101/v1",
                server_conda_env="summary-env",
                server_model="summary-server-model",
                gpu_memory_utilization=0.5,
                max_model_len=4096,
                tensor_parallel_size=1,
                dtype="bfloat16",
                quantization="fp8",
                supported_backends=["vllm", "transformers"],
            ),
            quiz=QuizModelPreset(
                id="quiz-test",
                label="Quiz Test",
                model_name="quiz-model",
                base_url="http://127.0.0.1:9100/v1",
                server_conda_env="quiz-env",
                server_model="quiz-server-model",
                lora_path="/tmp/fake-adapter",
                gpu_memory_utilization=0.6,
                max_model_len=8192,
                tensor_parallel_size=1,
                dtype="bfloat16",
                supported_backends=["vllm", "transformers"],
            ),
        )

    def test_live_pipeline_and_options_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            selection = self._make_selection()
            service = SequentialPipelineService(config, selected_models=selection)
            parameters = PipelineParameters(
                n_keywords=4,
                top_k=3,
                chunk_size=80,
                chunk_overlap=10,
                summary_model_id=selection.summary.id,
                quiz_model_id=selection.quiz.id,
                quiz_question_count=3,
                quiz_variant_count=2,
            )

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
            self.assertEqual(final_state.quiz_generation_count, 2)
            self.assertEqual(len(final_state.quiz_results), 2)
            self.assertEqual(len(final_state.quiz_result.questions), 3)
            self.assertEqual(final_state.selected_models, selection)

            transcript_path = Path(config.runs_dir) / final_state.run_id / "outputs" / "transcript.txt"
            self.assertTrue(transcript_path.exists())
            self.assertTrue((Path(config.runs_dir) / final_state.run_id / "outputs" / "quiz_v1.json").exists())
            self.assertTrue((Path(config.runs_dir) / final_state.run_id / "outputs" / "quiz_v2.json").exists())

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
            self.assertEqual(regenerated_state.quiz_generation_count, 3)
            self.assertEqual(len(regenerated_state.quiz_results), 3)

    def test_options_regeneration_can_use_custom_questions(self) -> None:
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

            regenerated_states = list(
                service.stream_regenerate_quiz(
                    run_state_payload=final_state.model_dump(mode="json"),
                    options_only=True,
                    custom_questions=[
                        "How does balancing affect a binary search tree?",
                        "Why does traversal order matter?",
                    ],
                )
            )
            regenerated_state = regenerated_states[-1]

            self.assertEqual(
                [question.question for question in regenerated_state.quiz_result.questions],
                [
                    "How does balancing affect a binary search tree?",
                    "Why does traversal order matter?",
                ],
            )
            self.assertEqual(
                service.fake_quiz_service.last_question_stems,
                [
                    "How does balancing affect a binary search tree?",
                    "Why does traversal order matter?",
                ],
            )
            self.assertEqual(len(regenerated_state.quiz_results), 2)

    def test_regeneration_clears_existing_quiz_before_new_output_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = SequentialPipelineService(config)
            parameters = PipelineParameters(n_keywords=4, top_k=3, chunk_size=80, chunk_overlap=10)

            final_state = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Binary search tree example.",
                    subtitle_path=None,
                )
            )[-1]

            regeneration_states = list(
                service.stream_regenerate_quiz(
                    run_state_payload=final_state.model_dump(mode="json"),
                    options_only=False,
                )
            )

            self.assertIsNone(regeneration_states[0].quiz_result)
            self.assertEqual(regeneration_states[0].steps["quiz"].status, "running")
            self.assertIsNotNone(regeneration_states[-1].quiz_result)
            self.assertEqual(regeneration_states[-1].steps["quiz"].status, "completed")
            self.assertEqual(len(regeneration_states[-1].quiz_results), 2)

    def test_rag_retrieval_can_use_custom_keywords_without_generating_quiz(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = SequentialPipelineService(config)
            parameters = PipelineParameters(n_keywords=4, top_k=3, chunk_size=80, chunk_overlap=10)

            final_state = list(
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
            )[-1]

            initial_generate_count = service.fake_quiz_service.generate_call_count
            rag_states = list(
                service.stream_rag_retrieval(
                    run_state_payload=final_state.model_dump(mode="json"),
                    custom_keywords=["balancing", "traversal"],
                )
            )

            self.assertEqual(service.fake_embedding_service.last_keywords, ["balancing", "traversal"])
            self.assertEqual(service.fake_quiz_service.generate_call_count, initial_generate_count)
            self.assertIsNone(rag_states[0].quiz_result)
            self.assertEqual(rag_states[0].steps["quiz"].status, "pending")
            self.assertIsNone(rag_states[-1].quiz_result)
            self.assertEqual(rag_states[-1].quiz_results, [])
            self.assertEqual(rag_states[-1].steps["retrieval"].status, "completed")
            self.assertEqual(rag_states[-1].steps["quiz"].status, "pending")
            self.assertIn("Manually regenerate quiz", rag_states[-1].steps["quiz"].message)

    def test_rag_retrieval_falls_back_to_auto_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = SequentialPipelineService(config)
            parameters = PipelineParameters(n_keywords=4, top_k=3, chunk_size=80, chunk_overlap=10)

            final_state = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Binary search tree example.",
                    subtitle_path=None,
                )
            )[-1]

            list(
                service.stream_rag_retrieval(
                    run_state_payload=final_state.model_dump(mode="json"),
                    custom_keywords=[],
                )
            )

            self.assertEqual(service.fake_embedding_service.last_keywords, ["tree", "search"])

    def test_keyword_regeneration_updates_keywords_and_clears_downstream_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = SequentialPipelineService(config)
            parameters = PipelineParameters(n_keywords=4, top_k=3, chunk_size=80, chunk_overlap=10)

            final_state = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Binary search tree example.",
                    subtitle_path=None,
                )
            )[-1]

            initial_chunk_calls = service.fake_chunk_service.call_count
            service.fake_summary_service.keywords = ["heap", "balance"]

            regenerated_states = list(
                service.stream_regenerate_keywords(
                    run_state_payload=final_state.model_dump(mode="json"),
                )
            )
            regenerated_state = regenerated_states[-1]

            self.assertEqual(service.fake_summary_service.call_count, 2)
            self.assertEqual(service.fake_chunk_service.call_count, initial_chunk_calls)
            self.assertEqual(regenerated_states[0].steps["summary"].status, "running")
            self.assertEqual(regenerated_state.keywords, ["heap", "balance"])
            self.assertEqual(regenerated_state.steps["summary"].status, "completed")
            self.assertEqual(regenerated_state.steps["retrieval"].status, "pending")
            self.assertEqual(regenerated_state.steps["quiz"].status, "pending")
            self.assertEqual(regenerated_state.retrieved_chunks, [])
            self.assertIsNone(regenerated_state.quiz_result)
            self.assertEqual(regenerated_state.quiz_results, [])

    def test_summary_cutoff_warning_is_persisted_in_pipeline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = WarningSummaryPipelineService(config)
            parameters = PipelineParameters(n_keywords=4, top_k=3, chunk_size=80, chunk_overlap=10)

            final_state = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Very long subtitle text.",
                    subtitle_path=None,
                )
            )[-1]

            self.assertEqual(
                final_state.summary_warning,
                "摘要輸入內容超過 summary model 的 max_model_len，已自動 cutoff。",
            )
            self.assertIn("已自動 cutoff", final_state.steps["summary"].message)

    def test_options_regeneration_accepts_custom_questions_without_existing_quiz(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = SequentialPipelineService(config)
            state = service._create_state(
                run_id="run_custom_only",
                mode="live",
                parameters=PipelineParameters(),
            )
            state.retrieved_chunks = [
                RetrievedChunk(
                    rank=1,
                    chunk_id="chunk_001",
                    text="tree search concept",
                    score=0.9,
                    matched_keywords=["tree"],
                )
            ]

            regenerated_states = list(
                service.stream_regenerate_quiz(
                    run_state_payload=state.model_dump(mode="json"),
                    options_only=True,
                    custom_questions=["What is a tree search concept?"],
                )
            )
            regenerated_state = regenerated_states[-1]

            self.assertEqual(regenerated_state.steps["quiz"].status, "completed")
            self.assertEqual(
                [question.question for question in regenerated_state.quiz_result.questions],
                ["What is a tree search concept?"],
            )
            self.assertEqual(len(regenerated_state.quiz_results), 1)

    def test_full_regeneration_uses_state_question_count(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(project_root=Path(tempdir))
            config.ensure_directories()
            service = SequentialPipelineService(config)
            parameters = PipelineParameters(
                n_keywords=4,
                top_k=3,
                chunk_size=80,
                chunk_overlap=10,
                quiz_question_count=4,
                quiz_variant_count=1,
            )

            final_state = list(
                service.stream_pipeline(
                    mode="live",
                    parameters=parameters,
                    video_path=None,
                    transcript_text="Binary search tree example.",
                    subtitle_path=None,
                )
            )[-1]

            regenerated_state = list(
                service.stream_regenerate_quiz(
                    run_state_payload=final_state.model_dump(mode="json"),
                    options_only=False,
                )
            )[-1]

            self.assertEqual(service.fake_quiz_service.last_question_count, 4)
            self.assertEqual(len(regenerated_state.quiz_result.questions), 4)

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

    def test_transformers_backend_skips_server_manager_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            config = AppConfig(
                project_root=Path(tempdir),
                inference_backend="transformers",
            )
            config.auto_start_model_servers = True
            config.model_server_start_strategy = "sequential"
            config.ensure_directories()
            fake_manager = FakeServerManager()
            service = SequentialPipelineService(config, fake_manager)
            parameters = PipelineParameters(
                inference_backend="transformers",
                n_keywords=2,
                top_k=1,
                chunk_size=80,
                chunk_overlap=10,
            )

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
            self.assertEqual(fake_manager.events, [])

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
