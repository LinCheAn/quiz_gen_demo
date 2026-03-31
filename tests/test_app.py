from __future__ import annotations

import unittest

from app import (
    APP_MODEL_REGISTRY,
    STATUS_TABLE_HEADERS,
    build_demo,
    format_progress_html,
    format_run_info,
    format_status_rows,
    render_rag_outputs,
    render_regeneration_outputs,
)
from utils.schemas import PipelineParameters, PipelineRunState, QuizQuestion, QuizResult, RetrievedChunk, StepStatus, TextChunk


def make_state() -> PipelineRunState:
    selection = APP_MODEL_REGISTRY.resolve_selection()
    steps = {
        "input": StepStatus(key="input", label="Input", status="completed", message="ready"),
        "asr": StepStatus(key="asr", label="ASR", status="skipped", message="used transcript"),
        "summary": StepStatus(key="summary", label="Summary", status="completed", message="keywords ready"),
        "chunking": StepStatus(key="chunking", label="Chunking", status="completed", message="chunks ready"),
        "retrieval": StepStatus(key="retrieval", label="Embedding Retrieval", status="completed", message="retrieval ready"),
        "quiz": StepStatus(key="quiz", label="Quiz Generation", status="completed", message="quiz ready"),
    }
    return PipelineRunState(
        run_id="run_123",
        mode="live",
        overview="demo",
        parameters=PipelineParameters(
            summary_model_id=selection.summary.id,
            quiz_model_id=selection.quiz.id,
        ),
        selected_models=selection,
        steps=steps,
        transcript="binary search tree transcript",
        keywords=["tree", "search"],
        chunks=[TextChunk(chunk_id="chunk_001", text="tree search concept", start_char=0, end_char=18)],
        retrieved_chunks=[
            RetrievedChunk(
                rank=1,
                chunk_id="chunk_001",
                text="tree search concept",
                score=0.9,
                matched_keywords=["tree", "search"],
            )
        ],
        quiz_result=QuizResult(
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
        ),
        quiz_generation_count=1,
    )


class AppRenderingTest(unittest.TestCase):
    def _build_demo_config(self) -> dict:
        return build_demo().get_config_file()

    def _find_component(self, config: dict, text: str) -> dict:
        for component in config["components"]:
            props = component.get("props", {})
            if props.get("label") == text or props.get("value") == text:
                return component
        self.fail(f"Component not found: {text}")

    def test_format_status_rows_returns_styled_dataframe(self) -> None:
        styled = format_status_rows(make_state())

        self.assertEqual(list(styled.data.columns), STATUS_TABLE_HEADERS)
        self.assertEqual(styled.data.iloc[0]["Status"], "completed")

    def test_render_rag_outputs_clears_only_retrieval_and_quiz(self) -> None:
        state = make_state()
        state.steps["retrieval"].status = "running"
        state.steps["quiz"].status = "pending"
        state.quiz_result = None

        outputs = render_rag_outputs(state, reset_unfinished=True)

        self.assertIn("Embedding Retrieval", outputs[1])
        self.assertEqual(outputs[4], state.transcript)
        self.assertEqual(outputs[5], {"auto_keywords": ["tree", "search"]})
        self.assertEqual(outputs[6], [state.chunks[0].model_dump(mode="json")])
        self.assertIsNone(outputs[7])
        self.assertIsNone(outputs[8])

    def test_render_regeneration_outputs_preserves_retrieval_but_clears_quiz(self) -> None:
        state = make_state()
        state.steps["quiz"].status = "running"
        state.quiz_result = None

        outputs = render_regeneration_outputs(state, reset_unfinished=True)

        self.assertEqual(outputs[7], [state.retrieved_chunks[0].model_dump(mode="json")])
        self.assertIsNone(outputs[8])

    def test_format_progress_html_marks_failed_step(self) -> None:
        state = make_state()
        state.steps["retrieval"].status = "failed"
        state.steps["retrieval"].error = "embedding crashed"

        html = format_progress_html(state)

        self.assertIn("Embedding Retrieval 失敗", html)
        self.assertIn("embedding crashed", html)

    def test_run_metadata_uses_selected_models_snapshot(self) -> None:
        state = make_state()

        run_info = format_run_info(state)

        self.assertEqual(run_info["summary_model_id"], state.selected_models.summary.id)
        self.assertEqual(run_info["quiz_model_id"], state.selected_models.quiz.id)
        self.assertEqual(run_info["selected_models"]["quiz"]["lora_path"], state.selected_models.quiz.lora_path)

    def test_build_demo_includes_model_dropdowns_with_registry_defaults(self) -> None:
        config = self._build_demo_config()

        summary_model = self._find_component(config, "Summary Model")
        quiz_model = self._find_component(config, "Quiz Model")

        self.assertEqual(summary_model["type"], "dropdown")
        self.assertEqual(quiz_model["type"], "dropdown")
        self.assertEqual(summary_model["props"]["value"], APP_MODEL_REGISTRY.defaults.summary_model_id)
        self.assertEqual(quiz_model["props"]["value"], APP_MODEL_REGISTRY.defaults.quiz_model_id)


if __name__ == "__main__":
    unittest.main()
