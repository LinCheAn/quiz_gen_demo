from __future__ import annotations

import unittest

from app import STATUS_TABLE_HEADERS, format_progress_html, format_status_rows, render_rag_outputs, render_regeneration_outputs
from utils.schemas import PipelineParameters, PipelineRunState, QuizQuestion, QuizResult, RetrievedChunk, StepStatus, TextChunk


def make_state() -> PipelineRunState:
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
        parameters=PipelineParameters(),
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


if __name__ == "__main__":
    unittest.main()
