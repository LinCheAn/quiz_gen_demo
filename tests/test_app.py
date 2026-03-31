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
    resolve_quiz_results,
)
from utils.schemas import PipelineParameters, PipelineRunState, QuizQuestion, QuizResult, RetrievedChunk, StepStatus, TextChunk


def make_state() -> PipelineRunState:
    selection = APP_MODEL_REGISTRY.resolve_selection()
    quiz_result = QuizResult(
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
        quiz_result=quiz_result,
        quiz_results=[quiz_result],
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

    def _layout_parents(self, config: dict) -> dict[int, int | None]:
        parents: dict[int, int | None] = {}

        def walk(node: dict, parent_id: int | None = None) -> None:
            node_id = node["id"]
            parents[node_id] = parent_id
            for child in node.get("children", []):
                walk(child, node_id)

        walk(config["layout"])
        return parents

    def _find_layout_node(self, node: dict, target_id: int) -> dict | None:
        if node["id"] == target_id:
            return node
        for child in node.get("children", []):
            found = self._find_layout_node(child, target_id)
            if found is not None:
                return found
        return None

    def _ancestor_ids(self, config: dict, component_id: int) -> list[int]:
        parents = self._layout_parents(config)
        ancestors: list[int] = []
        current = parents.get(component_id)
        while current is not None:
            ancestors.append(current)
            current = parents.get(current)
        return ancestors

    def _child_ids(self, config: dict, node_id: int) -> list[int]:
        node = self._find_layout_node(config["layout"], node_id)
        self.assertIsNotNone(node)
        return [child["id"] for child in node["children"]]

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
        self.assertEqual(outputs[6], {"auto_keywords": ["tree", "search"]})
        self.assertEqual(outputs[7], [state.chunks[0].model_dump(mode="json")])
        self.assertIsNone(outputs[8])

    def test_render_regeneration_outputs_preserves_retrieval_but_clears_quiz(self) -> None:
        state = make_state()
        state.steps["quiz"].status = "running"
        state.quiz_result = None

        outputs = render_regeneration_outputs(state, reset_unfinished=True)

        self.assertEqual(outputs[8], [state.retrieved_chunks[0].model_dump(mode="json")])

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

    def test_pipeline_status_tabs_are_present_in_order(self) -> None:
        config = self._build_demo_config()
        step_tab = self._find_component(config, "Step 表格")
        intermediate_results = self._find_component(config, "Intermediate Results")
        run_metadata = self._find_component(config, "Run Metadata")

        parent_ids = {
            self._layout_parents(config)[step_tab["id"]],
            self._layout_parents(config)[intermediate_results["id"]],
            self._layout_parents(config)[run_metadata["id"]],
        }
        self.assertEqual(len(parent_ids), 1)

        tabs_container_id = next(iter(parent_ids))
        self.assertIsNotNone(tabs_container_id)
        child_ids = self._child_ids(config, tabs_container_id)
        self.assertEqual(child_ids[:3], [step_tab["id"], intermediate_results["id"], run_metadata["id"]])

    def test_rag_controls_are_outside_intermediate_results(self) -> None:
        config = self._build_demo_config()

        intermediate_results = self._find_component(config, "Intermediate Results")
        custom_keywords = self._find_component(config, "Custom Keywords for RAG")
        current_auto_keywords = self._find_component(config, "Current Auto Keywords")
        run_rag_button = self._find_component(config, "進行RAG")
        extracted_keywords = self._find_component(config, "Extracted Keywords")
        parents = self._layout_parents(config)

        intermediate_results_id = intermediate_results["id"]

        self.assertNotIn(intermediate_results_id, self._ancestor_ids(config, custom_keywords["id"]))
        self.assertNotIn(intermediate_results_id, self._ancestor_ids(config, current_auto_keywords["id"]))
        self.assertNotIn(intermediate_results_id, self._ancestor_ids(config, run_rag_button["id"]))
        self.assertIn(intermediate_results_id, self._ancestor_ids(config, extracted_keywords["id"]))

        rag_container_id = parents[current_auto_keywords["id"]]
        self.assertIsNotNone(rag_container_id)
        child_ids = self._child_ids(config, rag_container_id)
        self.assertEqual(
            child_ids,
            [parents[custom_keywords["id"]], current_auto_keywords["id"], run_rag_button["id"]],
        )
        self.assertEqual(self._child_ids(config, parents[custom_keywords["id"]]), [custom_keywords["id"]])

    def test_regenerate_buttons_are_below_custom_question_input(self) -> None:
        config = self._build_demo_config()

        custom_question_input = self._find_component(config, "Custom Questions for Options-Only Regeneration")
        regenerate_button = self._find_component(config, "Regenerate Quiz")
        regenerate_options_button = self._find_component(config, "Regenerate Options Only")
        parents = self._layout_parents(config)

        button_row_id = parents[regenerate_button["id"]]
        self.assertIsNotNone(button_row_id)
        self.assertEqual(button_row_id, parents[regenerate_options_button["id"]])

        container_id = parents[button_row_id]
        self.assertIsNotNone(container_id)
        self.assertIn(container_id, self._ancestor_ids(config, custom_question_input["id"]))

        container_node = self._find_layout_node(config["layout"], container_id)
        self.assertIsNotNone(container_node)
        child_ids = [child["id"] for child in container_node["children"]]
        custom_question_branch_id = child_ids[0]

        self.assertEqual(child_ids[-1], button_row_id)
        self.assertIn(custom_question_branch_id, self._ancestor_ids(config, custom_question_input["id"]))

    def test_quiz_generation_controls_exist(self) -> None:
        config = self._build_demo_config()

        question_count = self._find_component(config, "Number of Questions")
        variant_count = self._find_component(config, "Number of Quiz Variants")

        self.assertEqual(question_count["type"], "slider")
        self.assertEqual(variant_count["type"], "slider")

    def test_resolve_quiz_results_supports_history_and_legacy_state(self) -> None:
        state = make_state()
        self.assertEqual(len(resolve_quiz_results(state)), 1)

        legacy_state = state.model_copy(deep=True)
        legacy_state.quiz_results = []

        self.assertEqual(len(resolve_quiz_results(legacy_state)), 1)

    def test_quiz_output_css_is_registered(self) -> None:
        config = self._build_demo_config()
        style_blocks = [
            component["props"]["value"]
            for component in config["components"]
            if component.get("type") == "html" and isinstance(component.get("props", {}).get("value"), str)
        ]
        style_html = "\n".join(style_blocks)

        self.assertIn(".quiz-output-question", style_html)
        self.assertIn("font-size: 1.35rem;", style_html)


if __name__ == "__main__":
    unittest.main()
