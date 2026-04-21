from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.quiz_service import QuizService
from utils.config import AppConfig
from utils.schemas import QuizQuestion


class QuizServiceTest(unittest.TestCase):
    def _build_service(self) -> QuizService:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        config = AppConfig(project_root=Path(tempdir.name))
        config.ensure_directories()
        return QuizService(config)

    def _build_transformers_service(self) -> QuizService:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        config = AppConfig(project_root=Path(tempdir.name), inference_backend="transformers")
        config.ensure_directories()
        return QuizService(config)

    def test_full_question_prompt_matches_reference_script_shape(self) -> None:
        service = self._build_service()
        references = service._format_reference_paragraphs(["A", "B", "C", "D", "E"])

        self.assertEqual(
            references,
            "\n".join(
                [
                    "Reference paragraph 1:A",
                    "Reference paragraph 2:B",
                    "Reference paragraph 3:C",
                    "Reference paragraph 4:D",
                    "Reference paragraph 5:E",
                ]
            ),
        )

        prompt = service._build_full_quiz_prompt(references)
        expected_prompt = (
            "Reference paragraph 1:A\n"
            "Reference paragraph 2:B\n"
            "Reference paragraph 3:C\n"
            "Reference paragraph 4:D\n"
            "Reference paragraph 5:E\n\n"
            "Format requirements:\n"
            "1. Use English to generate one multiple-choice question\n"
            "2. Must reference the key course content used\n"
            '3. Please avoid "All of the above" and "None of the above" options\n'
            "4. Output in JSON format:\n"
            "{\n"
            '    "question": [\n'
            "        {\n"
            '            "question": "question text",\n'
            '            "options": {\n'
            '                "A": "option A",\n'
            '                "B": "option B",\n'
            '                "C": "option C",\n'
            '                "D": "option D"\n'
            "            },\n"
            '            "answer": "A"\n'
            "        }\n"
            "    ]\n"
            "}\n"
            "5. Output in JSON format, without any other words."
        )
        self.assertEqual(prompt, expected_prompt)

    def test_continuation_prompt_uses_llama3_assistant_prefix(self) -> None:
        service = self._build_service()
        prompt = service._build_full_quiz_prompt(service._format_reference_paragraphs(["alpha"]))
        assistant_prefix = service._build_assistant_prefix('What is "alpha"?')
        formatted_prompt = service._format_llama3_prompt(
            service.FULL_QUESTION_SYSTEM_MESSAGE,
            prompt,
            assistant_prefix,
        )

        self.assertTrue(formatted_prompt.startswith("<|begin_of_text|><|start_header_id|>system"))
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>\n\n", formatted_prompt)
        self.assertIn('"question": "What is \\"alpha\\"?"', assistant_prefix)
        self.assertTrue(formatted_prompt.endswith(assistant_prefix))

    def test_parse_reference_question_payload_accepts_reference_schema(self) -> None:
        service = self._build_service()
        question = service._parse_reference_question_payload(
            {
                "question": [
                    {
                        "question": "What is alpha?",
                        "options": {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
                        "answer": "b",
                    }
                ]
            }
        )

        self.assertEqual(question.question, "What is alpha?")
        self.assertEqual(question.options["C"], "C1")
        self.assertEqual(question.answer, "B")

    def test_parse_reference_question_payload_preserves_expected_question_text(self) -> None:
        service = self._build_service()
        question = service._parse_reference_question_payload(
            {
                "question": [
                    {
                        "question": "Changed stem",
                        "options": {"A": "A1", "B": "B1", "C": "C1", "D": "D1"},
                        "answer": "A",
                    }
                ]
            },
            expected_question="Original stem",
        )

        self.assertEqual(question.question, "Original stem")

    def test_parse_reference_question_response_accepts_code_fenced_json(self) -> None:
        service = self._build_service()
        question = service._parse_reference_question_response(
            """```json
            {
              "question": [
                {
                  "question": "What is alpha?",
                  "options": {
                    "A": "A1",
                    "B": "B1",
                    "C": "C1",
                    "D": "D1"
                  },
                  "answer": "A"
                }
              ]
            }
            ```"""
        )

        self.assertEqual(question.question, "What is alpha?")
        self.assertEqual(question.options["D"], "D1")
        self.assertEqual(question.answer, "A")

    def test_parse_reference_question_response_extracts_json_from_wrapped_text(self) -> None:
        service = self._build_service()
        question = service._parse_reference_question_response(
            """
            Here is the quiz:
            {
              "question": [
                {
                  "question": "What is alpha?",
                  "options": {
                    "A": "A1",
                    "B": "B1",
                    "C": "C1",
                    "D": "D1"
                  },
                  "answer": "D"
                }
              ]
            }
            Thanks.
            """
        )

        self.assertEqual(question.question, "What is alpha?")
        self.assertEqual(question.answer, "D")

    def test_parse_reference_question_response_reports_json_syntax_error(self) -> None:
        service = self._build_service()

        with self.assertRaisesRegex(RuntimeError, r"Quiz response JSON syntax error at line \d+ column \d+"):
            service._parse_reference_question_response(
                """
                {
                  "question": [
                    {
                      "question": "What is alpha?",
                      "options": {
                        "A": "A1",
                        "B": B1,
                        "C": "C1",
                        "D": "D1"
                      },
                      "answer": "B"
                    }
                  ]
                }
                """
            )

    def test_parse_reference_question_response_reports_missing_option_d(self) -> None:
        service = self._build_service()

        with self.assertRaisesRegex(RuntimeError, "Quiz response missing option D"):
            service._parse_reference_question_response(
                """
                {
                  "question": [
                    {
                      "question": "What is alpha?",
                      "options": {
                        "A": "A1",
                        "B": "B1",
                        "C": "C1"
                      },
                      "answer": "B"
                    }
                  ]
                }
                """
            )

    def test_resolve_question_stems_prefers_custom_questions(self) -> None:
        service = self._build_service()
        stems = service._resolve_question_stems(
            existing_questions=[
                QuizQuestion(
                    question="Existing stem",
                    options={"A": "A", "B": "B", "C": "C", "D": "D"},
                    answer="A",
                )
            ],
            question_stems=["  Custom stem  ", "", "Second custom stem"],
        )

        self.assertEqual(stems, ["Custom stem", "Second custom stem"])

    def test_resolve_question_stems_falls_back_to_existing_questions(self) -> None:
        service = self._build_service()
        stems = service._resolve_question_stems(
            existing_questions=[
                QuizQuestion(
                    question="Existing stem",
                    options={"A": "A", "B": "B", "C": "C", "D": "D"},
                    answer="A",
                )
            ],
            question_stems=None,
        )

        self.assertEqual(stems, ["Existing stem"])

    def test_generate_full_questions_prefers_explicit_question_count(self) -> None:
        service = self._build_service()
        captured_indices: list[int] = []

        def fake_generate_single_full_question(client, prompt: str, question_index: int):
            del client, prompt
            captured_indices.append(question_index)
            return (
                QuizQuestion(
                    question=f"Question {question_index}",
                    options={"A": "A", "B": "B", "C": "C", "D": "D"},
                    answer="A",
                ),
                {"question_index": question_index},
            )

        service._generate_single_full_question = fake_generate_single_full_question  # type: ignore[method-assign]

        questions, raw_responses = service._generate_full_questions(
            client=object(),
            references="Reference paragraph 1:alpha",
            question_count=2,
        )

        self.assertEqual(captured_indices, [1, 2])
        self.assertEqual([question.question for question in questions], ["Question 1", "Question 2"])
        self.assertEqual([item["question_index"] for item in raw_responses], [1, 2])

    def test_generate_quiz_with_transformers_backend(self) -> None:
        service = self._build_transformers_service()

        class _FakeBackend:
            def __init__(self, config, *, role: str) -> None:
                self.config = config
                self.role = role

            def generate(self, **kwargs) -> str:
                return (
                    '{"question": [{"question": "What is alpha?", "options": '
                    '{"A": "A1", "B": "B1", "C": "C1", "D": "D1"}, "answer": "A"}]}'
                )

        with patch("services.quiz_service.TransformersTextGenerationBackend", _FakeBackend):
            result = service.generate_quiz(["alpha"], question_count=1)

        self.assertEqual(len(result.questions), 1)
        self.assertEqual(result.questions[0].question, "What is alpha?")
        self.assertEqual(result.questions[0].answer, "A")

    def test_request_completion_with_transformers_backend_uses_assistant_prefix(self) -> None:
        service = self._build_transformers_service()
        captured: dict[str, object] = {}

        class _FakeBackend:
            def __init__(self, config, *, role: str) -> None:
                self.config = config
                self.role = role

            def generate(self, **kwargs) -> str:
                captured.update(kwargs)
                return '"options": {"A": "A1", "B": "B1", "C": "C1", "D": "D1"}, "answer": "A"}]}'

        with patch("services.quiz_service.TransformersTextGenerationBackend", _FakeBackend):
            result = service._request_completion(
                None,
                prompt="unused",
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}],
                assistant_prefix='{"question": [',
            )

        self.assertIn('"answer": "A"', result)
        self.assertEqual(captured["assistant_prefix"], '{"question": [')


if __name__ == "__main__":
    unittest.main()
