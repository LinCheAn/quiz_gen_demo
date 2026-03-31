from __future__ import annotations

import unittest

from utils.schemas import QuizQuestion
from utils.ui_helpers import (
    append_custom_question,
    format_question_markdown,
    parse_custom_keywords,
    parse_custom_question_lines,
)


class UIHelpersTest(unittest.TestCase):
    def test_parse_custom_question_lines_ignores_blank_lines(self) -> None:
        self.assertEqual(
            parse_custom_question_lines("  First question  \n\nSecond question\n   \nThird question  "),
            ["First question", "Second question", "Third question"],
        )

    def test_append_custom_question_adds_new_line_when_needed(self) -> None:
        self.assertEqual(
            append_custom_question("First question", "Second question"),
            "First question\nSecond question",
        )
        self.assertEqual(
            append_custom_question("", "Second question"),
            "Second question",
        )

    def test_parse_custom_keywords_accepts_commas_and_newlines(self) -> None:
        self.assertEqual(
            parse_custom_keywords("tree search,\n graph traversal  ,\n\nheap"),
            ["tree search", "graph traversal", "heap"],
        )

    def test_format_question_markdown_separates_answer_from_option_list(self) -> None:
        markdown = format_question_markdown(
            QuizQuestion(
                question="What is alpha?",
                options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
                answer="B",
                explanation="Because it matches the definition.",
            ),
            index=1,
        )

        self.assertIn("- D. Option D\n\n**Answer:** `B`", markdown)
        self.assertIn("**Explanation:** Because it matches the definition.", markdown)


if __name__ == "__main__":
    unittest.main()
