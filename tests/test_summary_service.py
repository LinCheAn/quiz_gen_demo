from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from services.summary_service import SummaryService
from utils.config import AppConfig
from utils.errors import ModelResponseFormatError


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs) -> None:
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create),
        )

    @staticmethod
    def _create(*args, **kwargs):
        return _FakeResponse("keywords: tree, graph")


class SummaryServiceTest(unittest.TestCase):
    def test_extract_keywords_fails_on_non_json_response(self) -> None:
        fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAIClient)
        with patch.dict(sys.modules, {"openai": fake_module}):
            service = SummaryService(AppConfig())
            with self.assertRaises(ModelResponseFormatError) as context:
                service.extract_keywords("Binary search tree lecture content", 3)
        self.assertIn("parsable keywords JSON", str(context.exception))
        self.assertEqual(context.exception.raw_response, "keywords: tree, graph")


if __name__ == "__main__":
    unittest.main()
