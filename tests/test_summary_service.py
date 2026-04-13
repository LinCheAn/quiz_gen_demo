from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from services.summary_service import SUMMARY_AUTO_CUTOFF_WARNING, SummaryService
from utils.config import AppConfig
from utils.errors import ModelResponseFormatError


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class _FakeOpenAIClient:
    calls = 0

    def __init__(self, *args, **kwargs) -> None:
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create),
        )

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    @staticmethod
    def _create(*args, **kwargs):
        _FakeOpenAIClient.calls += 1
        return _FakeResponse("keywords: tree, graph")


class _FormatRetryOpenAIClient:
    calls = 0

    def __init__(self, *args, **kwargs) -> None:
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create),
        )

    @classmethod
    def reset(cls) -> None:
        cls.calls = 0

    @staticmethod
    def _create(*args, **kwargs):
        _FormatRetryOpenAIClient.calls += 1
        if _FormatRetryOpenAIClient.calls == 1:
            return _FakeResponse("keywords: tree, graph")
        return _FakeResponse('{"keywords": ["tree", "graph"]}')


class _CapturingOpenAIClient:
    last_request: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create),
        )

    @classmethod
    def reset(cls) -> None:
        cls.last_request = None

    @classmethod
    def _create(cls, *args, **kwargs):
        cls.last_request = kwargs
        return _FakeResponse('{"keywords": ["tree", "graph"]}')


class _RetryingOpenAIClient:
    calls: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create),
        )

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    @classmethod
    def _create(cls, *args, **kwargs):
        cls.calls.append(kwargs)
        user_prompt = kwargs["messages"][1]["content"]
        if len(user_prompt) > 400:
            raise Exception(
                "Error code: 400 - {'error': {'message': "
                "\"This model's maximum context length is 400 tokens. However, you requested 0 "
                "output tokens and your prompt contains at least 480 input tokens, for a total of "
                "at least 480 tokens. Please reduce the length of the input prompt or the number "
                "of requested output tokens. (parameter=input_tokens, value=480)\", "
                "'type': 'BadRequestError', 'param': 'input_tokens', 'code': 400}}"
            )
        return _FakeResponse('{"keywords": ["tree", "graph"]}')


class SummaryServiceTest(unittest.TestCase):
    def test_extract_keywords_fails_on_non_json_response(self) -> None:
        fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAIClient)
        with patch.dict(sys.modules, {"openai": fake_module}):
            service = SummaryService(AppConfig())
            _FakeOpenAIClient.reset()
            with self.assertRaises(ModelResponseFormatError) as context:
                service.extract_keywords("Binary search tree lecture content", 3)
        self.assertIn("parsable keywords JSON", str(context.exception))
        self.assertEqual(context.exception.raw_response, "keywords: tree, graph")
        self.assertEqual(_FakeOpenAIClient.calls, 2)

    def test_extract_keywords_retries_once_on_format_error(self) -> None:
        fake_module = types.SimpleNamespace(OpenAI=_FormatRetryOpenAIClient)
        with patch.dict(sys.modules, {"openai": fake_module}):
            service = SummaryService(AppConfig())
            _FormatRetryOpenAIClient.reset()
            with patch.object(service, "_load_tokenizer", return_value=None):
                result = service.extract_keywords("Binary search tree lecture content", 3)

        self.assertEqual(result.keywords, ["tree", "graph"])
        self.assertEqual(_FormatRetryOpenAIClient.calls, 2)

    def test_extract_keywords_auto_cuts_off_oversized_summary_input(self) -> None:
        fake_module = types.SimpleNamespace(OpenAI=_CapturingOpenAIClient)
        long_text = "binary search tree " * 100

        with patch.dict(sys.modules, {"openai": fake_module}):
            service = SummaryService(AppConfig(summary_server_max_model_len=520))
            _CapturingOpenAIClient.reset()
            with patch.object(
                service,
                "_estimate_messages_token_count",
                side_effect=lambda messages: sum(len(message["content"]) for message in messages),
            ):
                result = service.extract_keywords(long_text, 3)

        self.assertEqual(result.keywords, ["tree", "graph"])
        self.assertEqual(result.warning, SUMMARY_AUTO_CUTOFF_WARNING)
        self.assertIsNotNone(_CapturingOpenAIClient.last_request)

        sent_messages = _CapturingOpenAIClient.last_request["messages"]
        sent_prompt = sent_messages[1]["content"]
        original_prompt = service._build_prompt(long_text.strip(), 3)

        self.assertLess(len(sent_prompt), len(original_prompt))
        self.assertIn("lesson content:", sent_prompt)
        self.assertIn('"keywords"', sent_prompt)

    def test_extract_keywords_retries_with_stricter_cutoff_on_context_length_error(self) -> None:
        fake_module = types.SimpleNamespace(OpenAI=_RetryingOpenAIClient)
        long_text = "binary search tree " * 20
        trimmed_text = "binary search tree"

        with patch.dict(sys.modules, {"openai": fake_module}):
            service = SummaryService(AppConfig(summary_server_max_model_len=520))
            _RetryingOpenAIClient.reset()
            with patch.object(
                service,
                "_fit_text_to_context_window",
                side_effect=[
                    (long_text, None),
                    (trimmed_text, SUMMARY_AUTO_CUTOFF_WARNING),
                ],
            ):
                result = service.extract_keywords(long_text, 3)

        self.assertEqual(result.keywords, ["tree", "graph"])
        self.assertEqual(result.warning, SUMMARY_AUTO_CUTOFF_WARNING)
        self.assertEqual(len(_RetryingOpenAIClient.calls), 2)
        self.assertGreater(
            len(_RetryingOpenAIClient.calls[0]["messages"][1]["content"]),
            len(_RetryingOpenAIClient.calls[1]["messages"][1]["content"]),
        )


if __name__ == "__main__":
    unittest.main()
