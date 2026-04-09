from __future__ import annotations

import json
import math
import re
from typing import Callable

from utils.config import AppConfig
from utils.errors import ModelResponseFormatError
from utils.schemas import KeywordResult


ProgressCallback = Callable[[float, str], None]
SUMMARY_SYSTEM_PROMPT = (
    "You are a college student, you need to extract the most important keywords "
    "from the provided lesson content."
)
SUMMARY_AUTO_CUTOFF_WARNING = "摘要輸入內容超過 summary model 的 max_model_len，已自動 cutoff。"
SUMMARY_CONTEXT_SAFETY_MARGIN = 64


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_json_fragment(text: str) -> dict[str, object] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def endpoint_client_hint(base_url: str) -> str:
    if "0.0.0.0" in base_url:
        return " `0.0.0.0` should be used as a server bind address, not as a client target. Use `127.0.0.1` or the actual host IP instead."
    return ""


def endpoint_runtime_hint(config: AppConfig, component: str) -> str:
    if config.auto_start_model_servers:
        return (
            f" AUTO_START_MODEL_SERVERS=1, so inspect artifacts/server_logs and the {component} "
            "startup command in utils/server_manager.py."
        )
    return " AUTO_START_MODEL_SERVERS=0, so start the endpoint manually before running the pipeline."


class SummaryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._tokenizer_load_attempted = False

    def extract_keywords(
        self,
        text: str,
        n_keywords: int,
        progress_callback: ProgressCallback | None = None,
    ) -> KeywordResult:
        if progress_callback:
            progress_callback(0.1, "Preparing summary keywords")

        summary_text, warning = self._fit_text_to_context_window(
            text,
            n_keywords,
            context_limit=self.config.summary_server_max_model_len,
            reserved_tokens=SUMMARY_CONTEXT_SAFETY_MARGIN,
        )
        if warning and progress_callback:
            progress_callback(0.2, warning)

        try:
            keywords = self._extract_keywords_live(summary_text, n_keywords, progress_callback)
        except _SummaryContextLengthExceededError as exc:
            retry_reserved_tokens = max(
                SUMMARY_CONTEXT_SAFETY_MARGIN,
                (exc.input_tokens - exc.max_context_len + 1) if exc.input_tokens is not None else 1,
            ) + SUMMARY_CONTEXT_SAFETY_MARGIN
            summary_text, retry_warning = self._fit_text_to_context_window(
                text,
                n_keywords,
                context_limit=min(self.config.summary_server_max_model_len, exc.max_context_len),
                reserved_tokens=retry_reserved_tokens,
            )
            warning = warning or retry_warning or SUMMARY_AUTO_CUTOFF_WARNING
            if progress_callback:
                progress_callback(0.3, "Summary input exceeded model context, retrying with stricter cutoff")
            keywords = self._extract_keywords_live(summary_text, n_keywords, progress_callback)
        return KeywordResult(
            keywords=keywords,
            model=self.config.summary_model_name,
            warning=warning,
        )

    def _build_prompt(self, text: str, n_keywords: int) -> str:
        return f"""
lesson content:
{text}

Output in following json format:
{{
  "keywords": ["keyword 1", "keyword 2", "keyword 3"]
}}

Output only JSON format, no other explanation.
Return up to {n_keywords} keywords.
""".strip()

    def _build_summary_messages(self, text: str, n_keywords: int) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(text, n_keywords)},
        ]

    def _fit_text_to_context_window(
        self,
        text: str,
        n_keywords: int,
        *,
        context_limit: int,
        reserved_tokens: int,
    ) -> tuple[str, str | None]:
        normalized_text = text.strip()
        max_context_len = max(1, context_limit)
        max_model_len = max(1, max_context_len - max(0, reserved_tokens))
        empty_messages = self._build_summary_messages("", n_keywords)
        empty_message_token_count = self._estimate_messages_token_count(empty_messages)
        if empty_message_token_count > max_context_len:
            raise ValueError("Summary prompt exceeds summary_server_max_model_len even without transcript content")
        max_model_len = max(max_model_len, empty_message_token_count)
        messages = self._build_summary_messages(normalized_text, n_keywords)
        if self._estimate_messages_token_count(messages) <= max_model_len:
            return normalized_text, None

        low = 0
        high = len(normalized_text)
        while low < high:
            mid = (low + high + 1) // 2
            candidate_text = normalized_text[:mid].rstrip()
            candidate_messages = self._build_summary_messages(candidate_text, n_keywords)
            if self._estimate_messages_token_count(candidate_messages) <= max_model_len:
                low = mid
            else:
                high = mid - 1

        fitted_text = normalized_text[:low].rstrip()
        if not fitted_text:
            raise ValueError("Summary input becomes empty after applying max_model_len cutoff")
        return fitted_text, SUMMARY_AUTO_CUTOFF_WARNING

    def _estimate_messages_token_count(self, messages: list[dict[str, str]]) -> int:
        tokenizer = self._load_tokenizer()
        if tokenizer is not None:
            try:
                return len(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                )
            except Exception:
                pass
            try:
                return sum(
                    len(tokenizer.encode(message["content"], add_special_tokens=False))
                    for message in messages
                ) + (8 * len(messages))
            except Exception:
                pass

        return sum(
            self._estimate_text_tokens(message["content"])
            for message in messages
        ) + (8 * len(messages))

    def _load_tokenizer(self):
        if self._tokenizer_load_attempted:
            return self._tokenizer

        self._tokenizer_load_attempted = True
        try:
            from transformers import AutoTokenizer
        except ImportError:
            return None

        model_candidates = [
            self.config.summary_server_model,
            self.config.summary_base_model_name,
            self.config.summary_model_name,
        ]
        for model_name in dict.fromkeys(candidate for candidate in model_candidates if candidate):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True,
                )
                return self._tokenizer
            except Exception:
                continue
        return None

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        parts = re.findall(
            r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
            r"|[A-Za-z0-9_]+"
            r"|[^\s]",
            text,
        )
        estimate = 0
        for part in parts:
            if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", part):
                estimate += 1
            elif part.isascii() and re.fullmatch(r"[A-Za-z0-9_]+", part):
                estimate += max(1, math.ceil(len(part) / 4))
            else:
                estimate += 1
        return max(1, int(math.ceil(estimate * 1.1)))

    def _extract_keywords_live(
        self,
        text: str,
        n_keywords: int,
        progress_callback: ProgressCallback | None = None,
    ) -> list[str]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for live summary mode") from exc

        client = OpenAI(
            api_key=self.config.summary_api_key,
            base_url=self.config.summary_base_url,
        )
        prompt = self._build_prompt(text, n_keywords)

        if progress_callback:
            progress_callback(0.4, f"Calling summary model: {self.config.summary_model_name}")

        try:
            response = client.chat.completions.create(
                model=self.config.summary_model_name,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.summary_temperature,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    },
                },
            )
        except Exception as exc:
            context_limit_error = self._parse_context_length_error(exc)
            if context_limit_error is not None:
                raise context_limit_error from exc
            raise RuntimeError(
                "Failed to reach summary model endpoint "
                f"{self.config.summary_base_url} for model {self.config.summary_model_name}. "
                f"{endpoint_runtime_hint(self.config, 'summary')}"
                f"{endpoint_client_hint(self.config.summary_base_url)} "
                f"Original error: {exc}"
            ) from exc
        raw_content = response.choices[0].message.content or ""

        if progress_callback:
            progress_callback(0.8, "Parsing summary response")

        try:
            payload = extract_json_fragment(raw_content)
            if payload is None or "keywords" not in payload:
                raise RuntimeError("Summary model did not return parsable keywords JSON")

            raw_keywords = payload.get("keywords", [])
            if not isinstance(raw_keywords, list):
                raise RuntimeError("Summary model returned invalid keywords payload")
            keywords = [collapse_whitespace(str(item)) for item in raw_keywords if str(item).strip()]
            if not keywords:
                raise RuntimeError("Summary model returned empty keyword list")
            return keywords[:n_keywords]
        except RuntimeError as exc:
            raise ModelResponseFormatError(
                step="summary",
                message=str(exc),
                raw_response=raw_content,
                model_name=self.config.summary_model_name,
            ) from exc

    @staticmethod
    def _parse_context_length_error(exc: Exception) -> "_SummaryContextLengthExceededError | None":
        message = str(exc)
        max_context_match = re.search(r"maximum context length is (\d+) tokens", message)
        if not max_context_match:
            return None

        input_tokens_match = re.search(r"prompt contains at least (\d+) input tokens", message)
        input_tokens = int(input_tokens_match.group(1)) if input_tokens_match else None
        return _SummaryContextLengthExceededError(
            max_context_len=int(max_context_match.group(1)),
            input_tokens=input_tokens,
        )


class _SummaryContextLengthExceededError(RuntimeError):
    def __init__(self, *, max_context_len: int, input_tokens: int | None) -> None:
        super().__init__("Summary input exceeded model context length")
        self.max_context_len = max_context_len
        self.input_tokens = input_tokens
