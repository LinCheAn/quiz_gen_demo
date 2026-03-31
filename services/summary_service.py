from __future__ import annotations

import json
import re
from typing import Callable

from utils.config import AppConfig
from utils.errors import ModelResponseFormatError
from utils.schemas import KeywordResult


ProgressCallback = Callable[[float, str], None]

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

    def extract_keywords(
        self,
        text: str,
        n_keywords: int,
        progress_callback: ProgressCallback | None = None,
    ) -> KeywordResult:
        if progress_callback:
            progress_callback(0.1, "Preparing summary keywords")

        keywords = self._extract_keywords_live(text, n_keywords, progress_callback)
        return KeywordResult(keywords=keywords, model=self.config.summary_model_name)

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
        prompt = f"""
lesson content:
{text}

Output in following json format:
{{
  "keywords": ["keyword 1", "keyword 2", "keyword 3"]
}}

Output only JSON format, no other explanation.
Return up to {n_keywords} keywords.
""".strip()

        if progress_callback:
            progress_callback(0.4, f"Calling summary model: {self.config.summary_model_name}")

        try:
            response = client.chat.completions.create(
                model=self.config.summary_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a college student, you need to extract the most important keywords from the provided lesson content.",
                    },
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
