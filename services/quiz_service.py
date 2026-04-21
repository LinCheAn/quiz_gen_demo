from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable

from services.summary_service import (
    collapse_whitespace,
    endpoint_client_hint,
    endpoint_runtime_hint,
)
from services.transformers_backend import TransformersTextGenerationBackend
from utils.config import AppConfig
from utils.errors import ModelResponseFormatError
from utils.schemas import QuizQuestion, QuizResult


ProgressCallback = Callable[[float, str], None]


class QuizService:
    FULL_QUESTION_SYSTEM_MESSAGE = (
        "You are a professional educator, you need to generate a multiple-choice question by "
        "English based on the following key course content and reference paragraphs, with 4 "
        "options each, and indicate the correct answer."
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate_quiz(
        self,
        context_chunks: list[str],
        variant: int = 0,
        question_count: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> QuizResult:
        return self._generate_live(
            context_chunks,
            variant=variant,
            options_only_for=None,
            question_count=question_count,
            progress_callback=progress_callback,
        )

    def regenerate_full(
        self,
        context_chunks: list[str],
        variant: int = 1,
        question_count: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> QuizResult:
        return self.generate_quiz(
            context_chunks=context_chunks,
            variant=variant,
            question_count=question_count,
            progress_callback=progress_callback,
        )

    def regenerate_options_only(
        self,
        existing_questions: list[QuizQuestion] | None,
        context_chunks: list[str],
        question_stems: list[str] | None = None,
        variant: int = 1,
        progress_callback: ProgressCallback | None = None,
    ) -> QuizResult:
        normalized_stems = self._resolve_question_stems(
            existing_questions=existing_questions,
            question_stems=question_stems,
        )
        return self._generate_live(
            context_chunks=context_chunks,
            variant=variant,
            options_only_for=normalized_stems,
            progress_callback=progress_callback,
        )

    def _generate_live(
        self,
        context_chunks: list[str],
        variant: int,
        options_only_for: list[str] | None,
        question_count: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> QuizResult:
        del variant

        if not context_chunks:
            raise ValueError("context_chunks must not be empty")

        client = None
        if self.config.inference_backend == "vllm":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("openai package is required for live quiz mode") from exc
            client = OpenAI(api_key=self.config.quiz_api_key, base_url=self.config.quiz_base_url)
        references = self._format_reference_paragraphs(context_chunks)

        if options_only_for:
            if progress_callback:
                progress_callback(0.1, "Calling quiz model with continuation generation")
            questions, raw_response = self._generate_options_only_questions(
                client,
                references,
                options_only_for,
                progress_callback=progress_callback,
            )
            generation_mode = "options_only"
        else:
            if progress_callback:
                progress_callback(0.1, "Calling quiz model for full question generation")
            questions, raw_response = self._generate_full_questions(
                client,
                references,
                question_count=question_count,
                progress_callback=progress_callback,
            )
            generation_mode = "full"

        if progress_callback:
            progress_callback(0.9, "Quiz generation completed")

        return QuizResult(
            questions=questions,
            model=self.config.quiz_model_name,
            generation_mode=generation_mode,
            raw_response=raw_response,
        )

    def _generate_full_questions(
        self,
        client: Any,
        references: str,
        question_count: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[QuizQuestion], list[dict[str, Any]]]:
        prompt = self._build_full_quiz_prompt(references)
        total_questions = max(1, question_count if question_count is not None else self.config.quiz_question_count)
        questions: list[QuizQuestion] = []
        raw_responses: list[dict[str, Any]] = []

        for question_index in range(1, total_questions + 1):
            if progress_callback:
                progress = 0.15 + (0.65 * ((question_index - 1) / total_questions))
                progress_callback(progress, f"Generating full question {question_index}/{total_questions}")
            question, raw_response = self._generate_single_full_question(client, prompt, question_index)
            questions.append(question)
            raw_responses.append(raw_response)

        return questions, raw_responses

    def _generate_options_only_questions(
        self,
        client: Any,
        references: str,
        question_stems: list[str],
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[QuizQuestion], list[dict[str, Any]]]:
        prompt = self._build_full_quiz_prompt(references)
        total_questions = len(question_stems)
        questions: list[QuizQuestion] = []
        raw_responses: list[dict[str, Any]] = []

        for question_index, question_stem in enumerate(question_stems, start=1):
            if progress_callback:
                progress = 0.15 + (0.65 * ((question_index - 1) / max(1, total_questions)))
                progress_callback(
                    progress,
                    f"Continuation generating options {question_index}/{total_questions}",
                )
            question, raw_response = self._generate_single_continuation_question(
                client,
                prompt,
                question_stem,
                question_index,
            )
            questions.append(question)
            raw_responses.append(raw_response)

        return questions, raw_responses

    def _generate_single_full_question(
        self,
        client: Any,
        prompt: str,
        question_index: int,
    ) -> tuple[QuizQuestion, dict[str, Any]]:
        messages = [
            {"role": "system", "content": self.FULL_QUESTION_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        request_payload = {
            "model": self.config.quiz_model_name,
            "messages": messages,
            "temperature": self.config.quiz_temperature,
        }
        attempts: list[dict[str, Any]] = []

        for retry_idx in range(1, 4):
            raw_content = ""
            try:
                raw_content = self._request_chat_completion(client, messages)
                question = self._parse_reference_question_response(raw_content)
                return question, {
                    "question_index": question_index,
                    "request_payload": request_payload,
                    "raw_response": raw_content,
                }
            except Exception as exc:
                attempts.append(
                    {
                        "retry_idx": retry_idx,
                        "error_message": str(exc),
                        "request_payload": request_payload,
                        "raw_response": raw_content,
                    }
                )

        raise self._build_generation_failure(
            mode="full",
            question_index=question_index,
            attempts=attempts,
        )

    def _resolve_question_stems(
        self,
        *,
        existing_questions: list[QuizQuestion] | None,
        question_stems: list[str] | None,
    ) -> list[str]:
        normalized_custom_stems = [
            collapse_whitespace(stem)
            for stem in (question_stems or [])
            if collapse_whitespace(stem)
        ]
        if normalized_custom_stems:
            return normalized_custom_stems

        normalized_existing_stems = [
            collapse_whitespace(question.question)
            for question in (existing_questions or [])
            if collapse_whitespace(question.question)
        ]
        if normalized_existing_stems:
            return normalized_existing_stems

        raise ValueError("No question stems found. Generate a quiz first or provide custom questions.")

    def _generate_single_continuation_question(
        self,
        client: Any,
        prompt: str,
        existing_question_text: str,
        question_index: int,
    ) -> tuple[QuizQuestion, dict[str, Any]]:
        assistant_prefix = self._build_assistant_prefix(existing_question_text)
        messages = [
            {"role": "system", "content": self.FULL_QUESTION_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self._format_llama3_prompt(
            self.FULL_QUESTION_SYSTEM_MESSAGE,
            prompt,
            assistant_prefix,
        )
        request_payload = {
            "model": self.config.quiz_model_name,
            "messages": messages,
            "assistant_prefix": assistant_prefix,
            "temperature": self.config.quiz_temperature,
            "max_tokens": 2048,
            "stop": ["<|eot_id|>"],
        }
        if self.config.inference_backend == "vllm":
            request_payload["prompt"] = formatted_prompt
        attempts: list[dict[str, Any]] = []

        for retry_idx in range(1, 4):
            continuation_text = ""
            complete_response = assistant_prefix
            try:
                continuation_text = self._request_completion(
                    client,
                    prompt=formatted_prompt,
                    messages=messages,
                    assistant_prefix=assistant_prefix,
                )
                complete_response = assistant_prefix + continuation_text
                question = self._parse_reference_question_response(
                    complete_response,
                    expected_question=existing_question_text,
                )
                return question, {
                    "question_index": question_index,
                    "request_payload": request_payload,
                    "continuation_response": continuation_text,
                    "complete_response": complete_response,
                }
            except Exception as exc:
                attempts.append(
                    {
                        "retry_idx": retry_idx,
                        "error_message": str(exc),
                        "request_payload": request_payload,
                        "continuation_response": continuation_text,
                        "complete_response": complete_response,
                    }
                )

        raise self._build_generation_failure(
            mode="options_only",
            question_index=question_index,
            attempts=attempts,
            expected_question=existing_question_text,
        )

    def _request_chat_completion(
        self,
        client: Any,
        messages: list[dict[str, str]],
    ) -> str:
        if self.config.inference_backend == "transformers":
            backend = TransformersTextGenerationBackend(self.config, role="quiz")
            try:
                return backend.generate(
                    messages=messages,
                    temperature=self.config.quiz_temperature,
                    max_new_tokens=1024,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to generate quiz output with transformers backend "
                    f"for model {self.config.quiz_model_name}. Original error: {exc}"
                ) from exc

        try:
            response = client.chat.completions.create(
                model=self.config.quiz_model_name,
                messages=messages,
                temperature=self.config.quiz_temperature,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to reach quiz model endpoint "
                f"{self.config.quiz_base_url} for model {self.config.quiz_model_name}. "
                f"{endpoint_runtime_hint(self.config, 'quiz')}"
                f"{endpoint_client_hint(self.config.quiz_base_url)} "
                f"Original error: {exc}"
            ) from exc
        return response.choices[0].message.content or ""

    def _request_completion(
        self,
        client: Any,
        prompt: str,
        *,
        messages: list[dict[str, str]],
        assistant_prefix: str,
    ) -> str:
        if self.config.inference_backend == "transformers":
            backend = TransformersTextGenerationBackend(self.config, role="quiz")
            try:
                return backend.generate(
                    messages=messages,
                    assistant_prefix=assistant_prefix,
                    temperature=self.config.quiz_temperature,
                    max_new_tokens=2048,
                    stop_strings=["<|eot_id|>"],
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to continue quiz generation with transformers backend "
                    f"for model {self.config.quiz_model_name}. Original error: {exc}"
                ) from exc

        try:
            response = client.completions.create(
                model=self.config.quiz_model_name,
                prompt=prompt,
                temperature=self.config.quiz_temperature,
                max_tokens=2048,
                stop=["<|eot_id|>"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to reach quiz model endpoint "
                f"{self.config.quiz_base_url} for model {self.config.quiz_model_name}. "
                f"{endpoint_runtime_hint(self.config, 'quiz')}"
                f"{endpoint_client_hint(self.config.quiz_base_url)} "
                f"Original error: {exc}"
            ) from exc
        return response.choices[0].text or ""

    def _build_generation_failure(
        self,
        *,
        mode: str,
        question_index: int,
        attempts: list[dict[str, Any]],
        expected_question: str | None = None,
    ) -> ModelResponseFormatError:
        details = {
            "mode": mode,
            "question_index": question_index,
            "expected_question": expected_question,
            "attempts": attempts,
        }
        question_label = expected_question or f"question {question_index}"
        return ModelResponseFormatError(
            step="quiz",
            message=f"Quiz {mode} generation failed for {question_label} after {len(attempts)} attempts",
            raw_response=json.dumps(details, ensure_ascii=False, indent=2),
            model_name=self.config.quiz_model_name,
        )

    def _build_full_quiz_prompt(self, references: str) -> str:
        return (
            f"{references}\n\n"
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

    def _build_assistant_prefix(self, question_text: str) -> str:
        return (
            "{\n"
            '    "question": [\n'
            "        {\n"
            f'            "question": {json.dumps(question_text, ensure_ascii=False)},'
        )

    def _format_reference_paragraphs(self, context_chunks: Iterable[str]) -> str:
        references = list(context_chunks)[:5]
        while len(references) < 5:
            references.append("")
        return "\n".join(
            f"Reference paragraph {index}:{chunk}"
            for index, chunk in enumerate(references, start=1)
        )

    @staticmethod
    def _format_llama3_prompt(system_msg: str, user_msg: str, assistant_prefix: str) -> str:
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_prefix}"
        )

    def _parse_reference_question_response(
        self,
        raw_content: str,
        *,
        expected_question: str | None = None,
    ) -> QuizQuestion:
        payload = self._extract_quiz_payload(raw_content)
        return self._parse_reference_question_payload(payload, expected_question=expected_question)

    def _extract_quiz_payload(self, raw_content: str) -> dict[str, object]:
        cleaned = self._strip_code_fence(raw_content).strip()
        if not cleaned:
            raise RuntimeError("Quiz model returned an empty response")

        payload, decode_error = self._load_json_object(cleaned)
        if payload is not None:
            return payload

        fragment = self._extract_balanced_json_object(cleaned)
        if fragment is None:
            if decode_error is not None:
                raise RuntimeError(self._format_json_decode_error(decode_error))
            raise RuntimeError("Quiz model response does not contain a JSON object")

        payload, decode_error = self._load_json_object(fragment)
        if payload is not None:
            return payload
        if decode_error is not None:
            raise RuntimeError(self._format_json_decode_error(decode_error))
        raise RuntimeError("Quiz model did not return a JSON object payload")

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, count=1).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned, count=1).strip()
        return cleaned

    @staticmethod
    def _load_json_object(text: str) -> tuple[dict[str, object] | None, json.JSONDecodeError | None]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, exc
        if isinstance(payload, dict):
            return payload, None
        return None, None

    @staticmethod
    def _extract_balanced_json_object(text: str) -> str | None:
        start_idx: int | None = None
        depth = 0
        in_string = False
        escape = False

        for index, char in enumerate(text):
            if start_idx is None:
                if char == "{":
                    start_idx = index
                    depth = 1
                continue

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : index + 1]

        if start_idx is None:
            return None
        return text[start_idx:]

    @staticmethod
    def _format_json_decode_error(exc: json.JSONDecodeError) -> str:
        return (
            "Quiz response JSON syntax error "
            f"at line {exc.lineno} column {exc.colno}: {exc.msg}"
        )

    def _parse_reference_question_payload(
        self,
        payload: dict[str, object],
        *,
        expected_question: str | None = None,
    ) -> QuizQuestion:
        raw_questions = payload.get("question")
        if not isinstance(raw_questions, list) or not raw_questions:
            raise RuntimeError("Quiz response does not contain a question list")

        first_question = raw_questions[0]
        if not isinstance(first_question, dict):
            raise RuntimeError("Quiz response contains invalid question entries")

        if expected_question is None:
            question_text = collapse_whitespace(str(first_question.get("question", "")))
            if not question_text:
                raise RuntimeError("Quiz response is missing question text")
        else:
            question_text = collapse_whitespace(expected_question)

        options = first_question.get("options", {})
        answer = str(first_question.get("answer", "")).strip().upper()
        explanation = first_question.get("explanation")

        if not isinstance(options, dict):
            raise RuntimeError("Quiz response options must be a dictionary")

        normalized_options = {
            str(key).upper(): collapse_whitespace(str(value))
            for key, value in options.items()
        }
        for option_key in ("A", "B", "C", "D"):
            if option_key not in normalized_options or not normalized_options[option_key]:
                raise RuntimeError(f"Quiz response missing option {option_key}")

        if answer not in {"A", "B", "C", "D"}:
            raise RuntimeError("Quiz response answer must be one of A, B, C, D")

        return QuizQuestion(
            question=question_text,
            options={key: normalized_options[key] for key in ("A", "B", "C", "D")},
            answer=answer,
            explanation=collapse_whitespace(str(explanation)) if explanation else None,
        )
