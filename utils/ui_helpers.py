from __future__ import annotations

from utils.schemas import QuizQuestion


def parse_custom_question_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def append_custom_question(existing_text: str, question_text: str) -> str:
    normalized_question = question_text.strip()
    if not normalized_question:
        return existing_text

    normalized_existing = existing_text.rstrip()
    if not normalized_existing:
        return normalized_question
    return f"{normalized_existing}\n{normalized_question}"


def format_question_markdown(question: QuizQuestion, index: int | None = None) -> str:
    title = question.question if index is None else f"Q{index}. {question.question}"
    lines = [f"### {title}", ""]

    for option_key in ("A", "B", "C", "D"):
        lines.append(f"- {option_key}. {question.options[option_key]}")

    lines.append("")
    lines.append(f"**Answer:** `{question.answer}`")

    if question.explanation:
        lines.append(f"**Explanation:** {question.explanation}")

    return "\n".join(lines)
