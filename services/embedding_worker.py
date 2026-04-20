from __future__ import annotations

import json
import sys
from math import sqrt


def _vector_norm(vector: object) -> float:
    values = [float(value) for value in vector]  # type: ignore[arg-type]
    return sqrt(sum(value * value for value in values))


def _cosine_similarity(left: object, right: object, left_norm: float | None = None) -> float:
    left_values = [float(value) for value in left]  # type: ignore[arg-type]
    right_values = [float(value) for value in right]  # type: ignore[arg-type]
    numerator = sum(left_value * right_value for left_value, right_value in zip(left_values, right_values))
    effective_left_norm = left_norm if left_norm is not None else _vector_norm(left_values)
    right_norm = _vector_norm(right_values)
    if not effective_left_norm or not right_norm:
        return 0.0
    return numerator / (effective_left_norm * right_norm)


def main() -> int:
    try:
        from FlagEmbedding import FlagModel
    except ImportError as exc:
        raise RuntimeError("FlagEmbedding package is required in the embedding runtime environment") from exc

    payload = json.load(sys.stdin)
    chunks = payload.get("chunks", [])
    keywords = [str(item).strip() for item in payload.get("keywords", []) if str(item).strip()]
    top_k = int(payload.get("top_k", 5))
    model_name = str(payload.get("model_name", "")).strip()
    use_fp16 = bool(payload.get("use_fp16", True))

    if not model_name:
        raise RuntimeError("Embedding worker did not receive a model_name")
    if not isinstance(chunks, list):
        raise RuntimeError("Embedding worker expects a list of chunks")

    if not chunks:
        json.dump({"results": []}, sys.stdout, ensure_ascii=False)
        return 0

    model = FlagModel(model_name, use_fp16=use_fp16)
    chunk_texts = [str(chunk.get("text", "")) for chunk in chunks]
    query_text = ", ".join(keywords)
    query_vector = model.encode(query_text)
    chunk_vectors = model.encode(chunk_texts)
    query_norm = _vector_norm(query_vector)

    scored: list[dict[str, object]] = []
    for chunk, chunk_vector in zip(chunks, chunk_vectors):
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        text = str(chunk.get("text", ""))
        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword and keyword.lower() in text.lower()
        ]
        score = _cosine_similarity(query_vector, chunk_vector, left_norm=query_norm)
        scored.append(
            {
                "rank": 0,
                "chunk_id": chunk_id,
                "text": text,
                "score": round(float(score), 4),
                "matched_keywords": matched_keywords,
            }
        )

    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    for rank, item in enumerate(scored[: min(top_k, len(scored))], start=1):
        item["rank"] = rank

    json.dump({"results": scored[: min(top_k, len(scored))]}, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
