from __future__ import annotations

from services.summary_service import collapse_whitespace
from utils.schemas import ChunkResult, TextChunk


class ChunkService:
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> ChunkResult:
        normalized = collapse_whitespace(text)
        if not normalized:
            return ChunkResult(
                chunks=[],
                strategy="fixed_window_char",
                chunk_size=chunk_size,
                overlap=overlap,
            )

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        step = chunk_size - overlap
        cursor = 0
        chunks: list[TextChunk] = []
        while cursor < len(normalized):
            end = min(len(normalized), cursor + chunk_size)
            chunk_text = normalized[cursor:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"chunk_{len(chunks) + 1:03d}",
                        text=chunk_text,
                        start_char=cursor,
                        end_char=end,
                    )
                )
            if end >= len(normalized):
                break
            cursor += step

        return ChunkResult(
            chunks=chunks,
            strategy="fixed_window_char",
            chunk_size=chunk_size,
            overlap=overlap,
        )
