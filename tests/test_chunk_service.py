from __future__ import annotations

import unittest

from services.chunk_service import ChunkService


class ChunkServiceTest(unittest.TestCase):
    def test_chunking_respects_overlap(self) -> None:
        service = ChunkService()
        text = "abcdefghijklmnopqrstuvwxyz" * 6
        result = service.chunk_text(text, chunk_size=40, overlap=8)

        self.assertGreaterEqual(len(result.chunks), 2)
        self.assertEqual(result.chunks[0].start_char, 0)
        self.assertEqual(result.chunks[1].start_char, 32)
        self.assertEqual(result.chunks[0].text[-8:], result.chunks[1].text[:8])


if __name__ == "__main__":
    unittest.main()
