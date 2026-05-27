"""Per-page chunk collector.

Tracks chunk arrival for each page in flight and triggers stitching when all
expected chunks have returned from the GPU.
"""

import numpy as np
from inference.chunker import ChunkMeta
from pipeline.reassembler import reassemble


class PageBuffer:
    def __init__(self, scale: int = 4):
        self.scale = scale
        self._pages: dict = {}

    def add(self, upscaled: np.ndarray, meta: ChunkMeta) -> np.ndarray | None:
        """Stores an upscaled chunk. Returns the reassembled page if all chunks have arrived, None otherwise."""
        if meta.page_id not in self._pages:
            self._pages[meta.page_id] = []

        self._pages[meta.page_id].append((upscaled, meta))

        if len(self._pages[meta.page_id]) == meta.total_chunks:
            sorted_chunks = sorted(self._pages[meta.page_id], key=lambda x: x[1].chunk_index)
            result = reassemble(sorted_chunks, self.scale)
            del self._pages[meta.page_id]
            return result

        return None

    def is_empty(self) -> bool:
        """Returns True if all pages have been completed and cleared."""
        return len(self._pages) == 0