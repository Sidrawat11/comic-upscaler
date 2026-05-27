"""Standalone test: slice one 693x512 chunk from a test page and upscale it."""

"""Test chunker against the 690x4363 test page."""

import cv2
from pathlib import Path
from profiler.cache import load_cache
from profiler.benchmark import get_max_chunk_height
from inference.chunker import chunk_page

test_image_path = Path("test/060__001.jpg")
image = cv2.imread(str(test_image_path))
print(f"Full image: {image.shape}")

benchmark_map = load_cache()

chunks = chunk_page(image, 0, benchmark_map)
print(f"Total chunks: {len(chunks)}")

for c in chunks:
    print(f"  Chunk {c.chunk_index}: y={c.y_start}->{c.y_end}, shape={c.page_slice.shape}, overlap_top={c.overlap_top}, overlap_bottom={c.overlap_bottom}")