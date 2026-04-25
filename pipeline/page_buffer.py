"""Per-page chunk collector.

Tracks chunk arrival for each page in flight and triggers stitching when all
expected chunks have returned from the GPU.
"""
