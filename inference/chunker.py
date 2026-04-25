"""Variable-selection chunking strategy.

Queries the benchmark map to determine the largest safe chunk height per page
per machine, then slices pages into GPU-ready chunks with overlap for feathered
stitching.
"""
