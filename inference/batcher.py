"""Batch assembler.

Collects tagged chunks from the chunker, groups them into VRAM-safe batches
using the benchmark map's max_safe_batch_size, and manages torch.stack/unstack
with tag preservation for reassembly.
"""

from inference.chunker import ChunkMeta
from profiler.benchmark import get_lookup
from collections import defaultdict


def build_batches(chunks: list[ChunkMeta], benchmark_map: dict) -> list[list[ChunkMeta]]:
    """
    Groups same-shape chunks into VRAM-safe batches using benchmark map batch sizes. 
    Returns a list of batches, where each batch is a list of ChunkMetas that can be torch.stacked together.
    """
    slice_map = defaultdict(list)
    batches = []

    for chunk in chunks:
        h, w = chunk.page_slice.shape[:2]
        slice_map[(h, w)].append(chunk)
    
    ## Run through the dict keys and make batches
    for k, v in slice_map.items():
        h, w = k
        benchmark = get_lookup(w, h, benchmark_map)
        batch_size = benchmark["max_safe_batch_size"]

        for i in range(0, len(v), batch_size):
            batches.append(v[i:i+batch_size])

    return batches
