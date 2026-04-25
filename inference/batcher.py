"""Batch assembler.

Collects tagged chunks from the chunker, groups them into VRAM-safe batches
using the benchmark map's max_safe_batch_size, and manages torch.stack/unstack
with tag preservation for reassembly.
"""
