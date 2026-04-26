"""Persistent benchmark map.

Stores profiling results keyed by GPU model and exposes a three-case lookup
(exact, interpolate, extrapolate) for the chunker to query at runtime.
"""

import math
import torch


def build_map(raw_results: dict) -> dict:
    """Converts raw profiler output into a structured benchmark map with max_safe_batch_size per valid entry."""
    available_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
    entries = []

    for value in raw_results["entries"]:
        entry = {"width": value["width"], "height": value["height"], "valid": value["valid"]}

        if value["valid"]:
            entry["peak_vram_mb"] = value["peak_vram_mb"]
            entry["avg_duration_ms"] = value["avg_time_ms"]
            entry["max_safe_batch_size"] = math.floor((available_vram_mb * 0.85) / value["peak_vram_mb"])
        else:
            entry["reason"] = value.get("reason", "OOM")

        entries.append(entry)

    return {
        "widths_seen": raw_results["page_widths_seen"],
        "max_height": raw_results["max_page_height"],
        "available_vram": available_vram_mb,
        "entries": entries,
    }


def get_lookup(width: int, height: int, benchmark_map: dict) -> dict:
    """Three-case lookup (exact, interpolate, extrapolate) returning the best matching benchmark entry for a page shape."""
    benchmark_entries = benchmark_map["entries"]
    max_height = benchmark_map["max_height"]
    seen_width_set = set(benchmark_map["widths_seen"])
    available_vram_mb = benchmark_map["available_vram"]
    width_to_use, need_extrapolation = width, False

    if width not in seen_width_set:
        width_to_use = min(seen_width_set, key=lambda val: abs(val - width))
        need_extrapolation = True

    filtered_entries_by_width = [item for item in benchmark_entries if item["width"] == width_to_use]

    if not filtered_entries_by_width:
        return None

    matched_entries = [val for val in filtered_entries_by_width if val["height"] == height]
    matched_entry = dict(matched_entries[0]) if matched_entries else None

    if matched_entry:
        if need_extrapolation:
            matched_entry["max_safe_batch_size"] = math.floor(matched_entry["max_safe_batch_size"] * 0.75)
        return matched_entry

    if max_height < height:
        max_entry = max(filtered_entries_by_width, key=lambda x: x["height"])
        entry = dict(max_entry)
        if max_entry["valid"]:
            entry["max_safe_batch_size"] = math.floor(entry["max_safe_batch_size"] * 0.75)
        return entry
    else:
        valid_entries = sorted([e for e in filtered_entries_by_width if e["valid"]], key=lambda x: x["height"])
        lower = max((e for e in valid_entries if e["height"] < height), key=lambda x: x["height"], default=None)
        upper = min((e for e in valid_entries if e["height"] > height), key=lambda x: x["height"], default=None)

        if not lower:
            entry = dict(valid_entries[0])
            entry["max_safe_batch_size"] = math.floor(entry["max_safe_batch_size"] * 0.9)
            return entry

        if not upper:
            entry = dict(valid_entries[-1])
            entry["max_safe_batch_size"] = math.floor(entry["max_safe_batch_size"] * 0.9)
            return entry

        avg_time = (lower["avg_duration_ms"] + upper["avg_duration_ms"]) / 2
        avg_vram = (lower["peak_vram_mb"] + upper["peak_vram_mb"]) / 2
        max_batch_size = math.floor(math.floor((available_vram_mb * 0.85) / avg_vram) * 0.9)

        return {
            "width": width,
            "height": height,
            "valid": True,
            "peak_vram_mb": avg_vram,
            "avg_duration_ms": avg_time,
            "max_safe_batch_size": max_batch_size,
        }