import argparse
from core import config
from pipeline import batch as b
from pathlib import Path
from profiler.cache import cache_check, save_cache
from profiler.scanner import scan_library
from profiler.runner import run_profiler
from profiler.benchmark import build_map, get_lookup


def run_profiler_pipeline():
    if cache_check():
        print("Cache found, skipping profiling.")
        return

    library_dir = Path("Manwhas")
    model_path = Path("models/4x-UltraSharp.pth")

    print("Scanning library...")
    unique_widths, max_height = scan_library(library_dir)
    print(f"Widths found: {unique_widths}, Max height: {max_height}")

    print("Running profiler...")
    raw_results = run_profiler(unique_widths, max_height, model_path)

    print("Building benchmark map...")
    benchmark_map = build_map(raw_results)
    save_cache(benchmark_map)
    print("Cache saved.")

    print("Test lookups:")
    test_cases = [(720, 512), (720, 800), (720, 3200), (800, 1024)]
    for w, h in test_cases:
        result = get_lookup(w, h, benchmark_map)
        print(f"  {w}x{h} -> {result}")


def main():
    parser = argparse.ArgumentParser(description="Manwha HD Upscaler MVP")

    parser.add_argument('--comic-folder', type = str, help = "Path to the comic")
    parser.add_argument('--scale', type = int, default = 2, help = "Upscale factor (2 or 4)")
    parser.add_argument('--format', type = str, default='jpg', help="Extenstion to save images in")
    parser.add_argument('--quality', type = int, default = 92, help="JPG save quality default 92")
    parser.add_argument('--limit', type=int, default=0, help="Number of chapters to upscale, Default 0 means all.")

    args = parser.parse_args()

    output_config = config.OutputConfig(
        format=args.format,
        quality=args.quality,
        limit=args.limit
    )

    engine_config = config.EngineConfig(
        scale=args.scale,
    )

    directory_config = config.DirectoryConfig(
        manwha_dir=Path(args.comic_folder)
    )

    pipeline = config.PipelineConfig(dirs=directory_config, engine=engine_config, output=output_config)

    b.process_all_chapters(folder_path=Path(args.comic_folder), pipeline=pipeline)

if __name__ == "__main__":
    run_profiler_pipeline()
    ##main()
