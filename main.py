import argparse
from core import config
from pipeline import batch as b
from pathlib import Path


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
    main()
    