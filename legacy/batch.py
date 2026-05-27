"""Batch pipeline for upscaling all CBZ chapters in a directory."""

from legacy import engine as en
from core import config, extractor, packager
import os
from pathlib import Path
import logging
import time


def setup_logger(log_dir):
    """Configure logging to both file (DEBUG) and console (INFO)."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("ManwhaHDFyer")
    logger.setLevel(logging.DEBUG)

    # Don't add handlers if they already exist (prevents duplicates)
    if logger.handlers:
        return logger

    # File — everything, with timestamps
    fh = logging.FileHandler(os.path.join(log_dir, "upscale.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console — INFO and above, clean format
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def process_all_chapters(folder_path, pipeline: config.PipelineConfig):
    """Upscale every CBZ in folder_path and write results to the configured output directory."""
    logger = setup_logger(pipeline.dirs.log_dir if hasattr(pipeline.dirs, 'log_dir') else 'logs')

    engine = en.Engine(pipeline.engine, pipeline.dirs)
    cbz_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.cbz')])
    os.makedirs(pipeline.dirs.result_dir, exist_ok=True)

    logger.info(f"Found {len(cbz_files)} chapters in {folder_path}")
    total_start = time.time()

    for idx, n in enumerate(cbz_files):
        if pipeline.output.limit != 0 and idx + 1 > pipeline.output.limit:
            break
        chapter_path = os.path.join(folder_path, n)
        logger.info(f"\n[{idx + 1}/{len(cbz_files)}] {n}")
        output_path = os.path.join(pipeline.dirs.result_dir, n)
        if os.path.exists(output_path):
            logger.info(f"  Already processed, skipping")
            continue
        process_chapter(chapter_path, pipeline, engine, logger)

    total_time = time.time() - total_start
    logger.info(f"\nComplete: {len(cbz_files)} chapters in {total_time:.0f}s ({total_time/60:.1f} min)")
    if cbz_files:
        logger.info(f"Average: {total_time/len(cbz_files):.0f}s per chapter")

def process_chapter(chapter_path, pipeline: config.PipelineConfig, engine: en.Engine, logger):
    """Upscale a single CBZ chapter, skipping the trailing pages defined in pipeline config."""
    chapter_start = time.time()
    img_count = extractor.count_images(chapter_path)
    chapter_name = Path(chapter_path).name

    output_directory = os.path.join(pipeline.dirs.result_dir, chapter_name)

    logger.info(f"  {img_count} pages (skipping last {pipeline.output.skip_final})")

    with packager.Packager(output_directory, pipeline.output.format, pipeline.output.quality) as pkg:
        for i, (name, image, raw_bytes) in enumerate(extractor.extract_images(chapter_path)):
            page_start = time.time()

            if (img_count - i) <= pipeline.output.skip_final:
                pkg.add_raw(name, raw_bytes)
                logger.debug(f"    {name}: skipped")
            else:
                upscaled_image = engine.upscale(image)
                pkg.add_image(name, upscaled_image)
                page_time = time.time() - page_start
                logger.debug(f"    {name}: {page_time:.1f}s")

    chapter_time = time.time() - chapter_start
    logger.info(f"  Done in {chapter_time:.0f}s ({chapter_time/60:.1f} min)")

