"""Scans CBZ files for unique page dimensions without decoding pixels.

Returns the set of widths and maximum height encountered across all source files.
"""

from pathlib import Path
import zipfile
import time
import imagesize
import io

def scan_library(library_dir: Path) -> tuple[set[int], int]:
    """Returns unique page widths and global max height across all CBZ files in library_dir."""
    start = time.perf_counter()

    glob_start = time.perf_counter()
    cbz_paths = list(Path(library_dir).rglob("*.cbz"))
    glob_elapsed = time.perf_counter() - glob_start
    print(f"Found {len(cbz_paths)} CBZ files in {glob_elapsed:.3f}s")

    width_set = set()
    max_height = 0
    read_total = 0.0

    for path in cbz_paths:
        read_start = time.perf_counter()
        dimension_lists = _read_cbz_dimensions(path)
        read_total += time.perf_counter() - read_start
        for dimensions in dimension_lists:
            w, h = dimensions
            if w >= h:
                continue
            if h > 8000 or w > 1100:
                continue
            width_set.add(w)
            max_height = max(h, max_height)

    total_elapsed = time.perf_counter() - start
    print(f"Processed {len(cbz_paths)} CBZ files | _read_cbz_dimensions: {read_total:.3f}s | scan_library total: {total_elapsed:.3f}s")

    return width_set, max_height

def _read_cbz_dimensions(cbz_path: Path) -> list[tuple[int, int]]:
    """Extracts (width, height) from image headers in a CBZ without decoding pixels."""
    dimensions_cbz = []
    with zipfile.ZipFile(cbz_path) as z:
        include_filter = ('.jpg', '.png', '.jpeg', '.webp')
        exclude_filter = ('__MACOSX', '.')
        file_list = sorted([f for f in z.namelist() if f.endswith(include_filter) and not f.startswith(exclude_filter)])
        for name in file_list:
            with z.open(name) as raw:
                buf = io.BytesIO(raw.read(65536))
            dimensions_cbz.append(imagesize.get(buf))

    return dimensions_cbz