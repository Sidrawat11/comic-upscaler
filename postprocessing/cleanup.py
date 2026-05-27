"""Post-inference pixel cleanup.

Clamps near-black pixels to pure black to eliminate model hallucination patchiness
on solid regions. Currently implemented in core/engine.py (_post_sharpen, lines
128-134): pixels where all channels are below 15 are zeroed out. Implementation
lives in core/engine.py and will be migrated here in a later step.
"""

import numpy as np


def cleanup_near_black(image: np.ndarray, threshold: int = 15) -> np.ndarray:
    mask = np.all(image < threshold, axis=2)
    result = image.copy()
    result[mask] = 0
    return result
