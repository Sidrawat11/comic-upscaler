"""Sharpening algorithms.

Currently wraps the unsharp mask from core/engine.py (_post_sharpen, lines 120-127):
GaussianBlur followed by cv2.addWeighted to amplify high-frequency detail softened
by upscaling. Implementation lives in core/engine.py and will be migrated here in
a later step.

Phase 2 target: variance-weighted edge sharpening that distinguishes character
outlines from speed lines and flat fills.
"""

import cv2
import numpy as np


def sharpen(image: np.ndarray, strength: float = 0.3, radius: float = 1.0) -> np.ndarray:
    if strength == 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
