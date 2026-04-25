"""Sharpening algorithms.

Currently wraps the unsharp mask from core/engine.py (_post_sharpen, lines 120-127):
GaussianBlur followed by cv2.addWeighted to amplify high-frequency detail softened
by upscaling. Implementation lives in core/engine.py and will be migrated here in
a later step.

Phase 2 target: variance-weighted edge sharpening that distinguishes character
outlines from speed lines and flat fills.
"""
