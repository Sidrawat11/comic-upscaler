"""Post-inference pixel cleanup.

Clamps near-black pixels to pure black to eliminate model hallucination patchiness
on solid regions. Currently implemented in core/engine.py (_post_sharpen, lines
128-134): pixels where all channels are below 15 are zeroed out. Implementation
lives in core/engine.py and will be migrated here in a later step.
"""
