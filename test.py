"""Standalone test: slice one 693x512 chunk from a test page and upscale it."""

import cv2
from pathlib import Path
from core.model_loader import load_rrdbnet
from inference.engine import upscale

model_path = Path("models/4x-UltraSharp.pth")
test_image_path = Path("test/060__001.jpg")

print("Loading model...")
model = load_rrdbnet(model_path)

print("Reading test image...")
image = cv2.imread(str(test_image_path))
print(f"Full image shape: {image.shape}")

# Take first 512 rows — 693x512 is within VRAM budget
chunk = image[-512:, :]
print(f"Chunk shape: {chunk.shape}")

print("Upscaling...")
result = upscale(chunk, model, scale=4)
print(f"Output shape: {result.shape}")

output_path = Path("test/output_chunk_test.png")
cv2.imwrite(str(output_path), result)
print(f"Saved to {output_path}")