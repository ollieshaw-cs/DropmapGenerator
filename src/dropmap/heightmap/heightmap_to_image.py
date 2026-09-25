import numpy as np
from PIL import Image
from pathlib import Path

# ============================================================
# Paths
# ============================================================

NPY_PATH = Path("data/heightmap.npy")
PNG_PATH = Path("assets/images/Heightmap.png")

print("Loading heightmap...")
heightmap = np.load(NPY_PATH)

if heightmap.ndim != 2:
    raise ValueError("Heightmap must be a 2D array")

nan_mask = np.isnan(heightmap)
valid = heightmap[~nan_mask]

min_h = valid.min()
max_h = valid.max()

print(f"Min height: {min_h:.3f}")
print(f"Max height: {max_h:.3f}")

heightmap = heightmap.copy()
heightmap[nan_mask] = min_h

normalized = (heightmap - min_h) / (max_h - min_h)
normalized = np.clip(normalized, 0.0, 1.0)

height_16bit = (normalized * 65535).astype(np.uint16)

height_16bit = np.flipud(height_16bit)  # flip along vertical axis

PNG_PATH.parent.mkdir(parents=True, exist_ok=True)

img = Image.fromarray(height_16bit, mode="I;16")
img.save(PNG_PATH)

print(f"Saved 16-bit heightmap → {PNG_PATH.resolve()}")