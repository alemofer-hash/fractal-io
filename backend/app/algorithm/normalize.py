from __future__ import annotations
import numpy as np
from PIL import Image

def normalize_image(image: Image.Image, max_dimension: int = 512) -> np.ndarray:
    """Return a bounded RGBA float image in [0,1], preserving aspect ratio."""
    rgba = image.convert("RGBA")
    width, height = rgba.size
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1:
        rgba = rgba.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    return np.asarray(rgba, dtype=np.float32) / 255.0

