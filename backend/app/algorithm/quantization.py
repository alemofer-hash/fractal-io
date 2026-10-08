from __future__ import annotations
import cv2
import numpy as np

def quantize_colors(image: np.ndarray, clusters: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Quantize visible pixels in perceptual CIE Lab, never in RGB classes alone."""
    rgb = image[..., :3]
    alpha = image[..., 3]
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
    samples = lab.reshape(-1, 3).astype(np.float32)
    k = max(1, min(max(10, clusters), len(samples), 24))
    cv2.setRNGSeed(7)
    _, labels, centers = cv2.kmeans(samples, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(rgb.shape[:2])
    labels[alpha < 0.02] = -1
    return labels, centers

