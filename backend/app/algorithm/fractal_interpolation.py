from __future__ import annotations
import numpy as np
from .subdivision import subdivide_triangle

def _sample(image: np.ndarray, point: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = np.rint(point).astype(int)
    return image[np.clip(y, 0, h-1), np.clip(x, 0, w-1), :3]

def calculate_vertical_scaling(vertices: np.ndarray, image: np.ndarray, eta: float = .78, alpha_max: float = .85) -> float:
    colors = np.asarray([_sample(image, p) for p in vertices])
    residual = float(np.mean(np.linalg.norm(colors - colors.mean(axis=0), axis=1)))
    return float(np.clip(eta * residual / (residual + 1e-8), 0, alpha_max))

def calculate_q(vertices: np.ndarray, image: np.ndarray, alpha: float) -> dict:
    colors = np.asarray([_sample(image, p) for p in vertices])
    return {"colors": colors, "alpha": alpha}

def refine_fractal_triangle(node: dict, depth: int, output: list[dict], budget: int = 5000) -> None:
    def walk(vertices: np.ndarray, remaining: int):
        if len(output) >= budget: return
        if remaining == 0:
            alpha = calculate_vertical_scaling(vertices, node["image"])
            output.append({"vertices": vertices, **calculate_q(vertices, node["image"], alpha)})
            return
        for child in subdivide_triangle(vertices): walk(child, remaining - 1)
    walk(node["vertices"], depth)

