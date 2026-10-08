from __future__ import annotations
import numpy as np

def subdivide_triangle(vertices: np.ndarray) -> list[np.ndarray]:
    a, b, c = vertices
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    return [np.asarray([a, ab, ca]), np.asarray([ab, b, bc]), np.asarray([ca, bc, c]), np.asarray([ab, bc, ca])]

