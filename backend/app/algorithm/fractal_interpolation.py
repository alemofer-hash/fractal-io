from __future__ import annotations
import heapq
import time
from itertools import count
import numpy as np
from .subdivision import subdivide_triangle

_SAMPLES = np.asarray([[1/3,1/3,1/3],[.5,.5,0],[.5,0,.5],[0,.5,.5],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]], dtype=np.float32)

def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear RGB(A) sampling; avoids blocky nearest-pixel triangle colours."""
    h, w = image.shape[:2]; x = np.clip(points[:, 0], 0, w-1); y = np.clip(points[:, 1], 0, h-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); x1, y1 = np.minimum(x0+1,w-1), np.minimum(y0+1,h-1)
    fx, fy = (x-x0)[:,None], (y-y0)[:,None]
    return image[y0,x0]*(1-fx)*(1-fy) + image[y0,x1]*fx*(1-fy) + image[y1,x0]*(1-fx)*fy + image[y1,x1]*fx*fy

def _barycentric(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    a, b, c = vertices; matrix = np.column_stack((a-c, b-c)); det = float(np.linalg.det(matrix))
    if abs(det) < 1e-8: return np.tile(np.asarray([1.,0.,0.]), (len(points),1))
    uv = np.linalg.solve(matrix, (points-c).T).T
    return np.column_stack((uv[:,0], uv[:,1], 1-uv[:,0]-uv[:,1]))

def _points(vertices: np.ndarray, weights: np.ndarray = _SAMPLES) -> np.ndarray: return weights @ vertices

def _gradient_at(gradient: np.ndarray, points: np.ndarray) -> float:
    h, w = gradient.shape; x = np.clip(np.rint(points[:,0]).astype(int),0,w-1); y = np.clip(np.rint(points[:,1]).astype(int),0,h-1)
    return float(gradient[y, x].mean())

def _node(vertices: np.ndarray, image: np.ndarray, gradient: np.ndarray, plane_id: int, depth: int, parent_colors: np.ndarray | None = None, parent_texture: np.ndarray | None = None) -> dict:
    colors = _sample(image, vertices)[:, :3]; alpha_values = _sample(image, vertices)[:, 3]
    sample_points = _points(vertices); weights = _barycentric(vertices, sample_points)
    predicted = weights @ colors; observed = _sample(image, sample_points)[:, :3]; residuals = observed - predicted
    color_error = float(np.mean(np.sum(residuals * residuals, axis=1))); edge_error = _gradient_at(gradient, sample_points); center_texture = residuals[0]
    if parent_colors is None: parent_colors = colors.copy()
    rho = color_error; fractal_alpha = float(np.clip(.78 * rho / (rho + 1e-8), 0, .85)); q_colors = colors - fractal_alpha * parent_colors
    return {"vertices": vertices.astype(np.float32), "colors": colors, "opacity": float(alpha_values.mean()), "depth": depth, "plane_id": plane_id,
            "error": color_error, "edge_error": edge_error, "score": color_error + .085 * edge_error, "alpha": fractal_alpha,
            "q_colors": q_colors, "parent_colors": parent_colors, "parent_texture": center_texture if parent_texture is None else (parent_texture + center_texture) * .5}

def _children(node: dict, image: np.ndarray, gradient: np.ndarray) -> list[dict]:
    result = []
    for child_vertices in subdivide_triangle(node["vertices"]):
        inherited_parent = _barycentric(node["vertices"], child_vertices) @ node["colors"]
        result.append(_node(child_vertices, image, gradient, node["plane_id"], node["depth"] + 1, inherited_parent, node["parent_texture"]))
    return result

def calculate_vertical_scaling(vertices: np.ndarray, image: np.ndarray, eta: float = .78, alpha_max: float = .85) -> float:
    colors = _sample(image, vertices)[:, :3]; rho = float(np.var(colors, axis=0).mean())
    return float(np.clip(eta * rho / (rho + 1e-8), 0, alpha_max))

def calculate_q(vertices: np.ndarray, image: np.ndarray, alpha: float) -> dict:
    colors = _sample(image, vertices)[:, :3]
    return {"colors": colors, "q_colors": colors - alpha * colors, "alpha": alpha}

def refine_fractal_triangle(node: dict, depth: int, output: list[dict], budget: int = 5000) -> None:
    gradient = np.zeros(node["image"].shape[:2], dtype=np.float32)
    output.extend(refine_adaptive_triangles([node], node["image"], gradient, budget, depth, depth)[0])

def refine_adaptive_triangles(roots: list[dict], image: np.ndarray, gradient: np.ndarray, max_triangles: int = 3000, min_depth: int = 2, max_depth: int = 6, error_threshold: float = .006, deadline: float | None = None) -> tuple[list[dict], dict]:
    """Best-first 4-way recursion: detail gets budget before smooth regions do."""
    serial = count(); heap: list[tuple[float,int,dict]] = []
    for root in roots:
        item = _node(root["vertices"], image, gradient, root.get("plane_id",0), 0)
        heapq.heappush(heap, (-item["score"], next(serial), item))
    leaf_count = len(heap); fallback = False; leaves: list[dict] = []
    while heap:
        if deadline and time.monotonic() >= deadline:
            fallback = True; leaves.extend(item for _,_,item in heap); break
        _, _, item = heapq.heappop(heap); must_refine = item["depth"] < min_depth; should_refine = item["depth"] < max_depth and item["score"] > error_threshold
        if (must_refine or should_refine) and item["depth"] < max_depth and leaf_count + 3 <= max_triangles:
            for child in _children(item, image, gradient): heapq.heappush(heap, (-child["score"], next(serial), child))
            leaf_count += 3
        else: leaves.append(item)
    errors = np.asarray([leaf["error"] for leaf in leaves], dtype=np.float32)
    return leaves, {"mean_local_error": float(errors.mean()) if len(errors) else 0.0, "maximum_local_error": float(errors.max()) if len(errors) else 0.0, "fallback_used": fallback}

