from __future__ import annotations
import cv2
import numpy as np

def build_root_triangle(triangle: np.ndarray, image: np.ndarray, plane_id: int = 0) -> dict:
    return {"vertices": triangle.astype(np.float32), "image": image, "plane_id": plane_id}

def _inside(mask: np.ndarray, point: np.ndarray) -> bool:
    h, w = mask.shape
    x, y = np.rint(point).astype(int)
    return 0 <= x < w and 0 <= y < h and bool(mask[y, x])

def _valid_triangle(mask: np.ndarray, triangle: np.ndarray) -> bool:
    """Reject a Delaunay triangle crossing a concavity or hole, not merely off-centre ones."""
    a, b, c = triangle
    return all(_inside(mask, point) for point in (a, b, c, (a+b)/2, (b+c)/2, (c+a)/2, (a+b+c)/3))

def _unique_points(points: list[np.ndarray], width: int, height: int) -> list[tuple[float, float]]:
    seen: set[tuple[int, int]] = set(); result: list[tuple[float, float]] = []
    for point in points:
        x, y = float(point[0]), float(point[1]); key = (int(round(x)), int(round(y)))
        if key not in seen and 0 <= x < width and 0 <= y < height:
            seen.add(key); result.append((min(x, width-1.001), min(y, height-1.001)))
    return result

def triangulate_region(contour: np.ndarray, mask: np.ndarray, max_triangles: int = 140, gradient: np.ndarray | None = None) -> list[np.ndarray]:
    """Mask-filtered Delaunay seeds from contours, interior samples and strong edges."""
    h, w = mask.shape; subdiv = cv2.Subdiv2D((0, 0, w, h))
    points: list[np.ndarray] = [point for point in contour.astype(np.float32)]
    step = max(10, int(np.sqrt(max(1, mask.sum()) / max(12, max_triangles)) * 1.15))
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if mask[y, x]: points.append(np.asarray([x, y], dtype=np.float32))
    if gradient is not None and gradient.size:
        values = gradient[mask > 0]
        if len(values):
            threshold = float(np.percentile(values, 82)); edge = np.argwhere((gradient >= threshold) & (mask > 0))
            stride = max(1, len(edge) // max(20, max_triangles // 3))
            for y, x in edge[::stride]: points.append(np.asarray([x, y], dtype=np.float32))
    for point in _unique_points(points, w, h):
        try: subdiv.insert(point)
        except cv2.error: pass
    triangles = [np.asarray(flat, dtype=np.float32).reshape(3, 2) for flat in subdiv.getTriangleList()]
    triangles = [triangle for triangle in triangles if _valid_triangle(mask, triangle)]
    triangles.sort(key=lambda tri: (float(tri[:, 1].mean()), float(tri[:, 0].mean())))
    if not triangles and len(contour) >= 3:
        anchor = contour[0].astype(np.float32)
        triangles = [np.asarray([anchor, contour[i], contour[i+1]], dtype=np.float32) for i in range(1, len(contour)-1) if _valid_triangle(mask, np.asarray([anchor, contour[i], contour[i+1]], dtype=np.float32))]
    return triangles[:max_triangles]

def regular_triangular_fallback(image: np.ndarray) -> list[dict]:
    h, w = image.shape[:2]; mask = (image[..., 3] > 0.02).astype(np.uint8)
    contour = np.asarray([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
    return [{"mask": mask, "contour": contour, "holes": [], "color": tuple(image[..., :3].mean(axis=(0,1))), "area": float(mask.sum())}]

