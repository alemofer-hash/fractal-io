from __future__ import annotations
import cv2
import numpy as np

def build_root_triangle(triangle: np.ndarray, image: np.ndarray, plane_id: int = 0) -> dict:
    return {"vertices": triangle.astype(np.float32), "image": image, "plane_id": plane_id}

def _inside(mask: np.ndarray, point: np.ndarray) -> bool:
    h, w = mask.shape
    x, y = np.rint(point).astype(int)
    if not (0 <= x < w and 0 <= y < h): return False
    # Contours lie on pixel boundaries; a one-pixel tolerance preserves a legitimate
    # polygon ear without allowing a triangle to bridge a meaningful hole.
    return bool(mask[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)].any())

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

def _signed_area(contour: np.ndarray) -> float:
    return float(np.dot(contour[:, 0], np.roll(contour[:, 1], -1)) - np.dot(contour[:, 1], np.roll(contour[:, 0], -1))) / 2

def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle
    cross = np.asarray([(b-a)[0] * (point-a)[1] - (b-a)[1] * (point-a)[0], (c-b)[0] * (point-b)[1] - (c-b)[1] * (point-b)[0], (a-c)[0] * (point-c)[1] - (a-c)[1] * (point-c)[0]])
    # Collinear contour samples lie on an ear edge and must not disqualify that ear.
    return bool((cross > 1e-5).all() or (cross < -1e-5).all())

def _ear_clip(contour: np.ndarray) -> list[np.ndarray]:
    """Deterministic ear clipping preserves every original outer-boundary vertex."""
    clean = contour.astype(np.float32)
    if len(clean) < 3: return []
    keep = [i for i in range(len(clean)) if not np.allclose(clean[i], clean[(i - 1) % len(clean)])]
    clean = clean[keep]
    if len(clean) < 3: return []
    order = list(range(len(clean)))
    if _signed_area(clean) < 0: order.reverse()
    triangles: list[np.ndarray] = []; guard = 0
    while len(order) > 3 and guard < len(clean) * len(clean):
        found = False
        for pos, current in enumerate(order):
            previous, following = order[(pos - 1) % len(order)], order[(pos + 1) % len(order)]
            triangle = np.asarray([clean[previous], clean[current], clean[following]], dtype=np.float32)
            left, right = triangle[1] - triangle[0], triangle[2] - triangle[0]
            if left[0] * right[1] - left[1] * right[0] <= 1e-6: continue
            if any(_point_in_triangle(clean[index], triangle) for index in order if index not in (previous, current, following)): continue
            triangles.append(triangle); del order[pos]; found = True; break
        if not found: break
        guard += 1
    if len(order) == 3: triangles.append(np.asarray([clean[index] for index in order], dtype=np.float32))
    return triangles

def triangulate_region(contour: np.ndarray, mask: np.ndarray, max_triangles: int = 140, gradient: np.ndarray | None = None, holes: list[np.ndarray] | None = None) -> list[np.ndarray]:
    """Mask-filtered Delaunay seeds from contours, interior samples and strong edges."""
    assert len(contour) >= 3
    h, w = mask.shape; required = max(1, len(contour) - 2); max_triangles = max(max_triangles, required)
    # For a simple component, this is an exact covering using all contour points.
    # Adaptive recursion supplies the interior samples afterwards without overlapping roots.
    if not holes:
        ears = [triangle for triangle in _ear_clip(contour) if _valid_triangle(mask, triangle)]
        if len(ears) >= required:
            assert all(_valid_triangle(mask, triangle) for triangle in ears)
            return ears
    subdiv = cv2.Subdiv2D((0, 0, w, h))
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
    if not holes:
        assert len(triangles) >= required, f"simple contour lost geometry during triangulation ({len(contour)} points, {len(triangles)} triangles)"
    assert all(_valid_triangle(mask, triangle) for triangle in triangles)
    return triangles[:max_triangles]

def regular_triangular_fallback(image: np.ndarray) -> list[dict]:
    h, w = image.shape[:2]; mask = (image[..., 3] > 0.02).astype(np.uint8)
    contour = np.asarray([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
    return [{"mask": mask, "contour": contour, "holes": [], "color": tuple(image[..., :3].mean(axis=(0,1))), "area": float(mask.sum())}]

def build_projection_mesh(shape: tuple[int, int], cell_size: int = 56) -> list[np.ndarray]:
    """A continuous full-domain base mesh. It is the final projection layer, not a colour-class fill."""
    h, w = shape; triangles: list[np.ndarray] = []
    for y0 in range(0, h - 1, cell_size):
        for x0 in range(0, w - 1, cell_size):
            x1, y1 = min(w - 1, x0 + cell_size), min(h - 1, y0 + cell_size)
            triangles.append(np.asarray([[x0, y0], [x1, y0], [x1, y1]], dtype=np.float32))
            triangles.append(np.asarray([[x0, y0], [x1, y1], [x0, y1]], dtype=np.float32))
    return triangles

