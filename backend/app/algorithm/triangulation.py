from __future__ import annotations
import cv2
import numpy as np

def build_root_triangle(triangle: np.ndarray, image: np.ndarray) -> dict:
    return {"vertices": triangle.astype(np.float32), "image": image}

def triangulate_region(contour: np.ndarray, mask: np.ndarray, max_triangles: int = 180) -> list[np.ndarray]:
    h, w = mask.shape
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    points = contour.astype(np.float32).tolist()
    step = max(8, int((h * w / max_triangles) ** 0.5))
    ys, xs = np.where(mask > 0)
    if len(xs):
        for y in range(0, h, step):
            for x in range(0, w, step):
                if mask[y, x]: points.append([float(x), float(y)])
    for point in points:
        x, y = point
        try:
            subdiv.insert((min(max(x, 0.0), w - 1.001), min(max(y, 0.0), h - 1.001)))
        except cv2.error:
            pass
    triangles = []
    for flat in subdiv.getTriangleList():
        triangle = np.asarray(flat, dtype=np.float32).reshape(3, 2)
        centroid = triangle.mean(axis=0).astype(int)
        if (triangle[:, 0] >= 0).all() and (triangle[:, 0] < w).all() and (triangle[:, 1] >= 0).all() and (triangle[:, 1] < h).all() and mask[centroid[1], centroid[0]]:
            triangles.append(triangle)
            if len(triangles) >= max_triangles: break
    if not triangles and len(contour) >= 3:
        anchor = contour[0].astype(np.float32)
        triangles = [np.asarray([anchor, contour[i], contour[i+1]], dtype=np.float32) for i in range(1, len(contour)-1)]
    return triangles

def regular_triangular_fallback(image: np.ndarray, step: int = 48) -> list[dict]:
    h, w = image.shape[:2]
    mask = (image[..., 3] > 0.02).astype(np.uint8)
    contour = np.asarray([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
    return [{"mask": mask, "contour": contour, "color": tuple(image[..., :3].mean(axis=(0,1))), "area": float(mask.sum())}]

