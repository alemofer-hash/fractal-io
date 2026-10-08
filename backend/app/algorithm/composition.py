from __future__ import annotations
import cv2
import numpy as np

def compose_planes(planes: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    canvas = np.zeros((h, w, 4), dtype=np.float32)
    for plane in planes:
        layer = np.zeros((h, w, 4), dtype=np.float32)
        mask = plane["mask"].astype(bool)
        layer[mask, :3] = plane["color"]
        layer[mask, 3] = plane.get("opacity", 1.0)
        for item in plane["triangles"]:
            vertices = item["vertices"]
            xmin, ymin = np.maximum(np.floor(vertices.min(axis=0)).astype(int), 0)
            xmax, ymax = np.minimum(np.ceil(vertices.max(axis=0)).astype(int), [w - 1, h - 1])
            if xmax < xmin or ymax < ymin: continue
            xx, yy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
            points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
            a, b, c = vertices; matrix = np.column_stack((a-c, b-c)); determinant = float(np.linalg.det(matrix))
            if abs(determinant) < 1e-8: continue
            uv = np.linalg.solve(matrix, (points-c).T).T; weights = np.column_stack((uv[:,0], uv[:,1], 1-uv[:,0]-uv[:,1]))
            inside = (weights >= -1e-4).all(axis=1)
            local_mask = mask[yy.ravel(), xx.ravel()]
            inside &= local_mask
            if not inside.any(): continue
            # F = alpha * parent field + Q. The bubble carries inherited local texture and
            # vanishes at vertices, keeping sampled vertex colours exact.
            q = weights @ item["q_colors"]; parent = weights @ item["parent_colors"]
            bubble = (27 * weights[:,0] * weights[:,1] * weights[:,2])[:, None]
            rgb = np.clip(item["alpha"] * parent + q + item["alpha"] * bubble * item["parent_texture"], 0, 1)
            target_y, target_x = yy.ravel()[inside], xx.ravel()[inside]
            layer[target_y, target_x, :3] = rgb[inside]; layer[target_y, target_x, 3] = item["opacity"]
        alpha = layer[..., 3:4]
        canvas[..., :3] = layer[..., :3] * alpha + canvas[..., :3] * (1 - alpha)
        canvas[..., 3:4] = alpha + canvas[..., 3:4] * (1 - alpha)
    # A tiny final filter antialiases leaf boundaries without changing geometric export.
    canvas = cv2.GaussianBlur(canvas, (3, 3), .35)
    return np.clip(canvas, 0, 1)

