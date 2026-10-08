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
            polygon = np.rint(item["vertices"]).astype(np.int32)
            # Finite reconstruction: each leaf uses its vertex-color affine approximation.
            color = tuple(float(x) for x in item["colors"].mean(axis=0))
            cv2.fillConvexPoly(layer, polygon, (*color, 1.0), lineType=cv2.LINE_AA)
        alpha = layer[..., 3:4]
        canvas[..., :3] = layer[..., :3] * alpha + canvas[..., :3] * (1 - alpha)
        canvas[..., 3:4] = alpha + canvas[..., 3:4] * (1 - alpha)
    return np.clip(canvas, 0, 1)

