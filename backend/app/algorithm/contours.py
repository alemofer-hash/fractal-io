from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Region:
    mask: np.ndarray
    contour: np.ndarray
    holes: list[np.ndarray]
    color: tuple[float, float, float]
    area: float

def simplify_closed_contour(contour: np.ndarray, epsilon_ratio: float = 0.002) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    # Keep high-curvature facial/garment boundaries instead of flattening them.
    simplified = cv2.approxPolyDP(contour, max(0.45, perimeter * epsilon_ratio), True)
    return simplified.reshape(-1, 2)

def extract_closed_regions(labels: np.ndarray, centers: np.ndarray, min_area: int = 20, limit: int = 32, image: np.ndarray | None = None) -> list[Region]:
    regions: list[Region] = []
    for label in range(len(centers)):
        binary = (labels == label).astype(np.uint8)
        # Remove quantization pinholes while retaining spatially distinct components.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            mask = (components == component_id).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            external_indices = [i for i, entry in enumerate(hierarchy[0]) if entry[3] < 0]
            if not external_indices:
                continue
            index = max(external_indices, key=lambda i: cv2.contourArea(contours[i]))
            contour = contours[index]
            simple = simplify_closed_contour(contour)
            if len(simple) >= 3:
                holes = [simplify_closed_contour(contours[i]) for i, entry in enumerate(hierarchy[0]) if entry[3] == index and len(contours[i]) >= 3]
                # A hole belongs to the plane hierarchy, but is not transparent unless alpha made it so.
                color = tuple(float(v) for v in (image[mask > 0, :3].mean(axis=0) if image is not None else centers[label]))
                regions.append(Region(mask, simple, holes, color, float(area)))
    return sorted(regions, key=lambda r: r.area, reverse=True)[:limit]

