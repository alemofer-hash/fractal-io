from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Region:
    mask: np.ndarray
    contour: np.ndarray
    color: tuple[float, float, float]
    area: float

def simplify_closed_contour(contour: np.ndarray, epsilon_ratio: float = 0.006) -> np.ndarray:
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, max(0.75, perimeter * epsilon_ratio), True)
    return simplified.reshape(-1, 2)

def extract_closed_regions(labels: np.ndarray, centers: np.ndarray, min_area: int = 20, limit: int = 18) -> list[Region]:
    regions: list[Region] = []
    for label in range(len(centers)):
        binary = (labels == label).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            mask = (components == component_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            simple = simplify_closed_contour(contour)
            if len(simple) >= 3:
                regions.append(Region(mask, simple, tuple(float(v) for v in centers[label]), float(area)))
    return sorted(regions, key=lambda r: r.area, reverse=True)[:limit]

