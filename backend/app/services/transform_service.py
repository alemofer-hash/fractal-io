from __future__ import annotations
import base64
import io
import time
import cv2
from PIL import Image
import numpy as np
from fastapi import HTTPException
from app.algorithm.normalize import normalize_image
from app.algorithm.quantization import quantize_colors
from app.algorithm.contours import extract_closed_regions
from app.algorithm.triangulation import triangulate_region, build_root_triangle, build_projection_mesh, regular_triangular_fallback
from app.algorithm.fractal_interpolation import refine_adaptive_triangles
from app.algorithm.composition import compose_planes
from app.algorithm.wolfram import emit_wolfram_expression, compact_wolfram_expression, build_wolfram_url
from app.config import DEFAULT_WORKING_SIZE, MAX_PROCESSING_SECONDS

def _preview_data_url(image: np.ndarray) -> str:
    encoded = io.BytesIO()
    Image.fromarray(np.rint(image * 255).astype(np.uint8), "RGBA").save(encoded, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

def _gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((image[..., :3] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy); scale = float(np.percentile(magnitude, 97))
    return np.clip(magnitude / max(scale, 1e-6), 0, 1)

def transform_image(source: Image.Image, planes_count: int = 16, depth: int = 6, max_triangles: int = 4000, quality: str = "balanced") -> dict:
    started = time.monotonic()
    original_size = source.size
    working_size = {"draft": 512, "balanced": DEFAULT_WORKING_SIZE, "detail": min(768, DEFAULT_WORKING_SIZE)}.get(quality, DEFAULT_WORKING_SIZE)
    image = normalize_image(source, working_size)
    gradient = _gradient(image)
    labels, centers = quantize_colors(image, clusters=max(10, min(planes_count, 24)))
    regions = extract_closed_regions(labels, centers, min_area=max(12, round(image.shape[0] * image.shape[1] * .001)), limit=max(10, min(planes_count * 2, 32)), image=image)
    fallback = not bool(regions)
    if fallback:
        raw = regular_triangular_fallback(image)[0]
        from app.algorithm.contours import Region
        regions = [Region(**raw)]
    roots: list[dict] = []
    for plane_id, region in enumerate(regions):
        # Never cap a complex closed contour below n - 2 base triangles.
        base = triangulate_region(region.contour, region.mask, max_triangles=max(96, len(region.contour) - 2), gradient=gradient, holes=region.holes)
        roots.extend(build_root_triangle(triangle, image, plane_id) for triangle in base)
    if roots:
        # The closed-region meshes validate each plane. A full-domain projection mesh is
        # composed last so Lab labels never turn smooth lighting into flat class colours.
        projection_roots = [build_root_triangle(triangle, image, -1) for triangle in build_projection_mesh(image.shape[:2], cell_size=72)]
        leaves, adaptive_metrics = refine_adaptive_triangles(projection_roots, image, gradient, max_triangles=max_triangles, min_depth=2, max_depth=max(2, min(6, depth)), deadline=started + MAX_PROCESSING_SECONDS * .83)
    else:
        leaves, adaptive_metrics = [], {"mean_local_error": 0.0, "maximum_local_error": 0.0, "fallback_used": True}
    per_plane_base: dict[int, int] = {}
    for root in roots: per_plane_base[root["plane_id"]] = per_plane_base.get(root["plane_id"], 0) + 1
    planes = [{"mask": region.mask, "contour": region.contour, "color": region.color, "triangles": [], "base_triangle_count": per_plane_base.get(plane_id, 0), "opacity": 1.0} for plane_id, region in enumerate(regions) if per_plane_base.get(plane_id)]
    if not planes:
        # A valid raster always gets a finite fallback construction.
        raw = regular_triangular_fallback(image)[0]
        tri = np.asarray([[0,0], [image.shape[1]-1,0], [0,image.shape[0]-1]], dtype=np.float32)
        leaves, adaptive_metrics = refine_adaptive_triangles([build_root_triangle(tri, image, 0)], image, gradient, max_triangles=min(max_triangles, 64), min_depth=1, max_depth=2)
        planes = [{**raw, "triangles": leaves, "base_triangle_count": 1, "opacity": 1.0}]
        roots = [build_root_triangle(tri, image, 0)]; projection_roots = roots
        fallback = True
    projection = {"mask": (image[..., 3] > .02).astype(np.uint8), "contour": np.asarray([[0,0],[image.shape[1]-1,0],[image.shape[1]-1,image.shape[0]-1],[0,image.shape[0]-1]], dtype=np.float32), "color": tuple(image[..., :3].mean(axis=(0,1))), "triangles": leaves, "opacity": 1.0}
    reconstructed = compose_planes([projection], image.shape[:2])
    expression = emit_wolfram_expression([projection], image.shape[0])
    wolfram_expression, compact = compact_wolfram_expression([projection], image.shape[0])
    return {
        "success": True,
        "original": {"width": original_size[0], "height": original_size[1], "format": source.format or "UNKNOWN"},
        "processing_time_ms": round((time.monotonic() - started) * 1000),
        "plane_count": len(planes), "contour_count": len(regions), "triangle_count": len(leaves),
        "base_triangle_count": len(roots) + len(projection_roots), "leaf_triangle_count": len(leaves), "adaptive_refinement": True,
        "mean_local_error": round(adaptive_metrics["mean_local_error"], 7), "maximum_local_error": round(adaptive_metrics["maximum_local_error"], 7),
        "high_detail_regions": sum(1 for leaf in leaves if leaf["depth"] > 2), "fallback_used": bool(fallback or adaptive_metrics["fallback_used"]),
        "fractal_depth": max((leaf["depth"] for leaf in leaves), default=0), "preview_data_url": _preview_data_url(reconstructed),
        "expression": expression, "wolfram_expression": wolfram_expression,
        "wolfram_url": build_wolfram_url(wolfram_expression), "wolfram_uses_compact_expression": compact,
        "planes": [{"index": i + 1, "color": [round(v, 3) for v in p["color"]], "triangle_count": p["base_triangle_count"], "contour_points": len(p["contour"])} for i, p in enumerate(planes)],
        "quality_reduced": bool(fallback or adaptive_metrics["fallback_used"]),
    }
