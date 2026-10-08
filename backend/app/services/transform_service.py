from __future__ import annotations
import base64
import io
import time
from PIL import Image
import numpy as np
from fastapi import HTTPException
from app.algorithm.normalize import normalize_image
from app.algorithm.quantization import quantize_colors
from app.algorithm.contours import extract_closed_regions
from app.algorithm.triangulation import triangulate_region, build_root_triangle, regular_triangular_fallback
from app.algorithm.fractal_interpolation import refine_fractal_triangle
from app.algorithm.composition import compose_planes
from app.algorithm.wolfram import emit_wolfram_expression, compact_wolfram_expression, build_wolfram_url
from app.config import DEFAULT_WORKING_SIZE, MAX_PROCESSING_SECONDS

def _preview_data_url(image: np.ndarray) -> str:
    encoded = io.BytesIO()
    Image.fromarray(np.rint(image * 255).astype(np.uint8), "RGBA").save(encoded, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

def transform_image(source: Image.Image, planes_count: int = 8, depth: int = 3, max_triangles: int = 600, quality: str = "balanced") -> dict:
    started = time.monotonic()
    original_size = source.size
    working_size = {"draft": 320, "balanced": DEFAULT_WORKING_SIZE, "detail": min(768, DEFAULT_WORKING_SIZE)}.get(quality, DEFAULT_WORKING_SIZE)
    image = normalize_image(source, working_size)
    labels, centers = quantize_colors(image, clusters=max(1, min(planes_count, 16)))
    regions = extract_closed_regions(labels, centers, min_area=max(12, image.shape[0] * image.shape[1] // 12000), limit=max(1, min(planes_count, 18)))
    fallback = not bool(regions)
    if fallback:
        raw = regular_triangular_fallback(image)[0]
        from app.algorithm.contours import Region
        regions = [Region(**raw)]
    # A leaf cap makes CPU time and returned expressions predictable. Every root has 4^depth leaves.
    leaves_per_root = 4 ** depth
    available_roots = max(1, max_triangles // leaves_per_root)
    active_regions = regions[:available_roots]
    root_cap = max(1, available_roots // len(active_regions))
    planes = []
    all_count = 0
    quality_reduced = False
    for region in active_regions:
        if time.monotonic() - started > MAX_PROCESSING_SECONDS * .72:
            quality_reduced = True
            break
        roots = triangulate_region(region.contour, region.mask, max_triangles=root_cap)
        leaves: list[dict] = []
        for tri in roots:
            if all_count + len(leaves) + leaves_per_root > max_triangles: break
            refine_fractal_triangle(build_root_triangle(tri, image), depth, leaves, max_triangles)
        if leaves:
            all_count += len(leaves)
            planes.append({"mask": region.mask, "contour": region.contour, "color": region.color, "triangles": leaves, "opacity": 1.0})
    if not planes:
        # A valid raster always gets a finite fallback construction.
        raw = regular_triangular_fallback(image)[0]
        tri = np.asarray([[0,0], [image.shape[1]-1,0], [0,image.shape[0]-1]], dtype=np.float32)
        leaves = []
        refine_fractal_triangle(build_root_triangle(tri, image), min(depth, 2), leaves, max_triangles)
        planes = [{**raw, "triangles": leaves, "opacity": 1.0}]
        all_count = len(leaves)
        quality_reduced = True
    reconstructed = compose_planes(planes, image.shape[:2])
    expression = emit_wolfram_expression(planes, image.shape[0])
    wolfram_expression, compact = compact_wolfram_expression(planes, image.shape[0])
    return {
        "success": True,
        "original": {"width": original_size[0], "height": original_size[1], "format": source.format or "UNKNOWN"},
        "processing_time_ms": round((time.monotonic() - started) * 1000),
        "plane_count": len(planes), "contour_count": len(regions), "triangle_count": all_count,
        "fractal_depth": depth, "preview_data_url": _preview_data_url(reconstructed),
        "expression": expression, "wolfram_expression": wolfram_expression,
        "wolfram_url": build_wolfram_url(wolfram_expression), "wolfram_uses_compact_expression": compact,
        "planes": [{"index": i + 1, "color": [round(v, 3) for v in p["color"]], "triangle_count": len(p["triangles"]), "contour_points": len(p["contour"])} for i, p in enumerate(planes)],
        "quality_reduced": quality_reduced or fallback,
    }
