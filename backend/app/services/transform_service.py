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
from app.algorithm.contours import Region, extract_closed_regions
from app.algorithm.triangulation import triangulate_region, build_root_triangle, build_projection_mesh, regular_triangular_fallback
from app.algorithm.fractal_interpolation import refine_adaptive_triangles
from app.algorithm.composition import compose_planes
from app.algorithm.wolfram import build_expression_model
from app.config import DEFAULT_ANALYSIS_SIZE, DEFAULT_MESH_SIZE, DEFAULT_PREVIEW_SIZE, DEFAULT_RENDER_SIZE, MAX_PROCESSING_SECONDS

def _preview_data_url(image: np.ndarray, max_dimension: int) -> str:
    encoded = io.BytesIO()
    output = Image.fromarray(np.rint(image * 255).astype(np.uint8), "RGBA")
    scale = min(1.0, max_dimension / max(output.size))
    if scale < 1: output = output.resize((round(output.width * scale), round(output.height * scale)), Image.Resampling.LANCZOS)
    output.save(encoded, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")

def _gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((image[..., :3] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy); scale = float(np.percentile(magnitude, 97))
    return np.clip(magnitude / max(scale, 1e-6), 0, 1)

def _scale_regions(regions: list[Region], source_shape: tuple[int, int], target_image: np.ndarray) -> list[Region]:
    source_h, source_w = source_shape; target_h, target_w = target_image.shape[:2]
    if (source_h, source_w) == (target_h, target_w): return regions
    sx, sy = target_w / source_w, target_h / source_h; scaled = []
    for region in regions:
        mask = cv2.resize(region.mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        contour = region.contour * np.asarray([sx, sy], dtype=np.float32)
        holes = [hole * np.asarray([sx, sy], dtype=np.float32) for hole in region.holes]
        color = tuple(float(value) for value in target_image[mask > 0, :3].mean(axis=0)) if mask.any() else region.color
        scaled.append(Region(mask, contour, holes, color, float(mask.sum())))
    return scaled

def _region_mesh_budget(region: Region, gradient: np.ndarray) -> int:
    """Choose an individual root-mesh target from actual geometry, not a fixed grid."""
    boundary_minimum = max(1, len(region.contour) - 2)
    area_term = max(3, round(np.sqrt(region.area) / 10))
    local_edge = float(gradient[region.mask > 0].mean()) if region.mask.any() else 0.0
    # Edge-rich regions obtain more interior support; a small smooth component
    # stays close to its closed-contour requirement. The cap is per plane and
    # only prevents one pathological component consuming all triangulation work.
    return max(boundary_minimum, min(480, boundary_minimum + round(area_term * (1.0 + 2.0 * local_edge))))

def transform_image(source: Image.Image, planes_count: int = 16, depth: int = 6, max_triangles: int = 4000, quality: str = "balanced") -> dict:
    started = time.monotonic()
    original_size = source.size
    analysis_bound = {"draft": 512, "balanced": DEFAULT_ANALYSIS_SIZE, "detail": DEFAULT_ANALYSIS_SIZE}.get(quality, DEFAULT_ANALYSIS_SIZE)
    mesh_bound = {"draft": 512, "balanced": DEFAULT_MESH_SIZE, "detail": DEFAULT_MESH_SIZE}.get(quality, DEFAULT_MESH_SIZE)
    render_bound = {"draft": 768, "balanced": DEFAULT_RENDER_SIZE, "detail": DEFAULT_RENDER_SIZE}.get(quality, DEFAULT_RENDER_SIZE)
    analysis_image = normalize_image(source, analysis_bound)
    mesh_image = normalize_image(source, mesh_bound)
    render_image = normalize_image(source, render_bound)
    labels, centers = quantize_colors(analysis_image, clusters=max(10, min(planes_count, 24)))
    regions = extract_closed_regions(labels, centers, min_area=max(12, round(analysis_image.shape[0] * analysis_image.shape[1] * .001)), limit=max(10, min(planes_count * 2, 32)), image=analysis_image)
    regions = _scale_regions(regions, analysis_image.shape[:2], mesh_image)
    gradient = _gradient(mesh_image)
    fallback = not bool(regions)
    if fallback:
        raw = regular_triangular_fallback(mesh_image)[0]
        regions = [Region(**raw)]
    roots: list[dict] = []
    for plane_id, region in enumerate(regions):
        # Never cap a complex closed contour below n - 2 base triangles.
        base = triangulate_region(region.contour, region.mask, max_triangles=_region_mesh_budget(region, gradient), gradient=gradient, holes=region.holes)
        roots.extend(build_root_triangle(triangle, mesh_image, plane_id) for triangle in base)
    # The closed-region meshes preserve plane geometry. The continuous final field is built
    # at render resolution, keeping image sharpness independent of analysis downsampling.
    render_gradient = _gradient(render_image)
    projection_roots = [build_root_triangle(triangle, render_image, -1) for triangle in build_projection_mesh(render_image.shape[:2], cell_size=72)]
    leaves, adaptive_metrics = refine_adaptive_triangles(projection_roots, render_image, render_gradient, max_triangles=max_triangles, min_depth=2, max_depth=max(2, min(6, depth)), deadline=started + MAX_PROCESSING_SECONDS * .83)
    per_plane_base: dict[int, int] = {}
    for root in roots: per_plane_base[root["plane_id"]] = per_plane_base.get(root["plane_id"], 0) + 1
    planes = [{"mask": region.mask, "contour": region.contour, "color": region.color, "triangles": [], "base_triangle_count": per_plane_base.get(plane_id, 0), "opacity": 1.0} for plane_id, region in enumerate(regions) if per_plane_base.get(plane_id)]
    if not planes:
        raw = regular_triangular_fallback(mesh_image)[0]
        planes = [{**raw, "triangles": [], "base_triangle_count": 1, "opacity": 1.0}]
        roots = [build_root_triangle(np.asarray([[0,0], [mesh_image.shape[1]-1,0], [0,mesh_image.shape[0]-1]], dtype=np.float32), mesh_image, 0)]
        fallback = True
    projection = {"mask": (render_image[..., 3] > .02).astype(np.uint8), "contour": np.asarray([[0,0],[render_image.shape[1]-1,0],[render_image.shape[1]-1,render_image.shape[0]-1],[0,render_image.shape[0]-1]], dtype=np.float32), "color": tuple(render_image[..., :3].mean(axis=(0,1))), "triangles": leaves, "opacity": 1.0}
    reconstructed = compose_planes([projection], render_image.shape[:2])
    expression_model = build_expression_model([projection], render_image.shape[1], render_image.shape[0])
    preview_url = _preview_data_url(reconstructed, DEFAULT_PREVIEW_SIZE)
    metadata = {
        "width": render_image.shape[1], "height": render_image.shape[0],
        "planes": len(planes), "contours": len(regions),
        "triangles": len(leaves), "recursion_depth": max((leaf["depth"] for leaf in leaves), default=0),
        "render_time_ms": round((time.monotonic() - started) * 1000),
    }
    return {
        "success": True,
        "original": {"width": original_size[0], "height": original_size[1], "format": source.format or "UNKNOWN"},
        "processing_time_ms": round((time.monotonic() - started) * 1000),
        "plane_count": len(planes), "contour_count": len(regions), "triangle_count": len(leaves),
        "base_triangle_count": len(roots) + len(projection_roots), "leaf_triangle_count": len(leaves), "adaptive_refinement": True,
        "mean_local_error": round(adaptive_metrics["mean_local_error"], 7), "maximum_local_error": round(adaptive_metrics["maximum_local_error"], 7),
        "high_detail_regions": sum(1 for leaf in leaves if leaf["depth"] > 2), "fallback_used": bool(fallback or adaptive_metrics["fallback_used"]),
        "source_resolution": {"width": original_size[0], "height": original_size[1]},
        "analysis_resolution": {"width": analysis_image.shape[1], "height": analysis_image.shape[0]},
        "mesh_resolution": {"width": mesh_image.shape[1], "height": mesh_image.shape[0]},
        "render_resolution": {"width": render_image.shape[1], "height": render_image.shape[0]},
        "preview_resolution": {"max_dimension": DEFAULT_PREVIEW_SIZE},
        "fractal_depth": metadata["recursion_depth"], "preview_data_url": preview_url, "image_url": preview_url,
        "expression": expression_model, "metadata": metadata,
        # Flat aliases keep older local clients working while the structured
        # expression object becomes the canonical contract.
        "wolfram_expression": expression_model["wolfram_code"],
        "wolfram_url": expression_model["wolfram_url"],
        "wolfram_uses_compact_expression": expression_model["wolfram_uses_compact_expression"],
        "download_filename": expression_model["download_filename"],
        "planes": [{"index": i + 1, "color": [round(v, 3) for v in p["color"]], "triangle_count": p["base_triangle_count"], "contour_points": len(p["contour"])} for i, p in enumerate(planes)],
        "quality_reduced": bool(fallback or adaptive_metrics["fallback_used"]),
    }
