from __future__ import annotations
import numpy as np
from .fractal_interpolation import _TEXTURE_DEGREE


def _control_index(a: np.ndarray, b: np.ndarray, degree: int) -> np.ndarray:
    """Index controls emitted in (lambda0, lambda1, lambda2) order."""
    return a * (degree + 1) - (a * (a - 1)) // 2 + b


def _piecewise_texture(weights: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Evaluate a triangular, interpolating colour micro-mesh.

    The controls are samples of the source field at a degree-eight barycentric
    lattice.  Unlike a Bernstein fit, this field passes through those samples,
    so eyes, highlights and sharp silhouette changes do not acquire the blur of
    a smoothing basis.  It remains a compact finite collection of colours and
    triangles held by each fractal leaf.
    """
    degree = _TEXTURE_DEGREE
    # A tiny inward shift gives boundary pixels a well-defined micro-triangle
    # without moving a rendered pixel in any observable way.
    local = weights * (1.0 - 1e-5) + (1e-5 / 3.0)
    u = local[:, 1] * degree
    v = local[:, 2] * degree
    b = np.floor(u).astype(np.int32)
    c = np.floor(v).astype(np.int32)
    a = degree - b - c
    du, dv = u - b, v - c
    lower = (du + dv) <= 1.0
    result = np.empty((len(weights), 3), dtype=np.float32)

    if lower.any():
        indices = np.flatnonzero(lower)
        aa, bb = a[indices], b[indices]
        p00 = controls[_control_index(aa, bb, degree)]
        p10 = controls[_control_index(aa - 1, bb + 1, degree)]
        p01 = controls[_control_index(aa - 1, bb, degree)]
        result[indices] = ((1.0 - du[indices] - dv[indices])[:, None] * p00
                           + du[indices, None] * p10 + dv[indices, None] * p01)
    if (~lower).any():
        indices = np.flatnonzero(~lower)
        aa, bb = a[indices], b[indices]
        p10 = controls[_control_index(aa - 1, bb + 1, degree)]
        p01 = controls[_control_index(aa - 1, bb, degree)]
        p11 = controls[_control_index(aa - 2, bb + 1, degree)]
        result[indices] = ((1.0 - dv[indices])[:, None] * p10
                           + (1.0 - du[indices])[:, None] * p01
                           + (du[indices] + dv[indices] - 1.0)[:, None] * p11)
    return result

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
            # F = alpha * parent field + Q. The alpha/Q base exactly reproduces
            # the child affine field at its anchors. The finite triangular
            # interpolating residual carries texture; alpha gently modulates it instead
            # of adding a plane-wide wave (which used to make the mesh visible).
            q = weights @ item["q_colors"]; parent = weights @ item["parent_colors"]
            fractal_linear = item["alpha"] * parent + q
            texture = np.zeros_like(fractal_linear)
            texture[inside] = _piecewise_texture(weights[inside], item["texture_controls"])
            residual = texture - (weights @ item["colors"])
            rgb = np.clip(fractal_linear + (1.0 + 0.08 * item["alpha"]) * residual, 0, 1)
            target_y, target_x = yy.ravel()[inside], xx.ravel()[inside]
            layer[target_y, target_x, :3] = rgb[inside]; layer[target_y, target_x, 3] = item["opacity"]
        alpha = layer[..., 3:4]
        canvas[..., :3] = layer[..., :3] * alpha + canvas[..., :3] * (1 - alpha)
        canvas[..., 3:4] = alpha + canvas[..., 3:4] * (1 - alpha)
    return np.clip(canvas, 0, 1)

