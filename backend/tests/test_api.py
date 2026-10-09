import io
import time
import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.algorithm.fractal_interpolation import refine_adaptive_triangles
from app.algorithm.triangulation import build_root_triangle
from app.algorithm.composition import compose_planes
from app.algorithm.triangulation import triangulate_region

client = TestClient(app)

def png_bytes():
    image = Image.new("RGB", (64, 48), "#422780")
    for x in range(32, 64):
        for y in range(48): image.putpixel((x, y), (20, 195, 210))
    data = io.BytesIO(); image.save(data, "PNG"); return data.getvalue()

def image_bytes(kind: str, mode: str = "RGB"):
    size = 128; data = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size]
    if kind == "geometric":
        data[..., :3] = (24, 30, 42); data[(xx-64)**2 + (yy-64)**2 < 34**2, :3] = (226, 88, 108); data[20:108, 58:70, :3] = (85, 218, 215)
    elif kind == "face":
        data[..., :3] = (232, 238, 245); data[20:112, 35:93, :3] = (174, 117, 86); data[45:55, 44:56, :3] = 15; data[45:55, 72:84, :3] = 15; data[74:84, 53:75, :3] = (80, 36, 28)
    elif kind == "low":
        data[..., :3] = np.linspace(80, 110, size, dtype=np.uint8)[None, :, None]
    elif kind == "detail":
        data[..., :3] = np.stack(((xx*7+yy*3)%256, (xx*13)%256, (yy*11)%256), axis=-1)
    else:
        data[..., :3] = (30, 190, 160); data[20:108, 20:108, 3] = 0
    data[..., 3] = 255
    if kind == "transparent": data[20:108, 20:108, 3] = 0
    image = Image.fromarray(data if mode == "RGBA" else data[..., :3], mode)
    output = io.BytesIO(); image.save(output, "PNG"); return output.getvalue()

def adaptive_request(payload: bytes):
    started = time.monotonic()
    response = client.post("/transform", files={"file": ("case.png", payload, "image/png")}, data={"planes_count": 16, "fractal_depth": 6, "max_triangles": 700})
    assert response.status_code == 200
    assert time.monotonic() - started < 30
    return response.json()

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_transform_returns_preview_expression_and_wolfram_url():
    response = client.post("/transform", files={"file": ("test.png", png_bytes(), "image/png")}, data={"planes_count": 3, "fractal_depth": 1, "max_triangles": 64})
    assert response.status_code == 200
    body = response.json()
    assert body["success"] and body["plane_count"] >= 1 and body["triangle_count"] >= 1
    assert body["preview_data_url"].startswith("data:image/png;base64,")
    assert body["expression"].startswith("Graphics[")
    assert body["wolfram_url"].startswith("https://www.wolframalpha.com/input?i=")
    assert body["adaptive_refinement"] and body["leaf_triangle_count"] == body["triangle_count"]

def test_invalid_upload_is_rejected():
    response = client.post("/transform", files={"file": ("bad.txt", b"not image", "text/plain")})
    assert response.status_code == 415

def test_geometric_face_transparent_low_contrast_and_detail_images_are_valid():
    results = {kind: adaptive_request(image_bytes(kind, "RGBA" if kind == "transparent" else "RGB")) for kind in ("geometric", "face", "transparent", "low", "detail")}
    assert all(result["plane_count"] >= 1 and result["leaf_triangle_count"] >= 1 for result in results.values())
    assert results["transparent"]["original"]["format"] == "PNG"
    # Fine visual structure receives more adaptive geometry than a near-smooth field.
    assert results["detail"]["leaf_triangle_count"] >= results["low"]["leaf_triangle_count"]
    assert results["detail"]["base_triangle_count"] > results["detail"]["contour_count"]

def test_alpha_and_q_change_rendered_fractal_leaf_colours():
    image = np.zeros((48, 48, 4), dtype=np.float32); yy, xx = np.mgrid[:48, :48]
    image[..., :3] = np.stack((xx / 47, yy / 47, ((xx * yy) % 17) / 16), axis=-1); image[..., 3] = 1
    gradient = np.ones((48, 48), dtype=np.float32)
    leaves, _ = refine_adaptive_triangles([build_root_triangle(np.asarray([[0,0],[47,0],[0,47]], dtype=np.float32), image)], image, gradient, 64, 1, 2, 0)
    assert any(leaf["alpha"] > 0 and np.linalg.norm(leaf["q_colors"]) > 0 for leaf in leaves)
    plane = {"mask": np.ones((48,48), dtype=np.uint8), "contour": np.asarray([[0,0],[47,0],[0,47]]), "color": (0.,0.,0.), "triangles": leaves, "opacity": 1.0}
    fractal = compose_planes([plane], (48,48))
    for leaf in leaves: leaf["alpha"] = 0
    affine_only = compose_planes([plane], (48,48))
    assert not np.allclose(fractal, affine_only)

def test_complex_closed_contour_keeps_real_base_triangulation():
    contour = np.asarray([[4,4],[56,4],[56,16],[40,16],[40,28],[56,28],[56,56],[4,56],[4,40],[20,40],[20,28],[4,28]], dtype=np.float32)
    mask = np.zeros((64,64), dtype=np.uint8); cv2.fillPoly(mask, [contour.astype(np.int32)], 1)
    triangles = triangulate_region(contour, mask, max_triangles=4, holes=[])
    assert len(triangles) >= len(contour) - 2
    for triangle in triangles:
        centroid = np.rint(triangle.mean(axis=0)).astype(int)
        assert mask[centroid[1], centroid[0]]
