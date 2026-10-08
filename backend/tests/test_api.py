import io
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app

client = TestClient(app)

def png_bytes():
    image = Image.new("RGB", (64, 48), "#422780")
    for x in range(32, 64):
        for y in range(48): image.putpixel((x, y), (20, 195, 210))
    data = io.BytesIO(); image.save(data, "PNG"); return data.getvalue()

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

def test_invalid_upload_is_rejected():
    response = client.post("/transform", files={"file": ("bad.txt", b"not image", "text/plain")})
    assert response.status_code == 415
