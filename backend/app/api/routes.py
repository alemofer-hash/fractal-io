from __future__ import annotations
import io
import warnings
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from app.config import ALLOWED_EXTENSIONS, ALLOWED_FORMATS, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES
from app.schemas import TransformResponse
from app.services.transform_service import transform_image

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "service": "fractal-io-backend"}

def _validate_file(name: str, content_type: str | None, data: bytes) -> Image.Image:
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported file extension. Use PNG, JPEG, WEBP, BMP, or GIF.")
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(415, "The uploaded file must be an image.")
    if not data: raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES: raise HTTPException(413, "Images must be 50 MB or smaller.")
    try:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            image.verify()
        image = Image.open(io.BytesIO(data))
        image.seek(0)  # GIF processing intentionally uses its first frame.
        if image.format not in ALLOWED_FORMATS: raise HTTPException(415, "Unsupported image format.")
        if image.width * image.height > MAX_IMAGE_PIXELS: raise HTTPException(413, "Image dimensions are too large.")
        image.load()
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(422, "This image is corrupted or unsafe to process.")

@router.post("/transform", response_model=TransformResponse)
async def transform(file: UploadFile = File(...), quality: str = Form("balanced"), planes_count: int = Form(16), fractal_depth: int = Form(6), max_triangles: int = Form(4000)):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    image = _validate_file(file.filename or "upload", file.content_type, data)
    return transform_image(image, max(10, min(24, planes_count)), max(2, min(6, fractal_depth)), max(128, min(5000, max_triangles)), quality)
