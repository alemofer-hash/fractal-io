import os

HARD_UPLOAD_CEILING = 50 * 1024 * 1024
MAX_UPLOAD_BYTES = min(int(os.getenv("MAX_UPLOAD_BYTES", str(HARD_UPLOAD_CEILING))), HARD_UPLOAD_CEILING)
MAX_PROCESSING_SECONDS = max(1, int(os.getenv("MAX_PROCESSING_SECONDS", "30")))
DEFAULT_ANALYSIS_SIZE = max(64, min(1024, int(os.getenv("ANALYSIS_RESOLUTION", "768"))))
DEFAULT_MESH_SIZE = max(64, min(1024, int(os.getenv("MESH_RESOLUTION", str(DEFAULT_ANALYSIS_SIZE)))))
DEFAULT_RENDER_SIZE = max(64, min(2048, int(os.getenv("RENDER_RESOLUTION", "1024"))))
DEFAULT_PREVIEW_SIZE = max(64, min(1024, int(os.getenv("PREVIEW_RESOLUTION", "768"))))
# Backward-compatible name for deployments configured before resolution separation.
DEFAULT_WORKING_SIZE = DEFAULT_ANALYSIS_SIZE
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_EXTENSIONS = {"png", "jpeg", "jpg", "webp", "bmp", "gif"}
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "GIF"}

