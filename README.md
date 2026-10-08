# Fractal.io

Private, self-hosted MVP that turns an uploaded raster image into a finite X² fractal construction. The user receives a reconstructed image, plane/contour/triangle statistics, a complete Wolfram Language expression and a Wolfram|Alpha link.

## Privacy model

Uploads are decoded in process, held only in memory and discarded when the request completes. There is no database, object storage, account system, telemetry, external AI API or persistent upload directory. Do not commit `.env`, user images, credentials, or logs containing image data.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:3000`. The frontend talks to the API at `NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`). Confirm the backend with `curl http://localhost:8000/health`.

To run API tests outside Docker:

```bash
cd backend
python -m pip install -r requirements.txt
pytest -q
```

## X² finite construction

The backend normalizes the input to an RGBA function on Ω = `[0,1]²`, bounded to a working dimension (default 768). OpenCV deterministically quantizes perceptual CIE Lab pixels into 10–24 clusters. Morphology and connected components then form spatially independent regions `Ωk`; `RETR_CCOMP` retains parent/child contours rather than discarding internal boundaries. Transparent pixels do not form planes.

Each region is triangulated with an interior-filtered Delaunay mesh (and a polygon fan fallback). Seeds include contour vertices, regular interior samples and deterministic samples around strong luminance gradients; probes reject triangles crossing concavities or holes. For each root triangle `T=(p0,p1,p2)`, the affine map from the canonical triangle is `L(u,v)=p0+u(p1-p0)+v(p2-p0)`.

The X² tree is adaptive rather than uniform. It scores each triangle with `E_total = E_color + 0.085 B_edge`, where `E_color` is the mean sampled affine RGB residual and `B_edge` is normalized local luminance gradient. A priority queue refines highest-score triangles first until the error threshold, maximum depth 6, 3,000-leaf target (5,000 API hard ceiling), or time guard is reached. Minimum depth 2 prevents coarse holes; smooth areas stop before detailed boundaries. The response reports base and leaf counts, error metrics, detail-region count and whether the time fallback was used.

Vertex RGB samples define an affine color plane `P`. The local residual determines `α=clip(0.78ρ/(ρ+10^-8),0,0.85)`, so `|α|<1`; the correction is stored per leaf as `Q=Pchild-αPparent`. The renderer evaluates `α Pparent + Q` plus a barycentric, vertex-zero inherited texture residual. Thus α and Q directly alter the reconstructed pixels; with `α=0` it becomes ordinary affine interpolation. Planes are alpha-composited in descending region-area order. The output is therefore reconstructed from masks, contours, triangle leaves and finite rules—not saved as the only representation.

For normal images the server caps leaves and stops priority refinement before the 30-second budget. A valid but contour-poor image gets a regular full-image fallback plane.

## API

`GET /health` returns service status.

`POST /transform` accepts multipart `file`, `quality`, `planes_count`, `fractal_depth`, and `max_triangles`. Valid formats are PNG, JPEG/JPG, WEBP, BMP and first-frame GIF. The server verifies extension, image MIME, byte cap, decode integrity, decompression-bomb safety and dimensions. The hard limit is 50 MB.

The response contains the data-url preview, complete `.wl` expression, safe Wolfram expression/URL, statistics, and plane metadata. When the complete expression is too long for a robust URL, the link uses a visibly disclosed compact expression; the download remains complete.

## Deployment

This repository is intended to remain **private**. It is ready for a private Docker-compatible host: set `NEXT_PUBLIC_API_BASE_URL` to that host’s public API URL at frontend build time, use HTTPS/reverse-proxy termination, and expose only the frontend/API endpoints you intend to serve. GitHub Pages cannot host this FastAPI backend. GitHub Actions validates builds; it does not deploy.

## Project structure

```
backend/app/algorithm/  normalization, segmentation, contours, triangulation, recursion, composition, Wolfram
backend/app/api/        HTTP validation and routes
backend/app/services/   orchestration
frontend/components/    upload, stages, comparison, planes, expression, Wolfram action
```
