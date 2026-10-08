# Private deployment

1. Create a private repository named `fractal-io` and push this project.
2. On a private Docker-capable host, copy `.env.example` to `.env`; do not commit that file.
3. Set `NEXT_PUBLIC_API_BASE_URL` to the browser-accessible HTTPS API origin before `docker compose up --build -d`.
4. Put a TLS reverse proxy in front of ports 3000 and 8000, restrict network access as appropriate, and set CORS origins in `backend/app/main.py` to the deployed frontend origin.
5. Check `/health`, upload a non-sensitive test image, and verify the Wolfram link.

No deployment is implied merely by pushing GitHub. FastAPI must run on a compatible private host.
