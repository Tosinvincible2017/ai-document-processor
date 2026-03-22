import sys
import os
import io

# Fix Windows console encoding for unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from config import HOST, PORT, UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR
from routes import router
from database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app):
    """Ensure required directories exist and initialize database on startup."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    init_db()
    logging.getLogger(__name__).info("Document Processor API started (database ready)")
    yield


# Create app
app = FastAPI(
    title="Document Processor API",
    description="AI-powered document processing: OCR + NLP for PDFs and images",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")

# Root redirect to UI
@app.get("/")
async def root():
    return RedirectResponse(url="/ui/index.html")

# Serve static UI files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# Service worker must be served from root scope for PWA
@app.get("/sw.js")
async def service_worker():
    sw_path = os.path.join(STATIC_DIR, "sw.js")
    return FileResponse(sw_path, media_type="application/javascript",
                        headers={"Service-Worker-Allowed": "/"})


app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    # Check for SSL certs (required for mobile PWA install)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cert_file = os.path.join(BASE_DIR, "certs", "cert.pem")
    key_file = os.path.join(BASE_DIR, "certs", "key.pem")
    use_ssl = os.path.exists(cert_file) and os.path.exists(key_file)
    protocol = "https" if use_ssl else "http"

    print("=" * 50)
    print("  Document Processor API")
    print("=" * 50)
    print(f"  Server:  {protocol}://localhost:{PORT}")
    print(f"  Web UI:  {protocol}://localhost:{PORT}/ui/index.html")
    print(f"  API:     {protocol}://localhost:{PORT}/docs")
    print(f"  Health:  {protocol}://localhost:{PORT}/api/health")
    if use_ssl:
        # Show LAN address for mobile access
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"  Mobile:  {protocol}://{local_ip}:{PORT}")
        except Exception:
            pass
        print("  (SSL enabled — accept the cert warning on your phone)")
    else:
        print("  (No SSL — run with certs/ for mobile PWA install)")
    print("=" * 50)

    ssl_kwargs = {}
    if use_ssl:
        ssl_kwargs["ssl_certfile"] = cert_file
        ssl_kwargs["ssl_keyfile"] = key_file

    uvicorn.run("main:app", host=HOST, port=PORT, reload=False, **ssl_kwargs)
