import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.candle_build_manager import CandleBuildManager
from backend.candle_manager import CandleRuntimeManager
from backend.database import init_db
from backend.huggingface import set_huggingface_token
from backend.logging_config import get_logger, setup_logging
from backend.routes import models, status, gpu_info
from backend.routes import candle_builds
from backend.unified_monitor import unified_monitor
from backend.websocket_manager import websocket_manager


setup_logging(level="INFO")
logger = get_logger(__name__)


def ensure_data_directories():
    """Ensure data directories exist and are writable."""
    base_dir = os.getenv("CANDLE_STUDIO_DATA", "/app/data")
    subdirs = ["models", "configs", "logs", "candle-builds"]

    try:
        os.makedirs(base_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

        test_file = os.path.join(base_dir, ".write_test")
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("test")
        os.remove(test_file)
        logger.info("Verified data directory '%s' is writable", base_dir)
    except PermissionError as exc:
        logger.error("Data directory '%s' is not writable: %s", base_dir, exc)
        raise
    except Exception as exc:
        logger.error("Failed to ensure data directories: %s", exc)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_directories()
    await init_db()

    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if huggingface_api_key:
        set_huggingface_token(huggingface_api_key)
        logger.info("HuggingFace API key loaded from environment variable")

    runtime_manager = CandleRuntimeManager()
    build_manager = CandleBuildManager()
    app.state.candle_runtime_manager = runtime_manager
    app.state.candle_build_manager = build_manager

    await unified_monitor.start_monitoring()

    try:
        yield
    finally:
        await unified_monitor.stop_monitoring()
        try:
            await runtime_manager.stop_all()
        except Exception as exc:
            logger.error("Failed to stop candle runtimes gracefully: %s", exc)


app = FastAPI(
    title="Candle vLLM Studio",
    description="Web UI for managing candle-vllm models and builds",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins_env = os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:5173").strip()
allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:5173"
]

allow_credentials_env = os.getenv("BACKEND_CORS_ALLOW_CREDENTIALS", "false").lower() == "true"
if len(allow_origins) == 1 and allow_origins[0] == "*":
    allow_credentials_env = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials_env,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(gpu_info.router, prefix="/api", tags=["gpu"])
app.include_router(candle_builds.router, prefix="/api/candle-builds", tags=["candle-builds"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    import json

    try:
        await websocket_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                logger.debug("Received WebSocket message: %s", message.get("type"))
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)
        except Exception as exc:
            logger.error("WebSocket error: %s", exc)
            websocket_manager.disconnect(websocket)
    except Exception as exc:
        logger.error("Failed to establish WebSocket connection: %s", exc)
        websocket_manager.disconnect(websocket)


if os.path.exists("frontend/dist"):
    class CacheBustingStaticFiles(StaticFiles):
        def file_response(self, *args, **kwargs):
            response = super().file_response(*args, **kwargs)
            content_type = response.headers.get("content-type", "")
            if content_type.startswith(("text/css", "application/javascript")):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

    if os.path.exists("frontend/dist/assets"):
        app.mount("/assets", CacheBustingStaticFiles(directory="frontend/dist/assets"), name="assets")
    else:
        logger.warning("frontend/dist/assets not found, assets will not be served")

    @app.get("/vite.svg")
    async def serve_vite_svg():
        return FileResponse("frontend/public/vite.svg")

    @app.get("/favicon.ico")
    async def serve_favicon():
        return FileResponse("frontend/public/favicon.ico")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws"):
            return {"error": "Not found"}

        try:
            with open("frontend/dist/index.html", "r", encoding="utf-8") as handle:
                html_content = handle.read()
        except FileNotFoundError:
            logger.warning("frontend/dist/index.html not found, serving fallback page")
            return HTMLResponse(
                "<!DOCTYPE html><html><head><title>Candle Studio</title></head>"
                "<body><h1>Candle Studio Backend</h1>"
                "<p>Frontend build missing. Run <code>npm run build</code> in the frontend directory.</p>"
                "</body></html>"
            )

        timestamp = int(time.time() * 1000)
        import re

        html_content = re.sub(
            r'(src="/assets/[^"]+\.js")',
            rf'\1?v={timestamp}',
            html_content,
        )
        html_content = re.sub(
            r'(href="/assets/[^"]+\.css")',
            rf'\1?v={timestamp}',
            html_content,
        )

        response = HTMLResponse(html_content)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


if __name__ == "__main__":
    enable_reload = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")
    reload_dirs = ["/app/backend"] if enable_reload else None

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8080,
        reload=enable_reload,
        reload_dirs=reload_dirs,
        log_level="info",
    )
