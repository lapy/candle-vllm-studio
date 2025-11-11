import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.candle_build_manager import CandleBuildManager
from backend.candle_manager import CandleRuntimeManager
from backend.database import init_db, SessionLocal, Model, RunningInstance
from backend.huggingface import set_huggingface_token
from backend.logging_config import get_logger, setup_logging
from backend.routes import models, status, gpu_info
from backend.routes import candle_builds
from backend.unified_monitor import unified_monitor
from backend.websocket_manager import websocket_manager


setup_logging(level="INFO")
logger = get_logger(__name__)


def ensure_data_directories():
    """Ensure data directories exist and are writable, with fallbacks."""
    preferred = os.getenv("CANDLE_STUDIO_DATA")
    candidate_dirs = [
        preferred,
        "/app/data",
        "/data/candle-studio",
        str(Path.home() / ".local/share/candle-studio"),
        str(Path.home() / ".candle-studio"),
        os.path.join(tempfile.gettempdir(), "candle-studio"),
    ]

    subdirs = ["models", "configs", "logs", "candle-builds"]

    errors = []
    for candidate in candidate_dirs:
        if not candidate:
            continue
        base_path = Path(candidate).expanduser()
        try:
            base_path.mkdir(parents=True, exist_ok=True)
            for subdir in subdirs:
                (base_path / subdir).mkdir(parents=True, exist_ok=True)

            test_file = base_path / ".write_test"
            with open(test_file, "w", encoding="utf-8") as handle:
                handle.write("test")
            test_file.unlink(missing_ok=True)

            os.environ["CANDLE_STUDIO_DATA"] = str(base_path)
            logger.info("Using data directory '%s'", base_path)
            return str(base_path)
        except PermissionError as exc:
            logger.warning(
                "Data directory candidate '%s' is not writable: %s", base_path, exc
            )
            errors.append(f"{base_path}: {exc}")
        except OSError as exc:
            logger.warning(
                "Failed to initialise data directory candidate '%s': %s", base_path, exc
            )
            errors.append(f"{base_path}: {exc}")

    error_message = (
        "Failed to locate a writable data directory. "
        f"Tried: {', '.join(candidate for candidate in candidate_dirs if candidate)}. "
        f"Errors: {errors}"
    )
    logger.error(error_message)
    raise RuntimeError(error_message)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = ensure_data_directories()
    app.state.data_dir = data_dir
    await init_db()

    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if huggingface_api_key:
        set_huggingface_token(huggingface_api_key)
        logger.info("HuggingFace API key loaded from environment variable")

    runtime_manager = CandleRuntimeManager()

    async def runtime_state_listener(model_id: int, state: str, details: Dict[str, Any]):
        if state != "stopped":
            return

        db = SessionLocal()
        try:
            instance = (
                db.query(RunningInstance)
                .filter(RunningInstance.model_id == model_id)
                .first()
            )
            if instance:
                db.delete(instance)

            model = db.query(Model).filter(Model.id == model_id).first()
            if model:
                model.is_active = False

            db.commit()
        except Exception as exc:
            logger.error("Failed to update runtime state for model %s: %s", model_id, exc)
            db.rollback()
        finally:
            db.close()

        try:
            await unified_monitor._collect_and_send_unified_data()
        except Exception as exc:
            logger.warning("Unified monitor update failed after state change: %s", exc)

    runtime_manager.set_state_listener(runtime_state_listener)
    build_manager = CandleBuildManager(builds_dir=os.path.join(data_dir, "candle-builds"))
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
