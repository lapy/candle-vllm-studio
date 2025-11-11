from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import time
import asyncio
import json
from datetime import datetime

from backend.database import (
    CandleBuild,
    get_db,
    Model,
    RunningInstance,
    generate_runtime_alias,
)
from backend.dependencies import get_runtime_manager
from backend.huggingface import (
    search_models,
    download_model,
    download_model_with_websocket_progress,
    set_huggingface_token,
    get_huggingface_token,
    get_model_details,
    _extract_quantization,
    clear_search_cache,
)
from backend.smart_auto import SmartAutoConfig
from backend.smart_auto.model_metadata import get_model_metadata
from backend.gpu_detector import get_gpu_info
from backend.gguf_reader import get_model_layer_info
from backend.presets import get_architecture_and_presets
from backend.logging_config import get_logger
import psutil

from backend.candle_manager import CandleRuntimeManager
from backend.websocket_manager import websocket_manager

router = APIRouter()
logger = get_logger(__name__)

# Lightweight cache for GPU info to avoid repeated NVML calls during rapid estimate requests
_gpu_info_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
GPU_INFO_CACHE_TTL = 2.0  # seconds


async def get_cached_gpu_info() -> Dict[str, Any]:
    """Return cached GPU info when available to reduce NVML overhead."""
    now = time.monotonic()
    cached = _gpu_info_cache["data"]
    if cached is not None and now - _gpu_info_cache["timestamp"] < GPU_INFO_CACHE_TTL:
        return cached
    
    data = await get_gpu_info()
    _gpu_info_cache["data"] = data
    _gpu_info_cache["timestamp"] = now
    return data

# Global download tracking to prevent duplicates and track active downloads
active_downloads = {}  # {task_id: {"huggingface_id": str, "filename": str, "quantization": str}}
download_lock = asyncio.Lock()

class EstimationRequest(BaseModel):
    model_id: int
    config: dict
    usage_mode: Optional[str] = "single_user"


@router.get("")
@router.get("/")
async def list_models(db: Session = Depends(get_db)):
    """List all managed models grouped by base model"""
    # Sync is_active status before returning models
    from backend.database import sync_model_active_status
    sync_model_active_status(db)
    
    running_instances = {
        instance.model_id: instance for instance in db.query(RunningInstance).all()
    }
    models = db.query(Model).all()
    
    # Group models by huggingface_id and base_model_name
    grouped_models = {}
    for model in models:
        key = f"{model.huggingface_id}_{model.base_model_name}"
        if key not in grouped_models:
            # derive author/owner from huggingface_id
            hf_id = model.huggingface_id or ""
            author = hf_id.split('/')[0] if isinstance(hf_id, str) and '/' in hf_id else ""
            grouped_models[key] = {
                "base_model_name": model.base_model_name,
                "huggingface_id": model.huggingface_id,
                "model_type": model.model_type,
                "author": author,
                "quantizations": []
            }
        
        runtime = running_instances.get(model.id)
        grouped_models[key]["quantizations"].append({
            "id": model.id,
            "name": model.name,
            "file_path": model.file_path,
            "file_size": model.file_size,
            "quantization": model.quantization,
            "downloaded_at": model.downloaded_at,
            "is_active": model.is_active,
            "has_config": bool(model.config),
            "huggingface_id": model.huggingface_id,
            "base_model_name": model.base_model_name,
            "model_type": model.model_type,
            "config": model.config,
            "runtime_alias": model.runtime_alias,
            "runtime": {
                "port": runtime.port if runtime else None,
                "pid": runtime.pid if runtime else None,
                "endpoint": runtime.endpoint if runtime else None,
                "build_name": runtime.build_name if runtime else None,
                "started_at": runtime.started_at if runtime else None,
            } if runtime else None,
        })
    
    # Convert to list and sort quantizations by file size (smallest first)
    result = []
    for group in grouped_models.values():
        group["quantizations"].sort(key=lambda x: x["file_size"] or 0)
        result.append(group)
    
    # Sort groups by base model name
    result.sort(key=lambda x: x["base_model_name"])
    
    return result


@router.post("/search")
async def search_huggingface_models(request: dict):
    """Search HuggingFace for GGUF models"""
    try:
        query = request.get("query")
        limit = request.get("limit", 20)
        
        if not query:
            raise HTTPException(status_code=400, detail="query parameter is required")
        
        results = await search_models(query, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/clear-cache")
async def clear_search_cache_endpoint():
    """Clear the search cache to force fresh results"""
    try:
        clear_search_cache()
        return {"message": "Search cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/{model_id}/details")
async def get_model_details_endpoint(model_id: str):
    """Get detailed model information including config and architecture"""
    try:
        details = await get_model_details(model_id)
        return details
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_huggingface_model(
    request: dict,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Download model from HuggingFace"""
    try:
        huggingface_id = request.get("huggingface_id")
        filename = request.get("filename")
        total_bytes = request.get("total_bytes", 0)  # Get total size from search results
        
        if not huggingface_id or not filename:
            raise HTTPException(status_code=400, detail="huggingface_id and filename are required")
        
        # Check if this specific quantization already exists in database
        existing = db.query(Model).filter(
            Model.huggingface_id == huggingface_id,
            Model.name == filename.replace(".gguf", "")
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="This quantization is already downloaded")

        # Extract quantization for better task_id (use same function as search results)
        quantization = _extract_quantization(filename)

        # Generate unique task ID with quantization and milliseconds
        task_id = f"download_{huggingface_id.replace('/', '_')}_{quantization}_{int(time.time() * 1000)}"

        # Check if this specific file is already being downloaded
        async with download_lock:
            is_downloading = any(
                d["huggingface_id"] == huggingface_id and d["filename"] == filename
                for d in active_downloads.values()
            )
            if is_downloading:
                raise HTTPException(status_code=409, detail="This quantization is already being downloaded")
            
            # Register this download as active
            active_downloads[task_id] = {
                "huggingface_id": huggingface_id,
                "filename": filename,
                "quantization": quantization
            }

        # Start download in background (REMOVE db parameter, pass task_id)
        background_tasks.add_task(
            download_model_task,
            huggingface_id,
            filename,
            websocket_manager,
            task_id,
            total_bytes
        )
        
        return {"message": "Download started", "huggingface_id": huggingface_id, "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/huggingface-token")
async def get_huggingface_token_status():
    """Get HuggingFace API token status"""
    token = get_huggingface_token()
    env_token = os.getenv('HUGGINGFACE_API_KEY')
    
    return {
        "has_token": bool(token),
        "token_preview": f"{token[:8]}..." if token else None,
        "from_environment": bool(env_token),
        "environment_set": bool(env_token)
    }


@router.post("/huggingface-token")
async def set_huggingface_token_endpoint(request: dict):
    """Set HuggingFace API token"""
    try:
        # Check if token is set via environment variable
        env_token = os.getenv('HUGGINGFACE_API_KEY')
        if env_token:
            return {
                "message": "Token is set via environment variable and cannot be overridden via UI",
                "has_token": True,
                "from_environment": True
            }
        
        token = request.get("token", "").strip()
        
        if not token:
            set_huggingface_token("")
            return {"message": "HuggingFace token cleared", "has_token": False}
        
        # Validate token format (basic check)
        if len(token) < 10:
            raise HTTPException(status_code=400, detail="Invalid token format")
        
        set_huggingface_token(token)
        return {"message": "HuggingFace token set successfully", "has_token": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def download_model_task(huggingface_id: str, filename: str, 
                               websocket_manager=None, task_id: str = None, 
                               total_bytes: int = 0):
    """Background task to download model with WebSocket progress"""
    from backend.database import SessionLocal
    db = SessionLocal()
    
    try:
        if websocket_manager and task_id:
            file_path, file_size = await download_model_with_websocket_progress(
                huggingface_id, filename, websocket_manager, task_id, total_bytes
            )
        else:
            file_path, file_size = await download_model(huggingface_id, filename)
        
        # Save to database
        quantization = _extract_quantization(filename)
        model = Model(
            name=filename.replace(".gguf", ""),
            huggingface_id=huggingface_id,
            base_model_name=extract_base_model_name(filename),
            file_path=file_path,
            file_size=file_size,
            quantization=quantization,
            model_type=extract_model_type(filename),
            runtime_alias=generate_runtime_alias(huggingface_id, quantization),
            downloaded_at=datetime.utcnow()
        )
        weights_dir = os.path.dirname(file_path) if os.path.isfile(file_path) else file_path
        model.config = {
            "host": "0.0.0.0",
            "port": 41234,
            "weights_path": weights_dir,
            "kvcache_mem_gpu": 4,
            "dtype": None,
            "isq": None,
            "max_gen_tokens": 4096,
            "features": [],
            "extra_args": [],
            "env": {},
            "build_profile": "release",
        }
        db.add(model)
        db.commit()
        db.refresh(model)
        
        # Send download complete WebSocket event (NEW)
        if websocket_manager:
            await websocket_manager.broadcast({
                "type": "download_complete",
                "huggingface_id": huggingface_id,
                "filename": filename,
                "quantization": model.quantization,
                "runtime_alias": model.runtime_alias,
                "model_id": model.id,
                "base_model_name": model.base_model_name,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            await websocket_manager.send_notification(
                title="Download Complete",
                message=f"Successfully downloaded {filename}",
                type="success"
            )
        
    except Exception as e:
        if websocket_manager:
            await websocket_manager.send_notification(
                title="Download Failed",
                message=f"Failed to download {filename}: {str(e)}",
                type="error"
            )
    finally:
        # Cleanup: remove from active downloads and close session
        if task_id:
            async with download_lock:
                active_downloads.pop(task_id, None)
        db.close()


# Removed duplicate extract_quantization; use `_extract_quantization` from backend.huggingface


def extract_model_type(filename: str) -> str:
    """Extract model type from filename"""
    filename_lower = filename.lower()
    if "llama" in filename_lower:
        return "llama"
    elif "mistral" in filename_lower:
        return "mistral"
    elif "codellama" in filename_lower:
        return "codellama"
    elif "gemma" in filename_lower:
        return "gemma"
    return "unknown"


def extract_base_model_name(filename: str) -> str:
    """Extract base model name from filename by removing quantization"""
    import re
    
    # Remove file extension
    name = filename.replace('.gguf', '')
    
    # Remove quantization patterns
    quantization_patterns = [
        r'IQ\d+_[A-Z]+',  # IQ1_S, IQ2_M, etc.
        r'Q\d+_K_[A-Z]+',  # Q4_K_M, Q8_0, etc.
        r'Q\d+_[A-Z]+',   # Q4_0, Q5_1, etc.
        r'Q\d+[K_]?[A-Z]*',  # Q2_K, Q6_K, etc.
        r'Q\d+',  # Q4, Q8, etc.
    ]
    
    for pattern in quantization_patterns:
        name = re.sub(pattern, '', name)
    
    # Clean up any trailing underscores or dots
    name = name.rstrip('._')
    
    return name if name else filename


@router.get("/{model_id}/config")
async def get_model_config(model_id: int, db: Session = Depends(get_db)):
    """Get the stored candle runtime configuration for the model."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return model.config or {}


@router.put("/{model_id}/config")
async def update_model_config(
    model_id: int,
    config: dict,
    db: Session = Depends(get_db)
):
    """Persist a candle runtime configuration for the model."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.config = config
    db.commit()

    return {"message": "Configuration updated"}


@router.post("/{model_id}/auto-config")
async def generate_auto_config(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Generate optimal configuration using Smart-Auto"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        gpu_info = await get_gpu_info()
        smart_auto = SmartAutoConfig()
        config = await smart_auto.generate_config(model, gpu_info)
        
        model.config = config
        db.commit()
        
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/smart-auto")
async def generate_smart_auto_config(
    model_id: int,
    preset: Optional[str] = None,
    usage_mode: str = "single_user",
    speed_quality: Optional[int] = None,
    use_case: Optional[str] = None,
    debug: Optional[bool] = False,
    db: Session = Depends(get_db)
):
    """
    Generate smart auto configuration with optional preset tuning, speed/quality balance, and use case.
    
    preset: Optional preset name (coding, conversational, long_context) to use as tuning parameters
    usage_mode: 'single_user' (sequential, peak KV cache) or 'multi_user' (server, typical usage)
    speed_quality: Speed/quality balance (0-100), where 0 = max speed, 100 = max quality. Default: 50 (balanced)
    use_case: Optional use case ('chat', 'code', 'creative', 'analysis') for targeted optimization
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        gpu_info = await get_gpu_info()
        smart_auto = SmartAutoConfig()
        debug_map = {} if debug else None
        
        # Validate usage_mode
        if usage_mode not in ["single_user", "multi_user"]:
            usage_mode = "single_user"  # Default to single_user if invalid
        
        # Validate and normalize speed_quality (0-100, default 50)
        if speed_quality is not None:
            speed_quality = max(0, min(100, int(speed_quality)))
        else:
            speed_quality = 50
        
        # Validate use_case
        if use_case is not None and use_case not in ["chat", "code", "creative", "analysis"]:
            use_case = None  # Invalid use case, ignore it
        
        # If preset is provided, pass it to generate_config for tuning
        # Also pass speed_quality and use_case for wizard-based configuration
        config = await smart_auto.generate_config(
            model, gpu_info, 
            preset=preset, 
            usage_mode=usage_mode,
            speed_quality=speed_quality,
            use_case=use_case,
            debug=debug_map
        )
        if debug_map is not None:
            return {"config": config, "debug": debug_map}
        return config
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_id}/start")
async def start_model(
    model_id: int,
    runtime_manager: CandleRuntimeManager = Depends(get_runtime_manager),
    db: Session = Depends(get_db),
):
    """Start a model using the candle runtime manager."""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=400, detail="Model file not found on disk")

    existing = db.query(RunningInstance).filter(RunningInstance.model_id == model_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Model already running")

    stored_config: Dict[str, Any] = {}
    if isinstance(model.config, dict):
        stored_config = model.config
    elif isinstance(model.config, str) and model.config.strip():
        try:
            stored_config = json.loads(model.config)
        except json.JSONDecodeError:
            logger.warning("Model %s has invalid config JSON; ignoring.", model_id)
            stored_config = {}

    weights_path = stored_config.get("weights_path")
    if not weights_path:
        weights_path = model.file_path
    if not weights_path:
        raise HTTPException(status_code=400, detail="Weights path is not configured for this model")

    weights_path = os.path.expanduser(weights_path)
    if os.path.isfile(weights_path):
        weights_path = os.path.dirname(weights_path)
    if not os.path.isdir(weights_path):
        raise HTTPException(status_code=400, detail=f"Weights directory not found at {weights_path}")

    stored_config["weights_path"] = weights_path
    model.config = stored_config

    port_value = stored_config.get("port")
    port = None
    if port_value not in (None, "", 0, "0"):
        try:
            port_candidate = int(port_value)
            if port_candidate > 0:
                port = port_candidate
        except (TypeError, ValueError):
            port = None

    build_name = stored_config.get("build_name")
    if not build_name:
        active_build = (
            db.query(CandleBuild)
            .filter(CandleBuild.is_active.is_(True))
            .order_by(CandleBuild.created_at.desc())
            .first()
        )
        if active_build:
            build_name = active_build.name

    runtime_config: Dict[str, Any] = {
        "weights_path": weights_path,
        "host": stored_config.get("host", "0.0.0.0"),
        "port": port,
        "kvcache_mem_gpu": stored_config.get("kvcache_mem_gpu", 4),
        "dtype": stored_config.get("dtype"),
        "isq": stored_config.get("isq"),
        "max_gen_tokens": stored_config.get("max_gen_tokens"),
        "build_name": build_name,
        "features": stored_config.get("features", []),
        "extra_args": stored_config.get("extra_args", []),
        "env": stored_config.get("env", {}),
        "build_profile": stored_config.get("build_profile", "release"),
        "binary_path": stored_config.get("binary_path"),
    }

    try:
        await websocket_manager.send_model_status_update(
            model_id=model_id,
            status="starting",
            details={"message": f"Starting {model.name}"},
        )

        runtime_result = await runtime_manager.start_model(
            model_id=model_id,
            config=runtime_config,
            websocket_manager=websocket_manager,
        )

        running_instance = RunningInstance(
            model_id=model_id,
            build_name=build_name,
            port=runtime_result.get("port"),
            pid=runtime_result.get("pid"),
            endpoint=runtime_result.get("endpoint"),
            started_at=datetime.utcnow(),
            config=runtime_config,
        )
        db.add(running_instance)
        model.is_active = True
        db.commit()
        
        # Send success notification via unified monitoring
        from backend.unified_monitor import unified_monitor
        await unified_monitor._collect_and_send_unified_data()
        
        return {
            "model_id": model_id,
            "port": runtime_result.get("port"),
            "pid": runtime_result.get("pid"),
            "endpoint": runtime_result.get("endpoint"),
        }
        
    except Exception as e:
        logger.error("Failed to start model %s: %s", model_id, e)
        await websocket_manager.send_model_status_update(
            model_id=model_id,
            status="error",
            details={"message": f"Failed to start: {str(e)}"},
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{model_id}/stop")
async def stop_model(
    model_id: int,
    runtime_manager: CandleRuntimeManager = Depends(get_runtime_manager),
    db: Session = Depends(get_db),
):
    """Stop a running model."""
    running_instance = db.query(RunningInstance).filter(
        RunningInstance.model_id == model_id
    ).first()
    if not running_instance:
        raise HTTPException(status_code=404, detail="No running instance found")
    
    try:
        await runtime_manager.stop_model(model_id)

        db.delete(running_instance)
        model = db.query(Model).filter(Model.id == model_id).first()
        if model:
            model.is_active = False
        db.commit()
        await websocket_manager.send_model_status_update(
            model_id=model_id,
            status="stopped",
            details={"message": "Model stopped"},
        )
        
        # Send success notification via unified monitoring
        from backend.unified_monitor import unified_monitor
        await unified_monitor._collect_and_send_unified_data()
        
        return {"message": "Model stopped"}
        
    except Exception as e:
        logger.error("Failed to stop model %s: %s", model_id, e)
        await websocket_manager.send_model_status_update(
            model_id=model_id,
            status="error",
            details={"message": f"Failed to stop: {str(e)}"},
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/vram-estimate")
async def estimate_vram_usage(
    request: EstimationRequest,
    db: Session = Depends(get_db)
):
    """Estimate VRAM usage for given configuration"""
    model = db.query(Model).filter(Model.id == request.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        gpu_info = await get_cached_gpu_info()
        smart_auto = SmartAutoConfig()
        usage_mode = request.usage_mode if request.usage_mode in ["single_user", "multi_user"] else "single_user"
        metadata = get_model_metadata(model)
        vram_estimate = smart_auto.estimate_vram_usage(
            model,
            request.config,
            gpu_info,
            usage_mode=usage_mode,
            metadata=metadata,
        )
        
        return vram_estimate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ram-estimate")
async def estimate_ram_usage(
    request: EstimationRequest,
    db: Session = Depends(get_db)
):
    """Estimate RAM usage for given configuration"""
    try:
        model = db.query(Model).filter(Model.id == request.model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        smart_auto = SmartAutoConfig()
        usage_mode = request.usage_mode if request.usage_mode in ["single_user", "multi_user"] else "single_user"
        metadata = get_model_metadata(model)
        ram_estimate = smart_auto.estimate_ram_usage(
            model,
            request.config,
            usage_mode=usage_mode,
            metadata=metadata,
        )
        
        return ram_estimate
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantization-sizes")
async def get_quantization_sizes(request: dict):
    """Get actual file sizes for quantizations from HuggingFace API"""
    try:
        huggingface_id = request.get("huggingface_id")
        quantizations = request.get("quantizations", {})
        
        if not huggingface_id or not quantizations:
            raise HTTPException(status_code=400, detail="huggingface_id and quantizations are required")
        # Use centralized Hugging Face service helper
        from backend.huggingface import get_quantization_sizes_from_hf
        updated_quantizations = await get_quantization_sizes_from_hf(huggingface_id, quantizations)

        # Fallback: for any remaining without size, try HTTP HEAD
        if updated_quantizations is None:
            updated_quantizations = {}

        missing = [q for q in quantizations.keys() if q not in updated_quantizations]
        if missing:
            import requests
            for quant_name in missing:
                quant_data = quantizations.get(quant_name) or {}
                filename = quant_data.get("filename")
                if not filename:
                    continue
                url = f"https://huggingface.co/{huggingface_id}/resolve/main/{filename}"
                try:
                    response = requests.head(url, timeout=10)
                    if response.status_code == 200:
                        content_length = response.headers.get('content-length')
                        if content_length:
                            actual_size = int(content_length)
                            updated_quantizations[quant_name] = {
                                "filename": filename,
                                "size": actual_size,
                                "size_mb": round(actual_size / (1024 * 1024), 2)
                            }
                except Exception:
                    continue
        
        return {"quantizations": updated_quantizations}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class DeleteGroupRequest(BaseModel):
    huggingface_id: str

@router.post("/delete-group")
async def delete_model_group(
    request: DeleteGroupRequest,
    runtime_manager: CandleRuntimeManager = Depends(get_runtime_manager),
    db: Session = Depends(get_db),
):
    """Delete all quantizations of a model group"""
    huggingface_id = request.huggingface_id
    models = db.query(Model).filter(Model.huggingface_id == huggingface_id).all()
    if not models:
        raise HTTPException(status_code=404, detail="Model group not found")
    
    deleted_count = 0
    for model in models:
        # Stop if running
        running_instance = db.query(RunningInstance).filter(RunningInstance.model_id == model.id).first()
        if running_instance:
            try:
                await runtime_manager.stop_model(model.id)
            except Exception as e:
                logger.warning("Failed to stop model %s before deletion: %s", model.id, e)
            db.delete(running_instance)
        
        # Delete file
        if model.file_path and os.path.exists(model.file_path):
            os.remove(model.file_path)
        
        # Delete from database
        db.delete(model)
        deleted_count += 1
    
    db.commit()
    
    return {"message": f"Deleted {deleted_count} quantizations"}


@router.delete("/{model_id}")
async def delete_model(
    model_id: int,
    runtime_manager: CandleRuntimeManager = Depends(get_runtime_manager),
    db: Session = Depends(get_db),
):
    """Delete individual model quantization and its files"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Stop if running
    running_instance = db.query(RunningInstance).filter(RunningInstance.model_id == model_id).first()
    if running_instance:
        try:
            await runtime_manager.stop_model(model_id)
        except Exception as e:
            logger.warning("Failed to stop model %s before deletion: %s", model_id, e)
        db.delete(running_instance)
    
    # Delete file
    if model.file_path and os.path.exists(model.file_path):
        os.remove(model.file_path)
    
    # Delete from database
    db.delete(model)
    db.commit()
    
    return {"message": "Model quantization deleted"}


@router.get("/{model_id}/layer-info")
async def get_model_layer_info_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get model layer information from GGUF metadata"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        layer_info = get_model_layer_info(model.file_path)
        if layer_info:
            return {
                "layer_count": layer_info["layer_count"],
                "architecture": layer_info["architecture"],
                "context_length": layer_info["context_length"],
                "vocab_size": layer_info["vocab_size"],
                "embedding_length": layer_info["embedding_length"],
                "attention_head_count": layer_info["attention_head_count"],
                "attention_head_count_kv": layer_info["attention_head_count_kv"],
                "block_count": layer_info["block_count"],
                "is_moe": layer_info.get("is_moe", False),
                "expert_count": layer_info.get("expert_count", 0),
                "experts_used_count": layer_info.get("experts_used_count", 0)
            }
        else:
            # Fallback to default values if metadata reading fails
            return {
                "layer_count": 32,
                "architecture": "unknown",
                "context_length": 0,
                "vocab_size": 0,
                "embedding_length": 0,
                "attention_head_count": 0,
                "attention_head_count_kv": 0,
                "block_count": 0,
                "is_moe": False,
                "expert_count": 0,
                "experts_used_count": 0
            }
    except Exception as e:
        logger.error(f"Failed to get layer info for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read model metadata: {str(e)}")


@router.get("/{model_id}/recommendations")
async def get_model_recommendations_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get configuration recommendations for a model based on its architecture"""
    from backend.smart_auto.recommendations import get_model_recommendations
    
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        # Get layer info from GGUF metadata
        layer_info = get_model_layer_info(model.file_path)
        if not layer_info:
            # Fallback to basic defaults
            layer_info = {
                "layer_count": 32,
                "architecture": "unknown",
                "context_length": 0,
                "attention_head_count": 0,
                "embedding_length": 0
            }
        
        # Get recommendations using smart_auto with balanced preset
        recommendations = await get_model_recommendations(
            model_layer_info=layer_info,
            model_name=model.name or model.huggingface_id or "",
            file_path=model.file_path
        )
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get recommendations for model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/{model_id}/architecture-presets")
async def get_architecture_presets_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get architecture-specific presets for a model"""
    model = db.query(Model).filter(Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    architecture, presets = get_architecture_and_presets(model)
    return {
        "architecture": architecture,
        "presets": presets,
        "available_presets": list(presets.keys())
    }


@router.post("/{model_id}/regenerate-info")
async def regenerate_model_info_endpoint(
    model_id: int,
    db: Session = Depends(get_db)
):
    """
    Regenerate model information from GGUF metadata and update the database.
    This will re-read the model file and update architecture, layer count, and other metadata.
    """
    model = db.query(Model).filter(Model.id == model_id).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        # Read GGUF metadata from the model file
        layer_info = get_model_layer_info(model.file_path)
        
        if not layer_info:
            raise HTTPException(status_code=500, detail="Failed to read model metadata from file")
        
        # Get normalized architecture
        from backend.smart_auto.architecture_config import normalize_architecture, detect_architecture_from_name
        
        raw_architecture = layer_info.get("architecture", "")
        normalized_architecture = normalize_architecture(raw_architecture)
        
        # Fallback to name-based detection if normalization failed
        if not normalized_architecture or normalized_architecture == "unknown":
            normalized_architecture = detect_architecture_from_name(model.name or model.huggingface_id or "")
        
        # Update model information in database
        update_fields = {}
        
        # Update model_type (architecture)
        if normalized_architecture and normalized_architecture != "unknown":
            update_fields["model_type"] = normalized_architecture
            logger.info(f"Updating model {model_id} model_type to: {normalized_architecture}")
        
        # Update file_size if changed (model might have been updated)
        if os.path.exists(model.file_path):
            file_size = os.path.getsize(model.file_path)
            if file_size != model.file_size:
                update_fields["file_size"] = file_size
                logger.debug(f"Updating model {model_id} file_size from {model.file_size} to {file_size}")
        
        # Apply updates
        if update_fields:
            for key, value in update_fields.items():
                setattr(model, key, value)
            db.commit()
            logger.info(f"Successfully regenerated model info for model {model_id}: {update_fields}")
        
        # Return the updated model info
        return {
            "success": True,
            "model_id": model_id,
            "updated_fields": update_fields,
            "metadata": {
                "architecture": normalized_architecture,
                "layer_count": layer_info.get("layer_count", 0),
                "context_length": layer_info.get("context_length", 0),
                "vocab_size": layer_info.get("vocab_size", 0),
                "embedding_length": layer_info.get("embedding_length", 0),
                "attention_head_count": layer_info.get("attention_head_count", 0),
                "attention_head_count_kv": layer_info.get("attention_head_count_kv", 0),
                "block_count": layer_info.get("block_count", 0),
                "is_moe": layer_info.get("is_moe", False),
                "expert_count": layer_info.get("expert_count", 0),
                "experts_used_count": layer_info.get("experts_used_count", 0),
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to regenerate model info for model {model_id}: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to regenerate model info: {str(e)}")


@router.get("/supported-flags")
async def get_supported_flags_endpoint(db: Session = Depends(get_db)):
    """Report runtime flag capabilities (not applicable for candle)."""
    return {
        "supported_flags": [],
        "supported_config_keys": {},
        "binary_path": None,
        "message": "Candle-vllm manages sampling parameters via API requests; no CLI flag discovery available.",
    }
