from huggingface_hub import HfApi, hf_hub_download, list_models
from typing import List, Dict, Optional
import asyncio
import aiohttp
import json
import os
from tqdm import tqdm
import time
import re
import traceback
from datetime import datetime
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Initialize HF API - will be updated with token if provided
hf_api = HfApi()

# Check for environment variable on module initialization
_env_token = os.getenv('HUGGINGFACE_API_KEY')
if _env_token:
    hf_api = HfApi(token=_env_token)
    logger.info("HuggingFace API key loaded from environment variable")

# Simple cache for search results
_search_cache = {}
_cache_timeout = 300  # 5 minutes

def clear_search_cache():
    """Clear the search cache to force fresh results"""
    global _search_cache
    _search_cache = {}

# Rate limiting
_last_request_time = 0
_min_request_interval = 0.5  # Reduced to 0.5 seconds since we're making fewer calls


def _data_path(*parts: str) -> str:
    """Return a path inside the configured data directory."""
    base_dir = os.getenv("CANDLE_STUDIO_DATA", os.path.join(os.getcwd(), "data"))
    return os.path.join(base_dir, *parts)

def _sanitize_filename(filename: str) -> str:
    """Ensure filename is a safe basename without path traversal.
    Raises ValueError if invalid.
    """
    if not filename or filename.strip() == "":
        raise ValueError("filename is required")
    # Normalize and compare to basename
    base = os.path.basename(filename)
    if base != filename or base in (".", ".."):
        raise ValueError("invalid filename")
    # Disallow path separators just in case
    if "/" in base or "\\" in base:
        raise ValueError("invalid filename")
    return base


def _ensure_model_directory(huggingface_id: str) -> str:
    """Create (if needed) and return a safe directory for the given HuggingFace repo."""
    if not huggingface_id or huggingface_id.strip() == "":
        raise ValueError("huggingface_id is required")

    parts = [part.strip() for part in huggingface_id.split("/") if part.strip()]
    if not parts:
        raise ValueError("Invalid huggingface_id")

    sanitized_parts = []
    for part in parts:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", part)
        sanitized_parts.append(safe or "model")

    models_root = _data_path("models")
    os.makedirs(models_root, exist_ok=True)

    target_dir = os.path.join(models_root, *sanitized_parts)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

# Compiled regex patterns for better performance
QUANTIZATION_PATTERNS = [
    re.compile(r'IQ\d+_[A-Z]+'),  # IQ1_S, IQ2_M, etc.
    re.compile(r'Q\d+_K_[A-Z]+'),  # Q4_K_M, Q5_K_S, etc.
    re.compile(r'Q\d+_\d+'),      # Q4_0, Q5_1, etc.
    re.compile(r'Q\d+_K'),        # Q2_K, Q6_K, etc.
    re.compile(r'Q\d+'),          # Q3, Q4, etc. (fallback)
]

# Model size extraction pattern


def set_huggingface_token(token: str):
    """Set HuggingFace API token for authenticated requests"""
    global hf_api
    if token:
        hf_api = HfApi(token=token)
        logger.info("HuggingFace API token set - using authenticated requests")
    else:
        hf_api = HfApi()
        logger.info("HuggingFace API token cleared - using unauthenticated requests")


def get_huggingface_token() -> Optional[str]:
    """Get current HuggingFace API token"""
    return getattr(hf_api, 'token', None)


async def _rate_limit():
    """Async rate limiting to avoid hitting HuggingFace limits"""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    if time_since_last < _min_request_interval:
        sleep_time = _min_request_interval - time_since_last
        logger.warning(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
        await asyncio.sleep(sleep_time)
    _last_request_time = time.time()


async def search_models(query: str, limit: int = 20, model_type: str = "gguf") -> List[Dict]:
    """Search HuggingFace for models. Works with or without a token (unauthenticated if none).
    
    Args:
        query: Search query string
        limit: Maximum number of results to return
        model_type: Filter by model type - "gguf", "safetensors", or "all" (default: "gguf")
    """
    try:
        # Check cache first
        cache_key = f"{query.lower()}_{limit}_{model_type}"
        current_time = time.time()
        
        if cache_key in _search_cache:
            cached_data, cache_time = _search_cache[cache_key]
            if current_time - cache_time < _cache_timeout:
                logger.info(f"Returning cached results for '{query}' (type: {model_type})")
                return cached_data[:limit]  # Return only requested limit
        
        logger.info(f"Searching for models with query: '{query}', limit: {limit}, type: {model_type}")
        # Always attempt API search; authentication will be used automatically if a token is set
        return await _search_with_api(query, limit, model_type)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise Exception(f"Failed to search models: {e}")


async def _search_with_api(query: str, limit: int, model_type: str = "gguf") -> List[Dict]:
    """Search using HuggingFace Hub API (authenticated if token is configured)."""
    try:
        # Apply rate limiting
        await _rate_limit()
        
        # Determine filter based on model_type
        hf_filter = None
        if model_type == "gguf":
            hf_filter = "gguf"
        elif model_type == "safetensors":
            hf_filter = "safetensors"
        # model_type == "all" means no filter
        
        # Use real HuggingFace API search with expand parameter for rich metadata
        # NOTE: expand doesn't include file sizes, we'll fetch those on-demand
        models_generator = list_models(
            search=query,
            limit=min(limit * 2, 50),  # Get more models to filter from
            sort="downloads",
            direction=-1,
            filter=hf_filter,  # Filter based on model type
            expand=["author", "cardData"]  # Basic metadata without siblings to avoid size issues
        )
        
        # Convert generator to list
        models = list(models_generator)
        logger.info(f"Found {len(models)} models from HuggingFace API with expanded metadata (type: {model_type})")
        
        # Process models in parallel for better performance
        results = await _process_models_parallel(models, limit, model_type)
        
        # Cache the results
        cache_key = f"{query.lower()}_{limit}_{model_type}"
        _search_cache[cache_key] = (results, time.time())
        
        logger.info(f"Returning {len(results)} results from API (type: {model_type})")
        return results
        
    except Exception as e:
        logger.error(f"API search error: {e}")
        # Return empty results if API fails
        return []


async def _process_models_parallel(models: List, limit: int, model_type: str = "gguf", max_concurrent: int = 5) -> List[Dict]:
    """Process models in parallel with semaphore for concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_model(model):
        async with semaphore:
            return await _process_single_model(model, model_type)
    
    # Create tasks for all models
    tasks = [process_model(model) for model in models[:limit * 2]]
    
    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and None results
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Model processing error: {result}")
            continue
        if result is not None:
            valid_results.append(result)
    
    return valid_results[:limit]


async def _process_single_model(model, model_type: str = "gguf") -> Optional[Dict]:
    """Process a single model and extract all metadata
    
    Args:
        model: Model info from HuggingFace API
        model_type: Type of model to extract - "gguf", "safetensors", or "all"
    """
    try:
        logger.info(f"Processing model: {model.id} (type: {model_type})")
        
        # Fetch full model info with file sizes (siblings from expand don't have sizes)
        try:
            full_model_info = hf_api.model_info(model.id, files_metadata=True)
            siblings = full_model_info.siblings if hasattr(full_model_info, 'siblings') else []
        except Exception as e:
            logger.warning(f"Could not fetch full model info for {model.id}: {e}")
            siblings = model.siblings if hasattr(model, 'siblings') else []
        
        # Extract files based on model_type
        target_files = []
        quantizations = {}
        
        if siblings:
            if model_type == "gguf" or model_type == "all":
                # Extract GGUF files
                gguf_files = [sibling.rfilename for sibling in siblings 
                              if sibling.rfilename.endswith('.gguf')]
                target_files.extend(gguf_files)
                
                logger.info(f"Model {model.id}: {len(gguf_files)} GGUF files found")
                
                # Extract GGUF quantizations
                for file in gguf_files:
                    quantization = _extract_quantization(file)
                    if quantization == "unknown":
                        continue
                    
                    quantizations[quantization] = {
                        "filename": file,
                        "format": "gguf"
                    }
                    logger.info(f"Found GGUF quantization: {quantization} for file: {file}")
            
            if model_type == "safetensors" or model_type == "all":
                # For safetensors, we already have the metadata extracted
                # Just mark that this model has safetensors
                pass
        
        # For GGUF-only searches, skip if no GGUF quantizations found
        if model_type == "gguf" and not quantizations:
            return None
        
        # For safetensors-only searches, skip if no safetensors
        if model_type == "safetensors":
            # Check if model has safetensors in siblings
            has_safetensors = False
            if siblings:
                has_safetensors = any(s.rfilename.endswith('.safetensors') for s in siblings)
            
            if not has_safetensors:
                return None
        
        # Extract rich metadata from model and cardData
        # Create a temporary model object with updated siblings that have file sizes
        model_with_sizes = type('obj', (object,), {
            'id': model.id,
            'siblings': siblings,
            'cardData': model.cardData if hasattr(model, 'cardData') else None,
            'pipeline_tag': getattr(model, 'pipeline_tag', ''),
            'library_name': getattr(model, 'library_name', ''),
            'gated': getattr(model, 'gated', False),
            'private': getattr(model, 'private', False),
            'createdAt': getattr(model, 'createdAt', None),
            'lastModified': getattr(model, 'lastModified', None),
        })()
        
        metadata = _extract_model_metadata(model_with_sizes)
        
        result = {
            "id": model.id,
            "name": getattr(model, 'modelId', model.id),  # Use modelId if available, fallback to id
            "author": getattr(model, 'author', ''),
            "downloads": model.downloads,
            "likes": getattr(model, 'likes', 0),
            "tags": model.tags or [],
            "model_type": model_type,
            **metadata  # Include all extracted metadata (includes safetensors info)
        }
        
        # Only add quantizations if we found GGUF files
        if quantizations:
            result["quantizations"] = quantizations
        
        logger.info(f"Added model {model.id} to results (type: {model_type})")
        return result
        
    except Exception as e:
        logger.error(f"Error processing model {model.id}: {e}")
        return None


def _extract_model_metadata(model) -> Dict:
    """Extract rich metadata from ModelInfo and cardData"""
    metadata = {
        "description": "",
        "license": "",
        "pipeline_tag": getattr(model, 'pipeline_tag', ''),
        "library_name": getattr(model, 'library_name', ''),
        "language": [],
        "base_model": "",
        "architecture": "",
        "parameters": "",
        "context_length": None,
        "gated": getattr(model, 'gated', False),
        "private": getattr(model, 'private', False),
        "readme_url": f"https://huggingface.co/{model.id}",
        "created_at": getattr(model, 'createdAt', None),
        "updated_at": getattr(model, 'lastModified', None),
        "safetensors": {}
    }
    
    # Extract from cardData if available
    if hasattr(model, 'cardData') and model.cardData:
        card_data = model.cardData
        
        # Ensure card_data is not None and is a dict
        if card_data and isinstance(card_data, dict):
            # Extract basic info
            metadata["license"] = card_data.get('license', '')
            language_data = card_data.get('language', [])
            # Ensure language is always an array
            metadata["language"] = language_data if isinstance(language_data, list) else []
            metadata["base_model"] = card_data.get('base_model', '')
            
            # Extract from model_index if available
            model_index = card_data.get('model-index', [])
            if model_index:
                for item in model_index:
                    if isinstance(item, dict):
                        # Extract architecture
                        if 'name' in item:
                            metadata["architecture"] = item['name']
                        
                        # Extract parameters
                        if 'params' in item:
                            metadata["parameters"] = str(item['params'])
                        
                        # Extract context length
                        if 'context_length' in item:
                            metadata["context_length"] = item['context_length']
    
    # Extract model size from filename if not found in cardData
    if not metadata["parameters"]:
        # Try to extract model size from modelId using regex
        import re
        model_id = getattr(model, 'modelId', model.id)
        size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
        if size_match:
            metadata["parameters"] = f"{size_match.group(1)}B"
    
    # Extract safetensors metadata from siblings
    if hasattr(model, 'siblings') and model.siblings:
        metadata["safetensors"] = _extract_safetensors_metadata(model.siblings)
    
    return metadata


def _extract_quantization(filename: str) -> str:
    """Extract quantization from filename using compiled regex patterns.
    Works for GGUF files. Returns 'safetensors' for .safetensors files."""
    if filename.lower().endswith('.safetensors'):
        return "safetensors"
    
    for pattern in QUANTIZATION_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group()
    return "unknown"


def extract_base_model_name(filename: str, huggingface_id: str = None) -> str:
    """Extract base model name from filename or HuggingFace ID.
    Works for both GGUF and safetensors files.
    
    Args:
        filename: The filename (e.g., 'Qwen3-8B-Q4_K_M.gguf' or 'model-00001-of-00002.safetensors')
        huggingface_id: Optional HuggingFace repo ID (e.g., 'Qwen/Qwen3-8B')
    
    Returns:
        Base model name without quantization or shard indicators
    """
    import re
    
    # For safetensors, prefer using the HuggingFace ID if available
    if filename.lower().endswith('.safetensors') and huggingface_id:
        # Use the model name from the repo ID
        parts = huggingface_id.split('/')
        return parts[-1] if parts else filename
    
    # Remove file extension
    name = filename.replace('.gguf', '').replace('.safetensors', '')
    
    # Remove shard indicators for safetensors (e.g., '-00001-of-00002', 'model-00001', etc.)
    name = re.sub(r'-\d{5}(-of-\d{5})?$', '', name)
    name = re.sub(r'\.model\.\d+$', '', name)
    
    # Remove quantization patterns (for GGUF)
    quantization_patterns = [
        r'IQ\d+_[A-Z]+',  # IQ1_S, IQ2_M, etc.
        r'Q\d+_K_[A-Z]+',  # Q4_K_M, Q8_0, etc.
        r'Q\d+_[A-Z]+',   # Q4_0, Q5_1, etc.
        r'Q\d+[K_]?[A-Z]*',  # Q2_K, Q6_K, etc.
        r'Q\d+',  # Q4, Q8, etc.
    ]
    
    for pattern in quantization_patterns:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Clean up any trailing underscores, dots, or hyphens
    name = name.rstrip('._-')
    
    # If name is empty or too short, fall back to HuggingFace ID or original filename
    if not name or len(name) < 2:
        if huggingface_id:
            parts = huggingface_id.split('/')
            return parts[-1] if parts else filename
        return filename.replace('.gguf', '').replace('.safetensors', '')
    
    return name


def extract_model_type(filename: str, huggingface_id: str = None) -> str:
    """Extract model type/architecture from filename or HuggingFace ID.
    Works for both GGUF and safetensors files.
    
    Args:
        filename: The filename
        huggingface_id: Optional HuggingFace repo ID for better detection
    
    Returns:
        Model type (e.g., 'llama', 'mistral', 'qwen', etc.)
    """
    # Combine filename and HuggingFace ID for better detection
    search_text = filename.lower()
    if huggingface_id:
        search_text += " " + huggingface_id.lower()
    
    # Check for common model types
    if "llama" in search_text or "llama-" in search_text:
        return "llama"
    elif "mistral" in search_text:
        return "mistral"
    elif "qwen" in search_text or "qwq" in search_text:
        return "qwen"
    elif "codellama" in search_text:
        return "codellama"
    elif "gemma" in search_text:
        return "gemma"
    elif "phi" in search_text:
        return "phi"
    elif "yi" in search_text:
        return "yi"
    elif "deepseek" in search_text:
        return "deepseek"
    elif "stablelm" in search_text:
        return "stablelm"
    
    return "unknown"


def _extract_safetensors_metadata(siblings) -> Dict:
    """Extract safetensors metadata from siblings if available"""
    safetensors_info = {
        "has_safetensors": False,
        "safetensors_files": [],
        "total_tensors": 0,
        "total_size": 0
    }
    
    if not siblings:
        return safetensors_info
    
    safetensors_files = []
    total_size = 0
    
    for sibling in siblings:
        if sibling.rfilename.endswith('.safetensors'):
            # Get size from sibling object (might be 'size' or 'blob_size' attribute)
            file_size = 0
            if hasattr(sibling, 'size') and sibling.size:
                file_size = sibling.size
            elif hasattr(sibling, 'blob_size') and sibling.blob_size:
                file_size = sibling.blob_size
            
            safetensors_files.append({
                "filename": sibling.rfilename,
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2) if file_size > 0 else 0
            })
            total_size += file_size
    
    if safetensors_files:
        safetensors_info.update({
            "has_safetensors": True,
            "safetensors_files": safetensors_files,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0
        })
    
    return safetensors_info


async def get_model_details(model_id: str) -> Dict:
    """Get detailed model information including config and README"""
    try:
        # Get model info with expanded data
        model_info = hf_api.model_info(model_id, expand=["cardData", "siblings"])
        
        # Extract basic metadata
        metadata = _extract_model_metadata(model_info)
        
        # Add additional details
        details = {
            "id": model_info.id,
            "name": getattr(model_info, 'modelId', model_info.id),  # Use modelId if available, fallback to id
            "author": getattr(model_info, 'author', ''),
            "downloads": model_info.downloads,
            "likes": getattr(model_info, 'likes', 0),
            "tags": model_info.tags or [],
            **metadata
        }
        
        # Try to get config.json for architecture details
        try:
            config_files = [s for s in model_info.siblings if s.rfilename == 'config.json']
            if config_files:
                # Download and parse config.json
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename='config.json',
                    local_dir=_data_path("temp"),
                    local_dir_use_symlinks=False
                )
                
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Extract architecture details
                details["config"] = {
                    "architectures": config.get('architectures', []),
                    "model_type": config.get('model_type', ''),
                    "hidden_size": config.get('hidden_size'),
                    "num_attention_heads": config.get('num_attention_heads'),
                    "num_hidden_layers": config.get('num_hidden_layers'),
                    "vocab_size": config.get('vocab_size'),
                    "max_position_embeddings": config.get('max_position_embeddings')
                }
                
                # Clean up temp file
                os.remove(config_path)
                
        except Exception as e:
            logger.warning(f"Could not fetch config.json for {model_id}: {e}")
            details["config"] = {}
        
        return details
        
    except Exception as e:
        logger.error(f"Error getting model details for {model_id}: {e}")
        raise Exception(f"Failed to get model details: {e}")


async def download_model(huggingface_id: str, filename: str) -> tuple[str, int]:
    """Download model from HuggingFace"""
    try:
        # Create models directory for this repo
        target_dir = _ensure_model_directory(huggingface_id)
        
        # Sanitize filename
        filename = _sanitize_filename(filename)

        # Download the file
        file_path = hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return file_path, file_size
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


async def download_model_with_websocket_progress(huggingface_id: str, filename: str, 
                                               websocket_manager, task_id: str, total_bytes: int = 0):
    """Download model with WebSocket progress updates by tracking filesystem size"""
    import asyncio
    import time
    
    logger.info(f"=== DOWNLOAD PROGRESS START ===")
    logger.info(f"Download task: {task_id}")
    logger.info(f"HuggingFace ID: {huggingface_id}")
    logger.info(f"Filename: {filename}")
    logger.info(f"Total bytes from search: {total_bytes}")
    logger.info(f"WebSocket manager: {websocket_manager}")
    logger.info(f"Active connections: {len(websocket_manager.active_connections)}")
    
    try:
        target_dir = _ensure_model_directory(huggingface_id)
        
        # Sanitize filename and build path
        filename = _sanitize_filename(filename)
        # Send initial progress
        logger.info(f"Sending initial progress message...")
        await websocket_manager.send_download_progress(
            task_id=task_id,
            progress=0,
            message=f"Starting download of {filename}",
            bytes_downloaded=0,
            total_bytes=total_bytes,
            speed_mbps=0,
            eta_seconds=0,
            filename=filename
        )
        logger.info(f"Initial progress message sent")
        
        # Get file size from HuggingFace API if not provided
        if total_bytes == 0:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                file_info = api.repo_file_info(repo_id=huggingface_id, path=filename)
                total_bytes = file_info.size
                logger.info(f"Got file size from HuggingFace API: {total_bytes}")
            except Exception as e:
                logger.warning(f"Could not get file size from HuggingFace API: {e}")
                # If we can't get the size, we'll estimate it
                total_bytes = 0
        
        # Send total size update
        if total_bytes > 0:
            await websocket_manager.send_download_progress(
                task_id=task_id,
                progress=0,
                message=f"Downloading {filename}",
                bytes_downloaded=0,
                total_bytes=total_bytes,
                speed_mbps=0,
                eta_seconds=0,
                filename=filename
            )
        
        # Start the download with built-in progress tracking
        logger.info(f"🚀 Starting download with built-in progress tracking...")
        
        file_path, file_size = await download_with_progress_tracking(
            huggingface_id, filename, target_dir,
            websocket_manager, task_id, total_bytes
        )
        
        # Send final completion
        await websocket_manager.send_download_progress(
            task_id=task_id,
            progress=100,
            message=f"Download completed: {filename}",
            bytes_downloaded=file_size,
            total_bytes=file_size,
            speed_mbps=0,
            eta_seconds=0,
            filename=filename
        )
        
        return file_path, file_size
        
    except Exception as e:
        # Send error notification
        if websocket_manager and task_id:
            await websocket_manager.send_download_progress(
                task_id=task_id,
                progress=0,
                message=f"Download failed: {str(e)}",
                bytes_downloaded=0,
                total_bytes=0,
                speed_mbps=0,
                eta_seconds=0,
                filename=filename
            )
            await websocket_manager.send_notification(
                "error", "Download Failed", f"Failed to download {filename}: {str(e)}", task_id
            )
        raise


async def download_with_progress_tracking(
    huggingface_id: str,
    filename: str,
    destination_dir: str,
    websocket_manager,
    task_id: str,
    total_bytes: int,
    bundle_total_bytes: Optional[int] = None,
    bundle_base_bytes: int = 0,
    bundle_label: Optional[str] = None,
    bundle_start_time: Optional[float] = None,
):
    """Download the file using custom http_get method with progress tracking"""
    try:
        import aiofiles
        
        logger.info(f"📁 Starting download of {filename} ({total_bytes} bytes)")
        
        # Use the standard HuggingFace resolve URL (this is the default/preferred method)
        safe_filename = _sanitize_filename(filename)
        download_url = f"https://huggingface.co/{huggingface_id}/resolve/main/{safe_filename}"
        actual_file_size = total_bytes  # Start with the provided size
        
        # Optionally get exact file size from HuggingFace API
        try:
            api = HfApi()
            file_info = api.repo_file_info(repo_id=huggingface_id, filename=safe_filename)
            if hasattr(file_info, 'size') and file_info.size:
                actual_file_size = file_info.size
                logger.info(f"📊 Got file size from HuggingFace API: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
        except Exception as e:
            logger.debug(f"Could not get file size from API: {e}, using provided size: {total_bytes}")
        
        logger.info(f"📁 Download URL: {download_url}")
        
        # Build headers manually
        hf_headers = {
            "User-Agent": "candle-vllm-studio/1.0.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        
        # Create final destination path
        final_path = os.path.join(destination_dir, safe_filename)
        
        # Custom progress bar that sends WebSocket updates
        class WebSocketProgressBar(tqdm):
            def __init__(self, *args, **kwargs):
                self.websocket_manager = kwargs.pop("websocket_manager")
                self.task_id = kwargs.pop("task_id")
                self.filename = kwargs.pop("filename")
                self.bundle_total_bytes = kwargs.pop("bundle_total_bytes")
                self.bundle_base_bytes = kwargs.pop("bundle_base_bytes")
                self.bundle_label = kwargs.pop("bundle_label")
                self.bundle_start_time = kwargs.pop("bundle_start_time")
                super().__init__(*args, **kwargs)
                now = time.time()
                self.file_start_time = now
                if self.bundle_total_bytes is not None and self.bundle_start_time is None:
                    self.bundle_start_time = now
                self.last_update_time = now
            
            def update(self, n=1):
                super().update(n)
                # Send WebSocket update with current progress
                current_time = time.time()
                should_update = (
                    current_time - self.last_update_time >= 0.5 or self.n == self.total
                )
                if should_update and self.total > 0:
                    current_bytes = int(self.n)

                    if self.bundle_total_bytes:
                        aggregate_bytes = self.bundle_base_bytes + current_bytes
                        total_for_progress = max(self.bundle_total_bytes, 1)
                        progress = int((aggregate_bytes / total_for_progress) * 100)
                        bytes_downloaded = aggregate_bytes
                        total_bytes_report = total_for_progress
                        message = f"Downloading {self.bundle_label or self.filename}"
                        start_time = self.bundle_start_time or self.file_start_time
                    else:
                        progress = int((current_bytes / self.total) * 100)
                        bytes_downloaded = current_bytes
                        total_bytes_report = self.total
                        message = f"Downloading {self.filename}"
                        start_time = self.file_start_time

                    elapsed_time = max(current_time - start_time, 1e-3)
                    speed_bytes_per_sec = bytes_downloaded / elapsed_time
                    speed_mbps = speed_bytes_per_sec / (1024 * 1024)
                    remaining_bytes = max(total_bytes_report - bytes_downloaded, 0)
                    eta_seconds = int(remaining_bytes / speed_bytes_per_sec) if speed_bytes_per_sec > 0 else 0

                    logger.debug(
                        f"📊 Progress: {progress}% ({bytes_downloaded}/{total_bytes_report} bytes) - {speed_mbps:.1f} MB/s"
                    )

                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(
                                self.websocket_manager.send_download_progress(
                                    task_id=self.task_id,
                                    progress=progress,
                                    message=message,
                                    bytes_downloaded=bytes_downloaded,
                                    total_bytes=total_bytes_report,
                                    speed_mbps=speed_mbps,
                                    eta_seconds=eta_seconds,
                                    filename=self.bundle_label or self.filename,
                                )
                            )
                    except Exception as e:
                        logger.error(f"Error sending progress update: {e}")

                    self.last_update_time = current_time
        
        # Create our custom progress bar
        custom_progress_bar = WebSocketProgressBar(
            desc=safe_filename,
            total=actual_file_size,  # Use the actual file size
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            disable=False,
            websocket_manager=websocket_manager,
            task_id=task_id,
            filename=filename,
            bundle_total_bytes=bundle_total_bytes,
            bundle_base_bytes=bundle_base_bytes,
            bundle_label=bundle_label,
            bundle_start_time=bundle_start_time,
        )
        
        # Download using aiohttp with timeout and our custom progress bar
        timeout = aiohttp.ClientTimeout(total=3600, connect=30)  # 1 hour total, 30s connect
        async with aiohttp.ClientSession(headers=hf_headers, timeout=timeout) as session:
            async with session.get(download_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download: HTTP {response.status}")
                
                # Get actual file size from response headers
                content_length = response.headers.get('content-length')
                if content_length:
                    response_size = int(content_length)
                    if response_size != actual_file_size:
                        logger.debug(f"📏 Size difference: API said {actual_file_size}, response says {response_size} (diff: {abs(response_size - actual_file_size)} bytes)")
                        # Use the response size as it's more accurate
                        actual_file_size = response_size
                        custom_progress_bar.total = actual_file_size
                        logger.info(f"📊 Using response size: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
                
                # Download with progress tracking
                # Use 64KB chunks for better performance with large files
                chunk_size = 65536
                downloaded_bytes = 0
                async with aiofiles.open(final_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded_bytes += len(chunk)
                        custom_progress_bar.update(len(chunk))
        
        # Close the progress bar
        custom_progress_bar.close()
        
        logger.info(f"📁 Downloaded to: {final_path}")
        
        # Validate downloaded file size
        file_size = os.path.getsize(final_path)
        if file_size != actual_file_size:
            logger.warning(f"⚠️ Download size mismatch: expected {actual_file_size}, got {file_size}")
            # Allow small differences (like metadata)
            if abs(file_size - actual_file_size) > 1024:  # More than 1KB difference
                raise Exception(f"Download incomplete: expected {actual_file_size} bytes, got {file_size} bytes")
        
        return final_path, file_size
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


async def get_quantization_sizes_from_hf(huggingface_id: str, quantizations: Dict[str, Dict]) -> Dict[str, Dict]:
    """Return actual file sizes for provided quantizations using Hugging Face Hub API.
    Uses the shared hf_api instance and mirrors logic used elsewhere in this module.
    """
    try:
        # Prefer fetching only required files to reduce payload
        filenames = [qd.get("filename") for qd in (quantizations or {}).values() if isinstance(qd, dict) and qd.get("filename")]
        updated: Dict[str, Dict] = {}

        if filenames:
            try:
                # Newer API: batch query specific paths for metadata
                paths_info = hf_api.get_paths_info(repo_id=huggingface_id, paths=filenames)
                # Build lookup
                file_sizes: Dict[str, Optional[int]] = {pi.path: getattr(pi, 'size', None) for pi in paths_info}
            except Exception:
                # Fallback: fetch full metadata once
                model_info = hf_api.model_info(repo_id=huggingface_id, files_metadata=True)
                file_sizes = {}
                if hasattr(model_info, 'siblings') and model_info.siblings:
                    for sibling in model_info.siblings:
                        file_sizes[sibling.rfilename] = getattr(sibling, 'size', None)

            for quant_name, quant_data in (quantizations or {}).items():
                filename = quant_data.get("filename") if isinstance(quant_data, dict) else None
                if not filename:
                    continue
                actual_size = file_sizes.get(filename)
                if actual_size and actual_size > 0:
                    updated[quant_name] = {
                        "filename": filename,
                        "size": actual_size,
                        "size_mb": round(actual_size / (1024 * 1024), 2)
                    }

        return updated
    except Exception as e:
        logger.error(f"Failed to fetch quantization sizes for {huggingface_id}: {e}")
        return {}