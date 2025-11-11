"""Simplified constants for candle smart-auto."""

from typing import Dict


QUANTIZATION_FACTORS: Dict[str, float] = {
    "f32": 2.0,
    "f16": 1.0,
    "bf16": 1.0,
    "q8_0": 0.50,
    "q6k": 0.38,
    "q5k": 0.31,
    "q5_1": 0.31,
    "q5_0": 0.31,
    "q4k": 0.25,
    "q4_1": 0.25,
    "q4_0": 0.25,
    "q3k": 0.19,
    "q2k": 0.13,
}

QUANTIZATION_ORDER = [
    "f16",
    "q8_0",
    "q6k",
    "q5k",
    "q4k",
    "q3k",
    "q2k",
]

DEFAULT_KV_CACHE_GB = {
    "tiny": 2.0,   # <2 GB model
    "small": 4.0,  # 2-6 GB
    "medium": 6.0, # 6-12 GB
    "large": 8.0,  # 12-24 GB
    "xlarge": 12.0,  # >24 GB
}

MIN_KV_CACHE_GB = 1.0
MAX_KV_CACHE_GB = 32.0

COMPUTE_OVERHEAD_GB = 0.8  # CUDA context, cuBLAS workspace, etc.

SPEED_BALANCE_DEFAULT = 50

SPEED_TO_KV_SCALE = {
    "speed": 0.7,
    "balanced": 1.0,
    "quality": 1.25,
}

SPEED_THRESHOLDS = {
    "speed": 33,
    "balanced": 66,
}

HOST_DEFAULT = "0.0.0.0"

# Conservative fixed overhead (logs, orchestration, runtime bookkeeping).
RUNTIME_BASE_OVERHEAD_MB = 160

# Architecture defaults used during metadata normalisation.
DEFAULT_CONTEXT_LENGTH = 8192
ARCHITECTURE_CONTEXT_DEFAULTS = {
    "llama3": 8192,
    "llama2": 4096,
    "llama": 4096,
    "codellama": 4096,
    "mistral": 8192,
    "gemma": 8192,
    "gemma3": 8192,
    "glm": 8192,
    "glm4": 8192,
    "deepseek": 16384,
    "deepseek-v3": 16384,
    "qwen": 8192,
    "qwen2": 131072,
    "qwen3": 131072,
    "phi": 8192,
    "generic": DEFAULT_CONTEXT_LENGTH,
}
