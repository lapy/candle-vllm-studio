"""Simplified constants for candle smart-auto."""

from typing import Dict, Mapping


QUANTIZATION_FACTORS: Dict[str, float] = {
    "f32": 2.0,
    "f16": 1.0,
    "bf16": 1.0,
    "q8_0": 0.50,
    "q8_1": 0.50,
    "q8": 0.50,
    "q6k": 0.38,
    "q6_k": 0.38,
    "q5k": 0.31,
    "q5_k": 0.31,
    "q5_k_m": 0.31,
    "q5_k_s": 0.31,
    "q5_1": 0.31,
    "q5_0": 0.31,
    "q4k": 0.25,
    "q4_k": 0.25,
    "q4_k_m": 0.25,
    "q4_k_s": 0.23,
    "q4_1": 0.25,
    "q4_0": 0.25,
    "iq4_nl": 0.26,
    "iq4_xs": 0.24,
    "iq4_xxs": 0.22,
    "q3k": 0.19,
    "q3_k": 0.19,
    "iq3_m": 0.18,
    "q2k": 0.13,
    "q2_k": 0.13,
    "iq2_m": 0.12,
}

QUANTIZATION_SYNONYMS: Mapping[str, str] = {
    "q8": "q8_0",
    "q8_0": "q8_0",
    "q8_1": "q8_0",
    "q8_k": "q8_0",
    "q6_k": "q6k",
    "q5_k": "q5k",
    "q5_k_m": "q5k",
    "q5_k_s": "q5k",
    "q4_k": "q4k",
    "q4_k_m": "q4k",
    "q4_k_s": "q4k",
    "q4_1_m": "q4_1",
    "iq4_nl": "q4k",
    "iq4_xs": "q4k",
    "iq4_xxs": "q4k",
    "iq4_s": "q4k",
    "iq3_m": "q3k",
    "iq2_m": "q2k",
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

HOST_DEFAULT = "0.0.0.0"

# Conservative fixed overhead (logs, orchestration, runtime bookkeeping).
RUNTIME_BASE_OVERHEAD_MB = 160

# Expected parameter ranges for candle runtime flags.
PARAMETER_LIMITS: Mapping[str, Dict[str, float]] = {
    "temperature": {"min": 0.0, "max": 2.0},
    "top_p": {"min": 0.1, "max": 1.0},
    "top_k": {"min": 0.0, "max": 256.0},
    "min_p": {"min": 0.0, "max": 0.5},
    "frequency_penalty": {"min": 0.0, "max": 2.0},
    "presence_penalty": {"min": 0.0, "max": 2.0},
    "prefill_chunk_size": {"min": 256.0, "max": 4096.0},
    "block_size": {"min": 64.0, "max": 512.0},
    "max_num_seqs": {"min": 1.0, "max": 16.0},
}

# Architecture defaults used during metadata normalisation.
DEFAULT_CONTEXT_LENGTH = 8192
ARCHITECTURE_CONTEXT_DEFAULTS = {
    "llama3": 8192,
    "llama31": 131072,
    "llama2": 4096,
    "llama": 4096,
    "codellama": 4096,
    "mistral": 8192,
    "gemma": 8192,
    "gemma3": 8192,
    "glm": 8192,
    "glm4": 8192,
    "glm3": 8192,
    "deepseek": 16384,
    "deepseek-v3": 16384,
    "qwen": 8192,
    "qwen2": 131072,
    "qwen3": 131072,
    "phi": 8192,
    "phi3": 8192,
    "mixtral": 32768,
    "mamba": 8192,
    "generic": DEFAULT_CONTEXT_LENGTH,
}

# GPU capability hints derived from vendor documentation.
GPU_CAPABILITY_HINTS: Mapping[str, Dict[str, float]] = {
    "nvidia": {
        # Hopper and newer (SM 8.9+) expose FP8 KV cache acceleration
        "fp8_min_sm": 89.0,
    },
}
