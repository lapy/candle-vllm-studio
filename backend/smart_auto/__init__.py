from typing import Any, Dict, Optional, Tuple

import psutil

from backend.logging_config import get_logger
from backend.smart_auto.constants import (
    COMPUTE_OVERHEAD_GB,
    DEFAULT_KV_CACHE_GB,
    HOST_DEFAULT,
    MIN_KV_CACHE_GB,
    MAX_KV_CACHE_GB,
    QUANTIZATION_FACTORS,
    QUANTIZATION_ORDER,
    SPEED_BALANCE_DEFAULT,
    SPEED_THRESHOLDS,
    SPEED_TO_KV_SCALE,
)
from backend.smart_auto.model_metadata import get_model_metadata

logger = get_logger(__name__)


def _bytes_to_gb(value: float) -> float:
    return value / (1024 ** 3)


class SmartAutoConfig:
    """Candle smart auto-configuration."""

    async def generate_config(
        self,
        model,
        gpu_info: Dict[str, Any],
        preset: Optional[str] = None,
        usage_mode: str = "single_user",
        speed_quality: Optional[int] = None,
        use_case: Optional[str] = None,
        debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del preset  # Candle presets will be handled directly once tailored
        speed_quality = self._normalise_speed(speed_quality)

        metadata = get_model_metadata(model)
        available_vram_gb, total_vram_gb = self._vram_stats(gpu_info)
        model_size_gb = (model.file_size or 0) / (1024 ** 3)

        dtype, isq = self._select_quantisation(model, available_vram_gb, model_size_gb, speed_quality)
        kv_cache_gb = self._select_kv_cache(model_size_gb, available_vram_gb, speed_quality)
        max_tokens = self._adjust_max_tokens_by_use_case(
            self._select_max_tokens(metadata),
            use_case,
            metadata,
        )

        if debug is not None:
            debug.update(
                {
                    "model_size_gb": model_size_gb,
                    "available_vram_gb": available_vram_gb,
                    "total_vram_gb": total_vram_gb,
                    "selected_dtype": dtype,
                    "selected_isq": isq,
                    "kv_cache_gb": kv_cache_gb,
                    "max_tokens": max_tokens,
                }
            )

        config: Dict[str, Any] = {
            "weights_path": model.file_path,
            "host": HOST_DEFAULT,
            "port": 0,
            "kvcache_mem_gpu": round(kv_cache_gb, 2),
            "dtype": dtype,
            "isq": isq,
            "max_gen_tokens": max_tokens,
            "usage_mode": usage_mode,
            "speed_quality": speed_quality,
            "use_case": use_case,
            "features": [],
            "extra_args": [],
            "env": {},
            "build_profile": "release",
        }

        return config

    def estimate_vram_usage(
        self,
        model,
        config: Dict[str, Any],
        gpu_info: Dict[str, Any],
        usage_mode: str = "single_user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del usage_mode, metadata
        available_vram_gb, total_vram_gb = self._vram_stats(gpu_info)
        model_size_gb = (model.file_size or 0) / (1024 ** 3)

        quant_key = config.get("isq") or config.get("dtype") or "f16"
        quant_factor = QUANTIZATION_FACTORS.get(str(quant_key).lower(), 1.0)
        model_vram_gb = model_size_gb * quant_factor

        kv_cache_gb = float(config.get("kvcache_mem_gpu", MIN_KV_CACHE_GB))
        total_vram_required = model_vram_gb + kv_cache_gb + COMPUTE_OVERHEAD_GB

        return {
            "model_vram_gb": round(model_vram_gb, 2),
            "kv_cache_vram_gb": round(kv_cache_gb, 2),
            "overhead_vram_gb": COMPUTE_OVERHEAD_GB,
            "total_vram_gb": round(total_vram_required, 2),
            "available_vram_gb": round(available_vram_gb, 2),
            "total_device_vram_gb": round(total_vram_gb, 2),
            "fits_in_vram": total_vram_required <= available_vram_gb * 0.95,
        }

    def estimate_ram_usage(
        self,
        model,
        config: Dict[str, Any],
        usage_mode: str = "single_user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del usage_mode, metadata, config
        model_size_gb = (model.file_size or 0) / (1024 ** 3)

        base_overhead = 1.5  # process + runtime bookkeeping
        estimated_ram = base_overhead + model_size_gb * 0.05

        vm = psutil.virtual_memory()
        available_ram_gb = _bytes_to_gb(vm.available)
        total_ram_gb = _bytes_to_gb(vm.total)

        return {
            "estimated_ram_gb": round(estimated_ram, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "total_ram_gb": round(total_ram_gb, 2),
            "fits_in_ram": estimated_ram <= available_ram_gb * 0.95,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _vram_stats(self, gpu_info: Dict[str, Any]) -> Tuple[float, float]:
        gpus = gpu_info.get("gpus", [])
        if not gpus:
            return 0.0, 0.0
        total = sum(g.get("memory", {}).get("total", 0) for g in gpus)
        available = sum(g.get("memory", {}).get("free", 0) for g in gpus)
        return _bytes_to_gb(available), _bytes_to_gb(total)

    def _normalise_speed(self, speed_quality: Optional[int]) -> int:
        if speed_quality is None:
            return SPEED_BALANCE_DEFAULT
        return max(0, min(100, int(speed_quality)))

    def _speed_bucket(self, speed_quality: int) -> str:
        if speed_quality <= SPEED_THRESHOLDS["speed"]:
            return "speed"
        if speed_quality <= SPEED_THRESHOLDS["balanced"]:
            return "balanced"
        return "quality"

    def _select_quantisation(
        self,
        model,
        available_vram_gb: float,
        model_size_gb: float,
        speed_quality: int,
    ) -> Tuple[str, Optional[str]]:
        file_path = model.file_path or ""
        is_gguf = file_path.endswith(".gguf")

        model_quant = (model.quantization or "").lower() if model.quantization else None
        bucket = self._speed_bucket(speed_quality)

        if is_gguf:
            if model_quant and model_quant in QUANTIZATION_FACTORS:
                return model_quant, None
            # fallback to dtype derived from speed bucket
            return self._preferred_quant(bucket), None

        # Non-quantised weights: choose ISQ target
        quant = self._preferred_quant(bucket)

        # Ensure it actually fits available VRAM
        quant = self._adjust_quant_for_vram(quant, available_vram_gb, model_size_gb)

        return "f16", quant  # dtype f16 while in-situ quantising to target format

    def _preferred_quant(self, bucket: str) -> str:
        if bucket == "speed":
            return "q4k"
        if bucket == "balanced":
            return "q6k"
        return "q8_0"

    def _adjust_quant_for_vram(self, quant: str, available_vram_gb: float, model_size_gb: float) -> str:
        if available_vram_gb <= 0 or model_size_gb <= 0:
            return quant

        for candidate in QUANTIZATION_ORDER:
            factor = QUANTIZATION_FACTORS.get(candidate, 1.0)
            required = model_size_gb * factor + COMPUTE_OVERHEAD_GB
            if required <= max(available_vram_gb * 0.9, 1.0):
                return candidate
        return QUANTIZATION_ORDER[-1]

    def _select_kv_cache(self, model_size_gb: float, available_vram_gb: float, speed_quality: int) -> float:
        if available_vram_gb <= 0:
            return MIN_KV_CACHE_GB

        if model_size_gb < 2:
            base = DEFAULT_KV_CACHE_GB["tiny"]
        elif model_size_gb < 6:
            base = DEFAULT_KV_CACHE_GB["small"]
        elif model_size_gb < 12:
            base = DEFAULT_KV_CACHE_GB["medium"]
        elif model_size_gb < 24:
            base = DEFAULT_KV_CACHE_GB["large"]
        else:
            base = DEFAULT_KV_CACHE_GB["xlarge"]

        bucket = self._speed_bucket(speed_quality)
        scale = SPEED_TO_KV_SCALE[bucket]

        kv_budget = base * scale
        kv_budget = min(kv_budget, available_vram_gb * 0.75)
        kv_budget = max(MIN_KV_CACHE_GB, min(kv_budget, MAX_KV_CACHE_GB))
        return kv_budget

    def _select_max_tokens(self, metadata) -> int:
        context_length = metadata.context_length or 8192
        return max(512, min(4096, context_length // 5))

    def _adjust_max_tokens_by_use_case(self, base_tokens: int, use_case: Optional[str], metadata) -> int:
        if use_case == "code":
            return max(base_tokens, 2048)
        if use_case == "analysis":
            return max(base_tokens, min(metadata.context_length or 16384, 4096))
        if use_case == "creative":
            return min(4096, max(base_tokens, 1536))
        return base_tokens
