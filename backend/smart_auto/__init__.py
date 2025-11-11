from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import psutil

from backend.logging_config import get_logger
from backend.smart_auto.constants import (
    COMPUTE_OVERHEAD_GB,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_KV_CACHE_GB,
    HOST_DEFAULT,
    MAX_KV_CACHE_GB,
    MIN_KV_CACHE_GB,
    PARAMETER_LIMITS,
    QUANTIZATION_FACTORS,
    QUANTIZATION_ORDER,
    QUANTIZATION_SYNONYMS,
    SPEED_BALANCE_DEFAULT,
    SPEED_TO_KV_SCALE,
)
from backend.smart_auto.hardware import HardwareSnapshot, summarise_hardware, GpuDevice
from backend.smart_auto.model_metadata import get_model_metadata
from backend.smart_auto.profiles import UsageProfile, resolve_usage_profile
from backend.smart_auto.scoring import (
    clamp_speed_quality,
    interpolate_three,
    map_range,
    speed_bucket,
    weighted_choice,
)

logger = get_logger(__name__)


def _bytes_to_gb(value: float) -> float:
    return value / (1024 ** 3)


def _gb_to_mb(value: float) -> int:
    return int(round(value * 1024))


@dataclass
class ModelProfile:
    name: str
    size_gb: float
    quantization: Optional[str]
    file_path: str
    metadata: Any

    @property
    def is_gguf(self) -> bool:
        return self.file_path.endswith(".gguf")

    @property
    def context_length(self) -> int:
        return self.metadata.context_length or DEFAULT_CONTEXT_LENGTH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size_gb": round(self.size_gb, 2),
            "quantization": self.quantization,
            "context_length": self.context_length,
            "architecture": self.metadata.architecture,
            "is_moe": getattr(self.metadata, "is_moe", False),
        }


@dataclass
class PrecisionChoice:
    dtype: Optional[str]
    isq: Optional[str]
    quant_key: str
    model_vram_gb: float
    fp8_kv_cache: bool


class DecisionLog:
    """Collect reasoning for smart-auto selections."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def record(
        self,
        key: str,
        value: Any,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {"value": value, "reason": reason}
        if context:
            entry["context"] = context
        self._data[key] = entry

    def amend(
        self,
        key: str,
        *,
        value: Optional[Any] = None,
        reason: Optional[str] = None,
        note: Optional[str] = None,
        context_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = self._data.setdefault(key, {})
        if value is not None:
            entry["value"] = value
        if reason is not None:
            entry["reason"] = reason
        if note:
            entry.setdefault("notes", []).append(note)
        if context_update:
            entry.setdefault("context", {}).update(context_update)

    def to_dict(self) -> Dict[str, Any]:
        return self._data


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
        target_concurrency: Optional[int] = None,
        max_tokens_hint: Optional[int] = None,
        restrict_to_nvlink: bool = False,
        debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del preset  # keep hook for future preset-specific rules

        slider_value = clamp_speed_quality(
            speed_quality if speed_quality is not None else SPEED_BALANCE_DEFAULT
        )
        usage_profile = resolve_usage_profile(use_case)
        adjusted_slider = _apply_profile_bias(slider_value, usage_profile)

        metadata = get_model_metadata(model)
        model_name = (
            getattr(model, "name", "") or
            getattr(model, "huggingface_id", "") or
            Path(getattr(model, "file_path", "") or "").stem or
            "unknown"
        )
        model_profile = ModelProfile(
            name=model_name,
            size_gb=(getattr(model, "file_size", 0) or 0) / (1024 ** 3),
            quantization=getattr(model, "quantization", None),
            file_path=(getattr(model, "file_path", "") or "").strip(),
            metadata=metadata,
        )
        hardware = summarise_hardware(gpu_info or {})
        decision_log = DecisionLog()

        precision = self._choose_precision(
            model_profile, hardware, adjusted_slider, decision_log
        )

        concurrency = self._determine_concurrency(
            usage_profile,
            usage_mode,
            adjusted_slider,
            hardware,
            decision_log,
        )
        if target_concurrency is not None:
            requested = max(1, int(target_concurrency))
            max_limit = int(PARAMETER_LIMITS["max_num_seqs"]["max"])
            requested = min(requested, max_limit)
            if requested != concurrency:
                decision_log.amend(
                    "max_num_seqs",
                    value=requested,
                    note="User requested target concurrency override",
                    context_update={"requested": target_concurrency},
                )
            concurrency = requested

        block_limits = PARAMETER_LIMITS["block_size"]
        block_value = interpolate_three(usage_profile.block_size.as_dict(), adjusted_slider)
        block_size = int(round(map_range(block_value, block_limits["min"], block_limits["max"])))

        prefill_limits = PARAMETER_LIMITS["prefill_chunk_size"]
        prefill_value = interpolate_three(usage_profile.prefill_chunk_size.as_dict(), adjusted_slider)
        prefill_candidate = map_range(prefill_value, prefill_limits["min"], prefill_limits["max"])
        prefill_chunk_size = int(max(prefill_limits["min"], round(prefill_candidate / 1024) * 1024))
        if prefill_chunk_size < 1024:
            prefill_chunk_size = 1024

        max_tokens = self._determine_max_tokens(
            metadata,
            usage_profile,
            adjusted_slider,
            usage_mode,
            decision_log,
        )
        if max_tokens_hint is not None:
            requested_tokens = max(usage_profile.min_generation, int(max_tokens_hint))
            max_context = metadata.context_length or DEFAULT_CONTEXT_LENGTH
            cap_hint = min(max_context, 16384)
            requested_tokens = min(cap_hint, requested_tokens)
            if requested_tokens != max_tokens:
                decision_log.amend(
                    "max_gen_tokens",
                    value=requested_tokens,
                    note="User provided max token hint",
                    context_update={"requested": max_tokens_hint},
                )
            max_tokens = requested_tokens

        kv_gpu_mb, kv_cpu_mb, final_concurrency, cache_details = self._determine_kv_cache(
            model_profile,
            precision,
            hardware,
            concurrency,
            usage_mode,
            adjusted_slider,
            decision_log,
        )
        if final_concurrency != concurrency:
            decision_log.amend(
                "max_num_seqs",
                value=final_concurrency,
                note="Adjusted after VRAM budget check",
                context_update={"kv_cache_adjustment": cache_details},
            )
            concurrency = final_concurrency

        sampling = self._determine_sampling(
            usage_profile,
            adjusted_slider,
            decision_log,
        )

        cpu_mode = hardware.cpu_only
        fp8_enabled = precision.fp8_kv_cache and not cpu_mode

        topology_plan = self._build_topology_plan(
            hardware_snapshot=hardware,
            model_profile=model_profile,
            precision_choice=precision,
            fp8_enabled=fp8_enabled,
            usage_mode=usage_mode,
            adjusted_slider=adjusted_slider,
            base_concurrency=concurrency,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            restrict_to_nvlink=restrict_to_nvlink,
        )

        weights_path, weights_file = _resolve_weights_paths(model_profile.file_path)

        config: Dict[str, Any] = {
            "weights_path": weights_path,
            "weights_file": weights_file,
            "host": HOST_DEFAULT,
            "port": 0,
            "kvcache_mem_gpu": kv_gpu_mb,
            "kvcache_mem_cpu": kv_cpu_mb,
            "dtype": precision.dtype,
            "isq": precision.isq,
            "fp8_kvcache": fp8_enabled,
            "max_gen_tokens": max_tokens,
            "max_num_seqs": None if concurrency <= 1 else concurrency,
            "block_size": block_size,
            "prefill_chunk_size": prefill_chunk_size,
            "model_name": model_profile.name,
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "min_p": sampling["min_p"],
            "top_k": sampling["top_k"],
            "frequency_penalty": sampling["frequency_penalty"],
            "presence_penalty": sampling["presence_penalty"],
            "cpu": cpu_mode,
            "multithread": cpu_mode,
            "holding_time": 30 if usage_mode == "multi_user" else None,
            "record_conversation": False,
            "log": False,
            "verbose": False,
            "health_endpoints": ["/v1/models"],
            "health_interval": 30,
            "usage_mode": usage_mode,
            "speed_quality": slider_value,
            "use_case": use_case or usage_profile.key,
            "features": [],
            "extra_args": [],
            "env": {},
            "build_profile": "release",
            "device_id": 0,
            "topology_plan": topology_plan,
        }

        if debug is not None:
            debug.update(
                {
                    "slider_value": slider_value,
                    "adjusted_slider": adjusted_slider,
                    "requested_overrides": {
                        "target_concurrency": target_concurrency,
                        "max_tokens_hint": max_tokens_hint,
                    },
                    "hardware": hardware.to_dict(),
                    "model": model_profile.to_dict(),
                    "precision": {
                        "dtype": precision.dtype,
                        "isq": precision.isq,
                        "quant_key": precision.quant_key,
                        "model_vram_gb": round(precision.model_vram_gb, 2),
                        "fp8_kv_cache": fp8_enabled,
                    },
                    "health": {
                        "endpoints": ["/v1/models"],
                        "interval": 30,
                    },
                    "topology_plan": topology_plan,
                    "decisions": decision_log.to_dict(),
                }
            )

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
        hardware = summarise_hardware(gpu_info or {})
        model_size_gb = (getattr(model, "file_size", 0) or 0) / (1024 ** 3)

        quant_key = _normalise_quant_key(config.get("isq") or config.get("dtype") or "f16")
        quant_factor = QUANTIZATION_FACTORS.get(quant_key, 1.0)
        model_vram_gb = model_size_gb * quant_factor

        kv_cache_gb = _normalise_kv_cache_gb(config.get("kvcache_mem_gpu"))
        total_vram_required = model_vram_gb + kv_cache_gb + COMPUTE_OVERHEAD_GB

        return {
            "model_vram_gb": round(model_vram_gb, 2),
            "kv_cache_vram_gb": round(kv_cache_gb, 2),
            "overhead_vram_gb": COMPUTE_OVERHEAD_GB,
            "total_vram_gb": round(total_vram_required, 2),
            "available_vram_gb": round(hardware.available_vram_gb, 2),
            "total_device_vram_gb": round(hardware.total_vram_gb, 2),
            "fits_in_vram": total_vram_required <= hardware.available_vram_gb * 0.95,
        }

    def estimate_ram_usage(
        self,
        model,
        config: Dict[str, Any],
        usage_mode: str = "single_user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del metadata

        model_size_gb = (getattr(model, "file_size", 0) or 0) / (1024 ** 3)
        concurrency = config.get("max_num_seqs") or (2 if usage_mode == "multi_user" else 1)

        base_overhead = 1.5
        concurrency_penalty = 0.2 * max(0, concurrency - 1)
        estimated_ram = base_overhead + model_size_gb * 0.08 + concurrency_penalty

        vm = psutil.virtual_memory()
        available_ram_gb = _bytes_to_gb(vm.available)
        total_ram_gb = _bytes_to_gb(vm.total)

        return {
            "estimated_ram_gb": round(estimated_ram, 2),
            "available_ram_gb": round(available_ram_gb, 2),
            "total_ram_gb": round(total_ram_gb, 2),
            "fits_in_ram": estimated_ram <= available_ram_gb * 0.95,
        }

    def _build_topology_plan(
        self,
        hardware_snapshot: HardwareSnapshot,
        model_profile: ModelProfile,
        precision_choice: PrecisionChoice,
        fp8_enabled: bool,
        usage_mode: str,
        adjusted_slider: int,
        base_concurrency: int,
        block_size: int,
        prefill_chunk_size: int,
        restrict_to_nvlink: bool,
    ) -> Dict[str, Any]:
        devices = hardware_snapshot.devices
        
        # GGUF models cannot use multi-GPU tensor parallelism
        if model_profile.is_gguf and len(devices) > 1:
            logger.warning(
                "GGUF models do not support multi-GPU inference. "
                "Restricting to single GPU. Use safetensors with --isq for multi-GPU."
            )
            # Force single GPU selection
            devices = devices[:1]
        
        if not devices:
            return {
                "global": {
                    "usage_mode": usage_mode,
                    "tp_strategy": "cpu-only",
                    "total_kv_pool_mb": 0,
                    "recommended_max_num_seqs": base_concurrency,
                    "prefill_chunk_size": prefill_chunk_size,
                    "block_size": block_size,
                    "mixed_topology": False,
                    "warnings": ["No GPU devices detected; running in CPU-only mode."],
                },
                "groups": [],
            }

        total_weight_mb = max(precision_choice.model_vram_gb, model_profile.size_gb) * 1024
        kv_precision = "fp8" if fp8_enabled else "fp16"
        safety_factor = 0.95
        overhead_mb = 1536
        activation_ratio = 0.25

        device_map = {device.index: device for device in devices}
        clusters = [list(cluster) for cluster in (hardware_snapshot.nvlink_clusters or [])]
        clustered_devices = {idx for cluster in clusters for idx in cluster}

        groups: List[Dict[str, Any]] = []
        warnings: List[str] = []

        def describe_group(device_indices: List[int], link_type: str) -> Dict[str, Any]:
            sorted_indices = sorted(device_indices)
            tp_degree = max(1, len(sorted_indices))
            weight_mb_per_gpu = total_weight_mb / tp_degree if tp_degree else total_weight_mb

            group_devices: List[Dict[str, Any]] = []
            group_warnings: List[str] = []
            kv_pool_mb = 0

            for idx in sorted_indices:
                device: GpuDevice = device_map[idx]
                total_vram_mb = device.total_vram_gb * 1024
                free_vram_mb = device.free_vram_gb * 1024
                usable_vram_mb = max(total_vram_mb * 0.98, free_vram_mb)

                activation_mb = weight_mb_per_gpu * activation_ratio
                reserved_mb = weight_mb_per_gpu + activation_mb + overhead_mb
                if reserved_mb > usable_vram_mb:
                    shortfall_mb = int(reserved_mb - usable_vram_mb)
                    group_warnings.append(
                        f"GPU{idx} lacks {shortfall_mb} MB headroom for weights/activations; KV cache allocation set to zero."
                    )
                    kv_budget_mb = 0
                    safety_margin_mb = 0
                else:
                    raw_kv_mb = usable_vram_mb - reserved_mb
                    kv_budget_mb = max(int(raw_kv_mb * safety_factor), 0)
                    safety_margin_mb = int(raw_kv_mb - kv_budget_mb)

                kv_pool_mb += kv_budget_mb

                group_devices.append(
                    {
                        "index": idx,
                        "name": device.name,
                        "total_vram_gb": round(device.total_vram_gb, 2),
                        "free_vram_gb": round(device.free_vram_gb, 2),
                        "weights_mb": int(weight_mb_per_gpu),
                        "activations_mb": int(activation_mb),
                        "overhead_mb": overhead_mb,
                        "kv_budget_mb": int(kv_budget_mb),
                        "safety_margin_mb": int(safety_margin_mb),
                        "numa_node": device.numa_node,
                        "pcie_generation": device.pcie_generation,
                        "pcie_width": device.pcie_width,
                        "nvlink_peers": [peer.to_dict() for peer in device.nvlink_peers],
                        "supports_fp8": device.supports_fp8,
                    }
                )

            expected_penalty_pct = 0
            if link_type != "nvlink" and len(sorted_indices) > 1:
                expected_penalty_pct = 25
            elif link_type == "pcie" and len(sorted_indices) == 1 and hardware_snapshot.mixed_topology:
                expected_penalty_pct = 15

            group_warnings.extend(
                f"GPU{entry['index']} has zero KV cache budget; consider reducing tensor-parallel degree or excluding the device."
                for entry in group_devices
                if entry["kv_budget_mb"] <= 0
            )

            return {
                "id": f"group-{len(groups)}",
                "devices": group_devices,
                "tp_degree": tp_degree,
                "link_type": link_type,
                "kv_precision": kv_precision,
                "kv_pool_mb": int(kv_pool_mb),
                "expected_penalty_pct": expected_penalty_pct,
                "warnings": group_warnings,
            }

        for cluster in clusters:
            group = describe_group(cluster, link_type="nvlink")
            groups.append(group)
            warnings.extend(group["warnings"])

        include_residual = True
        excluded_devices: List[int] = []
        if restrict_to_nvlink and clusters:
            include_residual = False
        if restrict_to_nvlink and not clusters:
            warnings.append(
                "NVLink-only restriction requested but no NVLink groups are available; including all devices."
            )

        for device in devices:
            if device.index not in clustered_devices:
                if not include_residual:
                    excluded_devices.append(device.index)
                    continue
                group = describe_group([device.index], link_type="pcie")
                groups.append(group)
                warnings.extend(group["warnings"])

        if excluded_devices:
            warnings.append(
                "Excluded GPUs from plan due to NVLink-only restriction: "
                + ", ".join(f"GPU{idx}" for idx in sorted(excluded_devices))
            )

        total_kv_pool_mb = sum(group["kv_pool_mb"] for group in groups)
        tp_strategy = "tensor_parallel" if any(group["tp_degree"] > 1 for group in groups) else "replicated"

        if hardware_snapshot.mixed_topology:
            warnings.append(
                "Mixed NVLink/PCIe topology detected; expect synchronization overhead on PCIe-bound shards."
            )

        global_summary = {
            "usage_mode": usage_mode,
            "tp_strategy": tp_strategy,
            "speed_quality": adjusted_slider,
            "total_kv_pool_mb": int(total_kv_pool_mb),
            "recommended_max_num_seqs": base_concurrency,
            "prefill_chunk_size": prefill_chunk_size,
            "block_size": block_size,
            "mixed_topology": hardware_snapshot.mixed_topology,
            "nvlink_clusters": hardware_snapshot.nvlink_clusters,
            "isolated_gpus": hardware_snapshot.isolated_gpus,
            "fp8_enabled": fp8_enabled,
            "model_vram_mb": int(max(precision_choice.model_vram_gb, model_profile.size_gb) * 1024),
            "restrict_to_nvlink": restrict_to_nvlink and bool(clusters),
            "warnings": list(dict.fromkeys(warnings)),
        }

        return {
            "global": global_summary,
            "groups": groups,
        }

    # ------------------------------------------------------------------ #
    # Decision helpers
    # ------------------------------------------------------------------ #
    def _choose_precision(
        self,
        model_profile: ModelProfile,
        hardware: HardwareSnapshot,
        slider: int,
        decisions: DecisionLog,
    ) -> PrecisionChoice:
        bucket = speed_bucket(slider)
        context: Dict[str, Any] = {
            "bucket": bucket,
            "model_quantization": model_profile.quantization,
            "gpu_available_gb": round(hardware.available_vram_gb, 2),
        }

        if hardware.cpu_only:
            dtype = "bf16"
            quant_key = "bf16"
            decisions.record(
                "dtype",
                dtype,
                "CPU-only mode detected; using BF16 runtime",
                context,
            )
            choices = PrecisionChoice(
                dtype=dtype,
                isq=None,
                quant_key=quant_key,
                model_vram_gb=model_profile.size_gb * QUANTIZATION_FACTORS.get(quant_key, 1.0),
                fp8_kv_cache=False,
            )
            decisions.record(
                "isq",
                None,
                "No on-the-fly quantization for CPU runtime",
                context,
            )
            return choices

        if model_profile.is_gguf:
            quant = _normalise_quant_key(model_profile.quantization or bucket)
            if quant not in QUANTIZATION_FACTORS:
                quant = _prefer_quant(bucket)
            decisions.record(
                "dtype",
                None,
                "Model ships as GGUF; using on-disk quantised dtype",
                context,
            )
            return PrecisionChoice(
                dtype=None,
                isq=None,
                quant_key=quant,
                model_vram_gb=model_profile.size_gb * QUANTIZATION_FACTORS.get(quant, 1.0),
                fp8_kv_cache=_should_enable_fp8(hardware, slider),
            )

        preferred = _prefer_quant(bucket)
        quant = _adjust_quant_for_vram(
            preferred,
            hardware.available_vram_gb,
            model_profile.size_gb,
        )
        dtype = "f16"
        quant_key = quant or dtype

        decisions.record(
            "dtype",
            dtype,
            "Raw weights detected; falling back to F16 runtime",
            {**context, "preferred": preferred, "selected": quant_key},
        )
        decisions.record(
            "isq",
            quant,
            "Apply ISQ during load to hit VRAM budget",
            {**context, "preferred": preferred, "selected": quant},
        )

        return PrecisionChoice(
            dtype=dtype,
            isq=quant,
            quant_key=quant or dtype,
            model_vram_gb=model_profile.size_gb * QUANTIZATION_FACTORS.get(quant or dtype, 1.0),
            fp8_kv_cache=_should_enable_fp8(hardware, slider),
        )

    def _determine_concurrency(
        self,
        usage_profile: UsageProfile,
        usage_mode: str,
        slider: int,
        hardware: HardwareSnapshot,
        decisions: DecisionLog,
    ) -> int:
        anchors = (
            usage_profile.concurrency_multi_user
            if usage_mode == "multi_user"
            else usage_profile.concurrency_single_user
        )
        concurrency = int(round(interpolate_three(anchors.as_dict(), slider)))
        concurrency = max(1, concurrency)

        if hardware.cpu_only:
            concurrency = 1
            decisions.record(
                "max_num_seqs",
                concurrency,
                "CPU runtime only supports sequential requests",
                {"usage_mode": usage_mode},
            )
            return concurrency

        concurrency = max(1, concurrency * max(1, hardware.gpu_count))
        if hardware.gpu_count > 1 and usage_mode == "multi_user":
            concurrency = min(concurrency, hardware.gpu_count * 4)

        max_limit = int(PARAMETER_LIMITS["max_num_seqs"]["max"])
        concurrency = min(concurrency, max_limit)

        decisions.record(
            "max_num_seqs",
            concurrency,
            "Derived from usage profile anchors",
            {"usage_mode": usage_mode, "gpu_count": hardware.gpu_count},
        )
        return concurrency

    def _determine_max_tokens(
        self,
        metadata,
        usage_profile: UsageProfile,
        slider: int,
        usage_mode: str,
        decisions: DecisionLog,
    ) -> int:
        context_length = metadata.context_length or DEFAULT_CONTEXT_LENGTH
        ratio = interpolate_three(usage_profile.generation_ratio.as_dict(), slider)
        if usage_mode == "multi_user":
            ratio *= 0.9

        raw_tokens = max(
            usage_profile.min_generation,
            int(round(context_length * ratio / 64)) * 64,
        )
        cap = min(context_length, 16384)
        max_tokens = int(map_range(raw_tokens, usage_profile.min_generation, cap))

        decisions.record(
            "max_gen_tokens",
            max_tokens,
            "Scaled against model context window",
            {"context_length": context_length, "ratio": ratio},
        )
        return max_tokens

    def _determine_kv_cache(
        self,
        model_profile: ModelProfile,
        precision: PrecisionChoice,
        hardware: HardwareSnapshot,
        concurrency: int,
        usage_mode: str,
        slider: int,
        decisions: DecisionLog,
    ) -> Tuple[int, Optional[int], int, Dict[str, Any]]:
        if hardware.cpu_only:
            cpu_cache_gb = max(MIN_KV_CACHE_GB, concurrency * 2.0)
            decisions.record(
                "kvcache_mem_cpu",
                _gb_to_mb(cpu_cache_gb),
                "Allocate KV cache in system memory for CPU runtime",
                {"concurrency": concurrency},
            )
            return (
                0,
                _gb_to_mb(cpu_cache_gb),
                concurrency,
                {"per_sequence_gb": cpu_cache_gb / max(concurrency, 1)},
            )

        max_limit = int(PARAMETER_LIMITS["max_num_seqs"]["max"])
        adjusted_concurrency = min(concurrency, max_limit)

        base_cache = _kv_cache_base(model_profile.size_gb)
        speed_scale = weighted_choice(
            {
                "speed": SPEED_TO_KV_SCALE["speed"],
                "balanced": SPEED_TO_KV_SCALE["balanced"],
                "quality": SPEED_TO_KV_SCALE["quality"],
            },
            slider,
        )
        context_scale = map_range(model_profile.context_length / DEFAULT_CONTEXT_LENGTH, 0.7, 1.6)
        concurrency_scale = 0.9 if usage_mode == "multi_user" and concurrency > 1 else 1.0

        per_sequence_gb = base_cache * speed_scale * context_scale * concurrency_scale
        per_sequence_gb = map_range(per_sequence_gb, MIN_KV_CACHE_GB, MAX_KV_CACHE_GB / max(1, concurrency))

        total_cache_gb = per_sequence_gb * max(1, adjusted_concurrency)
        vram_budget = max(
            0.0,
            hardware.available_vram_gb - precision.model_vram_gb - COMPUTE_OVERHEAD_GB,
        )

        if total_cache_gb > vram_budget and vram_budget > 0:
            per_sequence_gb = max(MIN_KV_CACHE_GB, vram_budget / max(1, adjusted_concurrency))
            total_cache_gb = per_sequence_gb * adjusted_concurrency

            if total_cache_gb > vram_budget:
                max_possible = max(1, int(vram_budget / max(MIN_KV_CACHE_GB, 0.5)))
                adjusted_concurrency = max(1, min(concurrency, max_possible))
                per_sequence_gb = max(
                    MIN_KV_CACHE_GB,
                    vram_budget / max(1, adjusted_concurrency),
                )
                total_cache_gb = per_sequence_gb * adjusted_concurrency

        total_cache_gb = min(total_cache_gb, MAX_KV_CACHE_GB)
        per_sequence_gb = total_cache_gb / max(1, adjusted_concurrency)

        decisions.record(
            "kvcache_mem_gpu",
            _gb_to_mb(total_cache_gb),
            "Balanced KV cache against VRAM budget",
            {
                "per_sequence_gb": round(per_sequence_gb, 3),
                "concurrency": adjusted_concurrency,
                "available_vram_gb": round(hardware.available_vram_gb, 2),
                "model_vram_gb": round(precision.model_vram_gb, 2),
            },
        )

        return (
            _gb_to_mb(total_cache_gb),
            None,
            adjusted_concurrency,
            {
                "total_cache_gb": round(total_cache_gb, 3),
                "per_sequence_gb": round(per_sequence_gb, 3),
                "vram_budget_gb": round(vram_budget, 3),
            },
        )

    def _determine_sampling(
        self,
        usage_profile: UsageProfile,
        slider: int,
        decisions: DecisionLog,
    ) -> Dict[str, Any]:
        temp_limits = PARAMETER_LIMITS["temperature"]
        top_p_limits = PARAMETER_LIMITS["top_p"]
        top_k_limits = PARAMETER_LIMITS["top_k"]
        min_p_limits = PARAMETER_LIMITS["min_p"]
        freq_limits = PARAMETER_LIMITS["frequency_penalty"]
        presence_limits = PARAMETER_LIMITS["presence_penalty"]

        temperature = round(
            map_range(weighted_choice(usage_profile.temperature.as_dict(), slider), temp_limits["min"], temp_limits["max"]),
            3,
        )
        top_p = round(
            map_range(weighted_choice(usage_profile.top_p.as_dict(), slider), top_p_limits["min"], top_p_limits["max"]),
            3,
        )
        top_k = int(
            round(map_range(weighted_choice(usage_profile.top_k.as_dict(), slider), top_k_limits["min"], top_k_limits["max"]))
        )
        min_p = round(
            map_range(weighted_choice(usage_profile.min_p.as_dict(), slider), min_p_limits["min"], min_p_limits["max"]),
            3,
        )
        freq_penalty = round(
            map_range(
                weighted_choice(usage_profile.frequency_penalty.as_dict(), slider),
                freq_limits["min"],
                freq_limits["max"],
            ),
            3,
        )
        presence_penalty = round(
            map_range(
                weighted_choice(usage_profile.presence_penalty.as_dict(), slider),
                presence_limits["min"],
                presence_limits["max"],
            ),
            3,
        )

        decisions.record(
            "sampling",
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "frequency_penalty": freq_penalty,
                "presence_penalty": presence_penalty,
            },
            "Derived from use-case sampling anchors",
        )

        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "frequency_penalty": freq_penalty,
            "presence_penalty": presence_penalty,
        }


# ------------------------------------------------------------------ #
# Utility helpers
# ------------------------------------------------------------------ #
def _apply_profile_bias(slider: int, profile: UsageProfile) -> int:
    bias = int(round(profile.quality_bias * 100))
    return clamp_speed_quality(slider + bias)


def _prefer_quant(bucket: str) -> str:
    if bucket == "speed":
        return "q4k"
    if bucket == "balanced":
        return "q6k"
    return "q8_0"


def _adjust_quant_for_vram(quant: str, available_vram_gb: float, model_size_gb: float) -> str:
    if available_vram_gb <= 0 or model_size_gb <= 0:
        return quant
    for candidate in QUANTIZATION_ORDER:
        factor = QUANTIZATION_FACTORS.get(candidate, 1.0)
        required = model_size_gb * factor + COMPUTE_OVERHEAD_GB
        if required <= max(available_vram_gb * 0.9, 1.0):
            return candidate
    return QUANTIZATION_ORDER[-1]


def _kv_cache_base(model_size_gb: float) -> float:
    if model_size_gb < 2:
        return DEFAULT_KV_CACHE_GB["tiny"]
    if model_size_gb < 6:
        return DEFAULT_KV_CACHE_GB["small"]
    if model_size_gb < 12:
        return DEFAULT_KV_CACHE_GB["medium"]
    if model_size_gb < 24:
        return DEFAULT_KV_CACHE_GB["large"]
    return DEFAULT_KV_CACHE_GB["xlarge"]


def _normalise_kv_cache_gb(value: Any) -> float:
    if value in (None, "", 0, "0"):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric <= 0:
        return 0.0
    if numeric > 128:
        return numeric / 1024.0
    return numeric


def _normalise_quant_key(raw: Optional[str]) -> str:
    if not raw:
        return "f16"
    candidate = raw.lower().strip()
    candidate = candidate.replace("-", "_")
    candidate = QUANTIZATION_SYNONYMS.get(candidate, candidate)
    if candidate in QUANTIZATION_FACTORS:
        return candidate

    collapsed = candidate.replace("_", "")
    collapsed = QUANTIZATION_SYNONYMS.get(collapsed, collapsed)
    for key in QUANTIZATION_FACTORS.keys():
        if collapsed == key.replace("_", ""):
            return key

    while collapsed and not collapsed[-1].isdigit():
        collapsed = collapsed[:-1]
        collapsed = QUANTIZATION_SYNONYMS.get(collapsed, collapsed)
        for key in QUANTIZATION_FACTORS.keys():
            if collapsed == key.replace("_", ""):
                return key

    return candidate


def _should_enable_fp8(hardware: HardwareSnapshot, slider: int) -> bool:
    if not hardware.supports_fp8_kv:
        return False
    # Prefer FP8 when speed is prioritised and VRAM is limited
    return slider <= 40 and hardware.available_vram_gb < 48


def _resolve_weights_paths(file_path: str) -> Tuple[str, Optional[str]]:
    if not file_path:
        return "", None
    expanded = os.path.expanduser(file_path)
    if os.path.isfile(expanded):
        return os.path.dirname(expanded) or ".", expanded
    return expanded, None
