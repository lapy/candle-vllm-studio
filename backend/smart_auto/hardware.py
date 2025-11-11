"""Hardware introspection helpers for smart auto heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from backend.smart_auto.constants import GPU_CAPABILITY_HINTS


def _bytes_to_gb(value: float) -> float:
    return value / (1024 ** 3)


@dataclass
class NvLinkPeer:
    peer_index: int
    link_id: int
    version: Optional[int]
    speed_gbps: Optional[float]
    p2p_supported: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_index": self.peer_index,
            "link_id": self.link_id,
            "version": self.version,
            "speed_gbps": self.speed_gbps,
            "p2p_supported": self.p2p_supported,
        }


@dataclass
class GpuDevice:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: Optional[str]
    supports_fp8: bool
    numa_node: Optional[int]
    cpu_affinity: List[int]
    pcie_generation: Optional[int]
    pcie_width: Optional[int]
    nvlink_peers: List[NvLinkPeer]
    ancestor_types: Dict[int, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "free_vram_gb": round(self.free_vram_gb, 2),
            "compute_capability": self.compute_capability,
            "supports_fp8": self.supports_fp8,
            "numa_node": self.numa_node,
            "cpu_affinity": self.cpu_affinity,
            "pcie_generation": self.pcie_generation,
            "pcie_width": self.pcie_width,
            "nvlink_peers": [peer.to_dict() for peer in self.nvlink_peers],
            "ancestor_types": self.ancestor_types,
        }


@dataclass
class HardwareSnapshot:
    """Normalised view over the GPU inventory reported by gpu_detector."""

    vendor: Optional[str]
    gpu_count: int
    total_vram_gb: float
    available_vram_gb: float
    largest_gpu_vram_gb: float
    cpu_only: bool
    supports_fp8_kv: bool
    devices: List[GpuDevice]
    nvlink_clusters: List[List[int]]
    isolated_gpus: List[int]
    mixed_topology: bool
    ancestor_matrix: Dict[str, Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor": self.vendor,
            "gpu_count": self.gpu_count,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "available_vram_gb": round(self.available_vram_gb, 2),
            "largest_gpu_vram_gb": round(self.largest_gpu_vram_gb, 2),
            "cpu_only": self.cpu_only,
            "supports_fp8_kv": self.supports_fp8_kv,
            "devices": [device.to_dict() for device in self.devices],
            "nvlink_clusters": self.nvlink_clusters,
            "isolated_gpus": self.isolated_gpus,
            "mixed_topology": self.mixed_topology,
            "ancestor_matrix": self.ancestor_matrix,
        }


def summarise_hardware(gpu_info: Dict[str, Any]) -> HardwareSnapshot:
    """Collapse the raw gpu_detector payload into a HardwareSnapshot."""
    gpus = gpu_info.get("gpus", []) if isinstance(gpu_info, dict) else []

    total_vram = sum(g.get("memory", {}).get("total", 0) for g in gpus)
    available_vram = sum(g.get("memory", {}).get("free", 0) for g in gpus)
    largest_vram = max(
        (
            g.get("memory", {}).get("total", 0)
            for g in gpus
        ),
        default=0,
    )

    vendor = gpu_info.get("vendor") if isinstance(gpu_info, dict) else None
    cpu_only = bool(gpu_info.get("cpu_only_mode")) if isinstance(gpu_info, dict) else not gpus

    supports_fp8 = any(_supports_fp8_kv(g) for g in gpus)

    topology_info = gpu_info.get("topology", {}) if isinstance(gpu_info, dict) else {}
    ancestor_matrix = topology_info.get("ancestor_matrix", {})
    nvlink_clusters = topology_info.get("nvlink_clusters", [])
    isolated = topology_info.get("isolated_gpus", [])
    mixed = topology_info.get("mixed_topology", False)

    devices: List[GpuDevice] = []
    for gpu in gpus:
        index = gpu.get("index")
        memory = gpu.get("memory", {})
        total_gb = _bytes_to_gb(memory.get("total", 0))
        free_gb = _bytes_to_gb(memory.get("free", 0))
        topology = gpu.get("topology", {}) or {}
        cpu_affinity = topology.get("cpu_affinity", [])
        numa_node = topology.get("numa_node")
        nvlink_links = [
            NvLinkPeer(
                peer_index=peer.get("peer"),
                link_id=peer.get("link_id"),
                version=peer.get("version"),
                speed_gbps=peer.get("speed_gbps"),
                p2p_supported=peer.get("p2p_supported"),
            )
            for peer in topology.get("nvlink_links", [])
            if peer.get("peer") is not None
        ]

        pcie = gpu.get("pcie", {}) or {}
        ancestor_row_raw = ancestor_matrix.get(str(index), {})
        # Convert keys to int for easier downstream processing
        ancestor_row = {}
        for peer_idx_str, relation in ancestor_row_raw.items():
            try:
                ancestor_row[int(peer_idx_str)] = relation
            except ValueError:
                continue

        device = GpuDevice(
            index=index,
            name=gpu.get("name", f"GPU-{index}"),
            total_vram_gb=total_gb,
            free_vram_gb=free_gb,
            compute_capability=gpu.get("compute_capability"),
            supports_fp8=_supports_fp8_kv(gpu),
            numa_node=numa_node,
            cpu_affinity=cpu_affinity,
            pcie_generation=pcie.get("max_generation"),
            pcie_width=pcie.get("max_link_width"),
            nvlink_peers=nvlink_links,
            ancestor_types=ancestor_row,
        )
        devices.append(device)

    return HardwareSnapshot(
        vendor=vendor,
        gpu_count=len(gpus),
        total_vram_gb=_bytes_to_gb(total_vram),
        available_vram_gb=_bytes_to_gb(available_vram),
        largest_gpu_vram_gb=_bytes_to_gb(largest_vram),
        cpu_only=cpu_only or len(gpus) == 0,
        supports_fp8_kv=supports_fp8,
        devices=devices,
        nvlink_clusters=[list(cluster) for cluster in nvlink_clusters],
        isolated_gpus=isolated,
        mixed_topology=bool(mixed),
        ancestor_matrix={key: value for key, value in ancestor_matrix.items()},
    )


def _supports_fp8_kv(gpu: Dict[str, Any]) -> bool:
    """Rudimentary FP8 KV cache support detection based on compute capability."""
    capability = str(gpu.get("compute_capability", "")).strip()
    if not capability:
        return False

    try:
        major, _, minor = capability.partition(".")
        major_i = int(major)
        minor_i = int(minor or "0")
    except ValueError:
        return False

    threshold = GPU_CAPABILITY_HINTS.get("nvidia", {}).get("fp8_min_sm", 89.0)
    encoded = major_i * 10 + minor_i
    if encoded >= threshold:
        return True

    return False

