"""
GPU Detection and Capability Discovery

This module provides comprehensive GPU detection across multiple vendors:
- NVIDIA GPUs (CUDA support)
- AMD GPUs (ROCm and Vulkan support)
- GPU acceleration backends (CUDA, Vulkan, Metal, OpenBLAS)
"""

try:
    import pynvml  # type: ignore
except ImportError:
    pynvml = None  # NVML support not available (CPU-only environment)

import subprocess
import json
import os
from typing import Dict, List, Optional, Tuple, Any

from backend.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def _decode_bus_id(bus_id: Any) -> str:
    if isinstance(bus_id, bytes):
        return bus_id.decode("utf-8").strip().lower()
    return str(bus_id).strip().lower()


def _topology_level_to_str(level: int) -> str:
    if pynvml is None:
        return "unknown"
    mapping = {
        getattr(pynvml, "NVML_TOPOLOGY_INTERNAL", None): "internal",
        getattr(pynvml, "NVML_TOPOLOGY_SINGLE", None): "single",
        getattr(pynvml, "NVML_TOPOLOGY_MULTIPLE", None): "multiple",
        getattr(pynvml, "NVML_TOPOLOGY_HOSTBRIDGE", None): "hostbridge",
        getattr(pynvml, "NVML_TOPOLOGY_NODE", None): "node",
        getattr(pynvml, "NVML_TOPOLOGY_SYSTEM", None): "system",
    }
    return mapping.get(level, "unknown")


def _collect_cpu_affinity(handle) -> Dict[str, Any]:
    info: Dict[str, Any] = {"cpus": [], "numa_node": None}
    if pynvml is None or not hasattr(pynvml, "nvmlDeviceGetCpuAffinity"):
        return info
    try:
        cpu_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, 1)
        cpus: List[int] = []
        for idx, mask in enumerate(cpu_mask):
            for bit in range(64):
                if mask & (1 << bit):
                    cpus.append(idx * 64 + bit)
        info["cpus"] = cpus
        if cpus:
            # NUMA node heuristic: CPU id // (cpus per node) is expensive; expose first CPU
            info["numa_node"] = cpus[0] // max(1, os.cpu_count() or 1)
    except Exception:
        pass
    return info


def _gather_nvlink_links(handles: List[Any]) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[int]]]:
    nvlink_info: Dict[int, List[Dict[str, Any]]] = {}
    adjacency: Dict[int, List[int]] = {}

    if pynvml is None or not hasattr(pynvml, "NVML_NVLINK_MAX_LINKS"):
        return nvlink_info, adjacency

    pci_map: Dict[str, int] = {}
    for idx, handle in enumerate(handles):
        try:
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_map[_decode_bus_id(pci.busId)] = idx
        except Exception:
            continue

    for idx, handle in enumerate(handles):
        nvlink_info[idx] = []
        adjacency[idx] = []
        max_links = getattr(pynvml, "NVML_NVLINK_MAX_LINKS", 0)
        for link_id in range(max_links):
            try:
                if not pynvml.nvmlDeviceGetNvLinkState(handle, link_id):
                    continue
            except Exception:
                continue
            try:
                remote_pci = pynvml.nvmlDeviceGetNvLinkRemotePciInfo(handle, link_id)
            except Exception:
                continue

            remote_index = pci_map.get(_decode_bus_id(remote_pci.busId))
            if remote_index is None:
                continue

            link_entry: Dict[str, Any] = {"peer": remote_index, "link_id": link_id}
            try:
                version = pynvml.nvmlDeviceGetNvLinkVersion(handle, link_id)
                link_entry["version"] = version
            except Exception:
                link_entry["version"] = None

            try:
                speed = pynvml.nvmlDeviceGetNvLinkSpeed(handle, link_id)
                link_entry["speed_gbps"] = speed
            except Exception:
                link_entry["speed_gbps"] = None

            try:
                cap = pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, link_id, getattr(pynvml, "NVML_NVLINK_CAP_P2P_SUPPORTED", 0)
                )
                link_entry["p2p_supported"] = bool(cap)
            except Exception:
                link_entry["p2p_supported"] = None

            nvlink_info[idx].append(link_entry)
            adjacency[idx].append(remote_index)

    return nvlink_info, adjacency


def _discover_nvml_topology(handles: List[Any]) -> Dict[str, Any]:
    topology: Dict[str, Any] = {"ancestor_matrix": {}, "nvlink_clusters": [], "mixed_topology": False}
    if pynvml is None:
        return topology

    device_count = len(handles)
    nvlink_edges, nvlink_adj = _gather_nvlink_links(handles)
    ancestor_matrix: Dict[str, Dict[str, str]] = {}

    for i, handle_i in enumerate(handles):
        row: Dict[str, str] = {}
        for j, handle_j in enumerate(handles):
            if i == j:
                row[str(j)] = "self"
                continue
            try:
                ancestor = pynvml.nvmlDeviceGetTopologyCommonAncestor(handle_i, handle_j)
                row[str(j)] = _topology_level_to_str(ancestor)
            except Exception:
                row[str(j)] = "unknown"
        ancestor_matrix[str(i)] = row

    visited: set[int] = set()
    clusters: List[List[int]] = []
    for node in range(device_count):
        if node in visited:
            continue
        stack = [node]
        cluster: List[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            for neighbor in nvlink_adj.get(current, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(cluster) > 1:
            clusters.append(sorted(cluster))

    isolated = [idx for idx in range(device_count) if not nvlink_adj.get(idx)]
    topology["ancestor_matrix"] = ancestor_matrix
    topology["nvlink_edges"] = nvlink_edges
    topology["nvlink_clusters"] = clusters
    topology["isolated_gpus"] = isolated
    topology["mixed_topology"] = bool(clusters) and bool(isolated)
    return topology


def _parse_nvidia_smi_topology() -> Optional[Dict[str, Any]]:
    try:
        output = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None

    header_tokens = lines[0].split()
    gpu_labels = [token for token in header_tokens if token.startswith("GPU")]

    matrix: Dict[str, Dict[str, str]] = {}
    cpu_affinity: Dict[str, str] = {}

    for line in lines[1:]:
        tokens = line.split()
        if not tokens:
            continue
        row_label = tokens[0]
        if row_label.startswith("GPU"):
            matrix[row_label] = {}
            for col_label, value in zip(header_tokens[1:], tokens[1:]):
                if col_label.startswith("GPU"):
                    matrix[row_label][col_label] = value
                elif col_label.upper().startswith("CPU"):
                    cpu_affinity[row_label] = value

    return {
        "matrix": matrix,
        "gpu_labels": gpu_labels,
        "cpu_affinity": cpu_affinity,
    }

def _check_vulkan_drivers() -> bool:
    """Check if Vulkan drivers are installed"""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Check if vulkan libraries exist
        return os.path.exists("/usr/share/vulkan") or os.path.exists("/usr/lib/x86_64-linux-gnu/libvulkan.so")


def _check_openblas() -> bool:
    """Check if OpenBLAS is available"""
    try:
        result = subprocess.run(
            ["pkg-config", "--modversion", "openblas"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0
    except:
        # Check if library exists
        return os.path.exists("/usr/lib/x86_64-linux-gnu/libopenblas.so") or \
               os.path.exists("/usr/local/lib/libopenblas.so")


def _check_metal() -> bool:
    """Check if Metal is available (macOS only)"""
    try:
        if os.uname().sysname == "Darwin":
            return os.path.exists("/System/Library/Extensions/GeForceMTLDriver.bundle") or \
                   os.path.exists("/Library/Apple/System/Library/CoreServices/GPUWrangler.app")
    except:
        pass
    return False


# ============================================================================
# NVIDIA GPU Detection
# ============================================================================

def _query_nvml_cuda_version(initialized: bool = False) -> Optional[str]:
    """
    Retrieve CUDA driver version via NVML. If NVML is not initialized, this
    function will initialize and shut it down automatically.
    """
    if pynvml is None:
        return None
    try:
        if not initialized:
            pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetCudaDriverVersion()
        major = version // 1000
        minor = (version % 1000) // 10
        return f"{major}.{minor}"
    except Exception:
        return None
    finally:
        if not initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


async def detect_nvidia_gpu() -> Optional[Dict]:
    """Detect NVIDIA GPUs using pynvml and nvidia-smi"""
    cuda_version_hint: Optional[str] = None
    if pynvml is None:
        return await _detect_nvidia_via_smi()

    try:
        pynvml.nvmlInit()
        cuda_version_hint = _query_nvml_cuda_version(initialized=True)

        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            logger.debug("NVML reported zero NVIDIA devices; falling back to nvidia-smi detection")
            return await _detect_nvidia_via_smi(cuda_version_hint)
        
        gpus = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get basic info
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode('utf-8') if isinstance(raw_name, bytes) else str(raw_name)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get compute capability
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            compute_capability = f"{major}.{minor}"
            
            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
            
            # Get utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = None
                memory_util = None

            pci_info = {}
            bus_id = None
            try:
                pci = pynvml.nvmlDeviceGetPciInfo(handle)
                bus_id = _decode_bus_id(pci.busId)
                pci_info = {
                    "bus_id": bus_id,
                    "domain": getattr(pci, "domain", None),
                    "bus": getattr(pci, "bus", None),
                    "device": getattr(pci, "device", None),
                }
            except Exception:
                pci_info = {}

            pcie_info = {}
            try:
                gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
                width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
                pcie_info = {"max_generation": gen, "max_link_width": width}
            except Exception:
                pcie_info = {}

            affinity_info = _collect_cpu_affinity(handle)
            
            gpu_info = {
                "index": i,
                "name": name,
                "memory": {
                    "total": memory_info.total,
                    "free": memory_info.free,
                    "used": memory_info.used
                },
                "compute_capability": compute_capability,
                "temperature": temperature,
                "utilization": {
                    "gpu": gpu_util,
                    "memory": memory_util
                },
                "pci": pci_info,
                "pcie": pcie_info,
                "topology": {
                    "numa_node": affinity_info.get("numa_node"),
                    "cpu_affinity": affinity_info.get("cpus", []),
                    "nvlink_links": [],
                },
            }
            
            gpus.append(gpu_info)
        
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
        topology_info = _discover_nvml_topology(handles)

        nvlink_edges = topology_info.get("nvlink_edges", {})
        for gpu in gpus:
            gpu_index = gpu["index"]
            gpu["topology"]["nvlink_links"] = nvlink_edges.get(gpu_index, [])

        # Get CUDA version
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
            cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except:
            cuda_version_str = "Unknown"
        
        return {
            "vendor": "nvidia",
            "cuda_version": cuda_version_str,
            "device_count": device_count,
            "gpus": gpus,
            "total_vram": sum(gpu["memory"]["total"] for gpu in gpus),
            "available_vram": sum(gpu["memory"]["free"] for gpu in gpus),
            "cpu_only_mode": device_count == 0,
            "topology": topology_info,
        }
        
    except Exception as exc:
        logger.debug(f"Failed to detect NVIDIA GPUs via NVML: {exc}")
        # Fallback to nvidia-smi
        return await _detect_nvidia_via_smi(cuda_version_hint)
    finally:
        if pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


async def _detect_nvidia_via_smi(cuda_version_hint: Optional[str] = None) -> Optional[Dict]:
    """Fallback NVIDIA detection using nvidia-smi"""
    try:
        query_fields = [
            "index",
            "name",
            "memory.total",
            "memory.free",
            "memory.used",
            "compute_cap",
            "driver_version",
        ]

        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        gpus = []
        lines = result.stdout.strip().split('\n')

        reported_cuda_version = cuda_version_hint
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= len(query_fields):
                # nvidia-smi reports memory in MiB when using nounits
                try:
                    total_bytes = int(parts[2]) * 1024 * 1024
                    free_bytes = int(parts[3]) * 1024 * 1024
                    used_bytes = int(parts[4]) * 1024 * 1024
                except ValueError:
                    total_bytes = free_bytes = used_bytes = 0

                compute_capability = parts[5]
                driver_version = parts[6]
                cuda_version = None
                if len(parts) > 7:
                    cuda_version = parts[7].strip() or cuda_version
                if cuda_version and not reported_cuda_version:
                    reported_cuda_version = cuda_version

                gpu_info = {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory": {
                        "total": total_bytes,
                        "free": free_bytes,
                        "used": used_bytes
                    },
                    "compute_capability": compute_capability,
                    "driver_version": driver_version,
                    "cuda_version": cuda_version,
                }
                gpus.append(gpu_info)

        if reported_cuda_version in (None, "", "Unknown"):
            try:
                detail_output = subprocess.run(
                    ["nvidia-smi", "-q"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                for line in detail_output.splitlines():
                    if "CUDA Version" in line:
                        reported_cuda_version = line.split(":", 1)[1].strip() or reported_cuda_version
                        break
            except subprocess.CalledProcessError:
                pass

        return {
            "vendor": "nvidia",
            "cuda_version": reported_cuda_version or "Unknown",
            "device_count": len(gpus),
            "gpus": gpus,
            "total_vram": sum(gpu["memory"]["total"] for gpu in gpus),
            "available_vram": sum(gpu["memory"]["free"] for gpu in gpus),
            "cpu_only_mode": len(gpus) == 0,
            "topology": _parse_nvidia_smi_topology(),
        }

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    except PermissionError as exc:
        logger.debug("nvidia-smi not accessible: %s", exc)
        return None


# ============================================================================
# AMD GPU Detection
# ============================================================================

async def detect_amd_gpu() -> Optional[Dict]:
    """Detect AMD GPUs using rocm-smi or lspci"""
    try:
        # Try using rocm-smi first
        amd_info = await _detect_amd_via_rocm()
        if amd_info:
            return amd_info
        
        # Fallback to lspci
        amd_info = await _detect_amd_via_lspci()
        if amd_info:
            return amd_info
        
        return None
        
    except Exception:
        return None


async def _detect_amd_via_rocm() -> Optional[Dict]:
    """Detect AMD GPUs using rocm-smi"""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showid", "--showproductname", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True
        )
        data = json.loads(result.stdout)
        
        amd_gpus = []
        if isinstance(data, dict):
            for key, value in data.items():
                if 'GPU[' in key and isinstance(value, dict):
                    gpu_info = {
                        "index": len(amd_gpus),
                        "name": value.get("Card series", "Unknown AMD GPU"),
                        "pci_id": value.get("PCI ID", ""),
                        "memory": {
                            "total": int(value.get("Total Memory (Not-Formatted)", 0)) * 1024 * 1024,
                            "free": int(value.get("Free Memory (Not-Formatted)", 0)) * 1024 * 1024,
                            "used": 0
                        }
                    }
                    amd_gpus.append(gpu_info)
        
        if amd_gpus:
            return {
                "vendor": "amd",
                "device_count": len(amd_gpus),
                "gpus": amd_gpus,
                "total_vram": sum(gpu["memory"]["total"] for gpu in amd_gpus),
                "available_vram": sum(gpu["memory"]["free"] for gpu in amd_gpus),
                "cpu_only_mode": len(amd_gpus) == 0
            }
        
        return None
        
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return None


async def _detect_amd_via_lspci() -> Optional[Dict]:
    """Detect AMD GPUs using lspci"""
    try:
        result = subprocess.run(
            ["lspci", "-v"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        amd_gpus = []
        lines = result.stdout.split('\n')
        current_gpu = None
        
        for line in lines:
            if 'VGA' in line or 'Display' in line or '3D' in line:
                if 'AMD' in line or 'Advanced Micro Devices' in line or 'ATI' in line:
                    parts = line.split(':')
                    if len(parts) >= 3:
                        gpu_name = parts[2].strip()
                        pci_id = parts[0].strip()
                        
                        current_gpu = {
                            "index": len(amd_gpus),
                            "name": gpu_name,
                            "pci_id": pci_id,
                            "memory": {
                                "total": 0,
                                "free": 0,
                                "used": 0
                            }
                        }
        
        if current_gpu and 'AMD' in str(current_gpu.get('name', '')):
            amd_gpus.append(current_gpu)
            
        if amd_gpus:
            return {
                "vendor": "amd",
                "device_count": len(amd_gpus),
                "gpus": amd_gpus,
                "total_vram": sum(gpu.get("memory", {}).get("total", 0) for gpu in amd_gpus),
                "available_vram": sum(gpu.get("memory", {}).get("free", 0) for gpu in amd_gpus),
                "cpu_only_mode": len(amd_gpus) == 0
            }
        
        return None
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# ============================================================================
# Unified GPU Detection
# ============================================================================

async def get_gpu_info() -> Dict[str, any]:
    """Get comprehensive GPU information (tries all vendors)"""
    # Try NVIDIA first
    nvidia_info = await detect_nvidia_gpu()
    if nvidia_info:
        return nvidia_info
    
    # Try AMD
    amd_info = await detect_amd_gpu()
    if amd_info:
        return amd_info
    
    # No GPU detected
    return {
        "vendor": None,
        "cuda_version": "Unknown",
        "device_count": 0,
        "gpus": [],
        "total_vram": 0,
        "available_vram": 0,
        "cpu_only_mode": True
    }


# ============================================================================
# Backend Capability Detection
# ============================================================================

async def detect_build_capabilities() -> Dict[str, Dict[str, any]]:
    """Detect available build backends and their capabilities"""
    gpu_info = await get_gpu_info()
    
    # Determine CUDA availability
    cuda_available = False
    vendor = gpu_info.get("vendor")
    if gpu_info.get("device_count", 0) > 0:
        cuda_available = vendor == "nvidia"
    
    # Check other backends
    metal_available = _check_metal()
    openblas_available = _check_openblas()
    
    # Vulkan is only available if:
    # 1. An AMD GPU is detected AND Vulkan drivers are installed, OR
    # 2. A GPU device directory exists (indicating GPU passthrough in a container)
    vulkan_available = False
    if vendor == "amd":
        # For AMD GPUs, check if Vulkan drivers are available
        vulkan_available = _check_vulkan_drivers()
    elif vendor is None:
        # No specific GPU detected, but check if we have GPU access in a container
        if os.path.exists("/dev/dri"):
            vulkan_available = _check_vulkan_drivers()
    
    # Build capabilities response
    capabilities = {
        "cuda": {
            "available": cuda_available,
            "recommended": cuda_available and not vulkan_available and not openblas_available,
            "reason": f"{gpu_info.get('device_count', 0)} NVIDIA GPU(s) detected" if cuda_available else "No NVIDIA GPU detected"
        },
        "graph": {
            "available": cuda_available,
            "recommended": cuda_available,
            "reason": "CUDA Graphs reduce kernel launch overhead (requires NVIDIA GPU)"
        },
        "vulkan": {
            "available": vulkan_available,
            "recommended": (vulkan_available and not cuda_available) or (gpu_info.get("vendor") == "amd"),
            "reason": "Vulkan drivers available" if vulkan_available else ("Available for AMD GPUs in containers" if gpu_info.get("vendor") == "amd" else "Vulkan drivers not detected")
        },
        "metal": {
            "available": metal_available,
            "recommended": metal_available and not cuda_available and not vulkan_available,
            "reason": "Metal available (macOS)" if metal_available else "Not running on macOS"
        },
        "openblas": {
            "available": openblas_available,
            "recommended": openblas_available and not cuda_available and not vulkan_available,
            "reason": "OpenBLAS library available" if openblas_available else "OpenBLAS not installed"
        }
    }
    
    # Special handling for AMD GPUs
    if gpu_info.get("vendor") == "amd":
        capabilities["cuda"]["reason"] = "AMD GPU detected - use Vulkan instead"
        capabilities["cuda"]["available"] = False  # Explicitly disable CUDA for AMD
        capabilities["graph"]["available"] = False
        capabilities["graph"]["recommended"] = False
        capabilities["graph"]["reason"] = "CUDA Graphs require NVIDIA GPUs (AMD detected)"
        capabilities["vulkan"]["recommended"] = True
        capabilities["vulkan"]["reason"] = f"AMD GPU detected ({gpu_info.get('device_count', 0)} device(s)) - Vulkan recommended"
    
    # If no GPU available, recommend OpenBLAS for CPU acceleration
    if not cuda_available and not vulkan_available and not metal_available and openblas_available:
        capabilities["openblas"]["recommended"] = True
    
    return capabilities


# ============================================================================
# Legacy/Compatibility Functions
# ============================================================================

async def check_vulkan() -> bool:
    """Legacy function for Vulkan check (for backward compatibility)"""
    return _check_vulkan_drivers()


async def detect_gpu_capabilities() -> Dict[str, bool]:
    """Legacy function for GPU capabilities (for backward compatibility)"""
    try:
        gpu_info = await get_gpu_info()
        
        capabilities = {
            "cuda_available": gpu_info.get("device_count", 0) > 0 and gpu_info.get("vendor") == "nvidia",
            "multi_gpu": gpu_info.get("device_count", 0) > 1,
            "flash_attention": False,
            "tensor_parallel": False
        }
        
        if capabilities["cuda_available"]:
            gpus = gpu_info.get("gpus", [])
            
            # Check for flash attention support (Ampere and newer)
            capabilities["flash_attention"] = all(
                gpu.get("compute_capability", "0.0") >= "8.0" 
                for gpu in gpus
            )
            
            # Check for tensor parallelism support
            capabilities["tensor_parallel"] = capabilities["multi_gpu"]
        
        return capabilities
        
    except Exception as e:
        return {
            "cuda_available": False,
            "multi_gpu": False,
            "flash_attention": False,
            "tensor_parallel": False,
            "error": str(e)
        }


# ============================================================================
# NVLink Detection (NVIDIA specific)
# ============================================================================

def get_nvlink_topology(gpus: List[Dict]) -> Dict:
    """
    Analyze NVLink topology for multi-GPU systems
    
    Args:
        gpus: List of GPU info dictionaries
        
    Returns:
        Topology information including clusters and bandwidth
    """
    topology = {
        "has_nvlink": False,
        "clusters": [],
        "total_bandwidth": 0,
        "recommended_strategy": "none"
    }
    
    # This is a simplified implementation
    # Full NVLink detection would parse nvidia-smi topo -m output
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Check if NVLink exists in output
        if "NV" in result.stdout:
            topology["has_nvlink"] = True
            topology["recommended_strategy"] = "nvlink_enabled"
    except:
        pass
    
    return topology
