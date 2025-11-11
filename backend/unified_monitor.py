import asyncio
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from sqlalchemy.orm import Session

from backend.database import RunningInstance, SessionLocal
from backend.gpu_detector import get_gpu_info
from backend.logging_config import get_logger
from backend.websocket_manager import websocket_manager

logger = get_logger(__name__)


class UnifiedMonitor:
    """Collects system and runtime metrics and broadcasts them via websocket."""

    def __init__(self):
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.update_interval = 2.0
        self.recent_logs = deque(maxlen=100)
        self.subscribers: List[Any] = []

    async def add_subscriber(self, websocket):
        """Accept and register a WebSocket subscriber (minimal implementation)."""
        try:
            await websocket.accept()
        except Exception:
            return
        self.subscribers.append(websocket)

    async def remove_subscriber(self, websocket):
        """Remove a WebSocket subscriber and close if open."""
        try:
            if websocket in self.subscribers:
                self.subscribers.remove(websocket)
            try:
                await websocket.close()
            except Exception:
                pass
        except Exception:
            pass

    async def start_monitoring(self):
        """Start the unified monitoring background task."""
        if self.is_running:
            return

        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Unified monitoring started")

    async def stop_monitoring(self):
        """Stop the unified monitoring background task."""
        self.is_running = False

        if self.monitor_task:
            self.monitor_task.cancel()
        logger.info("Unified monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop that collects all metrics and sends unified stream."""
        while self.is_running:
            try:
                await self._collect_and_send_unified_data()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Unified monitoring error: %s", exc)
                await asyncio.sleep(self.update_interval)

    async def _collect_and_send_unified_data(self):
        """Collect all monitoring data and send as single unified message."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            memory = psutil.virtual_memory()
            data_dir = "data" if os.path.exists("data") else "/app/data"
            try:
                disk = psutil.disk_usage(data_dir)
            except FileNotFoundError:
                disk = psutil.disk_usage("/")

            db = SessionLocal()
            try:
                running_instances = db.query(RunningInstance).all()
                active_instances = []
                for instance in running_instances:
                    active_instances.append({
                        "id": instance.id,
                        "model_id": instance.model_id,
                        "port": instance.port,
                        "pid": instance.pid,
                        "endpoint": instance.endpoint,
                        "build_name": instance.build_name,
                        "started_at": instance.started_at.isoformat() if instance.started_at else None,
                    })
            finally:
                db.close()

            try:
                gpu_info = await get_gpu_info()
                vram_data = None
                if not gpu_info.get("cpu_only_mode", True):
                    vram_data = await self._get_vram_data(gpu_info)
            except Exception as exc:
                logger.error("Failed to get GPU info: %s", exc)
                gpu_info = {"cpu_only_mode": True, "device_count": 0}
                vram_data = None

            unified_data = {
                "type": "unified_monitoring",
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                        "used": memory.used,
                        "free": memory.free,
                        "cached": getattr(memory, "cached", 0),
                        "buffers": getattr(memory, "buffers", 0),
                        "swap_total": psutil.swap_memory().total,
                        "swap_used": psutil.swap_memory().used,
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": (disk.used / disk.total) * 100,
                    },
                },
                "gpu": {
                    "cpu_only_mode": gpu_info.get("cpu_only_mode", True),
                    "device_count": gpu_info.get("device_count", 0),
                    "total_vram": gpu_info.get("total_vram", 0),
                    "available_vram": gpu_info.get("available_vram", 0),
                    "vram_data": vram_data,
                },
                "models": {
                    "running_instances": active_instances,
                },
                "logs": list(self.recent_logs)[-20:],
            }

            await websocket_manager.broadcast(unified_data)
        except Exception as exc:
            logger.error("Error collecting unified monitoring data: %s", exc)

    async def _get_vram_data(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get current VRAM usage data."""
        try:
            import pynvml as nvml

            nvml.nvmlInit()

            device_count = gpu_info.get("device_count", 0)
            total_vram = 0
            used_vram = 0
            gpu_details = []

            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)

                gpu_total = mem_info.total
                gpu_used = mem_info.used
                gpu_free = mem_info.free

                total_vram += gpu_total
                used_vram += gpu_used

                gpu_details.append({
                    "device_id": i,
                    "total": gpu_total,
                    "used": gpu_used,
                    "free": gpu_free,
                    "utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                })

            return {
                "total": total_vram,
                "used": used_vram,
                "free": total_vram - used_vram,
                "percent": (used_vram / total_vram * 100) if total_vram > 0 else 0,
                "gpus": gpu_details,
                "cuda_version": gpu_info.get("cuda_version", "N/A"),
                "device_count": gpu_info.get("device_count", 0),
                "timestamp": time.time(),
            }
        except Exception as exc:
            logger.error("Failed to get VRAM data: %s", exc)
            return {
                "total": 0,
                "used": 0,
                "free": 0,
                "percent": 0,
                "gpus": [],
                "cuda_version": "N/A",
                "device_count": 0,
                "timestamp": time.time(),
            }

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return the most recent log entries."""
        logs = list(self.recent_logs)
        return logs[-limit:]

    def add_log(self, log_event: Dict[str, Any]):
        """Add a log event to the buffer."""
        self.recent_logs.append(log_event)


# Global unified monitor instance
unified_monitor = UnifiedMonitor()