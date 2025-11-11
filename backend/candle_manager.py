import asyncio
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import psutil

from backend.logging_config import get_logger

logger = get_logger(__name__)

CANDLE_BINARY_ENV = "CANDLE_VLLM_BINARY"
CANDLE_SOURCE_PATH_ENV = "CANDLE_VLLM_PATH"
CANDLE_DEFAULT_PORT_RANGE = (3000, 3999)


def _find_free_port(start: int = CANDLE_DEFAULT_PORT_RANGE[0],
                    end: int = CANDLE_DEFAULT_PORT_RANGE[1]) -> int:
    """Return the first free TCP port in the range."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Unable to find a free port in range {start}-{end}. "
        "Adjust CANDLE_DEFAULT_PORT_RANGE or stop conflicting processes."
    )


@dataclass
class RuntimeProcess:
    """Holds runtime process metadata."""

    model_id: int
    port: int
    process: asyncio.subprocess.Process
    config: Dict[str, Any] = field(default_factory=dict)
    log_task: Optional[asyncio.Task] = None
    health_task: Optional[asyncio.Task] = None


class CandleRuntimeManager:
    """
    Manage candle-vllm runtime processes.

    The manager supports two execution modes:
    1. Direct binary execution (preferred) - supply CANDLE_VLLM_BINARY env var.
    2. Cargo execution (fallback) - supply CANDLE_VLLM_PATH env var pointing to the candle-vllm repo.
    """

    def __init__(self):
        self._runtimes: Dict[int, RuntimeProcess] = {}
        self._binary_path = self._resolve_binary_path()
        self._repo_path = self._resolve_repo_path()
        self._data_dir = Path(
            os.getenv("CANDLE_STUDIO_DATA", os.path.join(os.getcwd(), "data"))
        ).expanduser()
        self._builds_dir = self._data_dir / "candle-builds"
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        if not self._binary_path and not self._repo_path:
            logger.warning(
                "candle-vllm binary/path not configured yet. "
                "Set CANDLE_VLLM_BINARY or CANDLE_VLLM_PATH before launching a model."
            )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    async def start_model(
        self,
        model_id: int,
        config: Dict[str, Any],
        websocket_manager=None
    ) -> Dict[str, Any]:
        """
        Launch a candle-vllm runtime for the given model with the supplied configuration.

        Args:
            model_id: Database identifier for the model.
            config: Dict containing runtime parameters produced by Smart Auto.
            websocket_manager: Optional websocket manager for live log streaming.
        """
        if model_id in self._runtimes:
            raise RuntimeError(f"Model {model_id} already running")

        runtime_config = dict(config)  # shallow copy to avoid mutation
        port = runtime_config.get("port") or _find_free_port()
        runtime_config["port"] = port

        cmd, env, workdir = self._build_command(runtime_config)
        logger.info("Starting candle-vllm model_id=%s on port=%s", model_id, port)
        logger.debug("Command: %s", " ".join(cmd))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=env
        )

        await self._wait_for_health(
            port=port,
            timeout=runtime_config.get("startup_timeout", 180),
            health_endpoints=runtime_config.get(
                "health_endpoints", ["/health", "/v1/models"]
            )
        )

        log_task = self._loop.create_task(
            self._stream_logs(model_id, process, websocket_manager)
        )
        health_task = self._loop.create_task(
            self._monitor_health(model_id, port, runtime_config, websocket_manager)
        )

        self._runtimes[model_id] = RuntimeProcess(
            model_id=model_id,
            port=port,
            process=process,
            config=runtime_config,
            log_task=log_task,
            health_task=health_task
        )

        return {
            "model_id": model_id,
            "port": port,
            "pid": process.pid,
            "endpoint": f"http://{runtime_config.get('host', '127.0.0.1')}:{port}/v1"
        }

    async def stop_model(self, model_id: int, force: bool = False) -> None:
        """Stop a running model."""
        runtime = self._runtimes.get(model_id)
        if not runtime:
            return

        process = runtime.process
        try:
            if force:
                logger.warning("Force killing candle-vllm model_id=%s", model_id)
                process.kill()
            else:
                logger.info("Stopping candle-vllm model_id=%s", model_id)
                process.terminate()
            await asyncio.wait_for(process.wait(), timeout=20.0)
        except (ProcessLookupError, asyncio.TimeoutError):
            logger.warning("Process did not terminate gracefully, killing")
            process.kill()
            await process.wait()
        finally:
            if runtime.log_task:
                runtime.log_task.cancel()
            if runtime.health_task:
                runtime.health_task.cancel()
            self._runtimes.pop(model_id, None)

    async def restart_model(self, model_id: int, websocket_manager=None) -> Dict[str, Any]:
        """Restart a running model with the same configuration."""
        runtime = self._runtimes.get(model_id)
        if not runtime:
            raise RuntimeError(f"Model {model_id} not running")

        config = runtime.config.copy()
        await self.stop_model(model_id)
        return await self.start_model(model_id, config, websocket_manager)

    async def stop_all(self) -> None:
        """Stop all running candle-vllm processes."""
        await asyncio.gather(*(self.stop_model(mid) for mid in list(self._runtimes.keys())))

    def get_runtime(self, model_id: int) -> Optional[RuntimeProcess]:
        return self._runtimes.get(model_id)

    def list_runtimes(self) -> Dict[int, RuntimeProcess]:
        return dict(self._runtimes)

    def get_process_stats(self, model_id: int) -> Optional[Dict[str, Any]]:
        runtime = self._runtimes.get(model_id)
        if not runtime:
            return None
        try:
            proc = psutil.Process(runtime.process.pid)
            return {
                "pid": proc.pid,
                "cpu_percent": proc.cpu_percent(interval=0.1),
                "memory_mb": proc.memory_info().rss / (1024 * 1024),
                "status": proc.status()
            }
        except psutil.Error as exc:
            logger.warning("Failed to fetch stats for model %s: %s", model_id, exc)
            return None

    # --------------------------------------------------------------------- #
    # Internal Helpers
    # --------------------------------------------------------------------- #
    def _resolve_binary_path(self) -> Optional[str]:
        binary = os.getenv(CANDLE_BINARY_ENV)
        if not binary:
            return None
        binary_path = Path(binary).expanduser().resolve()
        if not binary_path.exists():
            raise FileNotFoundError(f"CANDLE_VLLM_BINARY points to missing path: {binary_path}")
        if not os.access(binary_path, os.X_OK):
            raise PermissionError(f"CANDLE_VLLM_BINARY is not executable: {binary_path}")
        return str(binary_path)

    def _resolve_repo_path(self) -> Optional[str]:
        repo = os.getenv(CANDLE_SOURCE_PATH_ENV)
        if not repo:
            return None
        repo_path = Path(repo).expanduser().resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"CANDLE_VLLM_PATH points to missing path: {repo_path}")
        cargo_toml = repo_path / "Cargo.toml"
        if not cargo_toml.exists():
            raise FileNotFoundError(
                f"CANDLE_VLLM_PATH ({repo_path}) does not look like the candle-vllm repository."
            )
        return str(repo_path)

    def _build_command(self, config: Dict[str, Any]):
        """Construct the command to launch candle-vllm."""
        binary_path = self._binary_path or self._resolve_binary_from_config(config)
        repo_path = self._repo_path

        if not binary_path and not repo_path:
            raise RuntimeError(
                "Neither CANDLE_VLLM_BINARY nor CANDLE_VLLM_PATH is configured. "
                "Set CANDLE_VLLM_BINARY to a compiled candle-vllm binary or "
                "CANDLE_VLLM_PATH to the candle-vllm source repository."
            )

        host = config.get("host", "0.0.0.0")
        port = int(config["port"])
        weights_path = self._resolve_weights_directory(config)
        weights_file = self._resolve_weights_file(config, weights_path)
        config["weights_path"] = str(weights_path)
        config["weights_file"] = str(weights_file)

        dtype = config.get("dtype")
        isq = config.get("isq")
        kv_mem_gpu_mb = self._normalise_mem_to_mb(config.get("kvcache_mem_gpu"))
        kv_mem_cpu_mb = self._normalise_mem_to_mb(config.get("kvcache_mem_cpu"))
        if kv_mem_gpu_mb is not None:
            config["kvcache_mem_gpu"] = kv_mem_gpu_mb
        if kv_mem_cpu_mb is not None:
            config["kvcache_mem_cpu"] = kv_mem_cpu_mb
        max_num_seqs = config.get("max_num_seqs")
        block_size = config.get("block_size")
        hf_token = config.get("hf_token")
        hf_token_path = config.get("hf_token_path")
        verbose = bool(config.get("verbose"))
        cpu_flag = bool(config.get("cpu"))
        record_conversation = bool(config.get("record_conversation"))
        holding_time = config.get("holding_time")
        multithread = bool(config.get("multithread"))
        log_flag = bool(config.get("log"))
        temperature = config.get("temperature")
        top_p = config.get("top_p")
        min_p = config.get("min_p")
        top_k = config.get("top_k")
        frequency_penalty = config.get("frequency_penalty")
        presence_penalty = config.get("presence_penalty")
        prefill_chunk_size = config.get("prefill_chunk_size")
        fp8_kvcache = bool(config.get("fp8_kvcache"))
        additional_args = config.get("extra_args", [])
        features = config.get("features", [])
        env_overrides = config.get("env", {})

        env = os.environ.copy()
        env.update(env_overrides)

        if binary_path:
            cmd = [binary_path]
            workdir = None
        else:
            cmd = ["cargo", "run"]
            profile = config.get("build_profile", "release")
            if profile == "release":
                cmd.append("--release")
            if features:
                cmd.extend(["--features", ",".join(features)])
            cmd.append("--")
            workdir = self._repo_path

        cmd.extend(["--w", str(weights_path)])
        if weights_file:
            cmd.extend(["--f", str(weights_file)])
        cmd.extend(["--h", str(host)])
        cmd.extend(["--p", str(port)])

        if hf_token:
            cmd.extend(["--hf-token", str(hf_token)])
        if hf_token_path:
            cmd.extend(["--hf-token-path", str(hf_token_path)])
        if verbose:
            cmd.append("--verbose")
        if kv_mem_gpu_mb:
            cmd.extend(["--mem", str(kv_mem_gpu_mb)])
        if kv_mem_cpu_mb:
            cmd.extend(["--kvcache-mem-cpu", str(kv_mem_cpu_mb)])
        if max_num_seqs:
            cmd.extend(["--max-num-seqs", str(int(max_num_seqs))])
        if block_size:
            cmd.extend(["--block-size", str(int(block_size))])
        model_name = config.get("model_name")
        if model_name:
            cmd.extend(["--m", str(model_name)])
        if dtype:
            cmd.extend(["--dtype", str(dtype)])
        if isq:
            cmd.extend(["--isq", str(isq)])
        if cpu_flag:
            cmd.append("--cpu")
        if record_conversation:
            cmd.append("--record-conversation")
        if holding_time:
            cmd.extend(["--holding-time", str(int(holding_time))])
        if multithread:
            cmd.append("--multithread")
        if log_flag:
            cmd.append("--log")
        if temperature is not None:
            cmd.extend(["--temperature", str(float(temperature))])
        if top_p is not None:
            cmd.extend(["--top-p", str(float(top_p))])
        if min_p is not None:
            cmd.extend(["--min-p", str(float(min_p))])
        if top_k is not None:
            cmd.extend(["--top-k", str(int(top_k))])
        if frequency_penalty is not None:
            cmd.extend(["--frequency-penalty", str(float(frequency_penalty))])
        if presence_penalty is not None:
            cmd.extend(["--presence-penalty", str(float(presence_penalty))])
        if prefill_chunk_size:
            cmd.extend(["--prefill-chunk-size", str(int(prefill_chunk_size))])
        if fp8_kvcache:
            cmd.append("--fp8-kvcache")

        device = config.get("device_id")
        if device is not None:
            cmd.extend(["--d", str(device)])

        for arg in additional_args:
            cmd.append(str(arg))

        return cmd, env, workdir

    def _resolve_weights_directory(self, config: Dict[str, Any]) -> Path:
        """Ensure the weights path points to a directory candle-vllm expects."""
        raw_path = config.get("weights_path") or config.get("weights_dir")
        if not raw_path:
            raise RuntimeError("weights_path must be provided in runtime configuration")

        path = Path(str(raw_path)).expanduser()

        if path.is_file() or path.suffix.lower() == ".gguf":
            path = path.parent

        if not path.exists():
            raise FileNotFoundError(f"Weights directory not found: {path}")

        if not path.is_dir():
            raise NotADirectoryError(f"Weights path must be a directory, got: {path}")

        return path

    def _resolve_weights_file(self, config: Dict[str, Any], weights_dir: Path) -> Path:
        """Resolve the GGUF weights file path for candle-vllm."""
        raw_file = config.get("weights_file") or config.get("weights_filename")

        if raw_file:
            candidate = Path(str(raw_file)).expanduser()
            if not candidate.is_absolute():
                candidate = (weights_dir / candidate).resolve()

            if not candidate.exists():
                raise FileNotFoundError(f"Weights file not found at {candidate}")
            if not candidate.is_file():
                raise FileNotFoundError(f"Weights file path must be a file, got: {candidate}")
            return candidate

        gguf_files = sorted(p for p in weights_dir.glob("*.gguf") if p.is_file())
        if not gguf_files:
            raise FileNotFoundError(
                f"No GGUF files found in weights directory {weights_dir}"
            )
        return gguf_files[0]

    def _normalise_mem_to_mb(self, value: Any) -> Optional[int]:
        if value in (None, "", 0, "0"):
            return None
        try:
            mem_val = float(value)
        except (TypeError, ValueError):
            return None
        if mem_val <= 0:
            return None
        if mem_val <= 64:
            return int(mem_val * 1024)
        return int(mem_val)

    def _resolve_binary_from_config(self, config: Dict[str, Any]) -> Optional[str]:
        """Resolve candle-vllm binary from runtime configuration or active builds."""
        explicit = config.get("binary_path")
        if explicit:
            path = Path(explicit).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Configured candle-vllm binary not found: {path}")
            if not os.access(path, os.X_OK):
                raise PermissionError(f"Configured candle-vllm binary is not executable: {path}")
            return str(path)

        build_name = config.get("build_name")
        if not build_name:
            return None

        candidates = []
        if self._builds_dir:
            candidates.append(self._builds_dir / build_name / "candle-vllm")

        for candidate in candidates:
            if candidate.exists():
                if not os.access(candidate, os.X_OK):
                    try:
                        candidate.chmod(candidate.stat().st_mode | 0o111)
                    except PermissionError:
                        logger.warning("Unable to mark candle-vllm binary as executable: %s", candidate)
                return str(candidate)

        logger.error(
            "Build '%s' is active but no candle-vllm binary was found in %s",
            build_name,
            candidates[0].parent if candidates else "unknown location",
        )
        return None

    async def _wait_for_health(
        self,
        port: int,
        timeout: int = 180,
        health_endpoints: Optional[list] = None
    ) -> None:
        """Poll candle-vllm for readiness."""
        endpoints = health_endpoints or ["/v1/models"]
        start = (self._loop.time() if hasattr(self._loop, "time") else asyncio.get_event_loop().time())
        async with httpx.AsyncClient() as client:
            while True:
                elapsed = self._loop.time() - start
                if elapsed > timeout:
                    raise TimeoutError(f"candle-vllm did not become ready within {timeout}s")

                for endpoint in endpoints:
                    url = f"http://127.0.0.1:{port}{endpoint}"
                    try:
                        response = await client.get(url, timeout=5.0)
                        if response.status_code < 500:
                            logger.info("candle-vllm ready (endpoint=%s status=%s)", endpoint,
                                        response.status_code)
                            return
                    except (httpx.ConnectError, httpx.TimeoutException):
                        continue

                await asyncio.sleep(1.0)

    async def _stream_logs(self, model_id: int, process: asyncio.subprocess.Process,
                           websocket_manager):
        """Stream stderr logs to websocket clients."""
        if not websocket_manager or not process.stderr:
            return

        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").rstrip()
                if not text:
                    continue
                await websocket_manager.broadcast({
                    "type": "model_log",
                    "model_id": model_id,
                    "message": text
                })
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Failed streaming logs for model %s: %s", model_id, exc)

    async def _monitor_health(self, model_id: int, port: int, config: Dict[str, Any],
                              websocket_manager):
        """Periodic health checks."""
        interval = config.get("health_interval", 30)
        endpoints = config.get("health_endpoints", ["/health"])
        async with httpx.AsyncClient() as client:
            try:
                while True:
                    for endpoint in endpoints:
                        url = f"http://127.0.0.1:{port}{endpoint}"
                        try:
                            response = await client.get(url, timeout=10.0)
                            if response.status_code >= 500:
                                raise RuntimeError(
                                    f"Health endpoint {endpoint} returned {response.status_code}"
                                )
                        except Exception as exc:
                            logger.error("Health check failed for model %s: %s", model_id, exc)
                            if websocket_manager:
                                await websocket_manager.send_model_status_update(
                                    model_id=model_id,
                                    status="unhealthy",
                                    details={"message": str(exc)}
                                )
                            return
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                pass



