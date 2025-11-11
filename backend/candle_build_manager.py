import asyncio
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from backend.logging_config import get_logger
from backend.gpu_detector import get_gpu_info

logger = get_logger(__name__)


@dataclass
class CandleBuildConfig:
    """
    Build configuration for candle-vllm.

    High-level options are mapped to cargo features and environment variables.
    """

    build_profile: str = "release"  # "release" or "debug"
    enable_cuda: bool = False
    enable_metal: bool = False
    enable_nccl: bool = False
    enable_flash_attention: bool = False
    enable_graph: bool = False
    enable_marlin: bool = True
    cuda_architectures: str = ""
    custom_features: List[str] = field(default_factory=list)
    custom_rustflags: str = ""
    env: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        if self.enable_cuda and self.enable_metal:
            raise ValueError("CUDA and Metal backends cannot be enabled simultaneously.")
        if self.enable_nccl and not self.enable_cuda:
            logger.warning("NCCL requires CUDA; disabling NCCL feature.")
            self.enable_nccl = False
        if self.enable_flash_attention and not self.enable_cuda:
            logger.warning("Flash Attention requires CUDA; disabling feature.")
            self.enable_flash_attention = False
        if self.enable_graph and not self.enable_cuda:
            logger.warning("CUDA Graphs require CUDA; disabling graph feature.")
            self.enable_graph = False
        if self.enable_marlin and not self.enable_cuda:
            logger.info("Marlin kernels require CUDA; disabling marlin feature.")
            self.enable_marlin = False
        if not self.enable_cuda and not self.enable_metal:
            raise ValueError(
                "candle-vllm currently requires either the CUDA or Metal backend. "
                "Enable one of them before starting a build."
            )

    def cargo_features(self) -> List[str]:
        features: List[str] = []
        if self.enable_cuda:
            features.append("cuda")
            if self.enable_marlin:
                features.append("marlin")
            if self.enable_flash_attention:
                features.append("flash-attn")
            if self.enable_graph:
                features.append("graph")
        if self.enable_metal:
            features.append("metal")
        if self.enable_nccl:
            features.append("nccl")
        features.extend(self.custom_features)
        return sorted(set(features))


async def detect_cuda_architectures() -> Optional[str]:
    """Return CUDA architectures as semicolon separated list (e.g. '80;86')."""
    try:
        gpu_info = await get_gpu_info()
        if gpu_info.get("vendor") != "nvidia":
            return None
        arches = set()
        for gpu in gpu_info.get("gpus", []):
            capability = gpu.get("compute_capability")
            if not capability:
                continue
            arches.add(capability.replace(".", ""))
        if arches:
            return ";".join(sorted(arches))
    except Exception as exc:
        logger.warning("Failed to auto-detect CUDA architectures: %s", exc)
    return None


class CandleBuildManager:
    """Clone, build, and manage candle-vllm binaries."""

    def __init__(self, builds_dir: str = "data/candle-builds"):
        self._builds_dir = Path(builds_dir)
        self._builds_dir.mkdir(parents=True, exist_ok=True)
        # Ensure we have a writable cargo home when running as non-root
        cargo_home = Path(os.environ.get("CARGO_HOME", "/opt/cargo"))
        if not cargo_home.exists():
            cargo_home.mkdir(parents=True, exist_ok=True)
        if not os.access(cargo_home, os.W_OK):
            raise RuntimeError(
                f"CARGO_HOME {cargo_home} is not writable. Check filesystem permissions."
            )

    async def build(
        self,
        git_ref: str,
        config: CandleBuildConfig,
        websocket_manager=None,
        task_id: Optional[str] = None,
    ) -> Dict[str, str]:
        repo_dir = await self._clone_repository(git_ref, websocket_manager, task_id)
        build_name = self._make_build_name(git_ref, config)
        if config.enable_cuda and not config.cuda_architectures:
            config.cuda_architectures = await detect_cuda_architectures() or ""
        features = config.cargo_features()
        env = self._prepare_env(config)

        cmd = ["cargo", "build"]
        if config.build_profile == "release":
            cmd.append("--release")
        if features:
            cmd.extend(["--features", ",".join(features)])

        await self._run_with_progress(
            cmd,
            cwd=repo_dir,
            env=env,
            progress_stage="compile",
            progress_message="Compiling candle-vllm (this may take several minutes)...",
            websocket_manager=websocket_manager,
            task_id=task_id,
        )

        binary = Path(repo_dir) / "target" / config.build_profile / "candle-vllm"
        if not binary.exists():
            raise FileNotFoundError(f"Expected binary not found at {binary}")

        await self._verify_binary(binary)

        dest_dir = self._builds_dir / build_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_binary = dest_dir / "candle-vllm"
        shutil.copy2(binary, dest_binary)
        # Copy runtime resources that candle-vllm expects to find beside the binary.
        resource_dirs = ["assets", "kernels", "metal-kernels", "res"]
        for folder in resource_dirs:
            src = Path(repo_dir) / folder
            if not src.exists():
                continue
            dest = dest_dir / folder
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

        await self._emit_progress(
            websocket_manager,
            task_id,
            stage="complete",
            message=f"Build completed: {dest_binary}",
            progress=100,
        )

        return {
            "build_name": build_name,
            "binary_path": str(dest_binary),
            "git_ref": git_ref,
            "features": ",".join(features),
        }

    def list_builds(self) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for build_dir in self._builds_dir.iterdir():
            binary = build_dir / "candle-vllm"
            if binary.exists():
                stat = binary.stat()
                results.append(
                    {
                        "build_name": build_dir.name,
                        "binary_path": str(binary),
                        "size_mb": f"{stat.st_size / (1024 * 1024):.2f}",
                        "modified": stat.st_mtime,
                    }
                )
        return sorted(results, key=lambda item: item["modified"], reverse=True)

    def delete_build(self, build_name: str) -> bool:
        target = self._builds_dir / build_name
        if target.exists():
            shutil.rmtree(target)
            return True
        return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    async def _clone_repository(self, git_ref: str, websocket_manager, task_id: Optional[str]) -> str:
        await self._emit_progress(
            websocket_manager,
            task_id,
            stage="clone",
            message=f"Cloning candle-vllm ({git_ref})...",
            progress=10,
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="candle-build-")).resolve()
        repo_dir = tmp_dir / "candle-vllm"

        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            git_ref,
            "https://github.com/EricLBuehler/candle-vllm.git",
            str(repo_dir),
        ]
        process = await asyncio.create_subprocess_exec(
            *clone_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.warning("Branch clone failed, falling back to commit checkout.")
            await self._fallback_clone(git_ref, repo_dir)

        return str(repo_dir)

    async def _fallback_clone(self, git_ref: str, repo_dir: Path) -> None:
        process = await asyncio.create_subprocess_exec(
            "git",
            "clone",
            "https://github.com/EricLBuehler/candle-vllm.git",
            str(repo_dir),
        )
        await process.wait()
        checkout = await asyncio.create_subprocess_exec(
            "git",
            "checkout",
            git_ref,
            cwd=str(repo_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await checkout.communicate()
        if checkout.returncode != 0:
            raise RuntimeError(
                f"Failed to checkout git ref '{git_ref}'. stdout={stdout} stderr={stderr}"
            )

    def _make_build_name(self, git_ref: str, config: CandleBuildConfig) -> str:
        backend = "cuda" if config.enable_cuda else "metal" if config.enable_metal else "cpu"
        profile = "rel" if config.build_profile == "release" else "dbg"
        ref_slug = git_ref.replace("/", "-")[:12]
        features = "-".join(config.cargo_features()) or "default"
        return f"{backend}-{profile}-{ref_slug}-{features}"

    def _prepare_env(self, config: CandleBuildConfig) -> Dict[str, str]:
        env = os.environ.copy()
        env.update(config.env)
        if config.custom_rustflags:
            env["RUSTFLAGS"] = config.custom_rustflags
        if config.enable_cuda and config.cuda_architectures:
            env["CUDA_COMPUTE_CAP"] = config.cuda_architectures
        return env

    async def _run_with_progress(
        self,
        cmd: List[str],
        cwd: str,
        env: Dict[str, str],
        progress_stage: str,
        progress_message: str,
        websocket_manager,
        task_id: Optional[str],
    ):
        logger.debug("Executing command: %s (cwd=%s)", " ".join(cmd), cwd)
        await self._emit_progress(
            websocket_manager,
            task_id,
            stage=progress_stage,
            message=progress_message,
            progress=50,
        )

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _stream(pipe, channel: str):
            lines: List[str] = []
            while True:
                line = await pipe.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="ignore").rstrip()
                if text:
                    lines.append(text)
                    if websocket_manager and task_id:
                        await websocket_manager.broadcast(
                            {"type": "build_log", "task_id": task_id, "channel": channel, "message": text}
                        )
            return "\n".join(lines)

        stdout_task = asyncio.create_task(_stream(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(_stream(process.stderr, "stderr"))
        await process.wait()
        stdout_text, stderr_text = await asyncio.gather(stdout_task, stderr_task)

        if stdout_text:
            logger.debug("Command stdout:\n%s", stdout_text)
        if stderr_text:
            logger.debug("Command stderr:\n%s", stderr_text)

        if process.returncode != 0:
            error_message = (
                f"Command {' '.join(cmd)} failed with exit code {process.returncode}."
            )
            if stderr_text:
                error_message += f" stderr={stderr_text.strip()}"
            raise RuntimeError(error_message)

    async def _verify_binary(self, binary: Path) -> None:
        process = await asyncio.create_subprocess_exec(
            str(binary),
            "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(
                f"Validation of compiled binary failed. stdout={stdout} stderr={stderr}"
            )

    async def _emit_progress(
        self,
        websocket_manager,
        task_id: Optional[str],
        stage: str,
        message: str,
        progress: int,
    ):
        if not websocket_manager or not task_id:
            return
        await websocket_manager.send_build_progress(
            task_id=task_id,
            stage=stage,
            progress=progress,
            message=message,
            log_lines=[message],
        )

