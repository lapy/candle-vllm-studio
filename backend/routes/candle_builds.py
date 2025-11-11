import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi import status as http_status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.candle_build_manager import CandleBuildConfig
from backend.database import CandleBuild, SessionLocal, get_db
from backend.dependencies import get_build_manager
from backend.logging_config import get_logger
from backend.websocket_manager import websocket_manager
from backend.candle_build_manager import CandleBuildManager
from backend.gpu_detector import detect_build_capabilities

logger = get_logger(__name__)

router = APIRouter()


class CandleBuildResponse(BaseModel):
    id: int
    name: str
    git_ref: Optional[str]
    binary_path: Optional[str]
    is_active: bool
    created_at: datetime
    features: List[str]
    has_binary: bool
    size_mb: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BuildRequest(BaseModel):
    git_ref: str = Field(default="master", description="Git branch/tag/commit to build")
    build_profile: str = Field(default="release", pattern="^(release|debug)$")
    enable_cuda: bool = False
    enable_metal: bool = False
    enable_nccl: bool = False
    enable_flash_attention: bool = False
    enable_graph: bool = False
    enable_marlin: bool = True
    cuda_architectures: Optional[str] = Field(
        default=None,
        description="Semicolon separated CUDA architectures (e.g. '80;86')",
    )
    custom_features: List[str] = Field(default_factory=list)
    custom_rustflags: str = ""
    env: Dict[str, str] = Field(default_factory=dict)
    mark_active: bool = False


class BuildRequestPayload(BuildRequest):
    task_id: Optional[str] = None


class ActivateRequest(BaseModel):
    mark_active: bool = True


def _serialize_build(
    build: CandleBuild, artifacts: Dict[str, Dict[str, str]]
) -> CandleBuildResponse:
    feature_list: List[str] = []
    if isinstance(build.features, list):
        feature_list = build.features
    elif isinstance(build.features, str):
        feature_list = [f.strip() for f in build.features.split(",") if f.strip()]

    artifact = artifacts.get(build.name, {})
    metadata = build.build_metadata or {}

    return CandleBuildResponse(
        id=build.id,
        name=build.name,
        git_ref=build.git_ref,
        binary_path=build.binary_path or artifact.get("binary_path"),
        is_active=build.is_active,
        created_at=build.created_at or datetime.utcnow(),
        features=feature_list,
        has_binary=bool(build.binary_path or artifact.get("binary_path")),
        size_mb=artifact.get("size_mb"),
        metadata=metadata,
    )


@router.get("", response_model=List[CandleBuildResponse])
async def list_candle_builds(
    db: Session = Depends(get_db), build_manager: CandleBuildManager = Depends(get_build_manager)
):
    db_builds = (
        db.query(CandleBuild)
        .order_by(CandleBuild.created_at.desc())
        .all()
    )

    artifacts = {item["build_name"]: item for item in build_manager.list_builds()}
    return [_serialize_build(build, artifacts) for build in db_builds]


async def _execute_build_job(
    payload: BuildRequestPayload,
    build_manager: CandleBuildManager,
    task_id: str,
):
    logger.info("Starting candle build job task_id=%s git_ref=%s", task_id, payload.git_ref)
    db = SessionLocal()
    try:
        config = CandleBuildConfig(
            build_profile=payload.build_profile,
            enable_cuda=payload.enable_cuda,
            enable_metal=payload.enable_metal,
            enable_nccl=payload.enable_nccl,
            enable_flash_attention=payload.enable_flash_attention,
            enable_graph=payload.enable_graph,
            enable_marlin=payload.enable_marlin,
            cuda_architectures=payload.cuda_architectures or "",
            custom_features=payload.custom_features,
            custom_rustflags=payload.custom_rustflags,
            env=payload.env,
        )

        result = await build_manager.build(
            git_ref=payload.git_ref,
            config=config,
            websocket_manager=websocket_manager,
            task_id=task_id,
        )

        features = result.get("features", "")
        feature_list = [f.strip() for f in features.split(",") if f.strip()]

        build_name = result["build_name"]
        record = (
            db.query(CandleBuild)
            .filter(CandleBuild.name == build_name)
            .one_or_none()
        )
        if record is None:
            record = CandleBuild(
                name=build_name,
                created_at=datetime.utcnow(),
            )

        record.git_ref = payload.git_ref
        record.binary_path = result.get("binary_path")
        record.features = feature_list
        record.build_metadata = {
            "config": asdict(config),
        }
        if payload.mark_active:
            db.query(CandleBuild).update({"is_active": False})
            record.is_active = True

        db.add(record)
        db.commit()
        db.refresh(record)

        await websocket_manager.send_notification(
            title="Build Complete",
            message=f"Candle build '{record.name}' finished successfully.",
            type="success",
        )
    except Exception as exc:
        logger.exception("Candle build failed: %s", exc)
        await websocket_manager.send_build_progress(
            task_id=task_id,
            stage="error",
            progress=100,
            message=str(exc),
        )
        await websocket_manager.send_notification(
            title="Build Failed",
            message=str(exc),
            type="error",
        )
    finally:
        db.close()


@router.post("/build", status_code=http_status.HTTP_202_ACCEPTED)
async def enqueue_build(
    request: BuildRequestPayload,
    build_manager: CandleBuildManager = Depends(get_build_manager),
):
    task_id = request.task_id or f"build-{uuid4().hex[:8]}"
    request_dict = request.model_dump(exclude={"task_id"})
    payload = BuildRequestPayload(**request_dict, task_id=task_id)

    asyncio.create_task(_execute_build_job(payload, build_manager, task_id))

    return {
        "task_id": task_id,
        "message": "Build scheduled",
    }


@router.post("/{build_id}/activate")
async def activate_build(
    build_id: int,
    request: ActivateRequest,
    db: Session = Depends(get_db),
):
    build = db.query(CandleBuild).filter(CandleBuild.id == build_id).one_or_none()
    if build is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Build not found")

    if request.mark_active:
        db.query(CandleBuild).update({"is_active": False})
        build.is_active = True
        db.commit()
        await websocket_manager.send_notification(
            title="Active Build Updated",
            message=f"Candle build '{build.name}' marked as active.",
            type="info",
        )
    return {"status": "ok"}


@router.delete("/{build_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_build(
    build_id: int,
    build_manager: CandleBuildManager = Depends(get_build_manager),
    db: Session = Depends(get_db),
):
    build = db.query(CandleBuild).filter(CandleBuild.id == build_id).one_or_none()
    if build is None:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Build not found")

    build_manager.delete_build(build.name)
    db.delete(build)
    db.commit()
    return {"status": "deleted"}


@router.get("/capabilities")
async def build_capabilities():
    return await detect_build_capabilities()

