from fastapi import Request, Depends

from backend.candle_manager import CandleRuntimeManager
from backend.candle_build_manager import CandleBuildManager


def get_runtime_manager(request: Request) -> CandleRuntimeManager:
    manager = getattr(request.app.state, "candle_runtime_manager", None)
    if manager is None:
        raise RuntimeError("CandleRuntimeManager has not been initialised on app.state")
    return manager


def get_build_manager(request: Request) -> CandleBuildManager:
    manager = getattr(request.app.state, "candle_build_manager", None)
    if manager is None:
        raise RuntimeError("CandleBuildManager has not been initialised on app.state")
    return manager

