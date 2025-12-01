from __future__ import annotations

from fastapi import APIRouter, HTTPException

from . import get_runtime, register_builtin_strategies
from ..config import settings


router = APIRouter(prefix="/runtime", tags=["Runtime"])


@router.get("/status")
async def runtime_status():
    register_builtin_strategies()
    return get_runtime().status()


@router.get("/strategies")
async def list_strategies():
    register_builtin_strategies()
    return {"items": get_runtime().list_strategies()}


@router.post("/start")
async def start_runtime():
    register_builtin_strategies()
    runtime = get_runtime()
    if not settings.agent_runtime_enabled:
        raise HTTPException(status_code=400, detail="agent runtime disabled in configuration")
    await runtime.ensure_started()
    return runtime.status()


@router.post("/stop")
async def stop_runtime():
    runtime = get_runtime()
    await runtime.stop()
    return {"stopped": True}


@router.post("/strategies/{strategy_id}/tick")
async def run_strategy_now(strategy_id: str):
    runtime = get_runtime()
    register_builtin_strategies()
    state = next((s for s in runtime.list_strategies() if s["id"] == strategy_id), None)
    if not state:
        raise HTTPException(status_code=404, detail="strategy not found")
    started = await runtime.run_strategy_now(strategy_id)
    if not started:
        raise HTTPException(status_code=409, detail="strategy already running")
    return {"queued": True}


@router.post("/strategies/{strategy_id}/pause")
async def pause_strategy(strategy_id: str):
    runtime = get_runtime()
    if not runtime.pause_strategy(strategy_id):
        raise HTTPException(status_code=404, detail="strategy not found")
    return {"paused": True}


@router.post("/strategies/{strategy_id}/resume")
async def resume_strategy(strategy_id: str):
    runtime = get_runtime()
    if not runtime.resume_strategy(strategy_id):
        raise HTTPException(status_code=404, detail="strategy not found")
    return {"resumed": True}

