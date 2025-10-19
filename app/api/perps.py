from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core.chat import _get_agent
from ..core.perps import SimulationRequest, SimulationResult, build_simulator
from ..core.perps.policies import PolicyManager

router = APIRouter(prefix="/perps")
_SIMULATOR = build_simulator()


@router.post("/simulate", response_model=SimulationResult)
async def simulate_perps(req: SimulationRequest) -> SimulationResult:
    agent = _get_agent(None, None)
    try:
        return await _SIMULATOR.simulate(req, PolicyManager(getattr(agent, "context_manager", None)))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Simulation failed") from exc
