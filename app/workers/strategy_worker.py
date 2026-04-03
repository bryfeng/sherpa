"""
Strategy Worker — arq task definitions for DCA execution.

Tasks are enqueued by the DCA internal API endpoint and executed
by an arq worker process running app.workers.run.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def execute_dca_strategy(ctx: dict[str, Any], strategy_id: str) -> dict[str, Any]:
    """Execute a single DCA strategy cycle.

    This is the arq task that replaces inline execution in the API handler.
    The worker process pre-builds the DCA service and stores it in *ctx*
    via the ``startup`` hook in :pymod:`app.workers.run`.
    """
    service = ctx["dca_service"]
    logger.info("strategy_worker: executing DCA strategy %s", strategy_id)

    try:
        result = await service.execute_now(strategy_id)
        payload = {
            "success": result.success,
            "status": result.status.value,
            "txHash": result.tx_hash,
            "errorMessage": result.error_message,
            "nextExecutionAt": (
                result.next_execution_at.isoformat() if result.next_execution_at else None
            ),
        }
        logger.info(
            "strategy_worker: strategy %s finished — success=%s",
            strategy_id,
            result.success,
        )
        return payload
    except Exception:
        logger.exception("strategy_worker: strategy %s failed", strategy_id)
        raise
