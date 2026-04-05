"""SwapManager — thin facade delegating to focused services.

The actual logic lives in:
- app.services.intent_parser   — NL → structured trade intent
- app.services.quote_aggregator — Relay / Jupiter quote fetching
- app.services.trade_service    — orchestration (parse → resolve → quote → validate)
- app.services.panel_builder    — frontend panel formatting
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...providers.coingecko import CoingeckoProvider
from ...providers.jupiter import JupiterSwapProvider
from ...services.tokens import TokenService
from ...services.trade_service import TradeService
from ...types.requests import ChatRequest


class SwapManager:
    """Backward-compatible facade over :class:`TradeService`.

    New callers should use ``TradeService`` directly.
    """

    def __init__(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        price_provider: Optional[CoingeckoProvider] = None,
        jupiter_provider: Optional[JupiterSwapProvider] = None,
        token_service: Optional[TokenService] = None,
    ) -> None:
        self._trade_service = TradeService(
            log=logger,
            price_provider=price_provider,
            jupiter_provider=jupiter_provider,
            token_service=token_service,
        )

    async def maybe_handle(
        self,
        request: ChatRequest,
        conversation_id: str,
        *,
        wallet_address: Optional[str],
        default_chain: Optional[str],
        portfolio_tokens: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self._trade_service.maybe_handle(
            request,
            conversation_id,
            wallet_address=wallet_address,
            default_chain=default_chain,
            portfolio_tokens=portfolio_tokens,
        )
