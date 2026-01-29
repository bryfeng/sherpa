"""
Rhinestone provider for Smart Wallet and Intent infrastructure.

Handles Smart Account deployment, Smart Sessions, and Warp intent execution.
Uses Rhinestone's unified SDK for cross-chain operations.

Docs: https://docs.rhinestone.dev/
SDK: https://github.com/rhinestonewtf/sdk
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from .base import Provider
from ..config import settings

logger = logging.getLogger(__name__)


class RhinestoneError(Exception):
    """Base Rhinestone provider error."""
    pass


class RhinestoneApiError(RhinestoneError):
    """API request failed."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class RhinestoneIntentError(RhinestoneError):
    """Intent execution failed."""
    pass


class IntentStatus(str, Enum):
    """Status of an intent execution."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RhinestoneConfig:
    """Rhinestone provider configuration."""
    api_key: str = ""  # Optional for development
    base_url: str = "https://api.rhinestone.dev"
    timeout_s: float = 30.0


@dataclass
class ChainBalance:
    """Token balance on a specific chain."""
    chain_id: int
    chain_name: str
    token_address: str
    token_symbol: str
    balance: str  # Raw balance as string
    balance_formatted: str  # Human-readable balance
    usd_value: Optional[float] = None


@dataclass
class UnifiedBalance:
    """Unified balance across all chains."""
    token_symbol: str
    total_balance: str
    total_usd_value: float
    chain_balances: List[ChainBalance] = field(default_factory=list)


@dataclass
class IntentRequest:
    """Request to execute an intent."""
    account_address: str
    source_chains: List[int]  # Chain IDs to source funds from
    target_chain: int  # Chain ID for execution
    calls: List[Dict[str, Any]]  # Target contract calls
    token_requests: List[Dict[str, Any]]  # Tokens needed for execution


@dataclass
class IntentResult:
    """Result of an intent execution."""
    intent_id: str
    status: IntentStatus
    tx_hashes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    created_at: int = 0
    completed_at: Optional[int] = None


class RhinestoneProvider(Provider):
    """
    Rhinestone provider for Smart Wallet and Intent execution.

    Provides:
    - Intent submission and tracking via Warp Orchestrator
    - Unified multi-chain balance queries
    - Smart Account status checks

    Note: Smart Account deployment and Smart Sessions grant happen
    on the frontend via @rhinestone/sdk. This provider handles
    backend operations like intent submission for autonomous execution.

    Usage:
        provider = get_rhinestone_provider()

        # Submit an intent
        intent = await provider.submit_intent(
            account_address="0x...",
            source_chains=[8453],  # Base
            target_chain=42161,    # Arbitrum
            calls=[...],
            token_requests=[{"address": "USDC", "amount": "1000000"}],
        )

        # Check status
        result = await provider.get_intent_status(intent.intent_id)
    """

    name = "rhinestone"

    def __init__(self, config: Optional[RhinestoneConfig] = None) -> None:
        self._config = config or RhinestoneConfig(
            api_key=settings.rhinestone_api_key,
            base_url=settings.rhinestone_base_url,
        )
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Sherpa/1.0",
            }
            if self._config.api_key:
                headers["x-api-key"] = self._config.api_key

            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=headers,
                timeout=self._config.timeout_s,
            )
        return self._client

    async def ready(self) -> bool:
        """Check if provider is configured and ready."""
        return settings.enable_rhinestone

    async def health_check(self) -> Dict[str, Any]:
        """Check Rhinestone API health."""
        if not await self.ready():
            return {"status": "disabled", "reason": "Rhinestone not enabled"}

        try:
            client = self._get_client()
            response = await client.get("/health")
            if response.status_code == 200:
                return {"status": "healthy"}
            return {"status": "degraded", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    async def submit_intent(
        self,
        account_address: str,
        source_chains: List[int],
        target_chain: int,
        calls: List[Dict[str, Any]],
        token_requests: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> IntentResult:
        """
        Submit an intent for execution via Warp Orchestrator.

        Args:
            account_address: User's Smart Account address
            source_chains: Chain IDs to source funds from
            target_chain: Chain ID where calls will be executed
            calls: List of contract calls [{to, value, data}]
            token_requests: Tokens needed [{address, amount}]
            session_id: Optional Smart Session ID for permission validation

        Returns:
            IntentResult with intent_id for tracking
        """
        if not await self.ready():
            raise RhinestoneError("Rhinestone provider not enabled")

        payload = {
            "accountAddress": account_address,
            "sourceChains": source_chains,
            "targetChain": target_chain,
            "calls": calls,
            "tokenRequests": token_requests,
        }
        if session_id:
            payload["sessionId"] = session_id

        try:
            client = self._get_client()
            response = await client.post("/v1/intents", json=payload)

            if response.status_code != 200 and response.status_code != 201:
                error_text = response.text
                logger.error(f"Intent submission failed: {response.status_code} - {error_text}")
                raise RhinestoneApiError(
                    f"Intent submission failed: {error_text}",
                    status_code=response.status_code,
                )

            data = response.json()
            logger.info(f"Intent submitted: {data.get('intentId')}")

            return IntentResult(
                intent_id=data.get("intentId", ""),
                status=IntentStatus.PENDING,
                created_at=int(time.time()),
            )

        except httpx.RequestError as e:
            logger.error(f"Intent submission request failed: {e}")
            raise RhinestoneApiError(f"Request failed: {e}") from e

    async def get_intent_status(self, intent_id: str) -> IntentResult:
        """
        Get the status of an intent execution.

        Args:
            intent_id: Intent ID from submit_intent

        Returns:
            IntentResult with current status and tx hashes if completed
        """
        if not await self.ready():
            raise RhinestoneError("Rhinestone provider not enabled")

        try:
            client = self._get_client()
            response = await client.get(f"/v1/intents/{intent_id}")

            if response.status_code == 404:
                raise RhinestoneApiError(f"Intent not found: {intent_id}", status_code=404)

            if response.status_code != 200:
                raise RhinestoneApiError(
                    f"Failed to get intent status: {response.text}",
                    status_code=response.status_code,
                )

            data = response.json()

            status_str = data.get("status", "pending").lower()
            status = IntentStatus(status_str) if status_str in IntentStatus.__members__.values() else IntentStatus.PENDING

            return IntentResult(
                intent_id=intent_id,
                status=status,
                tx_hashes=data.get("txHashes", []),
                error=data.get("error"),
                created_at=data.get("createdAt", 0),
                completed_at=data.get("completedAt"),
            )

        except httpx.RequestError as e:
            logger.error(f"Get intent status failed: {e}")
            raise RhinestoneApiError(f"Request failed: {e}") from e

    async def wait_for_intent(
        self,
        intent_id: str,
        timeout_s: float = 120.0,
        poll_interval_s: float = 2.0,
    ) -> IntentResult:
        """
        Wait for an intent to complete or fail.

        Args:
            intent_id: Intent ID to wait for
            timeout_s: Maximum time to wait (default 2 minutes)
            poll_interval_s: Polling interval

        Returns:
            Final IntentResult

        Raises:
            RhinestoneIntentError: If intent fails or times out
        """
        import asyncio

        start_time = time.time()

        while True:
            result = await self.get_intent_status(intent_id)

            if result.status == IntentStatus.COMPLETED:
                return result

            if result.status == IntentStatus.FAILED:
                raise RhinestoneIntentError(
                    f"Intent {intent_id} failed: {result.error}"
                )

            elapsed = time.time() - start_time
            if elapsed >= timeout_s:
                raise RhinestoneIntentError(
                    f"Intent {intent_id} timed out after {timeout_s}s"
                )

            await asyncio.sleep(poll_interval_s)

    async def get_unified_balances(
        self,
        account_address: str,
        chains: Optional[List[int]] = None,
    ) -> List[UnifiedBalance]:
        """
        Get unified token balances across all chains.

        Args:
            account_address: Smart Account address
            chains: Optional list of chain IDs to query (defaults to all supported)

        Returns:
            List of UnifiedBalance with aggregated holdings
        """
        if not await self.ready():
            raise RhinestoneError("Rhinestone provider not enabled")

        try:
            client = self._get_client()
            params = {"address": account_address}
            if chains:
                params["chains"] = ",".join(str(c) for c in chains)

            response = await client.get("/v1/balances", params=params)

            if response.status_code != 200:
                raise RhinestoneApiError(
                    f"Failed to get balances: {response.text}",
                    status_code=response.status_code,
                )

            data = response.json()
            balances = []

            for token_data in data.get("tokens", []):
                chain_balances = [
                    ChainBalance(
                        chain_id=cb["chainId"],
                        chain_name=cb.get("chainName", ""),
                        token_address=cb["tokenAddress"],
                        token_symbol=token_data["symbol"],
                        balance=cb["balance"],
                        balance_formatted=cb.get("balanceFormatted", ""),
                        usd_value=cb.get("usdValue"),
                    )
                    for cb in token_data.get("chainBalances", [])
                ]

                balances.append(UnifiedBalance(
                    token_symbol=token_data["symbol"],
                    total_balance=token_data.get("totalBalance", "0"),
                    total_usd_value=token_data.get("totalUsdValue", 0.0),
                    chain_balances=chain_balances,
                ))

            return balances

        except httpx.RequestError as e:
            logger.error(f"Get balances failed: {e}")
            raise RhinestoneApiError(f"Request failed: {e}") from e

    async def get_smart_account_status(
        self,
        owner_address: str,
    ) -> Dict[str, Any]:
        """
        Check if an owner has a deployed Smart Account.

        Args:
            owner_address: EOA address that owns the Smart Account

        Returns:
            Dict with deployment status and Smart Account address if deployed
        """
        if not await self.ready():
            raise RhinestoneError("Rhinestone provider not enabled")

        try:
            client = self._get_client()
            response = await client.get(f"/v1/accounts/{owner_address}")

            if response.status_code == 404:
                return {"deployed": False, "address": None}

            if response.status_code != 200:
                raise RhinestoneApiError(
                    f"Failed to get account status: {response.text}",
                    status_code=response.status_code,
                )

            data = response.json()
            return {
                "deployed": True,
                "address": data.get("accountAddress"),
                "chains": data.get("deployedChains", []),
                "modules": data.get("installedModules", []),
            }

        except httpx.RequestError as e:
            logger.error(f"Get account status failed: {e}")
            raise RhinestoneApiError(f"Request failed: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Singleton instance
_rhinestone_provider: Optional[RhinestoneProvider] = None


def get_rhinestone_provider() -> RhinestoneProvider:
    """Get the singleton Rhinestone provider instance."""
    global _rhinestone_provider
    if _rhinestone_provider is None:
        _rhinestone_provider = RhinestoneProvider()
    return _rhinestone_provider
