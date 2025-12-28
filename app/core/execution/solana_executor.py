"""
Solana Transaction Executor.

Handles sending and monitoring Solana transactions built by Jupiter.
"""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx


class SolanaTransactionStatus(str, Enum):
    """Status of a Solana transaction."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class SolanaTransactionResult:
    """Result of a Solana transaction submission."""
    signature: str
    status: SolanaTransactionStatus
    slot: Optional[int] = None
    block_time: Optional[int] = None
    fee: Optional[int] = None
    error: Optional[str] = None
    confirmations: Optional[int] = None


@dataclass
class SolanaRpcConfig:
    """Configuration for Solana RPC connection."""
    rpc_url: str
    ws_url: Optional[str] = None
    commitment: str = "confirmed"
    max_retries: int = 3
    timeout_s: float = 30.0


class SolanaExecutorError(Exception):
    """Error in Solana transaction execution."""
    pass


class SolanaExecutor:
    """
    Executor for Solana transactions.

    Handles:
    - Sending signed transactions to the network
    - Monitoring transaction status
    - Confirmation waiting with exponential backoff

    Note: This executor sends pre-signed transactions. The signing
    happens client-side (in the frontend wallet) for security.

    Usage:
        executor = SolanaExecutor(SolanaRpcConfig(
            rpc_url="https://api.mainnet-beta.solana.com"
        ))

        # Send a pre-signed transaction (base64 encoded)
        result = await executor.send_transaction(signed_tx_base64)

        # Wait for confirmation
        result = await executor.wait_for_confirmation(result.signature)
    """

    def __init__(self, config: SolanaRpcConfig):
        self._config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._config.timeout_s)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rpc_call(
        self,
        method: str,
        params: List[Any],
    ) -> Dict[str, Any]:
        """Make an RPC call to Solana node."""
        client = await self._get_client()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }

        for attempt in range(self._config.max_retries):
            try:
                response = await client.post(
                    self._config.rpc_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    raise SolanaExecutorError(f"RPC error: {error_msg}")

                return data

            except httpx.HTTPStatusError as e:
                if attempt == self._config.max_retries - 1:
                    raise SolanaExecutorError(f"HTTP error: {e.response.status_code}")
                await asyncio.sleep(0.5 * (attempt + 1))
            except SolanaExecutorError:
                raise
            except Exception as e:
                if attempt == self._config.max_retries - 1:
                    raise SolanaExecutorError(str(e))
                await asyncio.sleep(0.5 * (attempt + 1))

        raise SolanaExecutorError("Max retries exceeded")

    async def send_transaction(
        self,
        signed_transaction: str,
        skip_preflight: bool = False,
        max_retries: Optional[int] = None,
    ) -> SolanaTransactionResult:
        """
        Send a signed transaction to the Solana network.

        Args:
            signed_transaction: Base64 encoded signed transaction
            skip_preflight: Skip preflight simulation
            max_retries: Maximum retries for sending

        Returns:
            SolanaTransactionResult with transaction signature
        """
        options = {
            "encoding": "base64",
            "skipPreflight": skip_preflight,
            "preflightCommitment": self._config.commitment,
            "maxRetries": max_retries or self._config.max_retries,
        }

        try:
            result = await self._rpc_call(
                "sendTransaction",
                [signed_transaction, options],
            )

            signature = result.get("result")
            if not signature:
                raise SolanaExecutorError("No signature returned from sendTransaction")

            return SolanaTransactionResult(
                signature=signature,
                status=SolanaTransactionStatus.PENDING,
            )

        except SolanaExecutorError:
            raise
        except Exception as e:
            return SolanaTransactionResult(
                signature="",
                status=SolanaTransactionStatus.FAILED,
                error=str(e),
            )

    async def get_transaction_status(
        self,
        signature: str,
    ) -> SolanaTransactionResult:
        """
        Get the current status of a transaction.

        Args:
            signature: Transaction signature (base58)

        Returns:
            SolanaTransactionResult with current status
        """
        options = {
            "commitment": self._config.commitment,
            "maxSupportedTransactionVersion": 0,
        }

        try:
            result = await self._rpc_call(
                "getTransaction",
                [signature, options],
            )

            tx_data = result.get("result")

            if tx_data is None:
                # Transaction not found - still pending or expired
                return SolanaTransactionResult(
                    signature=signature,
                    status=SolanaTransactionStatus.PENDING,
                )

            # Check for error
            meta = tx_data.get("meta", {})
            if meta.get("err") is not None:
                return SolanaTransactionResult(
                    signature=signature,
                    status=SolanaTransactionStatus.FAILED,
                    slot=tx_data.get("slot"),
                    block_time=tx_data.get("blockTime"),
                    fee=meta.get("fee"),
                    error=str(meta.get("err")),
                )

            # Transaction confirmed
            return SolanaTransactionResult(
                signature=signature,
                status=SolanaTransactionStatus.CONFIRMED,
                slot=tx_data.get("slot"),
                block_time=tx_data.get("blockTime"),
                fee=meta.get("fee"),
            )

        except SolanaExecutorError:
            raise
        except Exception as e:
            return SolanaTransactionResult(
                signature=signature,
                status=SolanaTransactionStatus.FAILED,
                error=str(e),
            )

    async def wait_for_confirmation(
        self,
        signature: str,
        timeout_s: float = 60.0,
        poll_interval_s: float = 1.0,
    ) -> SolanaTransactionResult:
        """
        Wait for a transaction to be confirmed.

        Uses exponential backoff for polling.

        Args:
            signature: Transaction signature (base58)
            timeout_s: Maximum time to wait
            poll_interval_s: Initial polling interval

        Returns:
            SolanaTransactionResult with final status
        """
        start_time = time.time()
        interval = poll_interval_s

        while (time.time() - start_time) < timeout_s:
            result = await self.get_transaction_status(signature)

            if result.status in (
                SolanaTransactionStatus.CONFIRMED,
                SolanaTransactionStatus.FINALIZED,
                SolanaTransactionStatus.FAILED,
            ):
                return result

            await asyncio.sleep(interval)
            # Exponential backoff, max 5 seconds
            interval = min(interval * 1.5, 5.0)

        return SolanaTransactionResult(
            signature=signature,
            status=SolanaTransactionStatus.EXPIRED,
            error="Transaction confirmation timed out",
        )

    async def get_recent_blockhash(self) -> Dict[str, Any]:
        """
        Get a recent blockhash for transaction building.

        Returns:
            Dict with blockhash and lastValidBlockHeight
        """
        result = await self._rpc_call(
            "getLatestBlockhash",
            [{"commitment": self._config.commitment}],
        )

        value = result.get("result", {}).get("value", {})
        return {
            "blockhash": value.get("blockhash"),
            "lastValidBlockHeight": value.get("lastValidBlockHeight"),
        }

    async def simulate_transaction(
        self,
        transaction: str,
    ) -> Dict[str, Any]:
        """
        Simulate a transaction without sending it.

        Args:
            transaction: Base64 encoded transaction

        Returns:
            Simulation result with logs and error info
        """
        options = {
            "encoding": "base64",
            "commitment": self._config.commitment,
            "replaceRecentBlockhash": True,
        }

        result = await self._rpc_call(
            "simulateTransaction",
            [transaction, options],
        )

        sim_result = result.get("result", {}).get("value", {})
        return {
            "error": sim_result.get("err"),
            "logs": sim_result.get("logs", []),
            "units_consumed": sim_result.get("unitsConsumed"),
        }

    async def get_balance(self, address: str) -> int:
        """
        Get SOL balance for an address.

        Args:
            address: Wallet address (base58)

        Returns:
            Balance in lamports
        """
        result = await self._rpc_call(
            "getBalance",
            [address, {"commitment": self._config.commitment}],
        )
        return result.get("result", {}).get("value", 0)

    async def get_token_accounts(
        self,
        owner: str,
        mint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get token accounts for an owner.

        Args:
            owner: Wallet address
            mint: Optional token mint to filter by

        Returns:
            List of token accounts with balances
        """
        filter_option = {"mint": mint} if mint else {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}

        result = await self._rpc_call(
            "getTokenAccountsByOwner",
            [
                owner,
                filter_option,
                {"encoding": "jsonParsed", "commitment": self._config.commitment},
            ],
        )

        accounts = []
        for item in result.get("result", {}).get("value", []):
            parsed = item.get("account", {}).get("data", {}).get("parsed", {})
            info = parsed.get("info", {})
            accounts.append({
                "address": item.get("pubkey"),
                "mint": info.get("mint"),
                "owner": info.get("owner"),
                "amount": int(info.get("tokenAmount", {}).get("amount", 0)),
                "decimals": info.get("tokenAmount", {}).get("decimals", 0),
            })

        return accounts


# Singleton instance
_solana_executor: Optional[SolanaExecutor] = None


def get_solana_executor(rpc_url: Optional[str] = None) -> SolanaExecutor:
    """
    Get the singleton Solana executor.

    RPC URL resolution order:
    1. Explicit rpc_url parameter
    2. SOLANA_RPC_URL environment variable
    3. Alchemy Solana URL (built from ALCHEMY_API_KEY)
    4. Public Solana RPC (fallback, rate limited)

    Args:
        rpc_url: Optional RPC URL override
    """
    global _solana_executor

    if _solana_executor is None:
        from ...config import settings

        # Try explicit URL first
        url = rpc_url or getattr(settings, "solana_rpc_url", None)

        # Fall back to Alchemy if we have an API key
        if not url:
            alchemy_key = getattr(settings, "alchemy_api_key", None)
            if alchemy_key:
                url = f"https://solana-mainnet.g.alchemy.com/v2/{alchemy_key}"

        # Last resort: public RPC (rate limited)
        if not url:
            url = "https://api.mainnet-beta.solana.com"

        _solana_executor = SolanaExecutor(
            SolanaRpcConfig(rpc_url=url)
        )

    return _solana_executor


__all__ = [
    "SolanaExecutor",
    "SolanaRpcConfig",
    "SolanaTransactionResult",
    "SolanaTransactionStatus",
    "SolanaExecutorError",
    "get_solana_executor",
]
