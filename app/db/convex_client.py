"""
Convex client for interacting with the Convex database from Python.

This client provides async methods to call Convex queries, mutations, and actions
via the Convex HTTP API.
"""

import httpx
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel
from functools import lru_cache

from app.config import settings


class ConvexError(Exception):
    """Base exception for Convex errors."""
    pass


class ConvexAuthError(ConvexError):
    """Authentication error when calling Convex."""
    pass


class ConvexQueryError(ConvexError):
    """Error executing a Convex query."""
    pass


class ConvexMutationError(ConvexError):
    """Error executing a Convex mutation."""
    pass


class ConvexClient:
    """
    Async client for interacting with Convex from Python.

    Example usage:
        client = ConvexClient(
            deployment_url="https://your-deployment.convex.cloud",
            deploy_key="prod:your-deploy-key"
        )

        # Query
        users = await client.query("users:listByWallet", {"walletId": "..."})

        # Mutation
        user_id = await client.mutation("users:create", {})

        # Action
        result = await client.action("strategies:triggerExecution", {"strategyId": "..."})
    """

    def __init__(
        self,
        deployment_url: Optional[str] = None,
        deploy_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.deployment_url = deployment_url or settings.convex_url
        self.deploy_key = deploy_key or settings.convex_deploy_key
        self.timeout = timeout

        if not self.deployment_url:
            raise ConvexError("CONVEX_URL is required")

        # Remove trailing slash if present
        self.deployment_url = self.deployment_url.rstrip("/")

        self._client: Optional[httpx.AsyncClient] = None

    @property
    def headers(self) -> Dict[str, str]:
        """Get headers for Convex API requests."""
        headers = {"Content-Type": "application/json"}
        if self.deploy_key:
            headers["Authorization"] = f"Convex {self.deploy_key}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.headers,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def query(
        self,
        function_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a Convex query function.

        Args:
            function_name: The query function path (e.g., "users:get" or "users:listByWallet")
            args: Arguments to pass to the query function

        Returns:
            The query result

        Raises:
            ConvexQueryError: If the query fails
        """
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.deployment_url}/api/query",
                json={
                    "path": function_name,
                    "args": args or {},
                },
            )

            if response.status_code == 401:
                raise ConvexAuthError("Invalid or missing deploy key")

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise ConvexQueryError(data["error"])

            return data.get("value")

        except httpx.HTTPStatusError as e:
            raise ConvexQueryError(f"Query failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise ConvexQueryError(f"Request failed: {str(e)}") from e

    async def mutation(
        self,
        function_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a Convex mutation function.

        Args:
            function_name: The mutation function path (e.g., "users:create")
            args: Arguments to pass to the mutation function

        Returns:
            The mutation result (usually the created/updated ID)

        Raises:
            ConvexMutationError: If the mutation fails
        """
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.deployment_url}/api/mutation",
                json={
                    "path": function_name,
                    "args": args or {},
                },
            )

            if response.status_code == 401:
                raise ConvexAuthError("Invalid or missing deploy key")

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise ConvexMutationError(data["error"])

            return data.get("value")

        except httpx.HTTPStatusError as e:
            raise ConvexMutationError(f"Mutation failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise ConvexMutationError(f"Request failed: {str(e)}") from e

    async def action(
        self,
        function_name: str,
        args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a Convex action function.

        Actions can have side effects and call external services.

        Args:
            function_name: The action function path
            args: Arguments to pass to the action function

        Returns:
            The action result

        Raises:
            ConvexError: If the action fails
        """
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.deployment_url}/api/action",
                json={
                    "path": function_name,
                    "args": args or {},
                },
            )

            if response.status_code == 401:
                raise ConvexAuthError("Invalid or missing deploy key")

            response.raise_for_status()
            data = response.json()

            if "error" in data:
                raise ConvexError(data["error"])

            return data.get("value")

        except httpx.HTTPStatusError as e:
            raise ConvexError(f"Action failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise ConvexError(f"Request failed: {str(e)}") from e

    # =========================================================================
    # Convenience methods for common operations
    # =========================================================================

    async def get_or_create_user(
        self,
        address: str,
        chain: str = "ethereum",
    ) -> Dict[str, Any]:
        """Get or create a user by wallet address."""
        return await self.mutation(
            "users:getOrCreateByWallet",
            {"address": address, "chain": chain},
        )

    async def get_wallet(
        self,
        address: str,
        chain: str = "ethereum",
    ) -> Optional[Dict[str, Any]]:
        """Get a wallet by address and chain."""
        return await self.query(
            "wallets:getByAddress",
            {"address": address, "chain": chain},
        )

    async def list_conversations(
        self,
        wallet_id: str,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """List conversations for a wallet."""
        return await self.query(
            "conversations:listByWallet",
            {"walletId": wallet_id, "includeArchived": include_archived},
        )

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a conversation with its messages."""
        return await self.query(
            "conversations:getWithMessages",
            {"conversationId": conversation_id},
        )

    async def create_conversation(
        self,
        wallet_id: str,
        title: Optional[str] = None,
    ) -> str:
        """Create a new conversation."""
        return await self.mutation(
            "conversations:create",
            {"walletId": wallet_id, "title": title},
        )

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        token_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a message to a conversation."""
        return await self.mutation(
            "conversations:addMessage",
            {
                "conversationId": conversation_id,
                "role": role,
                "content": content,
                "tokenCount": token_count,
                "metadata": metadata,
            },
        )

    async def list_strategies(
        self,
        user_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List strategies for a user."""
        args = {"userId": user_id}
        if status:
            args["status"] = status
        return await self.query("strategies:listByUser", args)

    async def get_strategy(
        self,
        strategy_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a strategy with its recent executions."""
        return await self.query(
            "strategies:getWithExecutions",
            {"strategyId": strategy_id},
        )

    async def create_strategy(
        self,
        user_id: str,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None,
        cron_expression: Optional[str] = None,
    ) -> str:
        """Create a new strategy."""
        return await self.mutation(
            "strategies:create",
            {
                "userId": user_id,
                "name": name,
                "description": description,
                "config": config,
                "cronExpression": cron_expression,
            },
        )

    async def activate_strategy(self, strategy_id: str) -> None:
        """Activate a strategy."""
        await self.mutation("strategies:activate", {"strategyId": strategy_id})

    async def pause_strategy(self, strategy_id: str) -> None:
        """Pause a strategy."""
        await self.mutation("strategies:pause", {"strategyId": strategy_id})

    async def start_execution(self, execution_id: str) -> None:
        """Mark an execution as started."""
        await self.mutation("executions:start", {"executionId": execution_id})

    async def complete_execution(
        self,
        execution_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Complete an execution successfully."""
        await self.mutation(
            "executions:complete",
            {"executionId": execution_id, "result": result},
        )

    async def fail_execution(
        self,
        execution_id: str,
        error: str,
    ) -> None:
        """Mark an execution as failed."""
        await self.mutation(
            "executions:fail",
            {"executionId": execution_id, "error": error},
        )

    async def record_decision(
        self,
        execution_id: str,
        decision_type: str,
        input_context: Dict[str, Any],
        reasoning: str,
        action_taken: Dict[str, Any],
        risk_assessment: Dict[str, Any],
    ) -> str:
        """Record an agent decision."""
        return await self.mutation(
            "executions:addDecision",
            {
                "executionId": execution_id,
                "decisionType": decision_type,
                "inputContext": input_context,
                "reasoning": reasoning,
                "actionTaken": action_taken,
                "riskAssessment": risk_assessment,
            },
        )

    async def create_transaction(
        self,
        wallet_id: str,
        chain: str,
        tx_type: str,
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None,
        value_usd: Optional[float] = None,
    ) -> str:
        """Create a new transaction record."""
        return await self.mutation(
            "transactions:create",
            {
                "walletId": wallet_id,
                "chain": chain,
                "type": tx_type,
                "inputData": input_data,
                "executionId": execution_id,
                "valueUsd": value_usd,
            },
        )

    async def update_transaction(
        self,
        transaction_id: str,
        status: str,
        tx_hash: Optional[str] = None,
        output_data: Optional[Dict[str, Any]] = None,
        gas_used: Optional[int] = None,
        gas_price: Optional[int] = None,
    ) -> None:
        """Update a transaction's status."""
        if status == "submitted" and tx_hash:
            await self.mutation(
                "transactions:markSubmitted",
                {"transactionId": transaction_id, "txHash": tx_hash},
            )
        elif status == "confirmed":
            await self.mutation(
                "transactions:markConfirmed",
                {
                    "transactionId": transaction_id,
                    "outputData": output_data,
                    "gasUsed": gas_used,
                    "gasPrice": gas_price,
                },
            )
        elif status == "failed":
            await self.mutation(
                "transactions:markFailed",
                {"transactionId": transaction_id, "outputData": output_data},
            )
        elif status == "reverted":
            await self.mutation(
                "transactions:markReverted",
                {
                    "transactionId": transaction_id,
                    "outputData": output_data,
                    "gasUsed": gas_used,
                    "gasPrice": gas_price,
                },
            )


# Singleton instance
_convex_client: Optional[ConvexClient] = None


def get_convex_client() -> ConvexClient:
    """Get the singleton Convex client instance."""
    global _convex_client
    if _convex_client is None:
        _convex_client = ConvexClient()
    return _convex_client
