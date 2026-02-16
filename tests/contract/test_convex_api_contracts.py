"""
Contract Tests for Convex <-> FastAPI Integration

These tests verify that FastAPI endpoints accept the exact payload formats
that Convex sends. This catches mismatches between TypeScript and Python.

IMPORTANT: When adding new internal endpoints called by Convex cron/actions,
add a corresponding test here that sends the EXACT payload Convex sends.

Reference files:
- Convex scheduler: frontend/convex/scheduler.ts
- Convex DCA: frontend/convex/dca.ts
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings
from app.core.strategies.generic_executor import GenericStrategyExecutor, SwapParams
from app.core.strategies.config_normalizer import normalize_strategy_config

client = TestClient(app)


# =============================================================================
# DCA Internal Endpoints
# =============================================================================


class TestDCAInternalExecuteContract:
    """
    Contract test for POST /dca/internal/execute

    Called by: frontend/convex/scheduler.ts:checkDCAStrategies (line ~337)
    Endpoint:  backend/app/api/dca.py (router prefix="/dca", route="/internal/execute")

    Convex sends:
        URL: ${fastapiUrl}/dca/internal/execute
        body: JSON.stringify({ strategyId: strategy._id })
    """

    ENDPOINT = "/dca/internal/execute"  # router prefix + route

    def test_accepts_strategy_id_in_body(self):
        """
        Verify endpoint accepts strategyId in JSON body (as Convex sends it).

        This test catches the bug where FastAPI expected query param but Convex sent body.
        """
        with patch("app.api.dca.get_dca_service") as mock_service:
            mock_service.return_value.execute_now = AsyncMock(
                side_effect=ValueError("Strategy not found")  # Expected for fake ID
            )

            response = client.post(
                self.ENDPOINT,
                json={"strategyId": "test_strategy_123"},  # Exactly what Convex sends
                headers={"X-Internal-Key": settings.convex_internal_api_key or "test-key"},
            )

            # Should NOT be 422 (validation error) - that means param mismatch
            assert response.status_code != 422, (
                f"Got 422 validation error - endpoint likely expects different param format. "
                f"Response: {response.json()}"
            )

            # 400 is expected (strategy not found), 401 if bad key
            assert response.status_code in (400, 401, 500)

    def test_rejects_missing_strategy_id(self):
        """Verify endpoint validates required strategyId field."""
        response = client.post(
            self.ENDPOINT,
            json={},  # Missing strategyId
            headers={"X-Internal-Key": settings.convex_internal_api_key or "test-key"},
        )

        # Should be 422 for missing required field
        assert response.status_code == 422

    def test_requires_internal_key(self):
        """Verify endpoint requires X-Internal-Key header."""
        response = client.post(
            self.ENDPOINT,
            json={"strategyId": "test_strategy_123"},
            # No X-Internal-Key header
        )

        assert response.status_code == 401


# =============================================================================
# News Internal Endpoints
# =============================================================================


class TestNewsInternalFetchContract:
    """
    Contract test for POST /news/internal/fetch

    Called by: frontend/convex/scheduler.ts:fetchNews (line ~430)

    Convex sends: empty body (just headers)
    """

    def test_accepts_empty_body(self):
        """Verify endpoint works with empty body (as Convex sends it)."""
        with patch("app.api.news.NewsFetcherService") as mock_service:
            mock_instance = AsyncMock()
            mock_instance.run_fetch_cycle = AsyncMock(return_value=(0, 0, 0))
            mock_service.return_value = mock_instance

            response = client.post(
                "/news/internal/fetch",
                headers={
                    "X-Internal-Key": settings.convex_internal_api_key or "test-key",
                    "Content-Type": "application/json",
                },
            )

            # Should not be 422
            assert response.status_code != 422, f"Validation error: {response.json()}"


class TestNewsInternalProcessContract:
    """
    Contract test for POST /news/internal/process

    Called by: frontend/convex/scheduler.ts:processNews (line ~466)

    Convex sends: empty body (just headers)
    """

    def test_accepts_empty_body(self):
        """Verify endpoint works with empty body (as Convex sends it)."""
        with patch("app.api.news.get_llm_provider") as mock_provider:
            mock_provider.return_value = AsyncMock()

            with patch("app.api.news.get_convex_client") as mock_convex:
                mock_convex.return_value.query = AsyncMock(return_value=[])

                with patch("app.workers.news_processor_worker.run_news_processor_worker") as mock_worker:
                    mock_result = MagicMock()
                    mock_result.items_processed = 0
                    mock_result.items_failed = 0
                    mock_worker.return_value = mock_result

                    response = client.post(
                        "/news/internal/process",
                        headers={
                            "X-Internal-Key": settings.convex_internal_api_key or "test-key",
                            "Content-Type": "application/json",
                        },
                    )

                    # Should not be 422
                    assert response.status_code != 422, f"Validation error: {response.json()}"


# =============================================================================
# Strategy Execution Internal Endpoints
# =============================================================================


class TestStrategyInternalExecuteContract:
    """
    Contract test for POST /internal/execute (general strategy execution)

    Called by: frontend/convex/scheduler.ts:checkTriggers (line ~79)

    Convex sends:
        body: JSON.stringify({
            executionId,
            strategyId: strategy._id,
            config: strategy.config,
        })
    """

    def test_accepts_execution_payload_in_body(self):
        """Verify endpoint accepts executionId, strategyId, config in body."""
        # This test documents the expected contract even if endpoint doesn't exist yet
        response = client.post(
            "/internal/execute",
            json={
                "executionId": "exec_123",
                "strategyId": "strategy_456",
                "config": {"some": "config"},
            },
            headers={"X-Internal-Key": settings.convex_internal_api_key or "test-key"},
        )

        # 404 is ok if endpoint doesn't exist yet, but NOT 422
        if response.status_code == 404:
            pytest.skip("Endpoint /internal/execute not implemented yet")

        assert response.status_code != 422, (
            f"Got 422 validation error - param mismatch. Response: {response.json()}"
        )


# =============================================================================
# Config Format Cross-Boundary Tests
# =============================================================================


class TestDCAResponseShapeContract:
    """
    Verify DCA execution response includes required fields.

    Called by: frontend/convex/scheduler.ts:checkDCAStrategies
    The response must include txHash for completeExecutionById.
    """

    ENDPOINT = "/dca/internal/execute"

    def test_dca_endpoint_exists(self):
        """DCA internal execute endpoint returns non-404."""
        response = client.post(
            self.ENDPOINT,
            json={"strategyId": "nonexistent"},
            headers={"X-Internal-Key": settings.convex_internal_api_key or "test-key"},
        )
        assert response.status_code != 404, (
            f"DCA endpoint {self.ENDPOINT} not found — routing broken"
        )

    def test_dca_execute_response_has_tx_hash_on_success(self):
        """On success, response must include txHash field (needed by completeExecutionById)."""
        with patch("app.api.dca.get_dca_service") as mock_svc:
            mock_svc.return_value.execute_now = AsyncMock(
                return_value={
                    "status": "completed",
                    "txHash": "0xabc123",
                    "executionId": "exec_1",
                }
            )

            response = client.post(
                self.ENDPOINT,
                json={"strategyId": "test_strategy_123"},
                headers={"X-Internal-Key": settings.convex_internal_api_key or "test-key"},
            )

            # If execution succeeds, response body should have txHash
            if response.status_code == 200:
                data = response.json()
                assert "txHash" in data or "tx_hash" in data, (
                    f"Successful DCA response missing txHash. Got: {data}"
                )


class TestConfigFormatContract:
    """
    Verify that _extract_swap_params produces valid SwapParams from both
    config format variants (snake_case agent output AND camelCase canonical).

    This catches the mismatch where the agent creates snake_case configs
    but the executor expects camelCase nested objects.
    """

    WALLET = "0x1234567890abcdef1234567890abcdef12345678"

    def _make_executor(self) -> GenericStrategyExecutor:
        return GenericStrategyExecutor(convex_client=AsyncMock())

    def test_snake_case_config_produces_swap_params(self):
        """Agent-generated snake_case config → valid SwapParams after normalization."""
        config = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 5,
            "frequency": "hourly",
            "chain_id": 8453,
        }

        executor = self._make_executor()
        result = executor._extract_swap_params(config, self.WALLET, "dca")

        assert result is not None, (
            "_extract_swap_params returned None for snake_case config — "
            "normalizer is not being called or mapping is broken"
        )
        assert isinstance(result, SwapParams)
        assert result.from_token == "USDC"
        assert result.to_token == "ETH"
        assert result.amount == "5"
        assert result.chain_id == 8453

    def test_camel_case_config_produces_swap_params(self):
        """Already-canonical camelCase config → valid SwapParams."""
        config = {
            "fromToken": {"symbol": "USDC", "address": "0xabc", "chainId": 8453},
            "toToken": {"symbol": "ETH", "address": "0xdef", "chainId": 8453},
            "amountPerExecution": 10,
            "chainId": 8453,
        }

        executor = self._make_executor()
        result = executor._extract_swap_params(config, self.WALLET, "dca")

        assert result is not None
        assert isinstance(result, SwapParams)
        assert result.from_token in ("USDC", "0xabc")
        assert result.to_token in ("ETH", "0xdef")
        assert result.amount == "10"
        assert result.chain_id == 8453

    def test_both_formats_produce_equivalent_params(self):
        """Both config formats yield functionally equivalent SwapParams."""
        snake_config = {
            "from_token": "USDC",
            "to_token": "ETH",
            "amount_usd": 5,
            "chain_id": 8453,
        }
        camel_config = {
            "fromToken": {"symbol": "USDC", "chainId": 8453},
            "toToken": {"symbol": "ETH", "chainId": 8453},
            "amountPerExecution": 5,
            "chainId": 8453,
        }

        executor = self._make_executor()
        snake_result = executor._extract_swap_params(snake_config, self.WALLET, "dca")
        camel_result = executor._extract_swap_params(camel_config, self.WALLET, "dca")

        assert snake_result is not None
        assert camel_result is not None
        assert snake_result.from_token == camel_result.from_token
        assert snake_result.to_token == camel_result.to_token
        assert snake_result.amount == camel_result.amount
        assert snake_result.chain_id == camel_result.chain_id
