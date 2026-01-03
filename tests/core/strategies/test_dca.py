"""
Tests for DCA Strategy Module

Tests the DCA models, scheduler, executor, and service.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.strategies.dca import (
    DCAStrategy,
    DCAConfig,
    DCAExecution,
    DCAFrequency,
    DCAStatus,
    ExecutionStatus,
    SkipReason,
    TokenInfo,
    MarketConditions,
    ExecutionQuote,
    SessionKeyRequirements,
)
from app.core.strategies.dca.scheduler import DCAScheduler
from app.core.strategies.dca.executor import DCAExecutor, ExecutionResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_from_token():
    """Sample USDC token info."""
    return TokenInfo(
        symbol="USDC",
        address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        chain_id=1,
        decimals=6,
    )


@pytest.fixture
def sample_to_token():
    """Sample ETH token info."""
    return TokenInfo(
        symbol="WETH",
        address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        chain_id=1,
        decimals=18,
    )


@pytest.fixture
def sample_config(sample_from_token, sample_to_token):
    """Sample DCA configuration."""
    return DCAConfig(
        from_token=sample_from_token,
        to_token=sample_to_token,
        amount_per_execution_usd=Decimal("100"),
        frequency=DCAFrequency.WEEKLY,
        execution_hour_utc=9,
        execution_day_of_week=0,  # Monday
        max_slippage_bps=100,
        max_gas_usd=Decimal("10"),
    )


@pytest.fixture
def sample_strategy(sample_config):
    """Sample DCA strategy."""
    return DCAStrategy(
        id="strategy_123",
        user_id="user_123",
        wallet_id="wallet_123",
        wallet_address="0x1234567890123456789012345678901234567890",
        name="Weekly ETH DCA",
        config=sample_config,
        status=DCAStatus.ACTIVE,
        session_key_id="session_123",
    )


@pytest.fixture
def mock_convex_client():
    """Mock Convex client."""
    client = AsyncMock()
    return client


# =============================================================================
# TokenInfo Tests
# =============================================================================


class TestTokenInfo:
    """Tests for TokenInfo model."""

    def test_to_dict(self, sample_from_token):
        """Test conversion to dictionary."""
        d = sample_from_token.to_dict()
        assert d["symbol"] == "USDC"
        assert d["address"] == "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        assert d["chainId"] == 1
        assert d["decimals"] == 6

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "symbol": "ETH",
            "address": "0x0000000000000000000000000000000000000000",
            "chainId": 1,
            "decimals": 18,
        }
        token = TokenInfo.from_dict(data)
        assert token.symbol == "ETH"
        assert token.chain_id == 1


# =============================================================================
# DCAConfig Tests
# =============================================================================


class TestDCAConfig:
    """Tests for DCAConfig model."""

    def test_to_dict(self, sample_config):
        """Test conversion to dictionary."""
        d = sample_config.to_dict()
        assert d["amountPerExecutionUsd"] == 100.0
        assert d["frequency"] == "weekly"
        assert d["executionHourUtc"] == 9
        assert d["maxSlippageBps"] == 100

    def test_from_dict(self, sample_config):
        """Test creation from dictionary."""
        d = sample_config.to_dict()
        config = DCAConfig.from_dict(d)
        assert config.amount_per_execution_usd == Decimal("100")
        assert config.frequency == DCAFrequency.WEEKLY

    def test_optional_fields(self, sample_from_token, sample_to_token):
        """Test config with optional fields."""
        config = DCAConfig(
            from_token=sample_from_token,
            to_token=sample_to_token,
            amount_per_execution_usd=Decimal("50"),
            frequency=DCAFrequency.DAILY,
            skip_if_gas_above_usd=Decimal("20"),
            pause_if_price_above_usd=Decimal("5000"),
            max_total_spend_usd=Decimal("10000"),
            max_executions=100,
        )
        d = config.to_dict()
        assert d["skipIfGasAboveUsd"] == 20.0
        assert d["pauseIfPriceAboveUsd"] == 5000.0
        assert d["maxExecutions"] == 100


# =============================================================================
# DCAScheduler Tests
# =============================================================================


class TestDCAScheduler:
    """Tests for DCA scheduling logic."""

    def test_next_hourly(self):
        """Test hourly scheduling."""
        now = datetime(2025, 1, 1, 10, 30, 0)
        next_exec = DCAScheduler._next_hourly(now)
        assert next_exec == datetime(2025, 1, 1, 11, 0, 0)

    def test_next_daily(self):
        """Test daily scheduling."""
        # Before execution hour
        now = datetime(2025, 1, 1, 8, 0, 0)
        next_exec = DCAScheduler._next_daily(now, hour_utc=9)
        assert next_exec == datetime(2025, 1, 1, 9, 0, 0)

        # After execution hour
        now = datetime(2025, 1, 1, 10, 0, 0)
        next_exec = DCAScheduler._next_daily(now, hour_utc=9)
        assert next_exec == datetime(2025, 1, 2, 9, 0, 0)

    def test_next_weekly(self):
        """Test weekly scheduling."""
        # Wednesday, should go to next Monday
        now = datetime(2025, 1, 1, 10, 0, 0)  # Wednesday
        next_exec = DCAScheduler._next_weekly(now, hour_utc=9, day_of_week=0)
        assert next_exec.weekday() == 0  # Monday
        assert next_exec.hour == 9

    def test_next_monthly(self):
        """Test monthly scheduling."""
        # Before execution day
        now = datetime(2025, 1, 10, 10, 0, 0)
        next_exec = DCAScheduler._next_monthly(now, hour_utc=9, day_of_month=15)
        assert next_exec == datetime(2025, 1, 15, 9, 0, 0)

        # After execution day
        now = datetime(2025, 1, 20, 10, 0, 0)
        next_exec = DCAScheduler._next_monthly(now, hour_utc=9, day_of_month=15)
        assert next_exec == datetime(2025, 2, 15, 9, 0, 0)

    def test_monthly_end_of_month_handling(self):
        """Test monthly scheduling handles months with fewer days."""
        # Day 31 in February should become Feb 28/29
        now = datetime(2025, 1, 31, 10, 0, 0)
        next_exec = DCAScheduler._next_monthly(now, hour_utc=9, day_of_month=31)
        assert next_exec.month == 2
        assert next_exec.day == 28  # 2025 is not a leap year

    def test_get_next_execution_with_config(self, sample_config):
        """Test full scheduler with config."""
        now = datetime(2025, 1, 1, 10, 0, 0)  # Wednesday
        next_exec = DCAScheduler.get_next_execution(sample_config, after=now)
        assert next_exec.weekday() == 0  # Monday
        assert next_exec.hour == 9

    def test_estimated_executions_per_year(self, sample_config):
        """Test execution count estimation."""
        assert DCAScheduler.get_estimated_executions_per_year(sample_config) == 52

        daily_config = DCAConfig(
            from_token=sample_config.from_token,
            to_token=sample_config.to_token,
            amount_per_execution_usd=Decimal("10"),
            frequency=DCAFrequency.DAILY,
        )
        assert DCAScheduler.get_estimated_executions_per_year(daily_config) == 365

    def test_validate_schedule(self, sample_config):
        """Test schedule validation."""
        assert DCAScheduler.validate_schedule(sample_config) is None

        # Invalid hour
        bad_config = DCAConfig(
            from_token=sample_config.from_token,
            to_token=sample_config.to_token,
            amount_per_execution_usd=Decimal("100"),
            frequency=DCAFrequency.DAILY,
            execution_hour_utc=25,  # Invalid
        )
        error = DCAScheduler.validate_schedule(bad_config)
        assert error is not None
        assert "hour" in error.lower()

    def test_format_schedule_description(self, sample_config):
        """Test human-readable schedule description."""
        desc = DCAScheduler.format_schedule_description(sample_config)
        assert "Monday" in desc
        assert "09:00 UTC" in desc


# =============================================================================
# SessionKeyRequirements Tests
# =============================================================================


class TestSessionKeyRequirements:
    """Tests for session key requirement generation."""

    def test_for_dca_strategy(self, sample_config):
        """Test generating requirements for DCA config."""
        requirements = SessionKeyRequirements.for_dca_strategy(sample_config)

        assert "swap" in requirements.permissions
        assert requirements.value_per_tx_usd == Decimal("110")  # 100 + 10% buffer
        assert requirements.token_allowlist == ["USDC", "WETH"]
        assert requirements.chain_allowlist == [1]
        assert requirements.duration_days == 30

    def test_with_max_spend(self, sample_config):
        """Test with max spend limit."""
        sample_config.max_total_spend_usd = Decimal("5000")
        requirements = SessionKeyRequirements.for_dca_strategy(sample_config)
        assert requirements.total_value_usd == Decimal("5500")  # 5000 + 10%


# =============================================================================
# DCAExecutor Tests
# =============================================================================


class TestDCAExecutor:
    """Tests for DCA execution logic."""

    @pytest.fixture
    def mock_executor(self, mock_convex_client):
        """Create executor with mocked dependencies."""
        return DCAExecutor(
            convex_client=mock_convex_client,
            swap_provider=AsyncMock(),
            pricing_provider=AsyncMock(),
            gas_provider=AsyncMock(),
            session_manager=AsyncMock(),
            policy_engine=AsyncMock(),
        )

    @pytest.mark.asyncio
    async def test_check_constraints_gas_too_high(self, mock_executor, sample_config):
        """Test that high gas causes skip."""
        sample_config.skip_if_gas_above_usd = Decimal("5")

        conditions = MarketConditions(
            token_price_usd=Decimal("3000"),
            gas_gwei=Decimal("100"),
            estimated_gas_usd=Decimal("10"),  # Above limit
        )

        skip_reason = mock_executor._check_constraints(sample_config, conditions)
        assert skip_reason == SkipReason.GAS_TOO_HIGH

    @pytest.mark.asyncio
    async def test_check_constraints_price_above_limit(self, mock_executor, sample_config):
        """Test that high price causes skip."""
        sample_config.pause_if_price_above_usd = Decimal("4000")

        conditions = MarketConditions(
            token_price_usd=Decimal("5000"),  # Above limit
            gas_gwei=Decimal("50"),
            estimated_gas_usd=Decimal("5"),
        )

        skip_reason = mock_executor._check_constraints(sample_config, conditions)
        assert skip_reason == SkipReason.PRICE_ABOVE_LIMIT

    @pytest.mark.asyncio
    async def test_check_constraints_price_below_limit(self, mock_executor, sample_config):
        """Test that low price causes skip."""
        sample_config.pause_if_price_below_usd = Decimal("2000")

        conditions = MarketConditions(
            token_price_usd=Decimal("1500"),  # Below limit
            gas_gwei=Decimal("50"),
            estimated_gas_usd=Decimal("5"),
        )

        skip_reason = mock_executor._check_constraints(sample_config, conditions)
        assert skip_reason == SkipReason.PRICE_BELOW_LIMIT

    @pytest.mark.asyncio
    async def test_check_constraints_pass(self, mock_executor, sample_config):
        """Test constraints pass when within limits."""
        sample_config.skip_if_gas_above_usd = Decimal("20")
        sample_config.pause_if_price_above_usd = Decimal("5000")

        conditions = MarketConditions(
            token_price_usd=Decimal("3000"),
            gas_gwei=Decimal("50"),
            estimated_gas_usd=Decimal("5"),
        )

        skip_reason = mock_executor._check_constraints(sample_config, conditions)
        assert skip_reason is None

    @pytest.mark.asyncio
    async def test_validate_session_key_expired(self, mock_executor, sample_strategy, mock_convex_client):
        """Test session key validation with expired key."""
        mock_convex_client.query.return_value = {
            "status": "active",
            "expiresAt": int((datetime.utcnow() - timedelta(hours=1)).timestamp() * 1000),
            "valueLimits": {"totalValueUsedUsd": "0", "maxTotalValueUsd": "10000"},
        }

        valid, error = await mock_executor._validate_session_key(sample_strategy)
        assert not valid
        assert "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_session_key_exhausted(self, mock_executor, sample_strategy, mock_convex_client):
        """Test session key validation with exhausted value."""
        mock_convex_client.query.return_value = {
            "status": "active",
            "expiresAt": int((datetime.utcnow() + timedelta(days=7)).timestamp() * 1000),
            "valueLimits": {"totalValueUsedUsd": "10000", "maxTotalValueUsd": "10000"},
        }

        valid, error = await mock_executor._validate_session_key(sample_strategy)
        assert not valid
        assert "exhausted" in error.lower()

    @pytest.mark.asyncio
    async def test_validate_session_key_valid(self, mock_executor, sample_strategy, mock_convex_client):
        """Test session key validation with valid key."""
        mock_convex_client.query.return_value = {
            "status": "active",
            "expiresAt": int((datetime.utcnow() + timedelta(days=7)).timestamp() * 1000),
            "valueLimits": {"totalValueUsedUsd": "500", "maxTotalValueUsd": "10000"},
        }

        valid, error = await mock_executor._validate_session_key(sample_strategy)
        assert valid
        assert error is None


# =============================================================================
# DCAExecution Tests
# =============================================================================


class TestDCAExecution:
    """Tests for DCAExecution model."""

    def test_from_convex(self):
        """Test creating execution from Convex document."""
        data = {
            "_id": "exec_123",
            "strategyId": "strategy_123",
            "executionNumber": 5,
            "chainId": 1,
            "status": "completed",
            "txHash": "0xabc123",
            "actualInputAmount": "100",
            "actualOutputAmount": "0.05",
            "actualPriceUsd": 2000,
            "gasUsed": 150000,
            "gasPriceGwei": 50,
            "gasUsd": 15,
            "scheduledAt": 1704067200000,
            "startedAt": 1704067260000,
            "completedAt": 1704067320000,
        }

        execution = DCAExecution.from_convex(data)
        assert execution.id == "exec_123"
        assert execution.execution_number == 5
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.tx_hash == "0xabc123"
        assert execution.actual_output_amount == Decimal("0.05")


# =============================================================================
# MarketConditions Tests
# =============================================================================


class TestMarketConditions:
    """Tests for MarketConditions model."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        conditions = MarketConditions(
            token_price_usd=Decimal("3000.50"),
            gas_gwei=Decimal("45.5"),
            estimated_gas_usd=Decimal("12.34"),
        )

        d = conditions.to_dict()
        assert d["tokenPriceUsd"] == 3000.50
        assert d["gasGwei"] == 45.5
        assert d["estimatedGasUsd"] == 12.34


# =============================================================================
# ExecutionQuote Tests
# =============================================================================


class TestExecutionQuote:
    """Tests for ExecutionQuote model."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        quote = ExecutionQuote(
            input_amount=Decimal("100"),
            expected_output_amount=Decimal("0.05"),
            minimum_output_amount=Decimal("0.0495"),
            price_impact_bps=10,
            route="USDC -> WETH via Uniswap V3",
        )

        d = quote.to_dict()
        assert d["inputAmount"] == "100"
        assert d["expectedOutputAmount"] == "0.05"
        assert d["minimumOutputAmount"] == "0.0495"
        assert d["priceImpactBps"] == 10
        assert "Uniswap" in d["route"]
