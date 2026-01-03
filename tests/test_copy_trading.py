"""
Copy Trading Tests

Comprehensive tests for the copy trading module.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.copy_trading import (
    CopyConfig,
    CopyRelationship,
    CopyExecution,
    CopyExecutionStatus,
    SizingMode,
    TradeSignal,
    LeaderProfile,
    CopyTradingManager,
    CopyExecutor,
    LeaderAnalytics,
    PercentageSizing,
    FixedSizing,
    ProportionalSizing,
    get_sizing_strategy,
)
from app.core.copy_trading.models import SkipReason, CopyTradingStats
from app.core.copy_trading.executor import ExecutionResult, MockCopyExecutor


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_config():
    """Create a sample copy config."""
    return CopyConfig(
        leader_address="0x1234567890123456789012345678901234567890",
        leader_chain="ethereum",
        leader_label="Test Whale",
        sizing_mode=SizingMode.PERCENTAGE,
        size_value=Decimal("5"),
        min_trade_usd=Decimal("10"),
        max_trade_usd=Decimal("1000"),
        token_whitelist=None,
        token_blacklist=["0xbadtoken"],
        allowed_actions=["swap"],
        delay_seconds=0,
        max_delay_seconds=300,
        max_slippage_bps=100,
        max_daily_trades=20,
        max_daily_volume_usd=Decimal("10000"),
    )


@pytest.fixture
def sample_relationship(sample_config):
    """Create a sample copy relationship."""
    return CopyRelationship(
        user_id="user123",
        follower_address="0xfollower1234567890123456789012345678901234",
        follower_chain="ethereum",
        config=sample_config,
    )


@pytest.fixture
def sample_signal():
    """Create a sample trade signal."""
    return TradeSignal(
        leader_address="0x1234567890123456789012345678901234567890",
        leader_chain="ethereum",
        tx_hash="0xabcdef1234567890",
        block_number=12345678,
        timestamp=datetime.now(timezone.utc),
        action="swap",
        token_in_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        token_in_symbol="WETH",
        token_in_amount=Decimal("1.5"),
        token_out_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
        token_out_symbol="USDC",
        token_out_amount=Decimal("3000"),
        value_usd=Decimal("3000"),
        dex="uniswap_v3",
    )


# =============================================================================
# Model Tests
# =============================================================================


class TestCopyConfig:
    """Tests for CopyConfig model."""

    def test_is_token_allowed_no_lists(self, sample_config):
        """Token is allowed when no whitelist/blacklist."""
        config = CopyConfig(
            leader_address="0x123",
            leader_chain="ethereum",
        )
        assert config.is_token_allowed("0xanytoken") is True

    def test_is_token_allowed_blacklisted(self, sample_config):
        """Token is blocked when blacklisted."""
        assert sample_config.is_token_allowed("0xbadtoken") is False
        assert sample_config.is_token_allowed("0xBadToken") is False  # Case insensitive

    def test_is_token_allowed_not_blacklisted(self, sample_config):
        """Token is allowed when not blacklisted."""
        assert sample_config.is_token_allowed("0xgoodtoken") is True

    def test_is_token_allowed_whitelist(self):
        """Token allowed only if in whitelist."""
        config = CopyConfig(
            leader_address="0x123",
            leader_chain="ethereum",
            token_whitelist=["0xtoken1", "0xtoken2"],
        )
        assert config.is_token_allowed("0xtoken1") is True
        assert config.is_token_allowed("0xtoken3") is False

    def test_is_action_allowed(self, sample_config):
        """Action filtering works correctly."""
        assert sample_config.is_action_allowed("swap") is True
        assert sample_config.is_action_allowed("SWAP") is True
        assert sample_config.is_action_allowed("bridge") is False


class TestCopyRelationship:
    """Tests for CopyRelationship model."""

    def test_can_execute_trade_active(self, sample_relationship):
        """Can execute trade when active."""
        can_execute, reason = sample_relationship.can_execute_trade(Decimal("100"))
        assert can_execute is True
        assert reason is None

    def test_can_execute_trade_paused(self, sample_relationship):
        """Cannot execute when paused."""
        sample_relationship.is_paused = True
        can_execute, reason = sample_relationship.can_execute_trade(Decimal("100"))
        assert can_execute is False
        assert reason == SkipReason.PAUSED

    def test_can_execute_trade_daily_limit(self, sample_relationship):
        """Cannot exceed daily trade limit."""
        sample_relationship.daily_trade_count = 20  # At limit
        can_execute, reason = sample_relationship.can_execute_trade(Decimal("100"))
        assert can_execute is False
        assert reason == SkipReason.DAILY_LIMIT_REACHED

    def test_can_execute_trade_volume_limit(self, sample_relationship):
        """Cannot exceed daily volume limit."""
        sample_relationship.daily_volume_usd = Decimal("9500")
        can_execute, reason = sample_relationship.can_execute_trade(Decimal("600"))
        assert can_execute is False
        assert reason == SkipReason.VOLUME_LIMIT_REACHED

    def test_record_trade_success(self, sample_relationship):
        """Recording successful trade updates stats."""
        sample_relationship.record_trade(success=True, volume_usd=Decimal("100"))
        assert sample_relationship.total_trades == 1
        assert sample_relationship.successful_trades == 1
        assert sample_relationship.total_volume_usd == Decimal("100")

    def test_record_trade_failed(self, sample_relationship):
        """Recording failed trade updates stats."""
        sample_relationship.record_trade(success=False, volume_usd=Decimal("0"))
        assert sample_relationship.total_trades == 1
        assert sample_relationship.failed_trades == 1

    def test_record_trade_skipped(self, sample_relationship):
        """Recording skipped trade updates stats."""
        sample_relationship.record_trade(success=False, volume_usd=Decimal("0"), skipped=True)
        assert sample_relationship.total_trades == 1
        assert sample_relationship.skipped_trades == 1


class TestLeaderProfile:
    """Tests for LeaderProfile model."""

    def test_risk_score_base(self):
        """Risk score starts at 0.5."""
        profile = LeaderProfile(address="0x123", chain="ethereum")
        assert profile.risk_score == 0.5

    def test_risk_score_high_drawdown(self):
        """High drawdown increases risk score."""
        profile = LeaderProfile(
            address="0x123",
            chain="ethereum",
            max_drawdown_pct=60,
        )
        assert profile.risk_score == 0.8

    def test_risk_score_leverage(self):
        """Using leverage increases risk score."""
        profile = LeaderProfile(
            address="0x123",
            chain="ethereum",
            uses_leverage=True,
        )
        assert profile.risk_score == 0.7

    def test_risk_score_low_win_rate(self):
        """Low win rate increases risk score."""
        profile = LeaderProfile(
            address="0x123",
            chain="ethereum",
            win_rate=35,
        )
        assert profile.risk_score == 0.7


# =============================================================================
# Sizing Strategy Tests
# =============================================================================


class TestPercentageSizing:
    """Tests for percentage sizing strategy."""

    def test_calculate_size_basic(self, sample_signal, sample_config):
        """Calculate size as percentage of portfolio."""
        strategy = PercentageSizing()
        size = strategy.calculate_size(
            signal=sample_signal,
            config=sample_config,
            follower_portfolio_value_usd=Decimal("10000"),
        )
        # 5% of $10,000 = $500
        assert size == Decimal("500")

    def test_calculate_size_min_bound(self, sample_signal, sample_config):
        """Size below minimum returns 0."""
        sample_config.size_value = Decimal("0.01")  # 0.01%
        strategy = PercentageSizing()
        size = strategy.calculate_size(
            signal=sample_signal,
            config=sample_config,
            follower_portfolio_value_usd=Decimal("100"),  # $0.01 result
        )
        assert size == Decimal("0")  # Below $10 minimum

    def test_calculate_size_max_bound(self, sample_signal, sample_config):
        """Size capped at maximum."""
        sample_config.size_value = Decimal("50")  # 50%
        strategy = PercentageSizing()
        size = strategy.calculate_size(
            signal=sample_signal,
            config=sample_config,
            follower_portfolio_value_usd=Decimal("10000"),
        )
        # Would be $5000 but capped at $1000
        assert size == Decimal("1000")


class TestFixedSizing:
    """Tests for fixed sizing strategy."""

    def test_calculate_size_basic(self, sample_signal, sample_config):
        """Calculate fixed size regardless of portfolio."""
        sample_config.sizing_mode = SizingMode.FIXED
        sample_config.size_value = Decimal("200")
        strategy = FixedSizing()
        size = strategy.calculate_size(
            signal=sample_signal,
            config=sample_config,
            follower_portfolio_value_usd=Decimal("100000"),
        )
        assert size == Decimal("200")


class TestProportionalSizing:
    """Tests for proportional sizing strategy."""

    def test_calculate_size_with_leader_portfolio(self, sample_signal, sample_config):
        """Calculate size proportional to leader's trade."""
        sample_config.sizing_mode = SizingMode.PROPORTIONAL
        sample_config.size_value = Decimal("1")  # 1x multiplier
        strategy = ProportionalSizing()

        # Leader traded $3000 out of $100000 (3%)
        # Follower has $10000, so 3% = $300
        size = strategy.calculate_size(
            signal=sample_signal,
            config=sample_config,
            follower_portfolio_value_usd=Decimal("10000"),
            leader_portfolio_value_usd=Decimal("100000"),
        )
        assert size == Decimal("300")

    def test_calculate_size_without_leader_portfolio(self, sample_signal, sample_config):
        """Falls back to size_value when leader portfolio unknown."""
        sample_config.sizing_mode = SizingMode.PROPORTIONAL
        sample_config.size_value = Decimal("100")
        strategy = ProportionalSizing()

        size = strategy.calculate_size(
            signal=sample_signal,
            config=sample_config,
            follower_portfolio_value_usd=Decimal("10000"),
            leader_portfolio_value_usd=None,
        )
        assert size == Decimal("100")


class TestGetSizingStrategy:
    """Tests for sizing strategy factory."""

    def test_get_percentage_strategy(self):
        """Returns percentage strategy for percentage mode."""
        strategy = get_sizing_strategy(SizingMode.PERCENTAGE)
        assert isinstance(strategy, PercentageSizing)

    def test_get_fixed_strategy(self):
        """Returns fixed strategy for fixed mode."""
        strategy = get_sizing_strategy(SizingMode.FIXED)
        assert isinstance(strategy, FixedSizing)

    def test_get_proportional_strategy(self):
        """Returns proportional strategy for proportional mode."""
        strategy = get_sizing_strategy(SizingMode.PROPORTIONAL)
        assert isinstance(strategy, ProportionalSizing)


# =============================================================================
# Executor Tests
# =============================================================================


class TestCopyExecutor:
    """Tests for CopyExecutor."""

    @pytest.mark.asyncio
    async def test_execute_no_provider(self, sample_signal):
        """Returns error when no provider configured."""
        executor = CopyExecutor()
        result = await executor.execute(
            signal=sample_signal,
            size_usd=Decimal("100"),
            follower_address="0xfollower",
            follower_chain="ethereum",
        )
        assert result.success is False
        assert "not configured" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_solana(self, sample_signal):
        """Routes Solana swaps to Jupiter."""
        mock_jupiter = MagicMock()
        mock_jupiter.get_quote = AsyncMock(return_value={
            "inputAmount": "100000000",
            "outputAmount": "200000000",
        })
        mock_jupiter.execute_swap = AsyncMock(return_value={
            "success": True,
            "txHash": "0xsolana123",
            "valueUsd": 100,
            "outputAmount": 200,
            "actualSlippageBps": 50,
        })

        executor = CopyExecutor(jupiter_provider=mock_jupiter)

        sample_signal.leader_chain = "solana"
        result = await executor.execute(
            signal=sample_signal,
            size_usd=Decimal("100"),
            follower_address="SolanaAddress123",
            follower_chain="solana",
        )

        assert result.success is True
        assert result.tx_hash == "0xsolana123"
        mock_jupiter.get_quote.assert_called_once()

    def test_chain_to_id(self):
        """Chain name to ID mapping."""
        executor = CopyExecutor()
        assert executor._chain_to_id("ethereum") == 1
        assert executor._chain_to_id("polygon") == 137
        assert executor._chain_to_id("base") == 8453


class TestMockCopyExecutor:
    """Tests for MockCopyExecutor."""

    @pytest.mark.asyncio
    async def test_mock_execute_success(self, sample_signal):
        """Mock executor records executions."""
        executor = MockCopyExecutor(success_rate=1.0)  # Always succeed
        result = await executor.execute(
            signal=sample_signal,
            size_usd=Decimal("100"),
            follower_address="0xfollower",
            follower_chain="ethereum",
        )

        assert result.success is True
        assert result.tx_hash is not None
        assert len(executor.executions) == 1


# =============================================================================
# Manager Tests
# =============================================================================


class TestCopyTradingManager:
    """Tests for CopyTradingManager."""

    @pytest.fixture
    def mock_manager(self):
        """Create manager with mocked dependencies."""
        executor = MockCopyExecutor(success_rate=1.0)
        return CopyTradingManager(
            convex_client=None,
            executor=executor,
        )

    @pytest.mark.asyncio
    async def test_start_copying(self, mock_manager, sample_config):
        """Start copying creates relationship."""
        relationship = await mock_manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )

        assert relationship is not None
        assert relationship.user_id == "user123"
        assert relationship.is_active is True
        assert relationship.config.leader_address == sample_config.leader_address

    @pytest.mark.asyncio
    async def test_stop_copying(self, mock_manager, sample_config):
        """Stop copying deactivates relationship."""
        relationship = await mock_manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )

        stopped = await mock_manager.stop_copying(relationship.id)
        assert stopped.is_active is False

    @pytest.mark.asyncio
    async def test_pause_resume_copying(self, mock_manager, sample_config):
        """Pause and resume work correctly."""
        relationship = await mock_manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )

        # Pause
        paused = await mock_manager.pause_copying(relationship.id, "Testing")
        assert paused.is_paused is True
        assert paused.pause_reason == "Testing"

        # Resume
        resumed = await mock_manager.resume_copying(relationship.id)
        assert resumed.is_paused is False
        assert resumed.pause_reason is None

    @pytest.mark.asyncio
    async def test_handle_trade_signal_no_followers(self, mock_manager, sample_signal):
        """No executions when no followers."""
        executions = await mock_manager.handle_trade_signal(sample_signal)
        assert len(executions) == 0

    @pytest.mark.asyncio
    async def test_handle_trade_signal_with_followers(self, mock_manager, sample_config, sample_signal):
        """Executions created for followers."""
        # Start following
        await mock_manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )

        # Handle signal
        executions = await mock_manager.handle_trade_signal(sample_signal)
        assert len(executions) == 1
        assert executions[0].status == CopyExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_handle_trade_signal_blacklisted_token(self, mock_manager, sample_config, sample_signal):
        """Signal skipped for blacklisted token."""
        # Set signal token to blacklisted
        sample_signal.token_in_address = "0xbadtoken"

        await mock_manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )

        executions = await mock_manager.handle_trade_signal(sample_signal)
        assert len(executions) == 1
        assert executions[0].status == CopyExecutionStatus.SKIPPED
        assert executions[0].skip_reason == SkipReason.TOKEN_BLACKLISTED

    @pytest.mark.asyncio
    async def test_handle_trade_signal_action_not_allowed(self, mock_manager, sample_config, sample_signal):
        """Signal skipped for disallowed action."""
        sample_signal.action = "bridge"

        await mock_manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )

        executions = await mock_manager.handle_trade_signal(sample_signal)
        assert len(executions) == 1
        assert executions[0].status == CopyExecutionStatus.SKIPPED
        assert executions[0].skip_reason == SkipReason.ACTION_NOT_ALLOWED

    @pytest.mark.asyncio
    async def test_get_user_stats(self, mock_manager, sample_config, sample_signal):
        """User stats calculated correctly."""
        # Create relationship and execute a trade
        await mock_manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )
        await mock_manager.handle_trade_signal(sample_signal)

        stats = await mock_manager.get_user_stats("user123")
        assert stats.active_relationships == 1
        assert stats.total_copy_trades == 1
        assert stats.successful_trades == 1


# =============================================================================
# Analytics Tests
# =============================================================================


class TestLeaderAnalytics:
    """Tests for LeaderAnalytics."""

    @pytest.fixture
    def mock_analytics(self):
        """Create analytics with mocked dependencies."""
        return LeaderAnalytics(convex_client=None, price_provider=None)

    @pytest.mark.asyncio
    async def test_get_profile_cached(self, mock_analytics):
        """Returns cached profile if recent."""
        # Pre-populate cache
        profile = LeaderProfile(
            address="0x123",
            chain="ethereum",
            last_analyzed_at=datetime.now(timezone.utc),
        )
        mock_analytics._profiles["ethereum:0x123"] = profile

        result = await mock_analytics.get_profile("0x123", "ethereum")
        assert result is profile

    @pytest.mark.asyncio
    async def test_get_profile_new(self, mock_analytics):
        """Creates new profile for unknown address."""
        result = await mock_analytics.get_profile("0xnewaddr", "ethereum")
        assert result is not None
        assert result.address == "0xnewaddr"
        assert result.chain == "ethereum"

    def test_calculate_win_rate(self, mock_analytics):
        """Win rate calculated from profitable trades."""
        profile = LeaderProfile(address="0x123", chain="ethereum")
        trades = [
            {"valueUsd": 100, "pnlUsd": 50},
            {"valueUsd": 100, "pnlUsd": -20},
            {"valueUsd": 100, "pnlUsd": 30},
            {"valueUsd": 100, "pnlUsd": -10},
        ]

        result = mock_analytics._calculate_win_rate(profile, trades)
        assert result.win_rate == 0.5  # 2 profitable out of 4

    def test_calculate_pnl_metrics(self, mock_analytics):
        """P&L metrics calculated correctly."""
        profile = LeaderProfile(address="0x123", chain="ethereum")
        trades = [
            {"pnlUsd": 100},
            {"pnlUsd": -50},
            {"pnlUsd": 200},
        ]

        result = mock_analytics._calculate_pnl_metrics(profile, trades)
        assert result.total_pnl_usd == Decimal("250")

    def test_calculate_risk_metrics(self, mock_analytics):
        """Risk metrics calculated correctly."""
        profile = LeaderProfile(address="0x123", chain="ethereum")
        trades = [
            {"pnlUsd": 100, "valueUsd": 1000},
            {"pnlUsd": -300, "valueUsd": 500},  # Large drawdown
            {"pnlUsd": 200, "valueUsd": 800},
        ]

        result = mock_analytics._calculate_risk_metrics(profile, trades)
        assert result.max_drawdown_pct is not None

    def test_calculate_data_quality_score(self, mock_analytics):
        """Data quality score increases with more data."""
        profile = LeaderProfile(address="0x123", chain="ethereum")

        # Few trades = low score
        trades_few = [{"timestamp": 1000000000000} for _ in range(5)]
        result = mock_analytics._calculate_data_quality(profile, trades_few)
        score_few = result.data_quality_score

        # Many trades = higher score
        trades_many = [{"timestamp": 1000000000000, "pnlUsd": 10} for _ in range(100)]
        result = mock_analytics._calculate_data_quality(profile, trades_many)
        score_many = result.data_quality_score

        assert score_many > score_few


# =============================================================================
# Integration Tests
# =============================================================================


class TestCopyTradingIntegration:
    """Integration tests for copy trading flow."""

    @pytest.mark.asyncio
    async def test_full_copy_flow(self, sample_config, sample_signal):
        """Test complete flow from setup to execution."""
        # Setup
        executor = MockCopyExecutor(success_rate=1.0)
        manager = CopyTradingManager(convex_client=None, executor=executor)

        # Start copying
        relationship = await manager.start_copying(
            user_id="user123",
            follower_address="0xfollower",
            follower_chain="ethereum",
            config=sample_config,
        )

        assert relationship.is_active

        # Simulate leader trade
        executions = await manager.handle_trade_signal(sample_signal)

        assert len(executions) == 1
        assert executions[0].status == CopyExecutionStatus.COMPLETED

        # Check stats updated
        assert relationship.total_trades == 1
        assert relationship.successful_trades == 1
        assert relationship.total_volume_usd > 0

        # Stop copying
        stopped = await manager.stop_copying(relationship.id)
        assert stopped.is_active is False

        # No more executions after stopping
        executions = await manager.handle_trade_signal(sample_signal)
        assert len(executions) == 0

    @pytest.mark.asyncio
    async def test_multiple_followers_single_leader(self, sample_config, sample_signal):
        """Multiple followers can copy same leader."""
        executor = MockCopyExecutor(success_rate=1.0)
        manager = CopyTradingManager(convex_client=None, executor=executor)

        # Three followers
        for i in range(3):
            await manager.start_copying(
                user_id=f"user{i}",
                follower_address=f"0xfollower{i}",
                follower_chain="ethereum",
                config=sample_config,
            )

        # Leader trades
        executions = await manager.handle_trade_signal(sample_signal)

        # All three should execute
        assert len(executions) == 3
        assert all(e.status == CopyExecutionStatus.COMPLETED for e in executions)

    @pytest.mark.asyncio
    async def test_different_sizing_strategies(self, sample_config, sample_signal):
        """Different sizing modes produce different sizes."""
        executor = MockCopyExecutor(success_rate=1.0)
        manager = CopyTradingManager(convex_client=None, executor=executor)

        # Percentage sizing
        config_pct = CopyConfig(**{**sample_config.model_dump(), "sizing_mode": SizingMode.PERCENTAGE, "size_value": Decimal("10")})
        await manager.start_copying(
            user_id="user_pct",
            follower_address="0xfollower_pct",
            follower_chain="ethereum",
            config=config_pct,
        )

        # Fixed sizing
        config_fixed = CopyConfig(**{**sample_config.model_dump(), "sizing_mode": SizingMode.FIXED, "size_value": Decimal("500")})
        await manager.start_copying(
            user_id="user_fixed",
            follower_address="0xfollower_fixed",
            follower_chain="ethereum",
            config=config_fixed,
        )

        executions = await manager.handle_trade_signal(sample_signal)
        assert len(executions) == 2

        sizes = [e.calculated_size_usd for e in executions if e.calculated_size_usd]
        # Should have different sizes due to different modes
        assert len(set(sizes)) == 2
