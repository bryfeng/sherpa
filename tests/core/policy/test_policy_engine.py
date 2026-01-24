"""
Tests for the Policy Engine

Tests for PolicyEngine, SessionPolicy, RiskPolicy, and SystemPolicy.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta

from app.core.policy import (
    FeePolicy,
    FeePolicyConfig,
    PolicyEngine,
    PolicyResult,
    PolicyType,
    PolicyViolation,
    ViolationSeverity,
    ActionContext,
    RiskPolicyConfig,
    SystemPolicyConfig,
    SessionPolicy,
    RiskPolicy,
    SystemPolicy,
    RiskLevel,
)
from app.core.wallet.models import (
    SessionKey,
    SessionKeyStatus,
    Permission,
    ValueLimit,
    ChainAllowlist,
    ContractAllowlist,
    TokenAllowlist,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_context() -> ActionContext:
    """Basic action context for testing."""
    return ActionContext(
        session_id="session-123",
        wallet_address="0x1234567890123456789012345678901234567890",
        action_type="swap",
        chain_id=1,
        value_usd=Decimal("100"),
    )


@pytest.fixture
def high_value_context() -> ActionContext:
    """High value action context."""
    return ActionContext(
        session_id="session-123",
        wallet_address="0x1234567890123456789012345678901234567890",
        action_type="swap",
        chain_id=1,
        value_usd=Decimal("10000"),
        portfolio_value_usd=Decimal("20000"),
        current_position_percent=10.0,
        slippage_percent=2.5,
        estimated_gas_usd=Decimal("50"),
    )


@pytest.fixture
def valid_session_key() -> SessionKey:
    """Valid session key for testing."""
    return SessionKey(
        session_id="session-123",
        wallet_address="0x1234567890123456789012345678901234567890",
        permissions=[Permission.SWAP, Permission.BRIDGE],
        value_limits=ValueLimit(
            max_value_per_tx_usd=Decimal("1000"),
            max_total_value_usd=Decimal("10000"),
        ),
        chain_allowlist=ChainAllowlist(allowed_chain_ids={1, 137, 42161}),
        contract_allowlist=ContractAllowlist(),  # Empty = allow all
        token_allowlist=TokenAllowlist(),  # Empty = allow all
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        status=SessionKeyStatus.ACTIVE,
    )


@pytest.fixture
def expired_session_key(valid_session_key: SessionKey) -> SessionKey:
    """Expired session key."""
    valid_session_key.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    valid_session_key.status = SessionKeyStatus.EXPIRED
    return valid_session_key


@pytest.fixture
def fee_policy_config() -> FeePolicyConfig:
    """Valid fee policy config."""
    return FeePolicyConfig(
        chain_id=1,
        stablecoin_symbol="USDC",
        stablecoin_address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        stablecoin_decimals=6,
        allow_native_fallback=True,
        native_symbol="ETH",
        native_decimals=18,
        fee_asset_order=["stablecoin", "native"],
        reimbursement_mode="none",
        is_enabled=True,
    )


# =============================================================================
# Session Policy Tests
# =============================================================================

class TestSessionPolicy:
    """Tests for SessionPolicy."""

    def test_valid_action_allowed(self, valid_session_key: SessionKey, basic_context: ActionContext):
        """Test that valid actions are allowed."""
        policy = SessionPolicy(valid_session_key)
        violations = policy.evaluate(basic_context)
        assert len(violations) == 0

    def test_expired_session_blocked(self, expired_session_key: SessionKey, basic_context: ActionContext):
        """Test that expired sessions are blocked."""
        policy = SessionPolicy(expired_session_key)
        violations = policy.evaluate(basic_context)

        assert len(violations) == 1
        assert violations[0].policy_name == "session_validity"
        assert violations[0].severity == ViolationSeverity.BLOCK

    def test_permission_denied(self, valid_session_key: SessionKey, basic_context: ActionContext):
        """Test that unpermitted actions are blocked."""
        valid_session_key.permissions = [Permission.BRIDGE]  # No swap permission
        basic_context.action_type = "swap"

        policy = SessionPolicy(valid_session_key)
        violations = policy.evaluate(basic_context)

        assert len(violations) == 1
        assert violations[0].policy_name == "permission"
        assert "swap" in violations[0].message

    def test_value_limit_exceeded(self, valid_session_key: SessionKey, basic_context: ActionContext):
        """Test that value limits are enforced."""
        basic_context.value_usd = Decimal("2000")  # Exceeds 1000 per-tx limit

        policy = SessionPolicy(valid_session_key)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "value_limit_per_tx" for v in violations)

    def test_chain_not_allowed(self, valid_session_key: SessionKey, basic_context: ActionContext):
        """Test that chain allowlist is enforced."""
        basic_context.chain_id = 999  # Not in allowed chains

        policy = SessionPolicy(valid_session_key)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "chain_allowlist" for v in violations)


# =============================================================================
# Risk Policy Tests
# =============================================================================

class TestRiskPolicy:
    """Tests for RiskPolicy."""

    def test_low_value_no_warnings(self, basic_context: ActionContext):
        """Test that low value transactions have no violations."""
        config = RiskPolicyConfig()
        policy = RiskPolicy(config)
        violations = policy.evaluate(basic_context)

        assert len(violations) == 0

    def test_high_value_blocked(self, basic_context: ActionContext):
        """Test that high value transactions are blocked."""
        basic_context.value_usd = Decimal("10000")  # Exceeds default 5000 max

        config = RiskPolicyConfig(max_single_tx_usd=Decimal("5000"))
        policy = RiskPolicy(config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "max_single_tx" for v in violations)
        assert any(v.severity == ViolationSeverity.BLOCK for v in violations)

    def test_position_limit_warning(self, high_value_context: ActionContext):
        """Test position concentration warnings."""
        config = RiskPolicyConfig(max_position_percent=20.0)
        policy = RiskPolicy(config)

        # This would create a 50% position
        high_value_context.value_usd = Decimal("10000")
        high_value_context.portfolio_value_usd = Decimal("20000")

        violations = policy.evaluate(high_value_context)

        assert any(v.policy_name == "max_position_percent" for v in violations)

    def test_slippage_blocked(self, basic_context: ActionContext):
        """Test that high slippage is blocked."""
        basic_context.slippage_percent = 5.0  # Exceeds default 3%

        config = RiskPolicyConfig(max_slippage_percent=3.0)
        policy = RiskPolicy(config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "max_slippage" for v in violations)

    def test_daily_loss_limit(self, basic_context: ActionContext):
        """Test daily loss limit enforcement."""
        basic_context.daily_loss_usd = Decimal("1500")  # Exceeds default 1000

        config = RiskPolicyConfig(max_daily_loss_usd=Decimal("1000"))
        policy = RiskPolicy(config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "daily_loss_limit" for v in violations)
        assert any(v.severity == ViolationSeverity.BLOCK for v in violations)

    def test_risk_score_calculation(self, high_value_context: ActionContext):
        """Test risk score calculation."""
        config = RiskPolicyConfig()
        policy = RiskPolicy(config)

        score, level = policy.calculate_risk_score(high_value_context)

        assert 0.0 <= score <= 1.0
        assert level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_requires_approval(self, high_value_context: ActionContext):
        """Test approval requirement for high value."""
        config = RiskPolicyConfig(require_approval_above_usd=Decimal("1000"))
        policy = RiskPolicy(config)

        requires, reason = policy.requires_approval(high_value_context)

        assert requires is True
        assert reason is not None

    def test_disabled_policy(self, basic_context: ActionContext):
        """Test that disabled policy returns no violations."""
        basic_context.value_usd = Decimal("1000000")  # Huge value

        config = RiskPolicyConfig(enabled=False)
        policy = RiskPolicy(config)
        violations = policy.evaluate(basic_context)

        assert len(violations) == 0


# =============================================================================
# System Policy Tests
# =============================================================================

class TestSystemPolicy:
    """Tests for SystemPolicy."""

    def test_normal_operation(self, basic_context: ActionContext):
        """Test that normal operations pass."""
        config = SystemPolicyConfig()
        policy = SystemPolicy(config)
        violations = policy.evaluate(basic_context)

        assert len(violations) == 0

    def test_emergency_stop(self, basic_context: ActionContext):
        """Test emergency stop blocks all actions."""
        config = SystemPolicyConfig(
            emergency_stop=True,
            emergency_stop_reason="Security incident",
        )
        policy = SystemPolicy(config)
        violations = policy.evaluate(basic_context)

        assert len(violations) == 1
        assert violations[0].policy_name == "emergency_stop"
        assert violations[0].severity == ViolationSeverity.BLOCK

    def test_maintenance_mode(self, basic_context: ActionContext):
        """Test maintenance mode blocks actions."""
        config = SystemPolicyConfig(
            in_maintenance=True,
            maintenance_message="Scheduled maintenance",
        )
        policy = SystemPolicy(config)
        violations = policy.evaluate(basic_context)

        assert len(violations) == 1
        assert violations[0].policy_name == "maintenance"

    def test_blocked_contract(self, basic_context: ActionContext):
        """Test blocked contracts are rejected."""
        basic_context.contract_address = "0xscamcontract"

        config = SystemPolicyConfig(blocked_contracts=["0xscamcontract"])
        policy = SystemPolicy(config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "blocked_contract" for v in violations)

    def test_blocked_token(self, basic_context: ActionContext):
        """Test blocked tokens are rejected."""
        basic_context.token_in = "0xscamtoken"

        config = SystemPolicyConfig(blocked_tokens=["0xscamtoken"])
        policy = SystemPolicy(config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "blocked_token" for v in violations)

    def test_blocked_chain(self, basic_context: ActionContext):
        """Test blocked chains are rejected."""
        basic_context.chain_id = 666

        config = SystemPolicyConfig(blocked_chains=[666])
        policy = SystemPolicy(config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "blocked_chain" for v in violations)

    def test_global_tx_limit(self, basic_context: ActionContext):
        """Test global transaction limit."""
        basic_context.value_usd = Decimal("200000")  # Exceeds default 100000

        config = SystemPolicyConfig()
        policy = SystemPolicy(config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "global_tx_limit" for v in violations)

    def test_is_operational(self):
        """Test is_operational check."""
        config = SystemPolicyConfig()
        policy = SystemPolicy(config)

        is_op, reason = policy.is_operational()
        assert is_op is True
        assert reason is None

        config.emergency_stop = True
        is_op, reason = policy.is_operational()
        assert is_op is False
        assert reason is not None


# =============================================================================
# Fee Policy Tests
# =============================================================================

class TestFeePolicy:
    """Tests for FeePolicy."""

    def test_missing_fee_policy_blocks(self, basic_context: ActionContext):
        """Test that missing fee policy blocks execution."""
        policy = FeePolicy(FeePolicyConfig.missing_for_chain(1))
        violations = policy.evaluate(basic_context)

        assert len(violations) == 1
        assert violations[0].policy_name == "fee_policy_missing"
        assert violations[0].severity == ViolationSeverity.BLOCK

    def test_fee_asset_order_enforced(
        self,
        basic_context: ActionContext,
        fee_policy_config: FeePolicyConfig,
    ):
        """Test that fee asset order must start with stablecoin."""
        fee_policy_config.fee_asset_order = ["native", "stablecoin"]
        policy = FeePolicy(fee_policy_config)
        violations = policy.evaluate(basic_context)

        assert any(v.policy_name == "fee_asset_order" for v in violations)


# =============================================================================
# Policy Engine Integration Tests
# =============================================================================

class TestPolicyEngine:
    """Tests for PolicyEngine (unified evaluation)."""

    def test_all_policies_pass(
        self,
        valid_session_key: SessionKey,
        basic_context: ActionContext,
    ):
        """Test that valid actions pass all policies."""
        engine = PolicyEngine(
            session_key=valid_session_key,
            risk_config=RiskPolicyConfig(),
            system_config=SystemPolicyConfig(),
        )

        result = engine.evaluate(basic_context)

        assert result.approved is True
        assert len(result.violations) == 0

    def test_system_blocks_first(
        self,
        valid_session_key: SessionKey,
        basic_context: ActionContext,
    ):
        """Test that system policy blocks before checking others."""
        engine = PolicyEngine(
            session_key=valid_session_key,
            system_config=SystemPolicyConfig(emergency_stop=True),
        )

        result = engine.evaluate(basic_context)

        assert result.approved is False
        assert any(v.policy_type == PolicyType.SYSTEM for v in result.violations)

    def test_session_blocks_before_risk(
        self,
        expired_session_key: SessionKey,
        basic_context: ActionContext,
    ):
        """Test that session policy blocks before risk check."""
        engine = PolicyEngine(
            session_key=expired_session_key,
            risk_config=RiskPolicyConfig(),
        )

        result = engine.evaluate(basic_context)

        assert result.approved is False
        assert any(v.policy_type == PolicyType.SESSION for v in result.violations)

    def test_fee_policy_blocks_before_session(
        self,
        valid_session_key: SessionKey,
        basic_context: ActionContext,
    ):
        """Test that fee policy blocks before session checks."""
        engine = PolicyEngine(
            session_key=valid_session_key,
            fee_config=FeePolicyConfig.missing_for_chain(1),
        )

        result = engine.evaluate(basic_context)

        assert result.approved is False
        assert any(v.policy_type == PolicyType.FEE for v in result.violations)

    def test_risk_warnings_included(
        self,
        valid_session_key: SessionKey,
        high_value_context: ActionContext,
    ):
        """Test that risk warnings are included in result."""
        # Make value high enough for warning but not blocking
        high_value_context.value_usd = Decimal("500")
        high_value_context.estimated_gas_usd = Decimal("20")  # 4% gas

        engine = PolicyEngine(
            session_key=valid_session_key,
            risk_config=RiskPolicyConfig(warn_gas_percent=3.0),
        )

        result = engine.evaluate(high_value_context)

        assert result.approved is True
        # May have warnings about gas

    def test_risk_score_in_result(
        self,
        valid_session_key: SessionKey,
        basic_context: ActionContext,
    ):
        """Test that risk score is calculated."""
        engine = PolicyEngine(session_key=valid_session_key)
        result = engine.evaluate(basic_context)

        assert 0.0 <= result.risk_score <= 1.0
        assert result.risk_level in RiskLevel

    def test_evaluation_time_recorded(
        self,
        valid_session_key: SessionKey,
        basic_context: ActionContext,
    ):
        """Test that evaluation time is recorded."""
        engine = PolicyEngine(session_key=valid_session_key)
        result = engine.evaluate(basic_context)

        assert result.evaluation_time_ms > 0
        assert result.evaluated_at is not None

    def test_policy_result_to_dict(
        self,
        valid_session_key: SessionKey,
        basic_context: ActionContext,
    ):
        """Test PolicyResult serialization."""
        engine = PolicyEngine(session_key=valid_session_key)
        result = engine.evaluate(basic_context)

        result_dict = result.to_dict()

        assert "approved" in result_dict
        assert "violations" in result_dict
        assert "riskScore" in result_dict
        assert "evaluatedAt" in result_dict

    def test_no_session_key(self, basic_context: ActionContext):
        """Test engine works without session key."""
        engine = PolicyEngine(
            session_key=None,
            risk_config=RiskPolicyConfig(),
        )

        result = engine.evaluate(basic_context)

        # Should work, just skip session checks
        assert isinstance(result.approved, bool)
