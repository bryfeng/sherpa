from typing import Dict, Any, Tuple


# Simple, offline price table for MVP quotes. In production, replace with
# aggregator APIs (0x, 1inch) or on-chain routing and real token metadata.
PRICE_TABLE_USD: Dict[str, float] = {
    "ETH": 2500.0,
    "WETH": 2500.0,
    "USDC": 1.0,
    "USDT": 1.0,
    "WBTC": 65000.0,
}


def _normalize_symbol(sym: str) -> str:
    return (sym or "").upper().strip()


def _get_price(sym: str) -> Tuple[float, bool]:
    s = _normalize_symbol(sym)
    if s in PRICE_TABLE_USD:
        return PRICE_TABLE_USD[s], True
    # Unknown token → fall back to $1 to avoid crashes; mark as estimated
    return 1.0, False


async def quote_swap_simple(
    token_in: str,
    token_out: str,
    amount_in: float,
    slippage_bps: int = 50,
) -> Dict[str, Any]:
    """Return a simple swap quote using a static price table and
    conservative fee/slippage assumptions. This is an MVP placeholder
    to unblock frontend UX while aggregator integration is pending.

    Returns a payload shaped for UI consumption.
    """

    sym_in = _normalize_symbol(token_in)
    sym_out = _normalize_symbol(token_out)

    price_in, known_in = _get_price(sym_in)
    price_out, known_out = _get_price(sym_out)

    # Constant product pool approximation with 0.3% LP fee and slippage reserve
    fee_rate = 0.003  # 30 bps typical for Uniswap V2/V3 pools
    fee_amount = max(0.0, amount_in * fee_rate)
    effective_in = max(0.0, amount_in - fee_amount)

    # Convert input value in USD, then to output units
    value_usd = effective_in * price_in
    # Reserve for slippage: slippage_bps (e.g., 50 = 0.5%)
    slip = max(0, slippage_bps) / 10_000.0
    value_usd_after_slip = value_usd * (1.0 - slip)

    # Avoid div by zero; use $1 fallback already applied in _get_price
    amount_out = value_usd_after_slip / price_out if price_out > 0 else 0.0

    warnings = []
    if not known_in or not known_out:
        warnings.append(
            "Using placeholder prices for unknown tokens; results are approximate."
        )

    return {
        "success": True,
        "from": sym_in,
        "to": sym_out,
        "amount_in": amount_in,
        "price_in_usd": price_in,
        "price_out_usd": price_out,
        "amount_out_est": amount_out,
        "fee_est": fee_amount,
        "slippage_bps": slippage_bps,
        "route": {
            "kind": "stub",
            "note": "Static price table with 0.3% fee and slippage reserve",
        },
        "sources": [
            {"name": "placeholder", "url": "https://0x.org/ or https://1inch.io/ (to integrate)"}
        ],
        "warnings": warnings,
    }

