"""Parse natural-language messages into structured trade intents."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_DOWN, DivisionByZero
from typing import Any, Dict, Optional, Tuple

from ..core.bridge.chain_registry import get_registry_sync, get_chain_registry, ChainId
from ..core.swap.constants import (
    GLOBAL_TOKEN_ALIASES,
    SWAP_FOLLOWUP_KEYWORDS,
    SWAP_KEYWORDS,
)


def is_swap_query(message: str) -> bool:
    return any(keyword in message for keyword in SWAP_KEYWORDS)


def is_swap_followup(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    if any(keyword in message for keyword in SWAP_FOLLOWUP_KEYWORDS):
        return True
    if stripped in {'yes', 'y', 'ok', 'okay', 'please', 'sure'}:
        return True
    return False


def normalize_text(message: str) -> str:
    normalized = message.lower()
    replacements = {
        '->': ' to ',
        '➡️': ' to ',
        ' into ': ' to ',
        ' in to ': ' to ',
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized


def detect_chain(message: str) -> Optional[ChainId]:
    """Detect chain from message using dynamic registry."""
    registry = get_registry_sync()
    return registry.detect_chain_in_text(message)


def detect_cross_chain(message: str) -> Tuple[Optional[ChainId], Optional[ChainId]]:
    """Detect origin and destination chains for cross-chain swaps.

    Parses patterns like:
    - "from ink to mainnet"
    - "on ink to ethereum"
    - "on ink chain to mainnet"
    - "USDC.e from ink to USDC on mainnet"

    Returns (origin_chain_id, destination_chain_id). Either may be None.
    """
    registry = get_registry_sync()

    origin_chain = registry.detect_chain_with_preposition(message, ["from"])
    if origin_chain is None:
        on_match = re.search(r'\bon\s+(\w+)(?:\s+chain)?\s+to\b', message, re.IGNORECASE)
        if on_match:
            origin_chain = registry.get_chain_id(on_match.group(1))

    destination_chain = registry.detect_chain_with_preposition(message, ["to"])
    if destination_chain is None:
        on_match = re.search(r'\bto\s+\w+\s+on\s+(\w+)', message, re.IGNORECASE)
        if on_match:
            destination_chain = registry.get_chain_id(on_match.group(1))

    if destination_chain is None and origin_chain is not None:
        trailing_chain = re.search(r'\b(?:on|to)\s+(\w+)\s*$', message, re.IGNORECASE)
        if trailing_chain:
            potential_chain = registry.get_chain_id(trailing_chain.group(1))
            if potential_chain is not None and potential_chain != origin_chain:
                destination_chain = potential_chain

    return (origin_chain, destination_chain)


def chain_from_default(chain_name: Optional[str]) -> Optional[ChainId]:
    """Resolve default chain name to chain ID using registry."""
    if not chain_name:
        return None
    registry = get_registry_sync()
    return registry.get_chain_id(chain_name)


def chain_name(chain_id: ChainId) -> str:
    """Get chain name from registry."""
    registry = get_registry_sync()
    return registry.get_chain_name(chain_id)


def parse_swap_request(
    message: str,
    extra_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Decimal], Optional[str], Optional[str], Optional[str]]:
    """Parse a swap request message into (amount, token_in, token_out, amount_currency).

    amount_currency is one of 'USD', 'PERCENT', 'TOKEN', or None.
    """
    amount: Optional[Decimal] = None
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    amount_currency: Optional[str] = None

    qualifier = r'(?:about|around|roughly|approximately)\s+'
    usd_prefix_pattern = re.search(
        rf'(?:swap|trade|convert|exchange)\s+(?:{qualifier})?\$(?P<usd>\d+(?:\.\d+)?)',
        message,
    )
    if usd_prefix_pattern:
        amount = _to_decimal(usd_prefix_pattern.group('usd'))
        amount_currency = 'USD'
    else:
        usd_suffix_pattern = re.search(
            rf'(?:swap|trade|convert|exchange)\s+(?:{qualifier})?(?P<usd>\d+(?:\.\d+)?)\s*(usd|dollars?)\b',
            message,
        )
        if usd_suffix_pattern:
            amount = _to_decimal(usd_suffix_pattern.group('usd'))
            amount_currency = 'USD'

    if amount is None:
        percent_pattern = re.search(
            r'(?:swap|trade|convert|exchange)[^\d%]*?(?P<percent>\d+(?:\.\d+)?)\s*(?:percent|pct|%)',
            message,
        )
        if percent_pattern:
            percent_value = _to_decimal(percent_pattern.group('percent'))
            if percent_value is not None:
                try:
                    amount = (percent_value / Decimal('100')).quantize(Decimal('1.000000000000000000'), rounding=ROUND_DOWN)
                    amount_currency = 'PERCENT'
                except (InvalidOperation, DivisionByZero):
                    amount = None
                    amount_currency = None

    if amount is None:
        amount_match = re.search(r'(?:swap|trade|convert|exchange)\s+(?P<amount>\d+(?:\.\d+)?)', message)
        if amount_match:
            amount = _to_decimal(amount_match.group('amount'))
            if amount is not None:
                amount_currency = 'TOKEN'

    alias_map = dict(GLOBAL_TOKEN_ALIASES)
    if extra_aliases:
        alias_map.update({k.lower(): v for k, v in extra_aliases.items()})

    pattern = re.search(
        r'(?:swap|trade|convert|exchange)\s+(?:\$?\d+(?:\.\d+)?\s*(?:usd|dollars?)?\s*)?(?:of\s+)?(?P<from>[a-zA-Z0-9$]{2,15})\s+to\s+(?P<to>[a-zA-Z0-9$]{2,15})',
        message,
    )
    if pattern:
        raw_in = pattern.group('from').lstrip('$')
        raw_out = pattern.group('to').lstrip('$')
        token_in = alias_map.get(raw_in.lower()) or raw_in.upper()
        token_out = alias_map.get(raw_out.lower()) or raw_out.upper()
    else:
        from typing import List
        tokens_ordered: List[str] = []
        for candidate in re.findall(r'[a-zA-Z0-9$]{2,20}', message):
            cleaned = candidate.lstrip('$').lower()
            if cleaned in alias_map:
                tokens_ordered.append(alias_map[cleaned])
        if len(tokens_ordered) >= 2:
            token_in, token_out = tokens_ordered[0], tokens_ordered[1]

    if amount is None:
        amount_match_generic = re.search(r'\b(\d+(?:\.\d+)?)\b', message)
        if amount_match_generic:
            amount = _to_decimal(amount_match_generic.group(1))
            if amount is not None and amount_currency is None:
                amount_currency = 'TOKEN'

    return amount, token_in, token_out, amount_currency


def _to_decimal(raw: Any) -> Optional[Decimal]:
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError):
        return None
