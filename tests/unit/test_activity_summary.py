from datetime import datetime, timedelta, timezone

from app.services.activity_summary import compute_totals, detect_notable_events


def _event(days_offset: int, usd: float, *, direction: str = 'outflow', counterparty: str = '0xabc', protocol: str = 'dex') -> dict:
    ts = (datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=days_offset)).isoformat()
    return {
        'tx_hash': f'0x{days_offset:02x}',
        'timestamp': ts,
        'direction': direction,
        'symbol': 'ETH',
        'native_amount': usd / 2000,
        'usd_value': usd,
        'counterparty': counterparty,
        'protocol': protocol,
        'fee_native': 0,
        'chain': 'ethereum',
    }


def test_detects_large_outflows():
    events = [_event(idx, usd) for idx, usd in enumerate([100, 120, 5_000, 6_500, 7_000])]
    totals = compute_totals(events)
    flags = detect_notable_events(events, totals)
    assert any(flag['type'] == 'large_outflow' for flag in flags)


def test_detects_dormant_period():
    events = [_event(0, 200, direction='inflow'), _event(5, 300), _event(50, 150)]
    totals = compute_totals(events)
    flags = detect_notable_events(events, totals)
    dormant = next(flag for flag in flags if flag['type'] == 'dormant_period')
    assert 'Dormant' in dormant['summary']


def test_detects_counterparty_concentration():
    events = [
        _event(0, 300, counterparty='0xaaa'),
        _event(1, 320, counterparty='0xaaa'),
        _event(2, 50, counterparty='0xbbbb'),
        _event(3, 30, counterparty='0xcccc'),
    ]
    totals = compute_totals(events)
    flags = detect_notable_events(events, totals)
    assert any(flag['type'] == 'concentrated_outflow' for flag in flags)
