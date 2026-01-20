from app.services.address import (
    normalize_chain,
    is_supported_chain,
    is_valid_address_for_chain,
)


def test_normalize_chain_defaults_to_ethereum():
    assert normalize_chain(None) == "ethereum"
    assert normalize_chain(" Ethereum ") == "ethereum"


def test_normalize_chain_aliases():
    assert normalize_chain("eth") == "ethereum"
    assert normalize_chain("SOL") == "solana"


def test_supported_chain_flags():
    assert is_supported_chain("ethereum") is True
    assert is_supported_chain("solana") is True
    assert is_supported_chain("base") is True
    assert is_supported_chain("polygon") is True
    assert is_supported_chain("avalanche") is False


def test_address_validation_evm():
    address = "0x1234567890abcdef1234567890ABCDEF12345678"
    assert is_valid_address_for_chain(address, "ethereum") is True
    assert is_valid_address_for_chain(address[:-1], "ethereum") is False


def test_address_validation_solana():
    solana_address = "So11111111111111111111111111111111111111112"
    assert is_valid_address_for_chain(solana_address, "solana") is True
    assert is_valid_address_for_chain("O0lNotBase58", "solana") is False


def test_address_validation_sui():
    sui_address = "0x2ccd4a37d0ac0ed8fb45a9ec7fa5cd6d10bd7d06bc6e2e31aa6a2d19f7aa0e4f"
    assert is_valid_address_for_chain(sui_address, "sui") is True
    assert is_valid_address_for_chain("0x123", "sui") is False
