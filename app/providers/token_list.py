"""
Token metadata fallback system using popular token lists
"""

# Common ERC-20 token metadata - well-known tokens
KNOWN_TOKENS = {
    # Major tokens
    "0xa0b86a33e6441d62e6c68b12b4f8b8ab89b1e4e4": {"symbol": "USDT", "name": "Tether USD", "decimals": 6},
    "0xa0b73e1ff0b80914ab6fe0444e65848c4c34450b": {"symbol": "CRO", "name": "Cronos", "decimals": 8},
    "0x6b175474e89094c44da98b954eedeac495271d0f": {"symbol": "DAI", "name": "Dai Stablecoin", "decimals": 18},
    "0xa0b86a33e6441d62e6c68b12b4f8b8ab89b1e4e4": {"symbol": "USDT", "name": "Tether USD", "decimals": 6},
    "0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce": {"symbol": "SHIB", "name": "Shiba Inu", "decimals": 18},
    "0x514910771af9ca656af840dff83e8264ecf986ca": {"symbol": "LINK", "name": "ChainLink Token", "decimals": 18},
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": {"symbol": "UNI", "name": "Uniswap", "decimals": 18},
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": {"symbol": "WBTC", "name": "Wrapped BTC", "decimals": 8},
    "0x7d1afa7b718fb893db30a3abc0cfc608aacfebb0": {"symbol": "MATIC", "name": "Polygon", "decimals": 18},
    "0x4fabb145d64652a948d72533023f6e7a623c7c53": {"symbol": "BUSD", "name": "Binance USD", "decimals": 18},
    
    # Additional common tokens that might appear as UNKNOWN
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": {"symbol": "WETH", "name": "Wrapped Ether", "decimals": 18},
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": {"symbol": "AAVE", "name": "Aave Token", "decimals": 18},
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": {"symbol": "MKR", "name": "Maker", "decimals": 18},
    "0x0d8775f648430679a709e98d2b0cb6250d2887ef": {"symbol": "BAT", "name": "Basic Attention Token", "decimals": 18},
    "0x1985365e9f78359a9b6ad760e32412f4a445e862": {"symbol": "REP", "name": "Reputation", "decimals": 18},
    "0x744d70fdbe2ba4cf95131626614a1763df805b9e": {"symbol": "SNT", "name": "Status Network Token", "decimals": 18},
    "0xe41d2489571d322189246dafa5ebde1f4699f498": {"symbol": "ZRX", "name": "0x Protocol Token", "decimals": 18},
    
    # DeFi tokens
    "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2": {"symbol": "SUSHI", "name": "SushiToken", "decimals": 18},
    "0xc00e94cb662c3520282e6f5717214004a7f26888": {"symbol": "COMP", "name": "Compound", "decimals": 18},
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e": {"symbol": "YFI", "name": "yearn.finance", "decimals": 18},
    "0x1494ca1f11d487c2bbe4543e90080aeba4ba3c2b": {"symbol": "DPI", "name": "DefiPulse Index", "decimals": 18},
    
    # Stablecoins
    "0xa0b86a33e6441d62e6c68b12b4f8b8ab89b1e4e4": {"symbol": "USDT", "name": "Tether USD", "decimals": 6},
    "0xdac17f958d2ee523a2206206994597c13d831ec7": {"symbol": "USDT", "name": "Tether USD", "decimals": 6},
    "0xa0b86a33e6441d62e6c68b12b4f8b8ab89b1e4e4": {"symbol": "USDC", "name": "USD Coin", "decimals": 6},
    "0x056fd409e1d7a124bd7017459dfea2f387b6d5cd": {"symbol": "GUSD", "name": "Gemini Dollar", "decimals": 2},
}

def get_fallback_token_metadata(address: str) -> dict:
    """
    Get fallback token metadata for well-known tokens
    """
    address_lower = address.lower()
    
    if address_lower in KNOWN_TOKENS:
        token_data = KNOWN_TOKENS[address_lower].copy()
        token_data["_source"] = {"name": "token_list", "url": "https://github.com/ethereum-lists/tokens"}
        return token_data
    
    # Return default unknown token metadata
    return {
        "symbol": "UNKNOWN",
        "name": "Unknown Token",
        "decimals": 18,
        "_source": {"name": "token_list", "url": "https://github.com/ethereum-lists/tokens"}
    }

def is_likely_spam_token(symbol: str, name: str, address: str) -> bool:
    """
    Detect likely spam/scam tokens based on common patterns
    """
    symbol_lower = symbol.lower()
    name_lower = name.lower()
    
    # Common spam patterns
    spam_patterns = [
        "visit", "claim", "reward", "airdrop", "free", 
        "bonus", "gift", "$", "1000", "10000", "million"
    ]
    
    for pattern in spam_patterns:
        if pattern in symbol_lower or pattern in name_lower:
            return True
    
    # Very long names are often spam
    if len(name) > 50:
        return True
        
    # Symbols with numbers are often spam
    if any(char.isdigit() for char in symbol):
        return True
    
    return False
