"""
Token Classifier

Classifies tokens into taxonomy based on multiple signals:
- Symbol patterns (WETH, LP tokens)
- Known token lists
- CoinGecko categories
- DefiLlama protocol data
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .models import (
    TokenTaxonomy,
    TokenSector,
    TokenSubsector,
    MarketCapTier,
    RelatedToken,
)


# Known stablecoins by symbol
STABLECOIN_SYMBOLS = {
    "USDC", "USDT", "DAI", "FRAX", "LUSD", "BUSD", "TUSD", "USDP",
    "GUSD", "USDD", "MIM", "UST", "FDUSD", "PYUSD", "CRVUSD",
    "GHO", "SUSD", "RAI", "ALUSD", "DOLA", "MAI", "CUSD",
}

# Known wrapped token patterns
WRAPPED_PREFIXES = {"W", "WE", "ST", "CB", "R"}  # WETH, WBTC, stETH, cbETH, rETH

# Known L1 tokens
L1_TOKENS = {
    "ETH": ("ethereum", TokenSubsector.L1),
    "BTC": ("bitcoin", TokenSubsector.L1),
    "SOL": ("solana", TokenSubsector.L1),
    "AVAX": ("avalanche", TokenSubsector.L1),
    "BNB": ("binance-smart-chain", TokenSubsector.L1),
    "MATIC": ("polygon", TokenSubsector.L1),
    "FTM": ("fantom", TokenSubsector.L1),
    "ATOM": ("cosmos", TokenSubsector.L1),
    "DOT": ("polkadot", TokenSubsector.L1),
    "ADA": ("cardano", TokenSubsector.L1),
    "NEAR": ("near", TokenSubsector.L1),
    "SUI": ("sui", TokenSubsector.L1),
    "APT": ("aptos", TokenSubsector.L1),
    "SEI": ("sei", TokenSubsector.L1),
}

# Known L2 tokens
L2_TOKENS = {
    "ARB": ("arbitrum", TokenSubsector.L2),
    "OP": ("optimism", TokenSubsector.L2),
    "MATIC": ("polygon", TokenSubsector.L2),  # Also listed as L1
    "METIS": ("metis", TokenSubsector.L2),
    "ZK": ("zksync", TokenSubsector.L2),
    "STRK": ("starknet", TokenSubsector.L2),
    "IMX": ("immutable", TokenSubsector.L2),
    "MANTA": ("manta", TokenSubsector.L2),
    "BLAST": ("blast", TokenSubsector.L2),
}

# Known DEX tokens
DEX_TOKENS = {
    "UNI": "uniswap",
    "SUSHI": "sushi",
    "CRV": "curve",
    "BAL": "balancer",
    "CAKE": "pancakeswap",
    "JOE": "traderjoe",
    "GMX": "gmx",
    "DYDX": "dydx",
    "1INCH": "1inch",
    "VELO": "velodrome",
    "AERO": "aerodrome",
    "RAY": "raydium",
    "ORCA": "orca",
    "JUP": "jupiter",
}

# Known lending tokens
LENDING_TOKENS = {
    "AAVE": "aave",
    "COMP": "compound",
    "MKR": "maker",
    "MORPHO": "morpho",
    "SPARK": "spark",
}

# Known oracle tokens
ORACLE_TOKENS = {
    "LINK": "chainlink",
    "PYTH": "pyth",
    "BAND": "band",
    "API3": "api3",
    "DIA": "dia",
    "UMA": "uma",
}

# Known meme tokens
MEME_TOKENS = {
    "DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "BRETT",
    "MEME", "WOJAK", "MONG", "LADYS", "TURBO", "MILADY",
    "GROK", "ANDY", "SNEK", "DOG",
}

# Known AI tokens
AI_TOKENS = {
    "FET": "fetch-ai",
    "AGIX": "singularitynet",
    "OCEAN": "ocean-protocol",
    "GRT": "the-graph",
    "RNDR": "render-token",
    "TAO": "bittensor",
    "AKT": "akash-network",
    "ARKM": "arkham",
    "WLD": "worldcoin",
}


class TokenClassifier:
    """
    Classifies tokens into taxonomy based on multiple signals.

    Uses a multi-stage classification approach:
    1. Symbol-based quick classification (stablecoins, wrapped, memes)
    2. Known token list matching
    3. Category analysis from external sources
    4. Fallback to Unknown
    """

    def classify(
        self,
        symbol: str,
        name: str,
        categories: Optional[List[str]] = None,
        coingecko_categories: Optional[List[str]] = None,
        defillama_category: Optional[str] = None,
        market_cap: Optional[float] = None,
    ) -> TokenTaxonomy:
        """
        Classify a token into our taxonomy.

        Args:
            symbol: Token symbol
            name: Token name
            categories: Pre-existing categories
            coingecko_categories: Categories from CoinGecko
            defillama_category: Category from DefiLlama
            market_cap: Market cap in USD

        Returns:
            TokenTaxonomy with sector, subsector, and flags
        """
        symbol_upper = symbol.upper()
        name_lower = name.lower()

        taxonomy = TokenTaxonomy(
            categories=list(categories or []),
        )

        # 1. Check if stablecoin
        if self._is_stablecoin(symbol_upper, name_lower):
            taxonomy.is_stablecoin = True
            taxonomy.sector = TokenSector.DEFI
            taxonomy.subsector = TokenSubsector.STABLECOIN
            if "stablecoin" not in taxonomy.categories:
                taxonomy.categories.append("stablecoin")

        # 2. Check if wrapped token
        if self._is_wrapped(symbol_upper, name_lower):
            taxonomy.is_wrapped = True
            if "wrapped" not in taxonomy.categories:
                taxonomy.categories.append("wrapped")

        # 3. Check if LP token
        if self._is_lp_token(symbol_upper, name_lower):
            taxonomy.is_lp_token = True
            taxonomy.sector = TokenSector.DEFI
            taxonomy.subsector = TokenSubsector.DEX
            if "lp" not in taxonomy.categories:
                taxonomy.categories.append("lp")

        # 4. Check if native token
        if self._is_native(symbol_upper):
            taxonomy.is_native = True
            if "native" not in taxonomy.categories:
                taxonomy.categories.append("native")

        # 5. Check known token lists (only if not already classified)
        if taxonomy.sector is None:
            sector, subsector, is_governance = self._classify_by_known_lists(symbol_upper)
            if sector:
                taxonomy.sector = sector
                taxonomy.subsector = subsector
                taxonomy.is_governance_token = is_governance

        # 6. Check meme tokens
        if self._is_meme(symbol_upper, name_lower):
            taxonomy.sector = TokenSector.MEME
            taxonomy.subsector = TokenSubsector.MEME_TOKEN
            if "meme" not in taxonomy.categories:
                taxonomy.categories.append("meme")

        # 7. Apply CoinGecko categories if we still don't have a classification
        if taxonomy.sector is None and coingecko_categories:
            cg_taxonomy = self._apply_coingecko_categories(coingecko_categories)
            if cg_taxonomy.sector:
                taxonomy.sector = cg_taxonomy.sector
                taxonomy.subsector = cg_taxonomy.subsector
                taxonomy.categories.extend(
                    c for c in cg_taxonomy.categories if c not in taxonomy.categories
                )

        # 8. Apply DefiLlama category
        if taxonomy.sector is None and defillama_category:
            dl_sector, dl_subsector = self._apply_defillama_category(defillama_category)
            if dl_sector:
                taxonomy.sector = dl_sector
                taxonomy.subsector = dl_subsector

        # 9. Calculate market cap tier
        if market_cap:
            taxonomy.market_cap_tier = self._calculate_market_cap_tier(market_cap)

        # 10. Default to Unknown if still unclassified
        if taxonomy.sector is None:
            taxonomy.sector = TokenSector.UNKNOWN

        return taxonomy

    def _is_stablecoin(self, symbol: str, name: str) -> bool:
        """Check if token is a stablecoin."""
        if symbol in STABLECOIN_SYMBOLS:
            return True

        # Check name patterns
        stablecoin_patterns = [
            "usd", "dollar", "stable", "peg",
        ]
        return any(p in name for p in stablecoin_patterns)

    def _is_wrapped(self, symbol: str, name: str) -> bool:
        """Check if token is a wrapped version."""
        # Check prefix
        if symbol.startswith("W") and len(symbol) > 1:
            base = symbol[1:]
            if base in L1_TOKENS:
                return True

        # Check name
        wrapped_patterns = ["wrapped", "bridged"]
        return any(p in name for p in wrapped_patterns)

    def _is_lp_token(self, symbol: str, name: str) -> bool:
        """Check if token is an LP token."""
        # Check symbol patterns first (most reliable)
        lp_symbol_patterns = ["LP", "SLP", "UNI-V2", "UNI-V3", "BPT", "G-UNI"]
        if any(p in symbol.upper() for p in lp_symbol_patterns):
            return True

        # Check name patterns (more specific to avoid false positives)
        lp_name_patterns = [
            "liquidity pool", "pool token", "lp token",
            "uni-v2", "uni-v3", "balancer pool", "curve pool",
        ]
        name_lower = name.lower()
        return any(p in name_lower for p in lp_name_patterns)

    def _is_native(self, symbol: str) -> bool:
        """Check if token is a native chain token."""
        return symbol in L1_TOKENS

    def _is_meme(self, symbol: str, name: str) -> bool:
        """Check if token is a meme token."""
        if symbol in MEME_TOKENS:
            return True

        meme_patterns = [
            "doge", "shiba", "pepe", "wojak", "chad", "frog",
            "inu", "elon", "moon", "rocket", "lambo",
        ]
        return any(p in name for p in meme_patterns)

    def _classify_by_known_lists(
        self,
        symbol: str,
    ) -> Tuple[Optional[TokenSector], Optional[TokenSubsector], bool]:
        """
        Classify by known token lists.

        Returns (sector, subsector, is_governance_token).
        """
        # L1 tokens
        if symbol in L1_TOKENS:
            return TokenSector.INFRASTRUCTURE, TokenSubsector.L1, False

        # L2 tokens
        if symbol in L2_TOKENS:
            return TokenSector.INFRASTRUCTURE, TokenSubsector.L2, False

        # DEX tokens (governance)
        if symbol in DEX_TOKENS:
            return TokenSector.DEFI, TokenSubsector.DEX, True

        # Lending tokens (governance)
        if symbol in LENDING_TOKENS:
            return TokenSector.DEFI, TokenSubsector.LENDING, True

        # Oracle tokens
        if symbol in ORACLE_TOKENS:
            return TokenSector.INFRASTRUCTURE, TokenSubsector.ORACLE, False

        # AI tokens
        if symbol in AI_TOKENS:
            return TokenSector.AI_DATA, TokenSubsector.AI, False

        return None, None, False

    def _apply_coingecko_categories(self, categories: List[str]) -> TokenTaxonomy:
        """Apply CoinGecko categories to taxonomy."""
        from .sources import COINGECKO_CATEGORY_MAP

        taxonomy = TokenTaxonomy()
        all_tags: Set[str] = set()

        for category in categories:
            if not category:
                continue

            category_key = category.lower().replace(" ", "-")
            mapping = COINGECKO_CATEGORY_MAP.get(category_key)

            if mapping:
                sector, subsector, tags = mapping
                if taxonomy.sector is None:
                    taxonomy.sector = sector
                if taxonomy.subsector is None:
                    taxonomy.subsector = subsector
                all_tags.update(tags)

        taxonomy.categories = list(all_tags)
        return taxonomy

    def _apply_defillama_category(
        self,
        category: str,
    ) -> Tuple[Optional[TokenSector], Optional[TokenSubsector]]:
        """Apply DefiLlama category to taxonomy."""
        from .sources import DefiLlamaSource

        source = DefiLlamaSource()
        return source.map_category_to_taxonomy(category)

    def _calculate_market_cap_tier(self, market_cap: float) -> MarketCapTier:
        """Calculate market cap tier."""
        if market_cap >= 10_000_000_000:
            return MarketCapTier.MEGA
        elif market_cap >= 1_000_000_000:
            return MarketCapTier.LARGE
        elif market_cap >= 100_000_000:
            return MarketCapTier.MID
        elif market_cap >= 10_000_000:
            return MarketCapTier.SMALL
        else:
            return MarketCapTier.MICRO

    def find_related_tokens(
        self,
        symbol: str,
        project_slug: Optional[str] = None,
        chain_id: int = 1,
    ) -> List[RelatedToken]:
        """
        Find related tokens for correlation analysis.

        Returns list of related tokens based on:
        - Same project (e.g., UNI on different chains)
        - Competitors (e.g., AAVE and COMP)
        - Derivatives (e.g., stETH and ETH)
        """
        related: List[RelatedToken] = []
        symbol_upper = symbol.upper()

        # Check for wrapped variants
        if symbol_upper in L1_TOKENS:
            wrapped_symbol = f"W{symbol_upper}"
            related.append(RelatedToken(
                address="",  # Would need to look up
                chain_id=chain_id,
                relationship="wrapped",
            ))

        # Check for staking derivatives
        staking_derivatives = {
            "ETH": ["STETH", "RETH", "CBETH", "WSTETH"],
            "SOL": ["MSOL", "JITOSOL", "BSOL"],
        }
        if symbol_upper in staking_derivatives:
            for derivative in staking_derivatives[symbol_upper]:
                related.append(RelatedToken(
                    address="",
                    chain_id=chain_id,
                    relationship="derivative",
                ))

        # Check for competitors in same sector
        competitors = {
            "AAVE": ["COMP", "MKR"],
            "COMP": ["AAVE", "MKR"],
            "UNI": ["SUSHI", "CRV", "BAL"],
            "SUSHI": ["UNI", "CRV"],
            "LINK": ["PYTH", "BAND"],
        }
        if symbol_upper in competitors:
            for comp in competitors[symbol_upper]:
                related.append(RelatedToken(
                    address="",
                    chain_id=chain_id,
                    relationship="competitor",
                ))

        return related
