"""
News Data Sources

Fetchers for different news sources:
- RSS feeds (CoinDesk, CoinTelegraph, TheBlock, etc.)
- CoinGecko status updates
- DefiLlama protocol updates
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import httpx

from .models import NewsItem, NewsSource, NewsSourceType

logger = logging.getLogger(__name__)


class NewsSourceFetcher(ABC):
    """Base class for news source fetchers."""

    @abstractmethod
    async def fetch(self, source: NewsSource) -> List[NewsItem]:
        """Fetch news items from the source."""
        pass

    @staticmethod
    def generate_source_id(url: str) -> str:
        """Generate a unique source ID from a URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16]

    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", " ", text)
        # Remove extra whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean


class RSSSource(NewsSourceFetcher):
    """
    Fetcher for RSS feed news sources.

    Supports standard RSS 2.0 and Atom feeds.
    """

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "User-Agent": "Sherpa News Bot/1.0",
                    "Accept": "application/rss+xml, application/xml, text/xml",
                },
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, source: NewsSource) -> List[NewsItem]:
        """Fetch news items from an RSS feed."""
        if source.type != NewsSourceType.RSS:
            raise ValueError(f"Expected RSS source, got {source.type}")

        client = await self._get_client()
        items: List[NewsItem] = []

        try:
            response = await client.get(source.url)
            response.raise_for_status()

            # Parse XML
            root = ElementTree.fromstring(response.text)

            # Try RSS 2.0 format first
            channel = root.find("channel")
            if channel is not None:
                items = self._parse_rss(channel, source)
            else:
                # Try Atom format
                items = self._parse_atom(root, source)

            logger.info(f"Fetched {len(items)} items from {source.name}")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {source.name}: {e}")
        except ElementTree.ParseError as e:
            logger.error(f"XML parse error for {source.name}: {e}")
        except Exception as e:
            logger.error(f"Error fetching {source.name}: {e}")

        return items

    def _parse_rss(self, channel: ElementTree.Element, source: NewsSource) -> List[NewsItem]:
        """Parse RSS 2.0 feed."""
        items: List[NewsItem] = []

        for item in channel.findall("item"):
            try:
                title = self._get_text(item, "title")
                link = self._get_text(item, "link")

                if not title or not link:
                    continue

                # Parse publication date
                pub_date = self._get_text(item, "pubDate")
                published_at = self._parse_date(pub_date) if pub_date else datetime.now(timezone.utc)

                # Get description/content
                description = self._get_text(item, "description")
                content = self._get_text(item, "content:encoded") or description

                # Get image if available
                image_url = None
                enclosure = item.find("enclosure")
                if enclosure is not None and enclosure.get("type", "").startswith("image"):
                    image_url = enclosure.get("url")

                # Also check media:content
                if not image_url:
                    media = item.find("{http://search.yahoo.com/mrss/}content")
                    if media is not None:
                        image_url = media.get("url")

                news_item = NewsItem(
                    source_id=self.generate_source_id(link),
                    source=f"rss:{source.name}",
                    title=self.clean_html(title),
                    url=link,
                    published_at=published_at,
                    summary=self.clean_html(description)[:500] if description else None,
                    raw_content=self.clean_html(content)[:10000] if content else None,
                    image_url=image_url,
                )
                items.append(news_item)

            except Exception as e:
                logger.warning(f"Error parsing RSS item: {e}")
                continue

        return items

    def _parse_atom(self, root: ElementTree.Element, source: NewsSource) -> List[NewsItem]:
        """Parse Atom feed."""
        items: List[NewsItem] = []
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns) or root.findall("entry"):
            try:
                title = self._get_text(entry, "atom:title", ns) or self._get_text(entry, "title")

                # Get link (prefer alternate)
                link = None
                for link_elem in entry.findall("atom:link", ns) or entry.findall("link"):
                    rel = link_elem.get("rel", "alternate")
                    if rel == "alternate":
                        link = link_elem.get("href")
                        break
                if not link:
                    link_elem = entry.find("atom:link", ns) or entry.find("link")
                    if link_elem is not None:
                        link = link_elem.get("href")

                if not title or not link:
                    continue

                # Parse publication date
                published = (
                    self._get_text(entry, "atom:published", ns)
                    or self._get_text(entry, "published")
                    or self._get_text(entry, "atom:updated", ns)
                    or self._get_text(entry, "updated")
                )
                published_at = self._parse_date(published) if published else datetime.now(timezone.utc)

                # Get summary/content
                summary = self._get_text(entry, "atom:summary", ns) or self._get_text(entry, "summary")
                content = self._get_text(entry, "atom:content", ns) or self._get_text(entry, "content")

                news_item = NewsItem(
                    source_id=self.generate_source_id(link),
                    source=f"rss:{source.name}",
                    title=self.clean_html(title),
                    url=link,
                    published_at=published_at,
                    summary=self.clean_html(summary)[:500] if summary else None,
                    raw_content=self.clean_html(content or summary)[:10000] if (content or summary) else None,
                )
                items.append(news_item)

            except Exception as e:
                logger.warning(f"Error parsing Atom entry: {e}")
                continue

        return items

    def _get_text(
        self,
        elem: ElementTree.Element,
        tag: str,
        namespaces: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Get text content of a child element."""
        child = elem.find(tag, namespaces) if namespaces else elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats."""
        # Common date formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822
            "%a, %d %b %Y %H:%M:%S %Z",  # With timezone name
            "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
            "%Y-%m-%dT%H:%M:%SZ",        # ISO 8601 UTC
            "%Y-%m-%d %H:%M:%S",         # Simple format
        ]

        # Clean up timezone abbreviations
        date_str = date_str.replace("GMT", "+0000").replace("UTC", "+0000")

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        # Fallback to now
        logger.warning(f"Could not parse date: {date_str}")
        return datetime.now(timezone.utc)


class CoinGeckoNewsSource(NewsSourceFetcher):
    """
    Fetcher for CoinGecko status updates and news.

    Uses the free CoinGecko API (no key required, rate limited).
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, source: NewsSource) -> List[NewsItem]:
        """Fetch trending coins and significant movers from CoinGecko."""
        client = await self._get_client()
        items: List[NewsItem] = []

        # Fetch trending coins
        try:
            url = f"{self.BASE_URL}/search/trending"
            response = await client.get(url)

            if response.status_code == 429:
                logger.warning("CoinGecko rate limit hit")
                return items

            response.raise_for_status()
            data = response.json()

            published_at = datetime.now(timezone.utc)

            for i, coin_data in enumerate(data.get("coins", [])[:7]):
                try:
                    coin = coin_data.get("item", {})
                    coin_id = coin.get("id", "")
                    name = coin.get("name", "Unknown")
                    symbol = coin.get("symbol", "").upper()
                    rank = coin.get("market_cap_rank")
                    price_data = coin.get("data", {})
                    price_change = price_data.get("price_change_percentage_24h", {}).get("usd", 0)

                    if not coin_id:
                        continue

                    # Build news item about trending coin
                    if price_change and abs(price_change) > 5:
                        direction = "up" if price_change > 0 else "down"
                        title = f"{name} ({symbol}) trending on CoinGecko, {direction} {abs(price_change):.1f}% in 24h"
                    else:
                        title = f"{name} ({symbol}) is trending on CoinGecko"

                    rank_info = f"Ranked #{rank} by market cap. " if rank else ""
                    summary = f"{rank_info}{name} is currently one of the top trending cryptocurrencies on CoinGecko."

                    coin_url = f"https://www.coingecko.com/en/coins/{coin_id}"
                    image_url = coin.get("small") or coin.get("thumb")

                    # Use date + rank for unique ID (trending resets daily)
                    source_id = f"cg-trending-{published_at.strftime('%Y%m%d')}-{i}"

                    news_item = NewsItem(
                        source_id=source_id,
                        source="coingecko:trending",
                        title=title,
                        url=coin_url,
                        published_at=published_at,
                        summary=summary,
                        raw_content=summary,
                        image_url=image_url,
                    )
                    items.append(news_item)

                except Exception as e:
                    logger.warning(f"Error parsing CoinGecko trending coin: {e}")
                    continue

            logger.info(f"Fetched {len(items)} trending items from CoinGecko")

        except httpx.HTTPStatusError as e:
            logger.error(f"CoinGecko HTTP error: {e}")
        except Exception as e:
            logger.error(f"CoinGecko error: {e}")

        return items

    def _parse_date(self, date_str: str) -> datetime:
        """Parse CoinGecko date format."""
        try:
            # CoinGecko uses ISO 8601
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt
        except Exception:
            return datetime.now(timezone.utc)


class DefiLlamaNewsSource(NewsSourceFetcher):
    """
    Fetcher for DefiLlama protocol news and updates.

    Monitors protocol TVL changes, new protocols, and significant events.
    """

    BASE_URL = "https://api.llama.fi"

    def __init__(self, timeout: float = 15.0):
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._last_tvl_snapshot: Dict[str, float] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={"Accept": "application/json"},
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, source: NewsSource) -> List[NewsItem]:
        """
        Fetch protocol updates from DefiLlama.

        Generates news items from:
        - New protocols added
        - Significant TVL changes (>20% in 24h)
        - Protocol hacks (from hacks endpoint)
        """
        client = await self._get_client()
        items: List[NewsItem] = []

        # Fetch hacks (security incidents)
        try:
            hacks = await self._fetch_hacks(client)
            items.extend(hacks)
        except Exception as e:
            logger.error(f"Error fetching DefiLlama hacks: {e}")

        # Fetch significant TVL changes
        try:
            tvl_updates = await self._fetch_tvl_changes(client)
            items.extend(tvl_updates)
        except Exception as e:
            logger.error(f"Error fetching DefiLlama TVL changes: {e}")

        logger.info(f"Fetched {len(items)} items from DefiLlama")
        return items

    async def _fetch_hacks(self, client: httpx.AsyncClient) -> List[NewsItem]:
        """Fetch recent hacks/exploits."""
        items: List[NewsItem] = []

        try:
            response = await client.get(f"{self.BASE_URL}/hacks")
            if response.status_code != 200:
                return items

            data = response.json()

            # Get hacks from last 7 days
            cutoff = datetime.now(timezone.utc).timestamp() - (7 * 24 * 60 * 60)

            for hack in data:
                try:
                    date_ts = hack.get("date")
                    if not date_ts or date_ts < cutoff:
                        continue

                    name = hack.get("name", "Unknown Protocol")
                    amount = hack.get("amount", 0)
                    technique = hack.get("technique", "Unknown")
                    chain = hack.get("chain", "")

                    # Format amount
                    amount_str = f"${amount:,.0f}" if amount else "Unknown amount"

                    title = f"{name} Hack: {amount_str} lost via {technique}"
                    description = f"Security incident at {name}"
                    if chain:
                        description += f" on {chain}"
                    description += f". Approximately {amount_str} was stolen using {technique}."

                    link = hack.get("link") or f"https://defillama.com/hacks"
                    published_at = datetime.fromtimestamp(date_ts, tz=timezone.utc)

                    source_id = f"dl-hack-{self.generate_source_id(f'{name}-{date_ts}')}"

                    news_item = NewsItem(
                        source_id=source_id,
                        source="defillama:hacks",
                        title=title,
                        url=link,
                        published_at=published_at,
                        summary=description,
                        raw_content=description,
                    )
                    items.append(news_item)

                except Exception as e:
                    logger.warning(f"Error parsing DefiLlama hack: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching DefiLlama hacks: {e}")

        return items

    async def _fetch_tvl_changes(self, client: httpx.AsyncClient) -> List[NewsItem]:
        """Fetch protocols with significant TVL changes."""
        items: List[NewsItem] = []

        try:
            response = await client.get(f"{self.BASE_URL}/protocols")
            if response.status_code != 200:
                return items

            data = response.json()

            for protocol in data:
                try:
                    name = protocol.get("name", "")
                    symbol = protocol.get("symbol", "").upper()
                    tvl = protocol.get("tvl", 0)
                    change_24h = protocol.get("change_1d")

                    if not name or not tvl or change_24h is None:
                        continue

                    # Only report significant changes (>20%)
                    if abs(change_24h) < 20:
                        continue

                    # Skip very small protocols
                    if tvl < 1_000_000:
                        continue

                    direction = "increased" if change_24h > 0 else "decreased"
                    tvl_str = f"${tvl / 1_000_000:.1f}M" if tvl < 1_000_000_000 else f"${tvl / 1_000_000_000:.2f}B"

                    title = f"{name} TVL {direction} {abs(change_24h):.1f}% to {tvl_str}"
                    description = f"{name}"
                    if symbol:
                        description += f" ({symbol})"
                    description += f" has seen its TVL {direction} by {abs(change_24h):.1f}% "
                    description += f"in the last 24 hours, now at {tvl_str}."

                    slug = protocol.get("slug", name.lower().replace(" ", "-"))
                    link = f"https://defillama.com/protocol/{slug}"

                    # Use current time as we don't have exact change time
                    published_at = datetime.now(timezone.utc)

                    source_id = f"dl-tvl-{self.generate_source_id(f'{slug}-{published_at.date()}')}"

                    news_item = NewsItem(
                        source_id=source_id,
                        source="defillama:tvl",
                        title=title,
                        url=link,
                        published_at=published_at,
                        summary=description,
                        raw_content=description,
                    )
                    items.append(news_item)

                except Exception as e:
                    logger.warning(f"Error parsing DefiLlama TVL change: {e}")
                    continue

            # Limit to top 20 most significant changes
            items.sort(key=lambda x: abs(float(x.summary.split("%")[0].split()[-1])), reverse=True)
            items = items[:20]

        except Exception as e:
            logger.error(f"Error fetching DefiLlama TVL changes: {e}")

        return items


async def fetch_all_sources(
    sources: List[NewsSource],
    rss_fetcher: Optional[RSSSource] = None,
    coingecko_fetcher: Optional[CoinGeckoNewsSource] = None,
    defillama_fetcher: Optional[DefiLlamaNewsSource] = None,
) -> List[NewsItem]:
    """
    Fetch news from all configured sources concurrently.

    Args:
        sources: List of news sources to fetch from
        rss_fetcher: Optional RSS fetcher instance
        coingecko_fetcher: Optional CoinGecko fetcher instance
        defillama_fetcher: Optional DefiLlama fetcher instance

    Returns:
        Combined list of news items from all sources
    """
    # Create fetchers if not provided
    rss = rss_fetcher or RSSSource()
    cg = coingecko_fetcher or CoinGeckoNewsSource()
    dl = defillama_fetcher or DefiLlamaNewsSource()

    tasks = []
    enabled_sources = [s for s in sources if s.enabled]

    for source in enabled_sources:
        if source.type == NewsSourceType.RSS:
            tasks.append(rss.fetch(source))
        elif source.name == "coingecko":
            tasks.append(cg.fetch(source))
        elif source.name == "defillama":
            tasks.append(dl.fetch(source))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_items: List[NewsItem] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Source fetch error: {result}")
        elif isinstance(result, list):
            all_items.extend(result)

    # Close fetchers if we created them
    if rss_fetcher is None:
        await rss.close()
    if coingecko_fetcher is None:
        await cg.close()
    if defillama_fetcher is None:
        await dl.close()

    return all_items
