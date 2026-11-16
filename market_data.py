import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import feedparser
import yfinance as yf

logger = logging.getLogger("oanda_agent.market_data")

# Cross-asset drivers that move FX majors
MACRO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "VIX": "^VIX",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "SPX": "^GSPC",
}

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (oanda-ai-trading-agent)"}
CACHE_DIR = Path(__file__).parent / ".cache" / "market_data"

FAILURE_BACKOFF_SECONDS = 60
RATE_LIMIT_BACKOFF_SECONDS = 15 * 60


class RateLimited(Exception):
    """Raised by a loader when the source answers HTTP 429."""

    def __init__(self, source: str, retry_after: Optional[str] = None):
        self.source = source
        self.backoff = int(retry_after) if retry_after and retry_after.isdigit() else RATE_LIMIT_BACKOFF_SECONDS
        super().__init__(f"{source} rate-limited us (HTTP 429); backing off {self.backoff // 60} min")


def _raise_for_status(resp: requests.Response, source: str):
    if resp.status_code == 429:
        raise RateLimited(source, resp.headers.get("Retry-After"))
    resp.raise_for_status()


class MarketDataProvider:
    """Free market context: yfinance news & cross-asset prices, Forex Factory economic calendar."""

    def __init__(self, news_ttl: int = 300, macro_ttl: int = 300, calendar_ttl: int = 3600):
        self.news_ttl = news_ttl
        self.macro_ttl = macro_ttl
        self.calendar_ttl = calendar_ttl  # Forex Factory rate-limits aggressive polling
        self._cache: Dict[str, tuple] = {}

    @staticmethod
    def _cache_file(key: str) -> Path:
        return CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def _store(self, key: str, stamp: float, value, persist: bool):
        self._cache[key] = (stamp, value)
        if persist:  # Survives process restarts (e.g. uvicorn --reload) so we don't re-hit rate-limited sources
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self._cache_file(key).write_text(json.dumps({"stamp": stamp, "value": value}), encoding="utf-8")
            except OSError as e:
                logger.warning(f"Market data cache write warning ({key}): {e}")

    def _cached(self, key: str, ttl: int, loader, persist: bool = False):
        hit = self._cache.get(key)
        if hit is None and persist:
            try:
                saved = json.loads(self._cache_file(key).read_text(encoding="utf-8"))
                hit = self._cache[key] = (saved["stamp"], saved["value"])
            except (OSError, ValueError, KeyError):
                pass
        if hit and time.time() - hit[0] < ttl:
            return hit[1]

        backoff = FAILURE_BACKOFF_SECONDS
        try:
            value = loader()
        except RateLimited as e:
            logger.warning(str(e))
            value, backoff = None, e.backoff
        if value:
            self._store(key, time.time(), value, persist)
            return value
        # Failed refresh: serve stale data if any, and wait `backoff` seconds before retrying the source
        stale = hit[1] if hit else (value if value is not None else [])
        self._store(key, time.time() - ttl + backoff, stale, persist)
        return stale

    @staticmethod
    def yf_symbol(instrument: str) -> str:
        """EUR_USD -> EURUSD=X"""
        return instrument.replace("_", "") + "=X"

    def get_pair_news(self, instrument: str, max_items: int = 8) -> List[Dict[str, str]]:
        """Yahoo Finance headlines for the currency pair via yfinance."""

        def load():
            items = []
            try:
                for raw in yf.Ticker(self.yf_symbol(instrument)).news[:max_items]:
                    content = raw.get("content", raw)
                    provider = content.get("provider") or {}
                    items.append({
                        "title": content.get("title", ""),
                        "source": f"Yahoo/{provider.get('displayName', 'Finance')}",
                        "summary": (content.get("summary") or "")[:240],
                        "published": content.get("pubDate", ""),
                    })
            except Exception as e:
                logger.warning(f"yfinance news warning ({instrument}): {e}")
            return items

        return self._cached(f"news:{instrument}", self.news_ttl, load)

    def get_rss_entries(self, source: str, url: str) -> List[Dict[str, str]]:
        """Title/summary entries of an RSS feed, shared across pairs to avoid feed rate limits."""

        def load():
            try:
                resp = requests.get(url, headers=HTTP_HEADERS, timeout=8)
                _raise_for_status(resp, source)
                return [
                    {"title": e.get("title", ""), "summary": e.get("summary", "")}
                    for e in feedparser.parse(resp.content).entries[:30]
                ]
            except RateLimited:
                raise
            except Exception as e:
                logger.warning(f"RSS feed warning ({source}): {e}")
                return []

        return self._cached(f"rss:{source}", self.news_ttl, load, persist=True)

    def get_macro_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Last close and daily % change for key cross-asset drivers."""

        def load():
            snapshot = {}
            try:
                data = yf.download(
                    list(MACRO_TICKERS.values()), period="5d", interval="1d",
                    progress=False, auto_adjust=False, threads=True,
                )
                closes = data["Close"]
                for name, symbol in MACRO_TICKERS.items():
                    series = closes[symbol].dropna() if symbol in closes else None
                    if series is None or len(series) < 2:
                        continue
                    last, prev = float(series.iloc[-1]), float(series.iloc[-2])
                    snapshot[name] = {
                        "last": round(last, 3),
                        "change_pct": round((last - prev) / prev * 100, 2) if prev else 0.0,
                    }
            except Exception as e:
                logger.warning(f"yfinance macro snapshot warning: {e}")
            return snapshot

        return self._cached("macro", self.macro_ttl, load)

    def get_economic_calendar(self) -> List[Dict[str, Any]]:
        """This week's Forex Factory calendar events with UTC datetimes."""

        def load():
            # Raw JSON rows are cached (and persisted to disk); datetimes are parsed on read
            try:
                resp = requests.get(FF_CALENDAR_URL, headers=HTTP_HEADERS, timeout=10)
                _raise_for_status(resp, "Forex Factory calendar")
                return resp.json()
            except RateLimited:
                raise
            except Exception as e:
                logger.warning(f"Forex Factory calendar warning: {e}")
                return []

        events = []
        for e in self._cached("calendar", self.calendar_ttl, load, persist=True):
            try:
                when = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            events.append({
                "title": e.get("title", ""),
                "currency": e.get("country", ""),
                "impact": e.get("impact", ""),
                "forecast": e.get("forecast", ""),
                "previous": e.get("previous", ""),
                "time_utc": when,
            })
        return events

    def get_upcoming_events(
        self, instrument: str, hours_ahead: int = 24, min_impact: tuple = ("High", "Medium")
    ) -> List[Dict[str, Any]]:
        """Calendar events affecting either leg of the pair in the next `hours_ahead` hours."""
        currencies = set(instrument.split("_"))
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(hours=hours_ahead)
        return sorted(
            (
                e for e in self.get_economic_calendar()
                if e["currency"] in currencies
                and e["impact"] in min_impact
                and now - timedelta(minutes=15) <= e["time_utc"] <= horizon
            ),
            key=lambda e: e["time_utc"],
        )

    def imminent_high_impact_events(self, instrument: str, window_minutes: int = 30) -> List[Dict[str, Any]]:
        """High-impact events for the pair released in the last 15 minutes or due within `window_minutes`."""
        now = datetime.now(timezone.utc)
        return [
            e for e in self.get_upcoming_events(instrument, hours_ahead=1, min_impact=("High",))
            if e["time_utc"] <= now + timedelta(minutes=window_minutes)
        ]
