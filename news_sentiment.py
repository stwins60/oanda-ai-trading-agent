import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any
import requests
from pydantic import BaseModel, Field
from openai import OpenAI
from config import settings
from market_data import MarketDataProvider
from db import LLMResultCache

logger = logging.getLogger("oanda_agent.news")

# Free forex news RSS feeds (no API key required)
FREE_RSS_FEEDS = {
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "Investing.com": "https://www.investing.com/rss/news_1.rss",
}

CURRENCY_KEYWORDS = {
    "USD": ["USD", "DOLLAR", "FED", "FOMC", "POWELL", "NFP", "NONFARM", "TREASURY"],
    "EUR": ["EUR", "EURO", "ECB", "LAGARDE", "EUROZONE"],
    "GBP": ["GBP", "POUND", "STERLING", "BOE", "BAILEY"],
    "JPY": ["JPY", "YEN", "BOJ", "UEDA"],
    "AUD": ["AUD", "AUSSIE", "RBA"],
    "NZD": ["NZD", "KIWI", "RBNZ"],
    "CAD": ["CAD", "LOONIE", "BOC"],
    "CHF": ["CHF", "FRANC", "SNB"],
}
MACRO_KEYWORDS = ["CPI", "INFLATION", "GDP", "PMI", "RATE DECISION", "CENTRAL BANK"]


class SentimentAnalysisResult(BaseModel):
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Directional sentiment for the currency pair from -1.0 (very bearish) to +1.0 (very bullish).",
    )
    high_impact_risk: bool = Field(
        ...,
        description="True if an imminent high-impact macro release (NFP, CPI, FOMC, central bank rate decision) or extreme geopolitical shock poses whip-saw risk within 30 minutes.",
    )
    catalyst_summary: str = Field(
        ...,
        description="Concise 1-2 sentence explanation of the key fundamental drivers and macro catalysts.",
    )


class NewsSentimentEngine:
    def __init__(self):
        self.client = (
            OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL)
            if settings.LLM_API_KEY
            else None
        )
        self.market_data = MarketDataProvider()
        self.llm_cache = LLMResultCache()

    @staticmethod
    def _cache_key(instrument: str, system_prompt: str, headlines: List[Dict[str, str]], events: List[Dict[str, Any]]) -> str:
        """
        Fingerprint of the news inputs: unchanged headlines + calendar mean an unchanged LLM verdict.
        The cross-asset snapshot is left out on purpose (it ticks every refresh); LLM_CACHE_TTL_MINUTES bounds staleness.
        """
        fingerprint = json.dumps(
            {
                "model": settings.LLM_MODEL,
                "instrument": instrument,
                "prompt": system_prompt,
                "headlines": sorted(h["title"] for h in headlines),
                "events": [(e["time_utc"], e["currency"], e["title"], e["forecast"]) for e in events],
            },
            sort_keys=True,
        )
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def fetch_latest_headlines(self, instrument: str, max_items: int = 20) -> List[Dict[str, str]]:
        """Aggregates latest forex headlines and economic calendar items."""
        base_ccy, quote_ccy = instrument.split("_")
        headlines: List[Dict[str, str]] = []

        # 1. Finnhub / NewsAPI if configured
        if settings.NEWS_API_KEY:
            try:
                url = "https://finnhub.io/api/v1/news"
                resp = requests.get(
                    url, params={"category": "forex", "token": settings.NEWS_API_KEY}, timeout=8
                )
                if resp.status_code == 200:
                    for item in resp.json()[:max_items]:
                        headlines.append({
                            "title": item.get("headline", ""),
                            "source": item.get("source", "Finnhub"),
                            "summary": item.get("summary", "")[:240],
                        })
            except Exception as e:
                logger.warning(f"Finnhub fetch warning: {e}")

        # 2. Yahoo Finance pair news via yfinance (free)
        headlines.extend(self.market_data.get_pair_news(instrument))

        # 3. Free forex RSS feeds, filtered for pair relevance
        keywords = (
            CURRENCY_KEYWORDS.get(base_ccy, [base_ccy])
            + CURRENCY_KEYWORDS.get(quote_ccy, [quote_ccy])
            + MACRO_KEYWORDS
        )
        for source, rss_url in FREE_RSS_FEEDS.items():
            matched = 0
            for entry in self.market_data.get_rss_entries(source, rss_url):
                if matched >= 6:  # Per-feed cap keeps the source mix balanced
                    break
                text_blob = f"{entry['title']} {entry['summary']}".upper()
                if any(k in text_blob for k in keywords):
                    matched += 1
                    headlines.append({
                        "title": entry["title"],
                        "source": source,
                        "summary": entry["summary"][:200],
                    })

        # Deduplicate by title
        seen = set()
        unique_headlines = []
        for h in headlines:
            if h["title"] and h["title"] not in seen:
                seen.add(h["title"])
                unique_headlines.append(h)

        return unique_headlines[:max_items]

    def analyze_pair_sentiment(self, instrument: str) -> Dict[str, Any]:
        """
        Evaluates news and macro catalysts for `instrument` via LLM and returns strict JSON:
        `sentiment_score`, `high_impact_risk`, `catalyst_summary`, and `headlines`.
        """
        headlines = self.fetch_latest_headlines(instrument)
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Free macro context: economic calendar (Forex Factory) + cross-asset moves (yfinance)
        upcoming_events = self.market_data.get_upcoming_events(instrument)
        imminent_events = self.market_data.imminent_high_impact_events(instrument)
        macro = self.market_data.get_macro_snapshot()
        context = {
            "upcoming_events": [
                {**e, "time_utc": e["time_utc"].strftime("%Y-%m-%d %H:%M UTC")} for e in upcoming_events[:8]
            ],
            "macro": macro,
        }
        calendar_risk_note = (
            "Calendar: high-impact release imminent - "
            + "; ".join(f"{e['currency']} {e['title']} @ {e['time_utc']:%H:%M} UTC" for e in imminent_events)
            if imminent_events
            else ""
        )

        if not self.client:
            return {
                "instrument": instrument,
                "sentiment_score": 0.0,
                "high_impact_risk": bool(imminent_events),
                "catalyst_summary": calendar_risk_note or "LLM_API_KEY not configured; sentiment defaulted to neutral (0.0).",
                "headlines": headlines,
                "timestamp": now_utc,
                **context,
            }

        formatted_news = "\n".join(
            [f"- [{h['source']}] {h['title']}: {h['summary']}" for h in headlines]
        ) or "No major pair-specific headlines in the last window."
        formatted_calendar = "\n".join(
            [
                f"- {e['time_utc']} | {e['currency']} | {e['impact']} | {e['title']}"
                f" (forecast {e['forecast'] or 'n/a'}, previous {e['previous'] or 'n/a'})"
                for e in context["upcoming_events"]
            ]
        ) or "No medium/high-impact events for this pair in the next 24h."
        formatted_macro = ", ".join(
            [f"{name} {v['last']} ({v['change_pct']:+.2f}%)" for name, v in macro.items()]
        ) or "Unavailable."

        system_prompt = (
            "You are an institutional FX quantitative macro analyst. "
            "Analyze the provided real-time forex headlines and calendar events for the requested currency pair. "
            "Remember: For pair BASE_QUOTE (e.g., EUR_USD), a positive sentiment_score (+0.1 to +1.0) means BASE is bullish against QUOTE. "
            "A negative sentiment_score (-0.1 to -1.0) means BASE is bearish against QUOTE. "
            "Weigh the economic calendar (scheduled releases with forecast vs previous) and the cross-asset snapshot "
            "(DXY, US10Y yield, VIX, gold, oil, S&P 500) as confirming or conflicting macro context. "
            "Set high_impact_risk to true if headlines indicate NFP, CPI, FOMC, or Central Bank rate decisions are occurring within 30 minutes or causing extreme disorderly volatility. "
            "Respond strictly in valid JSON matching the schema: "
            '{"sentiment_score": float, "high_impact_risk": bool, "catalyst_summary": string}.'
        )

        user_prompt = (
            f"Current UTC Time: {now_utc}\n"
            f"Target Currency Pair: {instrument}\n"
            f"Recent Market Headlines & Macro Releases:\n{formatted_news}\n\n"
            f"Economic Calendar (next 24h, {instrument.replace('_', '/')} currencies):\n{formatted_calendar}\n\n"
            f"Cross-Asset Snapshot (last daily close, % change):\n{formatted_macro}"
        )

        cache_key = self._cache_key(instrument, system_prompt, headlines, context["upcoming_events"])
        try:
            cached = self.llm_cache.get(cache_key, settings.LLM_CACHE_TTL_MINUTES)
            if cached:
                validated = SentimentAnalysisResult(**cached)
            else:
                response = self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=300,
                )
                content = response.choices[0].message.content or "{}"
                parsed_json = json.loads(content)
                validated = SentimentAnalysisResult(**parsed_json)
                self.llm_cache.set(cache_key, instrument, settings.LLM_MODEL, validated.model_dump())

            return {
                "instrument": instrument,
                "cached": bool(cached),
                "sentiment_score": round(validated.sentiment_score, 3),
                # Calendar is authoritative for scheduled releases; LLM covers unscheduled shocks
                "high_impact_risk": validated.high_impact_risk or bool(imminent_events),
                "catalyst_summary": " | ".join(filter(None, [calendar_risk_note, validated.catalyst_summary])),
                "headlines": headlines[:5],
                "timestamp": now_utc,
                **context,
            }
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed for {instrument}: {e}")
            return {
                "instrument": instrument,
                "sentiment_score": 0.0,
                "high_impact_risk": bool(imminent_events),
                "catalyst_summary": calendar_risk_note or f"Sentiment fallback (LLM error: {str(e)[:80]})",
                "headlines": headlines[:5],
                "timestamp": now_utc,
                **context,
            }

