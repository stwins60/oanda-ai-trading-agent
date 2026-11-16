"""
LLM discretionary trader.

Once per completed entry-timeframe bar per instrument, the model receives a full desk briefing
(multi-timeframe indicators, recent price action, live spread/costs, news sentiment & catalysts,
economic calendar, cross-asset moves, open book, and its own recent track record) and returns a
structured trade plan. The plan is advisory: agent.py enforces every hard risk rule
(sizing, spread, news blackout, max trades, daily loss breaker, SL/TP bounds) before any order.

Modes (STRATEGY_MODE): "rules" = AI off, "ai_shadow" = AI decides and is journaled but never trades,
"ai" = AI decisions drive entries.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

from config import settings

logger = logging.getLogger("oanda_agent.ai_trader")

SL_BOUNDS = (1.0, 4.0)  # ATR multiples the model may choose within; anything outside is clamped
TP_BOUNDS = (1.0, 6.0)


class TradePlan(BaseModel):
    action: Literal["BUY", "SELL", "NO_TRADE"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    stop_loss_atr: float = 1.5  # Clamped to SL_BOUNDS after parsing; ignored for NO_TRADE
    take_profit_atr: float = 2.5
    thesis: str = ""
    invalidation: str = ""
    key_risks: str = ""

    @field_validator("action", mode="before")
    @classmethod
    def _norm_action(cls, v):
        return str(v).strip().upper().replace(" ", "_")

    @field_validator("stop_loss_atr", "take_profit_atr", mode="before")
    @classmethod
    def _default_levels(cls, v):
        return v if isinstance(v, (int, float)) and v > 0 else 1.5

    @field_validator("thesis", "invalidation", "key_risks", mode="before")
    @classmethod
    def _truncate(cls, v):
        return str(v or "")[:600]


SYSTEM_PROMPT = """You are a disciplined senior FX trader running a small live account.
Your job is to decide, for ONE currency pair at the close of ONE entry bar, whether to open a trade now.

Hard facts about your situation:
- Every trade pays the spread up front. The briefing shows the spread as a fraction of ATR and of your stop.
- A backtest of simple trend-alignment rules on this account's pairs LOST money (about -0.1 to -0.2R per trade),
  mostly because the edge was smaller than costs. Do not repeat that: alignment alone is not a reason to trade.
- NO_TRADE is the default. Only trade when several independent factors line up (structure, momentum, location
  relative to recent range, macro/news drivers, cross-asset confirmation) AND reward clearly exceeds cost.
- Avoid entering into imminent high-impact events, after extended moves far from the mean, in thin/illiquid hours,
  or against the higher-timeframe trend unless there is a specific, stated catalyst.
- Stops are in ATR multiples of the ENTRY timeframe (allowed 1.0-4.0). Targets in ATR multiples (allowed 1.0-6.0).
  Place the stop beyond obvious structure, not arbitrarily tight.
- Confidence is your calibrated probability-weighted conviction (0-1). Be honest: 0.5 means a coin flip.
- Learn from your recent track record in the briefing; if recent trades in similar conditions failed, adapt.

Respond ONLY with JSON:
{"action": "BUY"|"SELL"|"NO_TRADE", "confidence": float, "stop_loss_atr": float, "take_profit_atr": float,
 "thesis": string, "invalidation": string, "key_risks": string}
Keep thesis to at most 3 sentences, invalidation and key_risks to 1 sentence each."""


def _fmt_tf(name: str, ev: Dict[str, Any]) -> str:
    if not ev.get("valid"):
        return f"{name}: insufficient data"
    return (f"{name}: bias={ev['bias']} close={ev['close']} EMA20={ev['ema_20']} EMA50={ev['ema_50']} "
            f"EMA200={ev['ema_200']} RSI14={ev['rsi_14']} ATR14={ev['atr_14']} MACDhist={ev['macd_hist']}")


class AITrader:
    def __init__(self, broker, journal=None):
        self.broker = broker
        self.journal = journal
        self.model = settings.AI_TRADER_MODEL or settings.LLM_MODEL
        self.client = OpenAI(api_key=settings.LLM_API_KEY, base_url=settings.LLM_BASE_URL) if settings.LLM_API_KEY else None
        self._cache: Dict[Tuple[str, Any], Dict[str, Any]] = {}  # (instrument, bar_time) -> decision

    @property
    def enabled(self) -> bool:
        return settings.STRATEGY_MODE in ("ai", "ai_shadow") and self.client is not None

    def _price_action(self, instrument: str, tf: str, bars: int = 32) -> str:
        df = self.broker.get_candles(instrument, granularity=tf, count=bars + 1)
        if df.empty:
            return "unavailable"
        if "complete" in df.columns:
            df = df[df["complete"]]
        df = df.tail(bars)
        rows = [f"{datetime.fromtimestamp(r.time, timezone.utc):%m-%d %H:%M} O{r.open} H{r.high} L{r.low} C{r.close}"
                for r in df.itertuples()]
        hi, lo = df["high"].max(), df["low"].min()
        last = df["close"].iloc[-1]
        pos = (last - lo) / (hi - lo) if hi > lo else 0.5
        return f"range {lo}-{hi}, last close at {pos:.0%} of range\n" + "\n".join(rows)

    def _track_record(self, instrument: str) -> str:
        trades = self.broker.get_closed_trades(count=20)
        if not trades:
            return "No closed trades yet."
        lines = []
        for t in trades[:10]:
            closed = datetime.fromtimestamp(float(t["close_time"]), timezone.utc).strftime("%m-%d %H:%M") if t.get("close_time") else "?"
            lines.append(f"{closed} {t['instrument']} {t['side']} {t['entry_price']}->{t['close_price']} P/L {t['realized_pnl']:+.2f}")
        wins = sum(1 for t in trades if t["realized_pnl"] > 0)
        return f"Last {len(trades)} closed trades: {wins} wins.\n" + "\n".join(lines)

    def decide(
        self,
        instrument: str,
        tech: Dict[str, Any],
        sentiment: Dict[str, Any],
        pricing: Dict[str, Any],
        account: Dict[str, Any],
        open_trades: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Returns a validated, clamped trade plan for the latest completed bar (cached per bar), or None."""
        if not self.enabled:
            return None
        bar_time = tech.get("bar_time")
        key = (instrument, bar_time)
        if key in self._cache:
            return self._cache[key]

        tfs = tech.get("timeframes", {})
        entry_tf, mid_tf, trend_tf = settings.STRATEGY_TIMEFRAMES
        atr = float(tech.get("atr") or 0.0)
        spread = float(pricing.get("spread") or 0.0) if "error" not in pricing else None
        cost_line = (
            f"bid {pricing['bid']} / ask {pricing['ask']}, spread {spread:.5f} = {spread / atr:.2f} x ATR "
            f"(= {spread / (atr * 1.5):.0%} of a 1.5-ATR stop)" if spread is not None and atr > 0 else "pricing unavailable"
        )
        events = "\n".join(
            f"- {e['time_utc']} {e['currency']} {e['impact']} {e['title']} (fcst {e.get('forecast') or 'n/a'}, prev {e.get('previous') or 'n/a'})"
            for e in sentiment.get("upcoming_events", [])[:8]
        ) or "none in next 24h"
        macro = ", ".join(f"{k} {v['last']} ({v['change_pct']:+.2f}%)" for k, v in (sentiment.get("macro") or {}).items()) or "unavailable"
        headlines = "\n".join(f"- [{h.get('source')}] {h.get('title')}" for h in sentiment.get("headlines", [])[:6]) or "none"
        book = "\n".join(
            f"- #{t['id']} {t['instrument']} {t['side']} {t['units']} @ {t['entry_price']} uPL {t['unrealized_pnl']:+.2f}"
            for t in open_trades
        ) or "flat"

        briefing = f"""UTC now: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} (bar closed {datetime.fromtimestamp(bar_time, timezone.utc):%H:%M} UTC)
PAIR: {instrument}
Rule-based signal for reference: {tech.get('signal')} - {tech.get('reason')}

MULTI-TIMEFRAME INDICATORS
{_fmt_tf(entry_tf, tfs.get(entry_tf, {}))}
{_fmt_tf(mid_tf, tfs.get(mid_tf, {}))}
{_fmt_tf(trend_tf, tfs.get(trend_tf, {}))}

RECENT {entry_tf} PRICE ACTION (oldest first)
{self._price_action(instrument, entry_tf)}

RECENT {trend_tf} PRICE ACTION
{self._price_action(instrument, trend_tf, bars=20)}

EXECUTION COSTS
{cost_line}

NEWS SENTIMENT (separate LLM analyst): score {sentiment.get('sentiment_score', 0):+.2f}, high_impact_risk={sentiment.get('high_impact_risk')}
Catalysts: {sentiment.get('catalyst_summary')}
Headlines:
{headlines}

ECONOMIC CALENDAR (pair currencies, next 24h)
{events}

CROSS-ASSET: {macro}

ACCOUNT: NAV {account.get('nav')} {account.get('currency')}, risk per trade {settings.RISK_PER_TRADE_PCT * 100:.2f}% of NAV
OPEN BOOK:
{book}

YOUR RECENT TRACK RECORD
{self._track_record(instrument)}"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": briefing}],
                temperature=0.2,
                max_tokens=500,
            )
            plan = TradePlan(**json.loads(resp.choices[0].message.content or "{}"))
        except Exception as e:
            logger.error(f"AI trader failed for {instrument}: {e}")
            return None  # Not cached: retried next cycle

        decision = {
            "instrument": instrument,
            "bar_time": bar_time,
            "model": self.model,
            "action": plan.action,
            "confidence": round(plan.confidence, 2),
            "stop_loss_atr": round(min(max(plan.stop_loss_atr, SL_BOUNDS[0]), SL_BOUNDS[1]), 2),
            "take_profit_atr": round(min(max(plan.take_profit_atr, TP_BOUNDS[0]), TP_BOUNDS[1]), 2),
            "thesis": plan.thesis,
            "invalidation": plan.invalidation,
            "key_risks": plan.key_risks,
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M UTC"),
        }
        self._cache = {k: v for k, v in self._cache.items() if k[1] == bar_time or k[0] != instrument}
        self._cache[key] = decision
        return decision
