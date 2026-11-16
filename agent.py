import logging
import math
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from config import settings
from broker import OandaBroker
from technicals import TechnicalAnalyzer
from news_sentiment import NewsSentimentEngine
from notifier import Notifier
from db import AgentStateStore, SignalJournal
from ai_trader import AITrader

logger = logging.getLogger("oanda_agent.core")

STATE_KEY_RUNNING = "is_running"
STATE_KEY_DAILY = "daily_baseline"


class AutonomousTradingAgent:
    def __init__(self, broker: OandaBroker, sentiment_engine: NewsSentimentEngine, notifier: Notifier):
        self.broker = broker
        self.sentiment_engine = sentiment_engine
        self.notifier = notifier
        self.state_store = AgentStateStore()
        self.journal = SignalJournal()
        self.ai_trader = AITrader(broker)
        self._journal_ctx: Optional[Dict[str, Any]] = None  # Signal context for the instrument being evaluated

        # The background loop and "Scan Now" run cycles in separate threads; serialising them prevents
        # both from seeing "no open trade" and placing duplicate orders on the same instrument.
        self._cycle_lock = threading.Lock()

        # Restore operator run/pause choice and today's loss baseline so a restart doesn't silently
        # pause the bot or reset the daily circuit breaker mid-day.
        self._is_running: bool = bool(self.state_store.get(STATE_KEY_RUNNING, False))
        daily = self.state_store.get(STATE_KEY_DAILY, {}) or {}
        self.current_day: str = daily.get("day", "")
        self.daily_starting_nav: float = float(daily.get("nav", 0.0))
        self.daily_halt_triggered: bool = bool(daily.get("halted", False))

        self._known_open_trade_ids: Optional[Set[str]] = None  # None until the first cycle seeds it

        # State snapshots exposed to the Web Dashboard
        self.latest_technicals: Dict[str, Any] = {}
        self.latest_sentiment: Dict[str, Any] = {}
        self.latest_ai: Dict[str, Any] = {}
        self.decision_log: list = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    @is_running.setter
    def is_running(self, value: bool):
        self._is_running = bool(value)
        self.state_store.set(STATE_KEY_RUNNING, self._is_running)

    def _persist_daily(self):
        self.state_store.set(
            STATE_KEY_DAILY,
            {"day": self.current_day, "nav": self.daily_starting_nav, "halted": self.daily_halt_triggered},
        )

    def _record_decision(self, instrument: str, action: str, details: str):
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "instrument": instrument,
            "action": action,
            "details": details,
        }
        self.decision_log.insert(0, entry)
        self.decision_log = self.decision_log[:100]
        self._journal(instrument, action, details)

    def _journal(self, instrument: str, action: str, details: str):
        """Persists the decision for the current rule signal and/or AI plan on this bar (one row per source per bar)."""
        ctx = self._journal_ctx
        if not ctx or ctx["instrument"] != instrument or not ctx["tech"].get("bar_time"):
            return
        tech, sent, ai = ctx["tech"], ctx["sentiment"], ctx.get("ai")
        base = {
            "instrument": instrument,
            "bar_time": tech["bar_time"],
            "price": tech.get("current_price"),
            "atr": tech.get("atr"),
            "sentiment_score": sent.get("sentiment_score"),
            "high_impact": bool(sent.get("high_impact_risk")),
            "decision": action,
            "details": details[:500],
        }
        if tech.get("signal") in ("BUY", "SELL"):
            self.journal.record({
                **base, "entry_tf": tech.get("entry_timeframe", "M15"), "direction": tech["signal"],
                "tech_reason": tech.get("reason"), "sl_atr": settings.ATR_SL_MULTIPLIER,
                "tp_atr": settings.ATR_TP_MULTIPLIER, "confidence": None,
            })
        if ai and ai["action"] in ("BUY", "SELL"):
            ai_action = action if settings.STRATEGY_MODE == "ai" else f"SHADOW:{action}"
            self.journal.record({
                **base, "entry_tf": f"AI-{tech.get('entry_timeframe', 'M15')}", "direction": ai["action"],
                "tech_reason": ai["thesis"][:500], "decision": ai_action, "sl_atr": ai["stop_loss_atr"],
                "tp_atr": ai["take_profit_atr"], "confidence": ai["confidence"],
            })

    def _check_daily_loss_limit(self, account_summary: Dict[str, Any]) -> bool:
        """
        Tracks daily starting NAV and enforces the max daily drawdown circuit breaker.
        Returns True if trading is allowed, False if daily loss limit was breached.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        nav = account_summary.get("nav", 0.0)
        if nav <= 0:
            return False

        if self.current_day != today_str or self.daily_starting_nav <= 0:
            if self.current_day and self.daily_starting_nav > 0 and settings.DAILY_SUMMARY_ENABLED:
                self._send_daily_summary(self.current_day, self.daily_starting_nav, nav)
            self.current_day = today_str
            self.daily_starting_nav = nav
            self.daily_halt_triggered = False
            self._persist_daily()

        daily_pnl_pct = (nav - self.daily_starting_nav) / self.daily_starting_nav
        if daily_pnl_pct <= -settings.MAX_DAILY_LOSS_PCT:
            if not self.daily_halt_triggered:
                self.daily_halt_triggered = True
                self._persist_daily()
                self.is_running = False
                msg = (
                    f"Max daily loss limit breached ({daily_pnl_pct * 100:.2f}% vs limit "
                    f"-{settings.MAX_DAILY_LOSS_PCT * 100:.2f}%). Liquidating positions and pausing agent."
                )
                logger.critical(msg)
                self.broker.close_all_positions()
                self.notifier.send_system_alert("Max Daily Loss Circuit Breaker Triggered", msg, critical=True)
            return False

        return True

    def _send_daily_summary(self, day: str, start_nav: float, end_nav: float):
        try:
            day_start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            closed = [
                t for t in self.broker.get_closed_trades(count=100)
                if t.get("close_time") and day_start <= float(t["close_time"]) < day_start + 86400
            ]
            pnl = sum(t["realized_pnl"] for t in closed)
            wins = sum(1 for t in closed if t["realized_pnl"] > 0)
            change = end_nav - start_nav
            lines = [f"NAV {start_nav:.2f} -> {end_nav:.2f} ({change:+.2f}, {change / start_nav * 100:+.2f}%)",
                     f"Closed trades: {len(closed)} ({wins} wins), realized P/L {pnl:+.2f}",
                     f"Mode: {settings.STRATEGY_MODE} | Automation {'ON' if self.is_running else 'PAUSED'}"]
            lines += [f"- #{t['id']} {t['instrument']} {t['side']} {t['realized_pnl']:+.2f}" for t in closed[:10]]
            self.notifier.send_system_alert(f"Daily Summary {day}", "\n".join(lines), critical=False)
        except Exception as e:
            logger.error(f"Daily summary failed: {e}")

    def calculate_position_units(
        self,
        instrument: str,
        nav: float,
        margin_available: float,
        entry_price: float,
        stop_loss_price: float,
        pricing: Dict[str, Any],
    ) -> Tuple[int, str]:
        """
        Fixed-fractional sizing: Risk Amount (account ccy) / (Stop Distance x quote->account conversion),
        then capped so a single trade never uses more than MAX_MARGIN_USAGE_PCT of free margin.
        Returns (units, sizing note). Units of 0 means the trade should be skipped.
        """
        stop_distance = abs(entry_price - stop_loss_price)
        if stop_distance <= 0 or entry_price <= 0:
            return 0, "invalid stop distance"

        risk_capital = nav * settings.RISK_PER_TRADE_PCT

        quote_to_home = pricing.get("quote_to_home") or 0.0
        if quote_to_home <= 0:
            # Fallback approximation for USD accounts when OANDA conversion factors are unavailable
            quote_to_home = 1.0 if instrument.endswith("_USD") else (1.0 / entry_price if instrument.startswith("USD_") else 1.0)
        risk_units = risk_capital / (stop_distance * quote_to_home)

        base_to_home = pricing.get("base_to_home") or 0.0
        if base_to_home <= 0:
            base_to_home = 1.0 if instrument.startswith("USD_") else entry_price * quote_to_home
        margin_per_unit = base_to_home * self.broker.get_instrument_margin_rate(instrument)
        margin_units = (
            margin_available * settings.MAX_MARGIN_USAGE_PCT / margin_per_unit if margin_per_unit > 0 else risk_units
        )

        units = int(math.floor(min(risk_units, margin_units)))
        note = f"risk={risk_capital:.2f} -> {int(risk_units)}u"
        if margin_units < risk_units:
            note += f", margin-capped to {units}u"
        return max(units, 0), note

    def _detect_closed_trades(self, open_trades: List[Dict[str, Any]]):
        """Alerts on trades that disappeared since the previous cycle (TP/SL hit, manual close, kill-switch)."""
        current_ids = {str(t["id"]) for t in open_trades}
        if self._known_open_trade_ids is None:
            self._known_open_trade_ids = current_ids
            return

        for trade_id in sorted(self._known_open_trade_ids - current_ids):
            trade = self.broker.get_trade(trade_id)
            if trade.get("state") != "CLOSED":
                continue
            pnl = float(trade.get("realizedPL", 0.0)) + float(trade.get("financing", 0.0))
            if trade.get("takeProfitOrder", {}).get("state") == "FILLED":
                reason = "Take-Profit"
            elif trade.get("stopLossOrder", {}).get("state") == "FILLED":
                reason = "Stop-Loss" if pnl < 0 else "Break-even/Locked Stop"
            elif trade.get("trailingStopLossOrder", {}).get("state") == "FILLED":
                reason = "Trailing Stop"
            else:
                reason = "Manual / Other"
            units = int(float(trade.get("initialUnits", 0)))
            instrument = trade.get("instrument", "?")
            self._record_decision(instrument, "TRADE_CLOSED", f"#{trade_id} {reason} | P/L {pnl:+.2f}")
            self.notifier.send_discord_trade_closed_alert(
                trade_id=trade_id,
                instrument=instrument,
                side="LONG" if units > 0 else "SHORT",
                units=units,
                entry_price=float(trade.get("price", 0.0)),
                close_price=float(trade.get("averageClosePrice", 0.0) or 0.0),
                realized_pnl=pnl,
                reason=reason,
            )
        self._known_open_trade_ids = current_ids

    def _manage_open_trades(self, open_trades: List[Dict[str, Any]]):
        """Moves the stop to break-even (+ a small lock-in) once a trade is BREAKEVEN_TRIGGER_R in profit."""
        if settings.BREAKEVEN_TRIGGER_R <= 0:
            return
        for t in open_trades:
            if not t.get("stop_loss"):
                continue
            instrument = t["instrument"]
            entry = float(t["entry_price"])
            sl = float(t["stop_loss"])
            is_long = t["units"] > 0
            risk = (entry - sl) if is_long else (sl - entry)
            if risk <= 0:
                continue  # Stop already at or beyond break-even

            pricing = self.broker.get_pricing(instrument)
            if "error" in pricing or not pricing.get("tradeable"):
                continue
            exit_price = pricing["bid"] if is_long else pricing["ask"]
            profit = (exit_price - entry) if is_long else (entry - exit_price)
            if profit < settings.BREAKEVEN_TRIGGER_R * risk:
                continue

            lock = settings.BREAKEVEN_LOCK_R * risk
            new_sl = entry + lock if is_long else entry - lock
            if (is_long and new_sl >= exit_price) or (not is_long and new_sl <= exit_price):
                continue  # Would be on the wrong side of the market; OANDA would reject it
            res = self.broker.set_trade_stop_loss(instrument, str(t["id"]), new_sl)
            if res.get("success"):
                self._record_decision(
                    instrument, "STOP_TO_BREAKEVEN",
                    f"#{t['id']} profit {profit / risk:.2f}R -> SL {sl} => {res['stop_loss']}",
                )
                self.notifier._record_audit(
                    "SYSTEM", f"Break-even stop #{t['id']} {instrument}", f"SL moved {sl} -> {res['stop_loss']}", "INFO"
                )
            else:
                self._record_decision(instrument, "SL_UPDATE_ERROR", f"#{t['id']}: {str(res.get('error'))[:160]}")

    def run_cycle(self) -> Dict[str, Any]:
        """Executes a full scan across all watchlist instruments."""
        with self._cycle_lock:
            return self._run_cycle_locked()

    def _run_cycle_locked(self) -> Dict[str, Any]:
        account = self.broker.get_account_summary()
        open_trades = self.broker.get_open_trades()

        self._detect_closed_trades(open_trades)

        # Check circuit breaker
        can_trade = self._check_daily_loss_limit(account)

        if self.is_running and can_trade:
            self._manage_open_trades(open_trades)

        open_instruments = {t["instrument"] for t in open_trades}
        open_count = len(open_trades)

        for instrument in settings.WATCHLIST:
            # 1. Multi-timeframe technical analysis
            tech = TechnicalAnalyzer.analyze_multi_timeframe(self.broker, instrument)
            self.latest_technicals[instrument] = tech

            # 2. LLM News & Sentiment analysis
            sentiment = self.sentiment_engine.analyze_pair_sentiment(instrument)
            self.latest_sentiment[instrument] = sentiment

            # 2b. AI trader plan (once per completed entry bar; shadow mode journals it without trading)
            pricing = self.broker.get_pricing(instrument)
            ai = None
            if self.ai_trader.enabled and tech.get("bar_time"):
                ai = self.ai_trader.decide(instrument, tech, sentiment, pricing, account, open_trades)
                if ai:
                    self.latest_ai[instrument] = ai
            self._journal_ctx = {"instrument": instrument, "tech": tech, "sentiment": sentiment, "ai": ai}
            ai_mode = settings.STRATEGY_MODE == "ai"

            if not self.is_running or not can_trade:
                self._record_decision(
                    instrument,
                    "PAUSED",
                    f"Tech={tech.get('signal')} | Sent={sentiment.get('sentiment_score'):+.2f} (Automation paused)",
                )
                continue

            # 3. Skip if position already open on this instrument
            if instrument in open_instruments:
                self._record_decision(instrument, "HOLD", "Position already open for instrument.")
                continue

            # 4. High-Impact News Filter (NFP / CPI / FOMC protection)
            if sentiment.get("high_impact_risk", False):
                self._record_decision(
                    instrument,
                    "BLOCKED_NEWS_RISK",
                    f"High-impact macro event detected: {sentiment.get('catalyst_summary')}",
                )
                continue

            # 5. Entry decision: AI plan (ai mode) or Technicals + Sentiment confluence (rules mode)
            signal = tech.get("signal", "NEUTRAL")
            score = sentiment.get("sentiment_score", 0.0)
            threshold = settings.MIN_SENTIMENT_THRESHOLD
            sl_mult, tp_mult = settings.ATR_SL_MULTIPLIER, settings.ATR_TP_MULTIPLIER

            trade_direction = None
            if ai_mode:
                if not ai:
                    self._record_decision(instrument, "AI_UNAVAILABLE", "No AI plan for this bar (LLM error or no bar).")
                    continue
                if ai["action"] == "NO_TRADE" or ai["confidence"] < settings.AI_MIN_CONFIDENCE:
                    self._record_decision(
                        instrument, "AI_NO_TRADE",
                        f"{ai['action']} conf={ai['confidence']:.2f} (min {settings.AI_MIN_CONFIDENCE}) | {ai['thesis'][:200]}",
                    )
                    continue
                trade_direction = ai["action"]
                sl_mult, tp_mult = ai["stop_loss_atr"], ai["take_profit_atr"]
            elif signal == "BUY" and score >= threshold:
                trade_direction = "BUY"
            elif signal == "SELL" and score <= -threshold:
                trade_direction = "SELL"
            else:
                self._record_decision(
                    instrument,
                    "NO_CONFLUENCE",
                    f"Tech={signal} ({tech.get('reason')}) | Sentiment={score:+.2f} (Req ±{threshold})",
                )
                continue

            # 6. Portfolio exposure cap
            if open_count >= settings.MAX_OPEN_TRADES:
                self._record_decision(
                    instrument, "BLOCKED_MAX_TRADES",
                    f"{trade_direction} signal skipped: {open_count}/{settings.MAX_OPEN_TRADES} trades already open.",
                )
                continue

            # 7. Live price & spread filter (signals use mid candles; we execute at bid/ask)
            atr = float(tech["atr"])
            if "error" in pricing or not pricing.get("tradeable"):
                self._record_decision(instrument, "BLOCKED_MARKET", "Instrument not tradeable or pricing unavailable.")
                continue
            if atr <= 0 or pricing["spread"] > settings.MAX_SPREAD_ATR_RATIO * atr:
                self._record_decision(
                    instrument, "BLOCKED_SPREAD",
                    f"Spread {pricing['spread']:.5f} > {settings.MAX_SPREAD_ATR_RATIO:.2f} x ATR {atr:.5f}",
                )
                continue

            # 8. Dynamic ATR Stop-Loss, Take-Profit & Position Sizing
            entry_price = pricing["ask"] if trade_direction == "BUY" else pricing["bid"]
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult

            if trade_direction == "BUY":
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            units, sizing_note = self.calculate_position_units(
                instrument=instrument,
                nav=account.get("nav", 0.0),
                margin_available=account.get("margin_available", 0.0),
                entry_price=entry_price,
                stop_loss_price=sl_price,
                pricing=pricing,
            )
            if units < 1:
                self._record_decision(instrument, "BLOCKED_SIZE", f"Position size rounds to 0 units ({sizing_note}).")
                continue
            signed_units = units if trade_direction == "BUY" else -units

            # 9. Execute Market Order with mandatory SL & TP
            order_res = self.broker.place_market_order(
                instrument=instrument,
                units=signed_units,
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
            )

            if order_res.get("success"):
                open_count += 1
                open_instruments.add(instrument)
                fill_px = order_res.get("fill_price") or entry_price
                rationale = (
                    f"AI conf {ai['confidence']:.2f}: {ai['thesis']} | Invalidation: {ai['invalidation']}"
                    if ai_mode else f"{tech.get('reason')} | Catalyst: {sentiment.get('catalyst_summary')}"
                )
                self._record_decision(
                    instrument,
                    f"EXECUTED_{trade_direction}",
                    f"Units={signed_units} @ {fill_px} | SL={order_res['stop_loss']} | TP={order_res['take_profit']} | {sizing_note}",
                )
                self.notifier.send_discord_trade_alert(
                    instrument=instrument,
                    direction=trade_direction,
                    units=signed_units,
                    entry_price=fill_px,
                    sl_price=float(order_res["stop_loss"]),
                    tp_price=float(order_res["take_profit"]),
                    sentiment_score=score,
                    reasoning=rationale,
                )
            else:
                self._record_decision(
                    instrument,
                    "ORDER_ERROR",
                    f"Broker rejected order: {order_res.get('error')}",
                )

        self._journal_ctx = None
        final_trades = self.broker.get_open_trades()
        if self._known_open_trade_ids is not None:
            # New fills become "known" now so their eventual close is reported next cycle
            self._known_open_trade_ids |= {str(t["id"]) for t in final_trades}

        return {
            "account": account,
            "open_trades": final_trades,
            "technicals": self.latest_technicals,
            "sentiment": self.latest_sentiment,
            "is_running": self.is_running,
        }

    def performance_summary(self) -> Dict[str, Any]:
        """Realized performance from OANDA closed-trade history."""
        trades = self.broker.get_closed_trades()
        pnls = [t["realized_pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_win, gross_loss = sum(wins), -sum(losses)
        return {
            "closed_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades), 3) if trades else 0.0,
            "realized_pnl": round(sum(pnls), 2),
            "avg_win": round(gross_win / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            "recent": trades[:15],
        }
