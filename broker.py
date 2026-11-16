import logging
from typing import Dict, Any, List
import requests
import pandas as pd
from config import settings

logger = logging.getLogger("oanda_agent.broker")


class OandaBroker:
    def __init__(self):
        self.base_url = settings.OANDA_BASE_URL
        self.account_id = settings.OANDA_ACCOUNT_ID
        self.session = requests.Session()
        self._margin_rates: Dict[str, float] = {}
        self.session.headers.update({
            "Authorization": f"Bearer {settings.OANDA_ACCESS_TOKEN}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "UNIX",
        })

    def _price_precision(self, instrument: str) -> int:
        """Returns decimal rounding precision for OANDA instruments (3 for JPY pairs, 5 for standard FX)."""
        return 3 if "JPY" in instrument else 5

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetches real-time account NAV, balance, margin used, free margin, and unrealized PnL."""
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            acc = resp.json().get("account", {})
            return {
                "balance": float(acc.get("balance", 0.0)),
                "nav": float(acc.get("NAV", 0.0)),
                "unrealized_pnl": float(acc.get("unrealizedPL", 0.0)),
                "lifetime_pl": float(acc.get("pl", 0.0)),
                "margin_used": float(acc.get("marginUsed", 0.0)),
                "margin_available": float(acc.get("marginAvailable", 0.0)),
                "open_trade_count": int(acc.get("openTradeCount", 0)),
                "currency": acc.get("currency", "USD"),
            }
        except Exception as e:
            logger.error(f"Failed to fetch OANDA account summary: {e}")
            return {
                "balance": 0.0,
                "nav": 0.0,
                "unrealized_pnl": 0.0,
                "margin_used": 0.0,
                "margin_available": 0.0,
                "open_trade_count": 0,
                "currency": "USD",
                "error": str(e),
            }

    def get_candles(self, instrument: str, granularity: str = "M15", count: int = 250) -> pd.DataFrame:
        """
        Fetches historical OHLCV candles from OANDA v20 API.
        Supported granularities: M5, M15, H1, H4, D.
        """
        url = f"{self.base_url}/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M",  # Midpoint candles
        }
        try:
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            candles = resp.json().get("candles", [])
            rows = []
            for c in candles:
                mid = c.get("mid", {})
                rows.append({
                    "time": int(float(c["time"])),
                    "open": float(mid["o"]),
                    "high": float(mid["h"]),
                    "low": float(mid["l"]),
                    "close": float(mid["c"]),
                    "volume": int(c.get("volume", 0)),
                    "complete": bool(c.get("complete", True)),
                })
            df = pd.DataFrame(rows)
            return df
        except Exception as e:
            logger.error(f"Error fetching candles for {instrument} ({granularity}): {e}")
            return pd.DataFrame()

    def get_pricing(self, instrument: str) -> Dict[str, Any]:
        """
        Live bid/ask, tradeable flag and home-currency conversion factors for `instrument`.
        `quote_to_home` converts a P/L amount in the quote currency into account currency (loss side, conservative);
        `base_to_home` converts a position size in base-currency units into account-currency notional.
        """
        url = f"{self.base_url}/accounts/{self.account_id}/pricing"
        try:
            resp = self.session.get(
                url, params={"instruments": instrument, "includeHomeConversions": "true"}, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            price = data["prices"][0]
            bid = float(price["bids"][0]["price"])
            ask = float(price["asks"][0]["price"])
            conversions = {c["currency"]: c for c in data.get("homeConversions", [])}
            base_ccy, quote_ccy = instrument.split("_")
            quote_conv = conversions.get(quote_ccy, {})
            base_conv = conversions.get(base_ccy, {})
            return {
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2,
                "spread": ask - bid,
                "tradeable": bool(price.get("tradeable", False)),
                "quote_to_home": float(quote_conv.get("accountLoss", 0.0) or 0.0),
                "base_to_home": float(base_conv.get("positionValue", 0.0) or 0.0),
            }
        except Exception as e:
            logger.error(f"Failed to fetch pricing for {instrument}: {e}")
            return {"error": str(e)}

    def get_instrument_margin_rate(self, instrument: str) -> float:
        """Margin rate OANDA applies to `instrument` on this account (e.g. 0.02 = 50:1). Cached per instrument."""
        if instrument in self._margin_rates:
            return self._margin_rates[instrument]
        url = f"{self.base_url}/accounts/{self.account_id}/instruments"
        try:
            resp = self.session.get(url, params={"instruments": instrument}, timeout=10)
            resp.raise_for_status()
            rate = float(resp.json()["instruments"][0]["marginRate"])
            self._margin_rates[instrument] = rate
            return rate
        except Exception as e:
            logger.error(f"Failed to fetch margin rate for {instrument}: {e}")
            return 0.05  # Conservative fallback (20:1); not cached so the next call retries

    def get_trade(self, trade_id: str) -> Dict[str, Any]:
        """Raw OANDA trade record (open or closed)."""
        url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json().get("trade", {})
        except Exception as e:
            logger.error(f"Failed to fetch trade {trade_id}: {e}")
            return {}

    def get_closed_trades(self, count: int = 500) -> List[Dict[str, Any]]:
        """Most recent closed trades (newest first) with realized P/L including financing."""
        url = f"{self.base_url}/accounts/{self.account_id}/trades"
        try:
            resp = self.session.get(url, params={"state": "CLOSED", "count": count}, timeout=10)
            resp.raise_for_status()
            parsed = []
            for t in resp.json().get("trades", []):
                units = int(float(t.get("initialUnits", 0)))
                parsed.append({
                    "id": t.get("id"),
                    "instrument": t.get("instrument"),
                    "units": units,
                    "side": "LONG" if units > 0 else "SHORT",
                    "entry_price": float(t.get("price", 0.0)),
                    "close_price": float(t.get("averageClosePrice", 0.0) or 0.0),
                    "realized_pnl": float(t.get("realizedPL", 0.0)) + float(t.get("financing", 0.0)),
                    "open_time": t.get("openTime"),
                    "close_time": t.get("closeTime"),
                })
            return parsed
        except Exception as e:
            logger.error(f"Error fetching closed trades: {e}")
            return []

    def set_trade_stop_loss(self, instrument: str, trade_id: str, price: float) -> Dict[str, Any]:
        """Creates or replaces the stop-loss attached to an open trade."""
        url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/orders"
        payload = {"stopLoss": {"price": f"{price:.{self._price_precision(instrument)}f}", "timeInForce": "GTC"}}
        try:
            resp = self.session.put(url, json=payload, timeout=10)
            if resp.status_code not in (200, 201):
                logger.error(f"Stop-loss update rejected for trade {trade_id}: {resp.text}")
                return {"success": False, "error": resp.text}
            return {"success": True, "trade_id": trade_id, "stop_loss": payload["stopLoss"]["price"]}
        except Exception as e:
            logger.error(f"Stop-loss update exception for trade {trade_id}: {e}")
            return {"success": False, "error": str(e)}

    def get_open_trades(self) -> List[Dict[str, Any]]:
        """Returns a list of all currently open trades on the account."""
        url = f"{self.base_url}/accounts/{self.account_id}/openTrades"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            trades = resp.json().get("trades", [])
            parsed = []
            for t in trades:
                units = int(float(t.get("currentUnits", 0)))
                parsed.append({
                    "id": t.get("id"),
                    "instrument": t.get("instrument"),
                    "units": units,
                    "side": "LONG" if units > 0 else "SHORT",
                    "entry_price": float(t.get("price", 0.0)),
                    "unrealized_pnl": float(t.get("unrealizedPL", 0.0)),
                    "open_time": t.get("openTime"),
                    "stop_loss": t.get("stopLossOrder", {}).get("price"),
                    "take_profit": t.get("takeProfitOrder", {}).get("price"),
                })
            return parsed
        except Exception as e:
            logger.error(f"Error fetching open trades: {e}")
            return []

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> Dict[str, Any]:
        """
        Places a Market Order with mandatory attached Stop-Loss and Take-Profit orders.
        Positive units = BUY (LONG), Negative units = SELL (SHORT).
        """
        if stop_loss_price <= 0 or take_profit_price <= 0:
            raise ValueError("Mandatory Stop-Loss and Take-Profit prices must be positive.")

        precision = self._price_precision(instrument)
        sl_str = f"{stop_loss_price:.{precision}f}"
        tp_str = f"{take_profit_price:.{precision}f}"

        payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(int(units)),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {"price": sl_str, "timeInForce": "GTC"},
                "takeProfitOnFill": {"price": tp_str, "timeInForce": "GTC"},
            }
        }

        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        try:
            resp = self.session.post(url, json=payload, timeout=10)
            data = resp.json()
            if resp.status_code not in (200, 201):
                logger.error(f"Order rejected by OANDA: {data}")
                return {"success": False, "error": data}

            fill_tx = data.get("orderFillTransaction", {})
            return {
                "success": True,
                "order_id": fill_tx.get("id"),
                "instrument": instrument,
                "units": units,
                "fill_price": float(fill_tx.get("price", 0.0)) if fill_tx.get("price") else None,
                "stop_loss": sl_str,
                "take_profit": tp_str,
                "raw": data,
            }
        except Exception as e:
            logger.error(f"Order placement exception for {instrument}: {e}")
            return {"success": False, "error": str(e)}

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """Closes a specific open trade by trade ID."""
        url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/close"
        try:
            resp = self.session.put(url, json={"units": "ALL"}, timeout=10)
            resp.raise_for_status()
            return {"success": True, "trade_id": trade_id, "data": resp.json()}
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {e}")
            return {"success": False, "trade_id": trade_id, "error": str(e)}

    def close_all_positions(self) -> Dict[str, Any]:
        """
        Emergency Kill-Switch: Immediately liquidates all open positions across the account.
        """
        url = f"{self.base_url}/accounts/{self.account_id}/openPositions"
        closed_summary = []
        errors = []
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            positions = resp.json().get("positions", [])

            for pos in positions:
                instrument = pos.get("instrument")
                long_units = int(float(pos.get("long", {}).get("units", 0)))
                short_units = int(float(pos.get("short", {}).get("units", 0)))

                close_payload = {}
                if long_units > 0:
                    close_payload["longUnits"] = "ALL"
                if short_units < 0:
                    close_payload["shortUnits"] = "ALL"

                if close_payload:
                    close_url = f"{self.base_url}/accounts/{self.account_id}/positions/{instrument}/close"
                    c_resp = self.session.put(close_url, json=close_payload, timeout=10)
                    if c_resp.status_code in (200, 201):
                        closed_summary.append(instrument)
                    else:
                        errors.append({instrument: c_resp.text})

            return {
                "success": len(errors) == 0,
                "closed_instruments": closed_summary,
                "errors": errors,
            }
        except Exception as e:
            logger.error(f"Emergency kill-switch error: {e}")
            return {"success": False, "closed_instruments": closed_summary, "errors": [str(e)]}

