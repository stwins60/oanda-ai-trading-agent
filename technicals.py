import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger("oanda_agent.technicals")


class TechnicalAnalyzer:
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Computes EMA 20/50/200, RSI(14), ATR(14), and MACD(12,26,9) natively on an OHLCV DataFrame."""
        if df.empty or len(df) < 205:
            return df

        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Exponential Moving Averages (EMA 20, 50, 200)
        df["ema_20"] = close.ewm(span=20, adjust=False).mean()
        df["ema_50"] = close.ewm(span=50, adjust=False).mean()
        df["ema_200"] = close.ewm(span=200, adjust=False).mean()

        # Relative Strength Index (RSI 14 - Wilder's smoothing)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Average True Range (ATR 14 - Wilder's smoothing)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    @classmethod
    def evaluate_single_timeframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates trend and momentum state on a single timeframe DataFrame."""
        if not df.empty and "complete" in df.columns:
            df = df[df["complete"]]  # The forming candle repaints; evaluate closed bars only
        df_ind = cls.compute_indicators(df)
        if df_ind.empty or "ema_200" not in df_ind.columns or pd.isna(df_ind.iloc[-1]["ema_200"]):
            return {"bias": "NEUTRAL", "valid": False}

        last = df_ind.iloc[-1]
        close = float(last["close"])
        ema20 = float(last["ema_20"])
        ema50 = float(last["ema_50"])
        ema200 = float(last["ema_200"])
        rsi = float(last["rsi_14"])
        atr = float(last["atr_14"])
        macd_hist = float(last["macd_hist"])

        # Trend stack + momentum check
        if close > ema20 > ema50 > ema200 and 45.0 <= rsi <= 72.0 and macd_hist > 0:
            bias = "BULLISH"
        elif close < ema20 < ema50 < ema200 and 28.0 <= rsi <= 55.0 and macd_hist < 0:
            bias = "BEARISH"
        elif close > ema50 > ema200 and macd_hist > 0:
            bias = "LEAN_BULLISH"
        elif close < ema50 < ema200 and macd_hist < 0:
            bias = "LEAN_BEARISH"
        else:
            bias = "NEUTRAL"

        return {
            "valid": True,
            "bias": bias,
            "close": close,
            "ema_20": round(ema20, 5),
            "ema_50": round(ema50, 5),
            "ema_200": round(ema200, 5),
            "rsi_14": round(rsi, 2),
            "atr_14": round(atr, 5),
            "macd_hist": round(macd_hist, 6),
        }

    @classmethod
    def analyze_multi_timeframe(cls, broker, instrument: str, timeframes=None) -> Dict[str, Any]:
        """
        Fetches entry / confirmation / trend timeframe candles (default M15, H1, H4; set STRATEGY_TIMEFRAMES)
        and requires the two higher timeframes to agree with a strong entry-timeframe trigger.
        """
        from config import settings

        tfs = list(timeframes or settings.STRATEGY_TIMEFRAMES)
        entry_tf, mid_tf, trend_tf = tfs
        frames = {tf: broker.get_candles(instrument, granularity=tf, count=250) for tf in tfs}
        evals = {tf: cls.evaluate_single_timeframe(frames[tf]) for tf in tfs}
        e, m, t = evals[entry_tf], evals[mid_tf], evals[trend_tf]

        if not all(v.get("valid") for v in evals.values()):
            return {
                "instrument": instrument,
                "signal": "NEUTRAL",
                "aligned": False,
                "reason": "Insufficient historical bars for EMA-200 calculation.",
                "timeframes": evals,
            }

        bullish_set = {"BULLISH", "LEAN_BULLISH"}
        bearish_set = {"BEARISH", "LEAN_BEARISH"}
        label = f"{entry_tf}={e['bias']}, {mid_tf}={m['bias']}, {trend_tf}={t['bias']}"

        if e["bias"] == "BULLISH" and m["bias"] in bullish_set and t["bias"] in bullish_set:
            signal, aligned, reason = "BUY", True, f"Bullish MTF alignment ({label})"
        elif e["bias"] == "BEARISH" and m["bias"] in bearish_set and t["bias"] in bearish_set:
            signal, aligned, reason = "SELL", True, f"Bearish MTF alignment ({label})"
        else:
            signal, aligned, reason = "NEUTRAL", False, f"No MTF confluence ({label})"

        entry_df = frames[entry_tf]
        completed = entry_df[entry_df["complete"]] if "complete" in entry_df.columns else entry_df
        return {
            "instrument": instrument,
            "signal": signal,
            "aligned": aligned,
            "reason": reason,
            "entry_timeframe": entry_tf,
            "bar_time": int(completed.iloc[-1]["time"]) if not completed.empty else None,
            "current_price": e["close"],
            "atr": e["atr_14"],
            "rsi": e["rsi_14"],
            "timeframes": evals,
        }
