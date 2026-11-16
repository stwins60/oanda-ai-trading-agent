"""
Walk-forward backtester for the agent's technical strategy (MTF EMA/RSI/MACD + ATR stops).

What it models: completed-candle signals, entry at the next M15 open at ask/bid, intrabar SL/TP on bid/ask,
SL-first when both are hit in one bar, break-even stop, spread filter, optional UTC session filter.
What it cannot model: the LLM news-sentiment confirmation and the news blackout (headlines can't be replayed),
swap/financing, and slippage beyond the quoted spread.

Usage (inside the app image, project mounted at /app):
    python backtest.py fetch                # download/cache ~2 years of candles
    python backtest.py optimize             # grid-search on in-sample, report out-of-sample
    python backtest.py journal              # forward-test live journaled signals & AI plans (needs DATABASE_URL)
    python backtest.py fetch --tfs D        # extra daily candles, then: optimize --tfs H1,H4,D
    python backtest.py run --sl 1.5 --tp 2.5 --be 1.0 --hours 0-24 --pairs EUR_USD,GBP_USD
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from technicals import TechnicalAnalyzer

CACHE = Path(__file__).parent / ".cache" / "backtest"
PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD"]
GRAN_SECONDS = {"M15": 900, "H1": 3600, "H4": 14400, "D": 86400}
DEFAULT_TFS = ("M15", "H1", "H4")
YEARS = 2
IS_FRACTION = 0.7  # First 70% of history = in-sample (tuning); last 30% = out-of-sample (validation)


# ---------------------------------------------------------------- data
def fetch_candles(broker, instrument: str, gran: str, start: float, end: float) -> pd.DataFrame:
    rows, cursor = [], start
    url = f"{broker.base_url}/instruments/{instrument}/candles"
    while cursor < end:
        for attempt in range(5):
            resp = broker.session.get(
                url, params={"granularity": gran, "price": "MBA", "from": f"{cursor:.0f}", "count": 5000}, timeout=30
            )
            if resp.status_code == 200:
                break
            time.sleep(2 * (attempt + 1))
        resp.raise_for_status()
        candles = [c for c in resp.json().get("candles", []) if c.get("complete")]
        if not candles:
            break
        for c in candles:
            m, b, a = c["mid"], c["bid"], c["ask"]
            rows.append([float(c["time"]), float(m["o"]), float(m["h"]), float(m["l"]), float(m["c"]),
                         float(b["o"]), float(b["h"]), float(b["l"]), float(b["c"]),
                         float(a["o"]), float(a["h"]), float(a["l"]), float(a["c"])])
        last = float(candles[-1]["time"])
        if last <= cursor:
            break
        cursor = last + 1
        time.sleep(0.1)
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "bo", "bh", "bl", "bc", "ao", "ah", "al", "ac"])
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def cmd_fetch(pairs: List[str], grans: List[str]):
    from broker import OandaBroker
    broker = OandaBroker()
    CACHE.mkdir(parents=True, exist_ok=True)
    end = time.time()
    start = end - YEARS * 365 * 86400 - 60 * 86400  # Extra 60 days of warm-up for EMA-200 on H4
    if "D" in grans:
        start -= 300 * 86400  # EMA-200 on daily bars needs ~10 more months of history
    for pair in pairs:
        for gran in grans:
            path = CACHE / f"{pair}_{gran}.pkl"
            df = fetch_candles(broker, pair, gran, start, end)
            df.to_pickle(path)
            print(f"{pair} {gran}: {len(df)} bars", flush=True)


def load(pair: str, gran: str) -> pd.DataFrame:
    return pd.read_pickle(CACHE / f"{pair}_{gran}.pkl")


# ---------------------------------------------------------------- signals
def bias_series(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised copy of TechnicalAnalyzer.evaluate_single_timeframe's bias rules (verified by self_check)."""
    d = TechnicalAnalyzer.compute_indicators(df)
    c, e20, e50, e200 = d["close"], d["ema_20"], d["ema_50"], d["ema_200"]
    rsi, h = d["rsi_14"], d["macd_hist"]
    bull = (c > e20) & (e20 > e50) & (e50 > e200) & rsi.between(45, 72) & (h > 0)
    bear = (c < e20) & (e20 < e50) & (e50 < e200) & rsi.between(28, 55) & (h < 0)
    lean_bull = (c > e50) & (e50 > e200) & (h > 0)
    lean_bear = (c < e50) & (e50 < e200) & (h < 0)
    bias = np.select([bull, bear, lean_bull, lean_bear], [2, -2, 1, -1], 0)  # 2=strong, 1=lean
    d["bias"] = bias
    d.loc[d.index < 205, "bias"] = 0
    return d


def build_signals(pair: str, tfs=DEFAULT_TFS) -> pd.DataFrame:
    """Entry-timeframe frame with higher-timeframe bias aligned as of each entry bar close (no look-ahead)."""
    entry_tf, mid_tf, trend_tf = tfs
    base, mid, trend = (bias_series(load(pair, g)) for g in tfs)
    base["close_time"] = base["time"] + GRAN_SECONDS[entry_tf]
    for name, tf, df in (("mid", mid_tf, mid), ("trend", trend_tf, trend)):
        df = df[["time", "bias"]].copy()
        if tf == "D":
            # OANDA daily candles open at 17:00 New York, so close-time = next bar's open; use the shift
            df["close_time"] = df["time"].shift(-1).fillna(df["time"] + 86400)
        else:
            df["close_time"] = df["time"] + GRAN_SECONDS[tf]
        base = pd.merge_asof(
            base.sort_values("close_time"), df[["close_time", "bias"]].rename(columns={"bias": f"bias_{name}"}),
            on="close_time", direction="backward",
        )
    base[["bias_mid", "bias_trend"]] = base[["bias_mid", "bias_trend"]].fillna(0)
    long_sig = (base["bias"] == 2) & (base["bias_mid"] > 0) & (base["bias_trend"] > 0)
    short_sig = (base["bias"] == -2) & (base["bias_mid"] < 0) & (base["bias_trend"] < 0)
    base["signal"] = np.where(long_sig, 1, np.where(short_sig, -1, 0))
    return base.reset_index(drop=True)


def self_check(pair: str = "EUR_USD", samples: int = 40):
    """Confirms the vectorised bias matches the live single-timeframe evaluator on random windows."""
    df = load(pair, "M15")
    names = {"BULLISH": 2, "BEARISH": -2, "LEAN_BULLISH": 1, "LEAN_BEARISH": -1, "NEUTRAL": 0}
    rng = np.random.default_rng(0)
    mismatches = 0
    for end in rng.integers(260, len(df), samples):
        window = df.iloc[end - 250:end][["time", "open", "high", "low", "close"]].reset_index(drop=True)
        live = names[TechnicalAnalyzer.evaluate_single_timeframe(window)["bias"]]
        vec = int(bias_series(window)["bias"].iloc[-1])
        mismatches += live != vec
    print(f"self-check {pair}: {samples - mismatches}/{samples} windows match live evaluator")


# ---------------------------------------------------------------- simulation
def simulate(sig: pd.DataFrame, sl_mult: float, tp_mult: float, be_r: float, hours: tuple,
             spread_ratio: float = 0.30, lock_r: float = 0.1) -> List[Dict]:
    """One position per pair; returns trades with R-multiple results (R = initial stop distance)."""
    n = len(sig)
    t = sig["close_time"].to_numpy()
    s = sig["signal"].to_numpy()
    atr = sig["atr_14"].to_numpy()
    ao, bo = sig["ao"].to_numpy(), sig["bo"].to_numpy()
    bh, bl, ah, al = sig["bh"].to_numpy(), sig["bl"].to_numpy(), sig["ah"].to_numpy(), sig["al"].to_numpy()
    bc, ac = sig["bc"].to_numpy(), sig["ac"].to_numpy()
    hour = ((t % 86400) // 3600).astype(int)
    h0, h1 = hours
    trades = []
    i = 0
    while i < n - 1:
        if s[i] == 0 or not (h0 <= hour[i] < h1) or not atr[i] > 0:
            i += 1
            continue
        j = i + 1  # Enter at next bar open
        spread = ao[j] - bo[j]
        if spread > spread_ratio * atr[i]:
            i += 1
            continue
        direction = s[i]
        entry = ao[j] if direction > 0 else bo[j]
        risk = atr[i] * sl_mult
        sl = entry - direction * risk
        tp = entry + direction * atr[i] * tp_mult
        moved = False
        exit_px, k = None, j
        while k < n:
            if direction > 0:
                if bl[k] <= sl:
                    exit_px = sl; break
                if bh[k] >= tp:
                    exit_px = tp; break
                if be_r > 0 and not moved and bh[k] - entry >= be_r * risk:
                    sl, moved = entry + lock_r * risk, True  # Applies from the next bar (conservative)
            else:
                if ah[k] >= sl:
                    exit_px = sl; break
                if al[k] <= tp:
                    exit_px = tp; break
                if be_r > 0 and not moved and entry - al[k] >= be_r * risk:
                    sl, moved = entry - lock_r * risk, True
            k += 1
        if exit_px is None:
            k = n - 1
            exit_px = bc[k] if direction > 0 else ac[k]
        r = direction * (exit_px - entry) / risk
        trades.append({"open": t[i], "close": t[k], "dir": int(direction), "r": float(r),
                       "entry": float(entry), "exit": float(exit_px), "spread_r": float(spread / risk)})
        i = k + 1  # Flat again after exit
    return trades


def stats(trades: List[Dict], years: float) -> Dict:
    if not trades:
        return {"n": 0, "per_month": 0, "win": 0, "exp_r": 0, "pf": 0, "total_r": 0, "max_dd_r": 0, "max_losing": 0}
    r = np.array([x["r"] for x in trades])
    wins, losses = r[r > 0], r[r < 0]
    equity = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.concatenate([[0], equity]))[1:] - equity).max())
    streak = best = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    return {
        "n": len(r),
        "per_month": round(len(r) / (years * 12), 1),
        "win": round(float((r > 0).mean()), 3),
        "exp_r": round(float(r.mean()), 3),
        "pf": round(float(wins.sum() / -losses.sum()), 2) if len(losses) else float("inf"),
        "total_r": round(float(r.sum()), 1),
        "max_dd_r": round(dd, 1),
        "max_losing": best,
    }


def split(trades: List[Dict], cut: float):
    return [x for x in trades if x["open"] < cut], [x for x in trades if x["open"] >= cut]


def period(sig_by_pair: Dict[str, pd.DataFrame]):
    start = max(df["close_time"].iloc[260] for df in sig_by_pair.values())
    end = min(df["close_time"].iloc[-1] for df in sig_by_pair.values())
    cut = start + (end - start) * IS_FRACTION
    return start, cut, end


def trim(trades, start):
    return [x for x in trades if x["open"] >= start]


def compound(trades: List[Dict], risk_pct: float, start_equity: float = 150.0) -> Dict:
    """Equity path at fixed-fractional risk, trades applied in close-time order."""
    eq, peak, max_dd = start_equity, start_equity, 0.0
    for x in sorted(trades, key=lambda x: x["close"]):
        eq *= 1 + risk_pct * x["r"]
        peak = max(peak, eq)
        max_dd = max(max_dd, 1 - eq / peak)
    return {"final": round(eq, 2), "return_pct": round((eq / start_equity - 1) * 100, 1), "max_dd_pct": round(max_dd * 100, 1)}


# ---------------------------------------------------------------- live journal evaluation
NOT_TRADED = ("BLOCKED", "SHADOW:BLOCKED", "PAUSED", "SHADOW:PAUSED")  # The agent would not have entered these


def _replay(frame_after: pd.DataFrame, r: Dict) -> Optional[Dict]:
    """Replays one journaled entry on the M15 bid/ask path after its signal bar, with the live exit rules."""
    if len(frame_after) < 2 or not r["atr"]:
        return None
    frame = frame_after.assign(close_time=frame_after["time"] + 900, signal=0, atr_14=r["atr"])
    signal_row = frame.iloc[[0]].copy()
    signal_row["signal"] = 1 if r["direction"] == "BUY" else -1
    frame = pd.concat([signal_row, frame], ignore_index=True)  # simulate() enters on the bar after the signal row
    from config import settings
    tr = simulate(frame, r["sl_atr"] or settings.ATR_SL_MULTIPLIER, r["tp_atr"] or settings.ATR_TP_MULTIPLIER,
                  settings.BREAKEVEN_TRIGGER_R, (0, 24), spread_ratio=settings.MAX_SPREAD_ATR_RATIO,
                  lock_r=settings.BREAKEVEN_LOCK_R)
    if not tr:
        return None  # Spread filter rejected the entry
    still_open = tr[0]["close"] >= frame["close_time"].iloc[-1]
    return {**tr[0], "still_open": still_open}


def cmd_journal():
    """
    Forward-test of live signals: replays journaled rule signals / AI plans against the M15 bid/ask path that
    followed them, like the live agent would trade them: ONE position per pair at a time (the journal re-logs the
    same idea every bar, which must not be counted as separate trades), spread filter, break-even stop, and entries
    the agent blocked (news risk, spread, paused) excluded from the "would have traded" lines.
    Also prints the real closed trades from the broker, which is the ground truth.
    """
    from broker import OandaBroker
    from db import SignalJournal

    rows = SignalJournal().fetch_all()
    if not rows:
        print("Journal is empty (needs DATABASE_URL and some live BUY/SELL signals).")
        return
    broker = OandaBroker()
    first = min(r["bar_time"] for r in rows)
    paths = {}
    for r in rows:
        if r["instrument"] not in paths:
            paths[r["instrument"]] = fetch_candles(broker, r["instrument"], "M15", first - 3600, time.time())

    def evaluate(subset: List[Dict]) -> pd.DataFrame:
        """Sequential per pair: a signal is skipped while the previous simulated trade on that pair is open."""
        out, busy_until = [], {}
        for r in sorted(subset, key=lambda x: x["bar_time"]):
            pair = r["instrument"]
            entry_time = r["bar_time"] + 900
            if entry_time < busy_until.get(pair, 0):
                continue
            c = paths[pair]
            res = _replay(c[c["time"] >= entry_time].reset_index(drop=True), r)
            if res is None:
                continue
            busy_until[pair] = res["close"]
            out.append({**r, **res})
        return pd.DataFrame(out)

    def tradable(r):
        return not str(r["decision"]).startswith(NOT_TRADED)

    def show(label, df):
        if df is None or df.empty:
            print(f"  {label:<46} n=0")
            return
        done = df[~df["still_open"]]
        if done.empty:
            print(f"  {label:<46} n=0 (all {len(df)} still open)")
            return
        r = done["r"]
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
        print(f"  {label:<46} n={len(done):<4} win={(r > 0).mean():.2f} expR={r.mean():+.3f} "
              f"totalR={r.sum():+.1f} PF={pf:.2f}" + (f"  ({len(df) - len(done)} open)" if len(df) > len(done) else ""))
        return done

    is_ai = lambda r: str(r["entry_tf"]).startswith("AI")
    rules = [r for r in rows if not is_ai(r)]
    ai = [r for r in rows if is_ai(r)]
    print(f"Journal rows: {len(rows)} ({len(ai)} AI, {len(rules)} rules); duplicates of an already-open idea are skipped.\n")

    print("RULES signals (simulated, one position per pair)")
    show("all technical signals", evaluate(rules))
    show("would have traded (not blocked)", evaluate([r for r in rules if tradable(r)]))

    print("\nAI TRADER proposals (simulated, one position per pair)")
    show("all proposals", evaluate(ai))
    traded_ai = [r for r in ai if tradable(r)]
    done = show("would have traded (not blocked)", evaluate(traded_ai))
    for lo in (0.7, 0.8):
        show(f"would have traded, confidence >= {lo}", evaluate([r for r in traded_ai if (r["confidence"] or 0) >= lo]))

    print("\nREAL CLOSED TRADES on the account since the journal started (ground truth)")
    try:
        real = [t for t in broker.get_closed_trades(500) if float(t.get("close_time") or 0) >= first]
        if real:
            pnl = np.array([float(t["realized_pnl"]) for t in real])
            print(f"  n={len(real)} win={(pnl > 0).mean():.2f} total=${pnl.sum():+.2f} "
                  f"avg=${pnl.mean():+.2f} best=${pnl.max():+.2f} worst=${pnl.min():+.2f}")
        else:
            print("  none")
    except Exception as e:
        print(f"  could not fetch: {e}")

    n_done = 0 if done is None else len(done)
    print(f"\nVerdict guide: need >= 50 resolved 'would have traded' AI trades with expR clearly > +0.10 and PF > 1.2,"
          f" AND real trades not contradicting it, before going live. Currently {n_done} resolved.")


# ---------------------------------------------------------------- commands
GRID = {
    "sl": [1.0, 1.5, 2.0, 3.0],
    "tp": [1.5, 2.5, 3.5, 5.0],
    "be": [0.0, 1.0],
    "hours": [(0, 24), (7, 17), (12, 17)],
}


def cmd_optimize(pairs: List[str], out: Optional[str], tfs=DEFAULT_TFS):
    self_check()
    sigs = {p: build_signals(p, tfs) for p in pairs}
    print(f"Timeframes: {','.join(tfs)}")
    start, cut, end = period(sigs)
    is_years, oos_years = (cut - start) / 31557600, (end - cut) / 31557600
    print(f"\nIn-sample {pd.to_datetime(start, unit='s'):%Y-%m-%d} -> {pd.to_datetime(cut, unit='s'):%Y-%m-%d} "
          f"({is_years:.2f}y) | Out-of-sample -> {pd.to_datetime(end, unit='s'):%Y-%m-%d} ({oos_years:.2f}y)\n")

    results = []
    for sl, tp, be, hours in itertools.product(*GRID.values()):
        per_pair = {p: trim(simulate(sigs[p], sl, tp, be, hours), start) for p in pairs}
        row = {"sl": sl, "tp": tp, "be": be, "hours": f"{hours[0]}-{hours[1]}", "pairs": {}}
        for p, tr in per_pair.items():
            a, b = split(tr, cut)
            row["pairs"][p] = {"is": stats(a, is_years), "oos": stats(b, oos_years)}
        all_is, all_oos = split([x for tr in per_pair.values() for x in tr], cut)
        row["is"], row["oos"] = stats(all_is, is_years), stats(all_oos, oos_years)
        results.append(row)

    # Rank on in-sample only, requiring a meaningful sample; OOS is reported, never used for selection
    eligible = [r for r in results if r["is"]["n"] >= 60]
    eligible.sort(key=lambda r: r["is"]["exp_r"] * np.sqrt(r["is"]["n"]), reverse=True)
    baseline = next(r for r in results if r["sl"] == 1.5 and r["tp"] == 2.5 and r["be"] == 1.0 and r["hours"] == "0-24")
    if not eligible:
        print("No configuration produced >= 60 in-sample trades.")
        return

    def line(r):
        i, o = r["is"], r["oos"]
        return (f"SL {r['sl']:<3} TP {r['tp']:<3} BE {r['be']:<3} hrs {r['hours']:<5} | "
                f"IS n={i['n']:<4} win={i['win']:.2f} expR={i['exp_r']:+.3f} PF={i['pf']} | "
                f"OOS n={o['n']:<4} win={o['win']:.2f} expR={o['exp_r']:+.3f} PF={o['pf']} DD={o['max_dd_r']}R streak={o['max_losing']}")

    print("CURRENT LIVE SETTINGS (all pairs):\n  " + line(baseline))
    print("\nTOP 10 BY IN-SAMPLE (all pairs):")
    for r in eligible[:10]:
        print("  " + line(r))
    best = eligible[0]
    print(f"\nPER-PAIR for best config (SL {best['sl']} TP {best['tp']} BE {best['be']} hrs {best['hours']}):")
    for p, v in best["pairs"].items():
        i, o = v["is"], v["oos"]
        print(f"  {p:8} IS n={i['n']:<4} expR={i['exp_r']:+.3f} PF={i['pf']:<5} | OOS n={o['n']:<4} expR={o['exp_r']:+.3f} PF={o['pf']}")
    print("\nPER-PAIR for current live settings:")
    for p, v in baseline["pairs"].items():
        i, o = v["is"], v["oos"]
        print(f"  {p:8} IS n={i['n']:<4} expR={i['exp_r']:+.3f} PF={i['pf']:<5} | OOS n={o['n']:<4} expR={o['exp_r']:+.3f} PF={o['pf']}")
    if out:
        Path(out).write_text(json.dumps({"start": start, "cut": cut, "end": end, "results": results}, default=str))


def cmd_run(pairs, sl, tp, be, hours, risks, tfs=DEFAULT_TFS):
    sigs = {p: build_signals(p, tfs) for p in pairs}
    start, cut, end = period(sigs)
    oos_years = (end - cut) / 31557600
    all_tr = [x for p in pairs for x in trim(simulate(sigs[p], sl, tp, be, hours), start)]
    _, oos = split(all_tr, cut)
    print(f"Config SL {sl} TP {tp} BE {be} hrs {hours} pairs {','.join(pairs)}")
    print("Full period:", stats(all_tr, (end - start) / 31557600))
    print("Out-of-sample:", stats(oos, oos_years))
    for rp in risks:
        print(f"  OOS compounding @ {rp * 100:.1f}% risk from $150 over {oos_years:.2f}y:", compound(oos, rp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "optimize", "run", "check", "journal"])
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--sl", type=float, default=1.5)
    ap.add_argument("--tp", type=float, default=2.5)
    ap.add_argument("--be", type=float, default=1.0)
    ap.add_argument("--hours", default="0-24")
    ap.add_argument("--risks", default="0.01,0.02")
    ap.add_argument("--out")
    ap.add_argument("--tfs", default=",".join(DEFAULT_TFS), help="entry,confirm,trend timeframes")
    a = ap.parse_args()
    pairs = [p.strip().upper() for p in a.pairs.split(",") if p.strip()]
    tfs = tuple(x.strip().upper() for x in a.tfs.split(","))
    if a.cmd == "fetch":
        cmd_fetch(pairs, list(tfs))
    elif a.cmd == "journal":
        cmd_journal()
    elif a.cmd == "check":
        self_check()
    elif a.cmd == "optimize":
        cmd_optimize(pairs, a.out, tfs)
    else:
        h = tuple(int(x) for x in a.hours.split("-"))
        cmd_run(pairs, a.sl, a.tp, a.be, h, [float(x) for x in a.risks.split(",")], tfs)


if __name__ == "__main__":
    sys.exit(main())
