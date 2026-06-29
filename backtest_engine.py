"""
BACKTEST ENGINE — REAL SIGNAL REPLAY
=====================================
Replays ACTUAL signals from signals/all_signals.json (or outcomes.parquet)
against historical bar data and reports:

  • Win rate / P&L by probability bucket  ← calibration check
  • Equity curve and max drawdown
  • Profit factor, Sharpe ratio
  • Per-strategy breakdown (REVERSION vs CONTINUATION)

TRADE LIFECYCLE (correct order of operations)
---------------------------------------------
  1. WAIT FOR ENTRY
     Scan forward from the bar AFTER signal generation.
     A BUY limit fills when bar.low  <= entry price.
     A SELL limit fills when bar.high >= entry price.
     If entry is never touched within max_horizon_bars → EXPIRED (no P&L).

  2. POST-ENTRY: WAIT FOR TP OR SL
     Once entry is filled, scan forward from the NEXT bar after fill.
     WIN  : bar.high >= TP  (BUY)  |  bar.low  <= TP  (SELL)
     LOSS : bar.low  <= SL  (BUY)  |  bar.high >= SL  (SELL)
     The entry bar itself is EXCLUDED from SL checks (fill occurred first).
     If neither TP nor SL hit within max_horizon_bars from entry → PENDING.

  3. EXPIRED ≠ LOSS
     Expired means price never reached the entry zone — no trade opened,
     no money at risk, no P&L impact.

CHANGE LOG
----------
v4 (current):
  - FIX-A: outputsize raised 500 → 5000 bars (~5 days → ~52 days of 15min data).
  - FIX-B: signal_time prefers frozen 'created_at' over refreshed 'timestamp'.
  - FIX-C: default max_horizon_bars raised 96 → 192 (24h → 48h per phase).
  - FIX-D: gap_hours threshold raised 2h → 6h to reduce redundant API fetches.
  - FIX-E: PRICE-SIDE GUARD — uses first forward bar open as signal-time price
    proxy. BUY zones above that price and SELL zones below it are immediately
    marked expired. Fixes phantom instant-fill losses where SELL entry was
    below current price and bar.high >= entry was trivially true on bar#0.
  - FIX-F: SELL fill logic comment clarified. bar.high >= entry is correct for
    a SELL limit (waiting for price to RISE to supply zone). FIX-E ensures the
    guard runs first so this only fires when the zone is genuinely above price.

v2/v3:
  - Horizon counter fixed to count from signal bar not bar index 0.
  - Post-entry horizon counted separately from entry-scan horizon.
  - Fill bar excluded from SL check for both BUY and SELL.
  - Bar-count logging added to trade record.

Usage
-----
    python backtest_engine.py                           # signals + live TwelveData bars
    python backtest_engine.py --source outcomes         # pre-resolved outcomes.parquet
    python backtest_engine.py --balance 5000 --risk 2   # custom capital / risk %
    python backtest_engine.py --horizon 192             # 192 bars = 48h on 15min (default)
"""

import json
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
SIGNALS_FILE  = Path("signals/all_signals.json")
OUTCOMES_FILE = Path("data/outcomes/outcomes.parquet")
BARS_FILE     = Path("data/bars_15min.parquet")
RESULTS_DIR   = Path("data/backtest_results")

PIP_SIZE = 0.0001   # default — overridden per-symbol in position sizing

# Per-symbol pip sizes — used in run_from_outcomes() and _position_size()
PIP_SIZE_MAP = {
    "EUR/USD": 0.0001, "EURUSD": 0.0001,
    "GBP/USD": 0.0001, "GBPUSD": 0.0001,
    "USD/JPY": 0.01,   "USDJPY": 0.01,
    "XAU/USD": 1.0,    "XAUUSD": 1.0,
}

# Kenya = UTC+3 (EAT). All internal processing uses UTC.
EAT_OFFSET_HOURS = 3

def to_eat(dt) -> str:
    """Convert UTC datetime/string → EAT string for display."""
    import pandas as _pd
    try:
        ts = _pd.to_datetime(dt)
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        eat = ts + _pd.Timedelta(hours=EAT_OFFSET_HOURS)
        return eat.strftime("%Y-%m-%d %H:%M EAT")
    except Exception:
        return str(dt)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_signals_from_json() -> pd.DataFrame:
    """Load raw signals from signals/all_signals.json."""
    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(f"Signals file not found: {SIGNALS_FILE}")

    with open(SIGNALS_FILE) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]

    df = pd.DataFrame(raw)
    # Keep only active signals that have the required fields
    required = {"direction", "entry", "sl", "tp", "probability"}
    df = df.dropna(subset=list(required & set(df.columns)))

    # Normalise direction
    df["direction"] = df["direction"].str.upper()

    # Parse timestamp — prefer created_at (frozen at generation) over timestamp
    if "created_at" in df.columns:
        df["signal_time"] = pd.to_datetime(
            df["created_at"].fillna(df.get("timestamp", pd.NaT)), errors="coerce"
        )
    else:
        df["signal_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["signal_time"].dt.tz is not None:
        df["signal_time"] = df["signal_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        
    else:
        df["signal_time"] = df["signal_time"].dt.tz_localize(None)
    df = df.dropna(subset=["signal_time"])
    df = df.sort_values("signal_time").reset_index(drop=True)

    # Normalise probability to 0–1
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").fillna(50)
    if df["probability"].max() > 1.5:          # stored as 0–100
        df["probability"] = df["probability"] / 100.0

    # Deduplicate — same entry/sl/tp = duplicate signal; keep highest probability
    before = len(df)
    df = (df.sort_values("probability", ascending=False)
            .drop_duplicates(subset=["direction", "entry", "sl", "tp"], keep="first")
            .sort_values("signal_time")
            .reset_index(drop=True))
    if len(df) < before:
        logger.info(f"Removed {before - len(df)} duplicate signals → {len(df)} unique")
    else:
        logger.info(f"Loaded {len(df)} signals from {SIGNALS_FILE}")
    return df


def load_outcomes() -> pd.DataFrame:
    """Load pre-resolved outcomes from outcome_tracker."""
    if not OUTCOMES_FILE.exists():
        raise FileNotFoundError(
            f"Outcomes file not found: {OUTCOMES_FILE}\n"
            "Run outcome_tracker.py first."
        )
    df = pd.read_parquet(OUTCOMES_FILE)
    logger.info(f"Loaded {len(df)} outcomes from {OUTCOMES_FILE}")
    return df


def fetch_bars_twelvedata(symbol: str = "EUR/USD",
                          interval: str = "15min",
                          outputsize: int = 5000) -> Optional[pd.DataFrame]:
    """
    Fetch recent bars from TwelveData API.
    Used when local bar file is stale or does not cover signal dates.
    """
    import os, requests as _req
    env_file = Path(".env")
    if env_file.exists() and not os.getenv("TWELVE_API_KEY"):
        for line in env_file.read_text().splitlines():
            line = line.strip().lstrip("export").strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    api_key = os.getenv("TWELVE_API_KEY", "152db33cd9e64bd4976cbbc0264bc753")
    params = {
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     api_key,
        "format":     "JSON",
    }
    try:
        r = _req.get("https://api.twelvedata.com/time_series",
                     params=params, timeout=30)
        data = r.json()
    except Exception as exc:
        logger.warning(f"TwelveData fetch failed: {exc}")
        return None

    if "values" not in data:
        logger.warning(f"TwelveData returned no values: {data.get('message','')}")
        return None

    rows = []
    for c in data["values"]:
        try:
            rows.append({
                "time":  pd.to_datetime(c["datetime"]),
                "open":  float(c["open"]),
                "high":  float(c["high"]),
                "low":   float(c["low"]),
                "close": float(c["close"]),
            })
        except Exception:
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    logger.info(f"Fetched {len(df)} bars from TwelveData "
                f"({df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})")
    return df


def load_bars(signals_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """
    Load 15-min bar data for all unique symbols in signals_df.
    If signals_df is None, loads bars for all standard symbols: EURUSD, GBPUSD, USDJPY, XAUUSD.
    For each symbol, if the local file is stale or missing, automatically fetches live bars via TwelveData.
    """
    symbols = []
    if signals_df is not None and "symbol" in signals_df.columns:
        symbols = [s for s in signals_df["symbol"].dropna().unique() if str(s).strip()]
    
    if not symbols:
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]

    bars_by_sym = {}

    for raw_sym in symbols:
        sym = raw_sym.upper().replace("/", "")
        # Format the twelve data query symbol (with slash, except gold which TwelveData usually expects as XAU/USD)
        query_sym = raw_sym
        if "/" not in query_sym:
            if query_sym in ("EURUSD", "GBPUSD", "USDJPY", "XAUUSD"):
                query_sym = f"{query_sym[:3]}/{query_sym[3:]}"
        
        local_df = None
        # Try finding local file bars_15min_{sym}.parquet or bars_15min_{raw_sym}.parquet
        for filename in [f"bars_15min_{sym}.parquet", f"bars_15min_{raw_sym}.parquet"]:
            p = Path("data") / filename
            if p.exists():
                try:
                    local_df = pd.read_parquet(p)
                    if "time" not in local_df.columns:
                        local_df = local_df.reset_index().rename(columns={"index": "time"})
                    local_df["time"] = pd.to_datetime(local_df["time"])
                    local_df = local_df.sort_values("time").reset_index(drop=True)
                    logger.info(f"Local bars for {sym}: {len(local_df)} rows "
                                f"({local_df['time'].iloc[0].date()} → "
                                f"{local_df['time'].iloc[-1].date()})")
                    break
                except Exception as exc:
                    logger.warning(f"Could not read local bars file {p}: {exc}")

        needs_live = True
        if local_df is not None:
            if signals_df is not None and "signal_time" in signals_df.columns:
                # filter signals for this symbol
                sym_signals = signals_df[signals_df["symbol"].str.upper().str.replace("/", "") == sym]
                if not sym_signals.empty:
                    latest_signal = pd.to_datetime(sym_signals["signal_time"]).max()
                    latest_local  = local_df["time"].max()
                    gap_hours = (latest_signal - latest_local).total_seconds() / 3600
                    if gap_hours <= 6:
                        needs_live = False
                        logger.info(f"Local bars for {sym} cover signal dates — skipping TwelveData fetch")
                    else:
                        logger.warning(
                            f"Local bars for {sym} end {latest_local.date()} but latest signal is "
                            f"{latest_signal.date()} ({gap_hours:.0f}h gap) — fetching live bars"
                        )
                else:
                    needs_live = False  # No signals for this symbol, no need to update
            else:
                needs_live = False  # No signals dataframe provided to check freshness

        if needs_live:
            live_df = fetch_bars_twelvedata(symbol=query_sym, outputsize=5000)
            if live_df is not None:
                if local_df is not None:
                    combined = pd.concat([local_df, live_df], ignore_index=True)
                    combined = (combined
                                .drop_duplicates(subset=["time"])
                                .sort_values("time")
                                .reset_index(drop=True))
                    logger.info(f"Combined bars for {sym}: {len(combined)} rows")
                    bars_by_sym[sym] = combined
                else:
                    bars_by_sym[sym] = live_df
            elif local_df is not None:
                logger.warning(f"TwelveData fetch failed for {sym} — falling back to local bars")
                bars_by_sym[sym] = local_df
            else:
                logger.warning(f"No local bars for {sym} and TwelveData fetch failed. Skipping this symbol.")
        elif local_df is not None:
            bars_by_sym[sym] = local_df

    if not bars_by_sym:
        raise FileNotFoundError(
            "No local bars found and TwelveData fetch failed for all symbols. "
            "Verify TwelveData API key and that tick data has been resampled."
        )

    return bars_by_sym


# ─────────────────────────────────────────────────────────────────────────────
# Trade simulation
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Replays real signals against historical bars with correct limit-order logic.

    ENTRY MODEL
    -----------
    All signals are treated as LIMIT ORDERS — price must pull back to the zone
    (entry level) before the trade opens. This matches how SMC zones work in
    practice: you identify a zone and wait for price to return to it.

    HORIZON MODEL
    -------------
    max_horizon_bars applies to each phase independently:
      Phase 1 — entry scan : up to max_horizon_bars bars to find entry fill
      Phase 2 — TP/SL scan : up to max_horizon_bars bars AFTER entry to resolve

    This is intentional. On a 15min chart with max_horizon_bars=96 (24h):
      - You wait up to 24h for price to reach your zone
      - Once filled, you hold up to 24h for TP or SL

    The total maximum time per signal is therefore 2 × 24h = 48h.
    Adjust max_horizon_bars via CLI --horizon flag.
    """

    def __init__(self,
                 initial_balance: float = 10_000,
                 risk_per_trade_pct: float = 1.0,
                 slippage_pips: float = 0.3,
                 commission_pips: float = 0.0,
                 max_horizon_bars: int = 192,
                 max_concurrent: int = 3,
                 pip_size: float = PIP_SIZE):

        self.initial_balance    = initial_balance
        self.risk_per_trade_pct = risk_per_trade_pct / 100.0
        self.slippage_pips      = slippage_pips
        self.commission_pips    = commission_pips
        self.max_horizon_bars   = max_horizon_bars
        self.max_concurrent     = max_concurrent
        self.pip_size           = pip_size

        self.balance      = initial_balance
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    # ── position sizing ────────────────────────────────────────────────────────

    def _position_size(self, entry: float, sl: float, symbol: str = "") -> float:
        """Fixed-fractional lot size: risk = risk_per_trade_pct × balance."""
        risk_amount = self.balance * self.risk_per_trade_pct
        pip_size    = PIP_SIZE_MAP.get(symbol.upper(), self.pip_size)
        risk_pips   = abs(entry - sl) / pip_size
        if risk_pips <= 0:
            return 0.01
        value_per_pip = 10.0           # $10/pip for 1 standard lot (EUR/USD)
        size = risk_amount / (risk_pips * value_per_pip)
        return max(0.01, round(size, 2))

    # ── single trade simulation ────────────────────────────────────────────────

    def _simulate(self, sig: Dict, forward_bars: pd.DataFrame) -> Optional[Dict]:
        """
        Simulate a single limit-order trade against forward bar data.

        Parameters
        ----------
        sig          : signal dict with direction, entry, sl, tp, probability
        forward_bars : bars AFTER signal generation time, reset_index'd to 0-based

        PHASE 1 — ENTRY SCAN
        --------------------
        Walk up to max_horizon_bars bars looking for the limit fill:
          BUY  limit: fills when bar.low  <= entry  (price drops to zone)
          SELL limit: fills when bar.high >= entry  (price rises to zone)

        If no fill in max_horizon_bars → outcome = 'expired', no P&L.

        PHASE 2 — OUTCOME SCAN
        ----------------------
        Walk up to max_horizon_bars bars FROM THE FILL BAR:
          WIN  (BUY):  bar.high >= TP
          WIN  (SELL): bar.low  <= TP
          LOSS (BUY):  bar.low  <= SL  (not checked on fill bar itself)
          LOSS (SELL): bar.high >= SL  (not checked on fill bar itself)

        The fill bar is excluded from SL checks — on that bar price already
        touched entry (the fill level), so checking SL on the same bar would
        simulate an instantaneous SL hit which doesn't reflect reality for a
        limit order where fill happened at the proximal edge.

        If neither TP nor SL hit in max_horizon_bars → outcome = 'pending'.
        """
        direction = str(sig.get("direction", "BUY")).upper()
        entry_raw = float(sig["entry"])
        sl        = float(sig["sl"])
        tp        = float(sig["tp"])
        symbol    = str(sig.get("symbol", ""))
        pip_size  = PIP_SIZE_MAP.get(symbol.upper(), self.pip_size)

        # Apply slippage to fill price (adverse for the trader)
        slip  = self.slippage_pips * pip_size
        entry = entry_raw + slip if direction == "BUY" else entry_raw - slip

        # PRICE-SIDE GUARD
        # The first bar's open is the best proxy for price at signal generation time.
        # BUY  limit: entry must be BELOW signal price (zone is below, wait for drop)
        # SELL limit: entry must be ABOVE signal price (zone is above, wait for rise)
        # If this check fails the zone is already behind price → expired immediately.
        if len(forward_bars) > 0:
            signal_open = float(forward_bars.iloc[0]["open"])
            if direction == "BUY" and entry >= signal_open:
                return {
                    "direction": direction, "entry": round(entry, 5),
                    "exit_price": None, "sl": sl, "tp": tp,
                    "outcome": "expired", "pips": 0.0, "pnl": 0.0,
                    "balance": round(self.balance, 2), "bars_to_entry": None,
                    "bars_held": 0, "pos_size": 0.0,
                    "probability": float(sig.get("probability", 0.5)),
                    "strategy": sig.get("strategy", ""),
                    "signal_time": sig.get("signal_time", sig.get("timestamp", "")),
                    "signal_time_eat": to_eat(sig.get("signal_time", sig.get("timestamp", ""))),
                }
            if direction == "SELL" and entry <= signal_open:
                return {
                    "direction": direction, "entry": round(entry, 5),
                    "exit_price": None, "sl": sl, "tp": tp,
                    "outcome": "expired", "pips": 0.0, "pnl": 0.0,
                    "balance": round(self.balance, 2), "bars_to_entry": None,
                    "bars_held": 0, "pos_size": 0.0,
                    "probability": float(sig.get("probability", 0.5)),
                    "strategy": sig.get("strategy", ""),
                    "signal_time": sig.get("signal_time", sig.get("timestamp", "")),
                    "signal_time_eat": to_eat(sig.get("signal_time", sig.get("timestamp", ""))),
                }

        # ── PHASE 1: Scan for entry fill ──────────────────────────────────────
        # Walk at most max_horizon_bars bars from signal time.
        # Use enumerate() so bar_num is always a clean 0-based counter
        # regardless of what the DataFrame index contains.
        entry_bar_num = None

        for bar_num, (_, row) in enumerate(forward_bars.iterrows()):
            if bar_num >= self.max_horizon_bars:
                break   # hit horizon without fill → expired

            # LIMIT ORDER FILL LOGIC (SMC zone-based):
            # BUY  limit → price drops DOWN to demand zone → fill when bar.low  <= entry
            # SELL limit → price rises UP  to supply zone → fill when bar.high >= entry
            # Both directions: zone must be on the correct side of price at signal time.
            if direction == "BUY" and row["low"] <= entry:
                entry_bar_num = bar_num
                break
            if direction == "SELL" and row["high"] >= entry:
                entry_bar_num = bar_num
                break

        # No fill within horizon → EXPIRED (no trade, no P&L)
        if entry_bar_num is None:
            return {
                "direction":       direction,
                "entry":           round(entry, 5),
                "exit_price":      None,
                "sl":              sl,
                "tp":              tp,
                "outcome":         "expired",
                "pips":            0.0,
                "pnl":             0.0,
                "balance":         round(self.balance, 2),
                "bars_to_entry":   None,
                "bars_held":       0,
                "pos_size":        0.0,
                "probability":     float(sig.get("probability", 0.5)),
                "strategy":        sig.get("strategy", ""),
                "signal_time":     sig.get("signal_time", sig.get("timestamp", "")),
                "signal_time_eat": to_eat(sig.get("signal_time", sig.get("timestamp", ""))),
            }

        # ── PHASE 2: Scan for TP or SL from fill bar onward ──────────────────
        fill_row = forward_bars.iloc[entry_bar_num]
        logger.info(
            f"  FILL [{direction}] bar#{entry_bar_num} "
            f"H={fill_row['high']:.5f} L={fill_row['low']:.5f} | "
            f"entry={entry:.5f} sl={sl:.5f} tp={tp:.5f} | "
            f"risk={abs(entry-sl)/pip_size:.1f}p "
            f"reward={abs(tp-entry)/pip_size:.1f}p"
        )

        # Sanity check: if SL is on the wrong side of entry the trade geometry
        # is inverted — skip it rather than produce a guaranteed instant loss.
        if direction == "BUY" and sl >= entry:
            logger.warning(
                f"  SKIPPED [{direction}]: SL {sl:.5f} >= entry {entry:.5f} "
                "— inverted geometry from signal generator. Check production_predictor."
            )
            return None
        if direction == "SELL" and sl <= entry:
            logger.warning(
                f"  SKIPPED [{direction}]: SL {sl:.5f} <= entry {entry:.5f} "
                "— inverted geometry from signal generator. Check production_predictor."
            )
            return None

        pos_size   = self._position_size(entry, sl, symbol=sig.get("symbol", ""))
        outcome    = "pending"
        # Default exit at close of last bar in the post-entry window
        post_entry = forward_bars.iloc[entry_bar_num:]
        last_bar_idx = min(self.max_horizon_bars - 1, len(post_entry) - 1)
        exit_price   = float(post_entry.iloc[last_bar_idx]["close"])
        bars_held    = 0

        for bars_held_count, (_, row) in enumerate(post_entry.iterrows()):
            if bars_held_count >= self.max_horizon_bars:
                bars_held = bars_held_count
                break

            is_fill_bar = (bars_held_count == 0)
            bars_held   = bars_held_count + 1

            if direction == "BUY":
                # TP check: can hit on fill bar (strong bullish candle)
                if row["high"] >= tp:
                    outcome    = "win"
                    exit_price = tp - slip
                    break
                # SL check: SKIP on fill bar to avoid same-candle false loss
                if not is_fill_bar and row["low"] <= sl:
                    outcome    = "loss"
                    exit_price = sl + slip
                    break

            else:  # SELL
                # TP check: can hit on fill bar (strong bearish candle)
                if row["low"] <= tp:
                    outcome    = "win"
                    exit_price = tp + slip
                    break
                # SL check: SKIP on fill bar
                if not is_fill_bar and row["high"] >= sl:
                    outcome    = "loss"
                    exit_price = sl - slip
                    break

        # ── P&L (only for filled+closed trades) ──────────────────────────────
        if direction == "BUY":
            pips = (exit_price - entry) / pip_size
        else:
            pips = (entry - exit_price) / pip_size
        pips -= self.commission_pips

        pnl = pips * 10.0 * pos_size if outcome in ("win", "loss") else 0.0
        if outcome in ("win", "loss"):
            self.balance += pnl

        return {
            "direction":       direction,
            "entry":           round(entry, 5),
            "exit_price":      round(exit_price, 5) if exit_price is not None else None,
            "sl":              sl,
            "tp":              tp,
            "outcome":         outcome,      # expired | win | loss | pending
            "pips":            round(pips, 1) if outcome in ("win", "loss") else 0.0,
            "pnl":             round(pnl, 2),
            "balance":         round(self.balance, 2),
            "bars_to_entry":   entry_bar_num,   # how many bars before fill
            "bars_held":       bars_held,        # how many bars post-fill
            "pos_size":        pos_size,
            "probability":     float(sig.get("probability", 0.5)),
            "strategy":        sig.get("strategy", ""),
            "signal_time":     sig.get("signal_time", sig.get("timestamp", "")),
            "signal_time_eat": to_eat(sig.get("signal_time", sig.get("timestamp", ""))),
        }

    # ── full backtest ──────────────────────────────────────────────────────────

    def run(self, signals: pd.DataFrame, bars: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Dict:
        """
        Main backtest loop.

        Parameters
        ----------
        signals : DataFrame from load_signals_from_json()
        bars    : DataFrame or Dict of DataFrames from load_bars()

        NOTE on forward bar slicing
        ---------------------------
        We slice bars to AFTER signal_time (strictly greater than) so we never
        use the bar the signal was generated within. On 15-min bars this means
        we start from the next complete candle — correct for a limit order
        placed at the current moment.
        """
        self.balance      = self.initial_balance
        self.trades       = []
        
        if isinstance(bars, dict):
            # Use the time from the first available DataFrame in the dictionary
            first_df = next(iter(bars.values()))
            start_time = first_df["time"].iloc[0] if not first_df.empty else datetime.utcnow()
        else:
            start_time = bars["time"].iloc[0] if not bars.empty else datetime.utcnow()
            
        self.equity_curve = [{"time": start_time, "balance": self.balance}]

        open_count = 0

        for _, sig in signals.iterrows():
            if open_count >= self.max_concurrent:
                open_count = max(0, open_count - 1)

            sig_time = pd.to_datetime(sig.get("signal_time",
                                               sig.get("timestamp", "")))
            if pd.isna(sig_time):
                continue

            symbol = sig.get("symbol", "EURUSD")
            
            # Look up correct bars DataFrame
            if isinstance(bars, dict):
                sym_key = symbol.upper().replace("/", "")
                sym_bars = None
                for k, df_b in bars.items():
                    if k.upper().replace("/", "") == sym_key:
                        sym_bars = df_b
                        break
                if sym_bars is None:
                    logger.warning(f"No bars found for symbol {symbol} in backtest")
                    continue
            else:
                sym_bars = bars

            # All bars strictly after the signal bar
            # reset_index so enumerate() gives clean 0-based bar counts
            forward = (sym_bars[sym_bars["time"] > sig_time]
                       .reset_index(drop=True))

            if len(forward) < 3:
                # Not enough bars to evaluate — mark pending rather than skip
                logger.debug(f"Skipping signal at {sig_time} — only {len(forward)} forward bars")
                continue

            trade = self._simulate(sig.to_dict(), forward)
            if trade is None:
                continue

            self.trades.append(trade)
            open_count += 1
            self.equity_curve.append({
                "time":    sig_time,
                "balance": self.balance,
            })

        return self._metrics()

    # ── from pre-resolved outcomes ─────────────────────────────────────────────

    def run_from_outcomes(self, outcomes: pd.DataFrame) -> Dict:
        """
        Build trade results directly from outcome_tracker output.
        No bar simulation needed — outcomes are already known.
        """
        self.balance      = self.initial_balance
        self.trades       = []
        self.equity_curve = [{"time": datetime.utcnow(), "balance": self.balance}]

        for _, row in outcomes.iterrows():
            direction   = str(row.get("direction", "BUY")).upper()
            entry       = float(row.get("entry", 0))
            sl          = float(row.get("sl", 0))
            exit_price  = float(row.get("exit_price", entry))
            outcome_str = str(row.get("outcome", "timeout"))
            symbol      = str(row.get("symbol", ""))

            # per-symbol pip size — falls back to row-level pip_size if present
            pip_size = float(row.get("pip_size", 0)) or PIP_SIZE_MAP.get(
                symbol.upper(), self.pip_size
            )

            if outcome_str == "win":
                outcome = "win"
            elif outcome_str == "loss":
                outcome = "loss"
            elif outcome_str == "expired":
                outcome = "expired"
            elif outcome_str == "pending":
                outcome = "pending"
            else:
                outcome = "timeout"

            if direction == "BUY":
                pips = (exit_price - entry) / pip_size
            else:
                pips = (entry - exit_price) / pip_size

            pos_size = self._position_size(entry, sl, symbol=symbol)
            pnl      = pips * 10.0 * pos_size
            self.balance += pnl

            self.trades.append({
                "direction":   direction,
                "entry":       entry,
                "exit_price":  exit_price,
                "sl":          sl,
                "tp":          float(row.get("tp", entry)),
                "outcome":     outcome,
                "pips":        round(pips, 1),
                "pnl":         round(pnl, 2),
                "balance":     round(self.balance, 2),
                "bars_to_entry": None,
                "bars_held":   int(row.get("bars_held", 0)),
                "pos_size":    pos_size,
                "probability": float(row.get("predicted_prob", 0.5)),
                "strategy":    row.get("strategy", ""),
                "signal_time": row.get("signal_timestamp", ""),
            })
            self.equity_curve.append({
                "time":    row.get("resolved_at", datetime.utcnow()),
                "balance": self.balance,
            })

        return self._metrics()

    # ── metrics ───────────────────────────────────────────────────────────────

    def _metrics(self) -> Dict:
        if not self.trades:
            logger.warning("No trades to calculate metrics.")
            return {}

        df       = pd.DataFrame(self.trades)
        total    = len(df)
        wins     = (df["outcome"] == "win").sum()
        losses   = (df["outcome"] == "loss").sum()
        expired  = (df["outcome"] == "expired").sum()
        pending  = (df["outcome"] == "pending").sum()
        filled   = wins + losses
        timeout  = (df["outcome"] == "timeout").sum()   # legacy

        win_rate = wins / filled if filled > 0 else 0

        total_pnl    = df["pnl"].sum()
        avg_win      = df[df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_loss     = df[df["pnl"] < 0]["pnl"].mean() if losses > 0 else 0
        gross_profit = df[df["pnl"] > 0]["pnl"].sum()
        gross_loss   = abs(df[df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        total_return = ((self.balance / self.initial_balance) - 1) * 100

        eq_df = pd.DataFrame(self.equity_curve)
        eq_df["peak"] = eq_df["balance"].cummax()
        eq_df["dd"]   = (eq_df["balance"] - eq_df["peak"]) / eq_df["peak"]
        max_dd = eq_df["dd"].min() * 100

        rets   = df["pnl"] / self.initial_balance
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

        calibration = self._calibration_table(df)

        return {
            "total_trades":     total,
            "wins":             int(wins),
            "losses":           int(losses),
            "expired":          int(expired),
            "pending":          int(pending),
            "filled":           int(filled),
            "win_rate":         round(win_rate, 4),
            "total_pnl":        round(total_pnl, 2),
            "avg_win":          round(avg_win, 2),
            "avg_loss":         round(avg_loss, 2),
            "profit_factor":    round(profit_factor, 2),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio":     round(sharpe, 2),
            "final_balance":    round(self.balance, 2),
            "calibration":      calibration,
            "trades_df":        df,
            "equity_curve":     eq_df,
        }

    def _calibration_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Win rate vs model probability — tells you if the model is calibrated."""
        filled_df = df[df["outcome"].isin(["win", "loss"])].copy()
        if filled_df.empty:
            return pd.DataFrame()

        bins   = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
        labels = ["<55%", "55-60%", "60-65%", "65-70%", "70-75%", "75%+"]

        filled_df["bucket"] = pd.cut(
            filled_df["probability"], bins=bins, labels=labels
        )

        tbl = (filled_df.groupby("bucket", observed=True)
               .apply(lambda x: pd.Series({
                   "signals":   len(x),
                   "win_rate":  round((x["outcome"] == "win").mean(), 3),
                   "avg_pips":  round(x["pips"].mean(), 1),
                   "avg_pnl":   round(x["pnl"].mean(), 2),
               }))
               .reset_index())
        return tbl


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(metrics: Dict, engine: BacktestEngine):
    if not metrics:
        print("No results to display.")
        return

    print("\n" + "═" * 70)
    print("  BACKTEST REPORT — REAL SIGNAL REPLAY")
    print("═" * 70)
    print(f"  Initial Balance   : ${engine.initial_balance:>10,.2f}")
    print(f"  Final Balance     : ${metrics['final_balance']:>10,.2f}")
    print(f"  Total Return      : {metrics['total_return_pct']:>+10.2f}%")
    print(f"  Max Drawdown      : {metrics['max_drawdown_pct']:>10.2f}%")
    print(f"  Sharpe Ratio      : {metrics['sharpe_ratio']:>10.2f}")
    print()
    filled  = metrics.get("filled", metrics["wins"] + metrics["losses"])
    expired = metrics.get("expired", 0)
    pending = metrics.get("pending", 0)
    print(f"  Total Signals     : {metrics['total_trades']:>10}")
    print(f"  ├─ Expired        : {expired:>10}  (price never reached entry within horizon)")
    print(f"  ├─ Filled         : {filled:>10}  (entry zone was touched)")
    print(f"  │   ├─ Wins  (TP) : {metrics['wins']:>10}  ({metrics['win_rate']:.1%} of filled)")
    print(f"  │   └─ Losses(SL) : {metrics['losses']:>10}")
    print(f"  └─ Pending        : {pending:>10}  (entry hit, TP/SL not reached yet)")
    print()
    print(f"  Avg Win           : ${metrics['avg_win']:>10.2f}")
    print(f"  Avg Loss          : ${metrics['avg_loss']:>10.2f}")
    print(f"  Profit Factor     : {metrics['profit_factor']:>10.2f}")
    print()
    print(f"  Horizon Setting   : {engine.max_horizon_bars} bars per phase")
    print(f"  (Phase 1: up to {engine.max_horizon_bars} bars to reach entry zone)")
    print(f"   Phase 2: up to {engine.max_horizon_bars} bars after entry to hit TP/SL)")
    print()

    # Calibration table
    cal = metrics.get("calibration")
    if cal is not None and not cal.empty:
        print("  ── CALIBRATION (win rate vs predicted probability) ──────────────")
        print(f"  {'Bucket':<10} {'Signals':>8} {'Win Rate':>10} {'Avg Pips':>10} {'Avg P&L':>10}")
        print("  " + "─" * 55)
        for _, row in cal.iterrows():
            wr     = row["win_rate"]
            bucket = row["bucket"]
            flag   = "✅" if wr >= 0.55 else "⚠️ "
            print(f"  {flag} {str(bucket):<8} {int(row['signals']):>8} "
                  f"{wr:>9.1%} {row['avg_pips']:>10.1f} ${row['avg_pnl']:>9.2f}")

        wrs        = cal["win_rate"].dropna().tolist()
        calibrated = len(wrs) < 2 or all(wrs[i] <= wrs[i+1] + 0.05 for i in range(len(wrs)-1))
        verdict    = "✅  Model appears CALIBRATED" if calibrated else "⚠️  Model may be OVERCONFIDENT — check labels"
        print(f"\n  {verdict}")

    # Per-strategy breakdown
    trades_df = metrics.get("trades_df")
    if trades_df is not None and "strategy" in trades_df.columns:
        strategies = trades_df["strategy"].dropna().unique()
        if len(strategies) > 1:
            print("\n  ── PER-STRATEGY BREAKDOWN ───────────────────────────────────────")
            for strat in strategies:
                s      = trades_df[trades_df["strategy"] == strat]
                wins_s = (s["outcome"] == "win").sum()
                wr_s   = wins_s / len(s) if len(s) > 0 else 0
                pnl_s  = s["pnl"].sum()
                print(f"  {strat:<20} trades={len(s):>4}  "
                      f"win_rate={wr_s:.1%}  total_pnl=${pnl_s:+.2f}")

    # Per-trade detail
    if trades_df is not None and len(trades_df) > 0:
        print()
        print("  ── SIGNAL DETAIL (EAT = UTC+3, your TradingView timezone) ──────────")
        print(f"  {'#':<3} {'Signal Time (EAT)':<22} {'Dir':<5} {'Entry':>8} "
              f"{'SL':>8} {'TP':>8} {'Prob':>6}  {'Outcome':<10} {'Pips':>6} {'BarsToFill':>11}")
        print("  " + "─" * 96)
        emoji_map  = {"win": "✅", "loss": "❌", "expired": "⏸ ", "pending": "⏳"}
        dir_emoji  = {"BUY": "🟢", "SELL": "🔴"}
        for i, row in trades_df.reset_index(drop=True).iterrows():
            eat_raw  = row.get("signal_time_eat", row.get("signal_time", ""))
            eat_str  = str(eat_raw) if "EAT" in str(eat_raw) else to_eat(eat_raw)
            outcome  = str(row.get("outcome", "")).lower()
            emoji    = emoji_map.get(outcome, "❓")
            d        = str(row.get("direction", "BUY"))
            pips_str = f"{row['pips']:+.1f}" if outcome in ("win", "loss") else "—"
            b2e     = row.get("bars_to_entry")
            try:
                b2e_str = f"{int(b2e)}bars" if (b2e is not None and not pd.isna(b2e)) else "—"
            except (TypeError, ValueError):
                b2e_str = "—"
            print(f"  {i+1:<3} {eat_str:<22} "
                  f"{dir_emoji.get(d,'')}{d:<4} "
                  f"{float(row['entry']):>8.5f} {float(row['sl']):>8.5f} "
                  f"{float(row['tp']):>8.5f} {float(row['probability']):>5.1%}  "
                  f"{emoji}{outcome.upper():<9} {pips_str:>6} {b2e_str:>11}")
        print()
        print("  ⏸  EXPIRED = price never reached entry zone within horizon — no trade opened")
        print("  ✅  WIN     = entry zone touched (filled), then TP hit")
        print("  ❌  LOSS    = entry zone touched (filled), then SL hit")
        print("  ⏳  PENDING = entry zone touched, TP/SL not reached yet (still live)")
        print()
        print("  BarsToFill = number of 15-min bars from signal to entry fill")
        print("               If this is 0 or 1, the zone was hit very quickly —")
        print("               check signal generation time vs bar open time.")
    print("═" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(metrics: Dict, tag: str = ""):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"backtest_{tag}_{ts}.json"

    cal_json = None
    if metrics.get("calibration") is not None:
        cal_json = metrics["calibration"].to_dict(orient="records")

    result = {
        "run_at":           datetime.utcnow().isoformat(),
        "total_trades":     metrics.get("total_trades"),
        "win_rate":         metrics.get("win_rate"),
        "total_return_pct": metrics.get("total_return_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "profit_factor":    metrics.get("profit_factor"),
        "sharpe_ratio":     metrics.get("sharpe_ratio"),
        "final_balance":    metrics.get("final_balance"),
        "calibration":      cal_json,
    }

    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"Results saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest Engine — real signal replay")
    parser.add_argument(
        "--source",
        choices=["signals", "outcomes"],
        default="signals",
        help="'signals' = replay from JSON + local bars  |  'outcomes' = use pre-resolved outcomes.parquet",
    )
    parser.add_argument("--balance",  type=float, default=10_000, help="Starting balance (default 10000)")
    parser.add_argument("--risk",     type=float, default=1.0,    help="Risk per trade %% (default 1)")
    parser.add_argument("--slippage", type=float, default=0.3,    help="Slippage pips (default 0.3)")
    parser.add_argument("--horizon",  type=int,   default=192,
                        help="Max bars per phase (default 192 = 48h on 15min). "
                             "Applies to BOTH entry scan and post-entry TP/SL scan.")
    parser.add_argument("--no-save",  action="store_true", help="Don't save results to disk")
    parser.add_argument("--min-prob", type=float, default=0.0,
                        help="Minimum probability threshold to trade (default 0.0 = all signals). "
                             "Recommended: 0.65 for high-confidence trades only.")
    args = parser.parse_args()

    engine = BacktestEngine(
        initial_balance    = args.balance,
        risk_per_trade_pct = args.risk,
        slippage_pips      = args.slippage,
        max_horizon_bars   = args.horizon,
    )

    if args.source == "outcomes":
        logger.info("Source: pre-resolved outcomes.parquet")
        outcomes = load_outcomes()
        if args.min_prob > 0.0:
            prob_col = "predicted_prob" if "predicted_prob" in outcomes.columns else "probability"
            before   = len(outcomes)
            outcomes = outcomes[outcomes[prob_col] >= args.min_prob].reset_index(drop=True)
            logger.info(f"Filtered by min_prob={args.min_prob:.2f}: {before} → {len(outcomes)} signals")
        metrics  = engine.run_from_outcomes(outcomes)
        tag = "outcomes"
    else:
        logger.info("Source: signals/all_signals.json + local bar data")
        signals = load_signals_from_json()
        bars    = load_bars(signals_df=signals)
        metrics = engine.run(signals, bars)
        tag = "signals"

    print_report(metrics, engine)

    if not args.no_save:
        save_results(metrics, tag=tag)


if __name__ == "__main__":
    main()