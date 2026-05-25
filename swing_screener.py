"""
8-EMA Pullback Swing Screener (+ market leadership filters)
============================================================
Find stocks in an established uptrend that have just pulled back to the 8-EMA
on shrinking volume, with a reversal candle signaling continuation —
AND that are leading the market, not lagging.

Strategy:

  1. TREND FILTER (Daily):
     - Higher highs & higher lows (recent structure)
     - Price > 20-EMA > 50-EMA  (stacked alignment)
     - 50-EMA sloping up
     - Weekly trend also up (price > weekly 20-EMA)

  2. LEADERSHIP FILTER (vs S&P 500):
     - Relative Strength (1M, 3M, 6M, 12M) — percentile rank vs SPY
     - Price within 25% of 52-week high (no falling knives)
     - Healthy Average Daily Range (ADR ~3%+) — moves enough to be tradeable

  3. PULLBACK FILTER (Daily):
     - Price pulled back to within ~1 ATR of the 8-EMA
     - Pullback volume DECLINING vs the prior impulse move
     - Pullback depth shallow-to-moderate (didn't break 20-EMA badly)

  4. ENTRY TRIGGER (latest 1-2 daily bars):
     - Bullish reversal candle near 8-EMA:
         * Bullish engulfing
         * Hammer / pin bar
         * Inside-bar break on the upside
     - Close back above 8-EMA on the trigger bar

  4. LIQUIDITY:
     - 20-day average dollar volume >= $20M  (configurable)

  5. TRADE PLAN:
     - Entry: at close of trigger bar (or next open)
     - Stop: below trigger bar low / recent swing low (whichever lower)
     - Targets (scaled):
         * T1: prior swing high (1/4 out)
         * T2: 2R from entry (1/4 out)
         * Runner: trail behind 8-EMA

  Composite score 0-100 + tier (A_SETUP / B_SETUP / WATCH / PASS).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import logging
import warnings
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# Silence yfinance's per-ticker error logging (404s on delisted/renamed tickers).
# yfinance uses a Python logger named 'yfinance' — raising its level to CRITICAL
# suppresses the noise without breaking stdout for our progress lines.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("yfinance").propagate = False


# =============================================================================
# INDICATORS
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# =============================================================================
# DATA
# =============================================================================

def fetch(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV. yfinance's noise is silenced globally via logging config at import time."""
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


# =============================================================================
# CANDLESTICK PATTERN DETECTION
# =============================================================================

def is_bullish_engulfing(prev: pd.Series, curr: pd.Series) -> bool:
    """Current green candle's body fully engulfs prior red candle's body."""
    prev_red = prev["Close"] < prev["Open"]
    curr_green = curr["Close"] > curr["Open"]
    body_engulfs = (curr["Open"] <= prev["Close"]) and (curr["Close"] >= prev["Open"])
    curr_body = abs(curr["Close"] - curr["Open"])
    prev_body = abs(prev["Close"] - prev["Open"])
    meaningful = curr_body > prev_body * 0.9
    return bool(prev_red and curr_green and body_engulfs and meaningful)


def is_hammer_or_pinbar(bar: pd.Series) -> bool:
    """Long lower wick (>=2x body), small upper wick, close in upper half."""
    rng = bar["High"] - bar["Low"]
    if rng <= 0:
        return False
    body = abs(bar["Close"] - bar["Open"])
    lower_wick = min(bar["Open"], bar["Close"]) - bar["Low"]
    upper_wick = bar["High"] - max(bar["Open"], bar["Close"])
    if body <= 0:
        body = rng * 0.01  # avoid divide-by-zero for doji
    long_lower = lower_wick >= 2.0 * body
    small_upper = upper_wick <= body * 0.7
    close_upper_half = (bar["Close"] - bar["Low"]) >= 0.6 * rng
    return bool(long_lower and small_upper and close_upper_half)


def is_inside_bar_breakout(prev: pd.Series, curr: pd.Series, two_back: pd.Series) -> bool:
    """Prior bar was inside two_back; current closes above prior high (bullish break)."""
    inside = (prev["High"] <= two_back["High"]) and (prev["Low"] >= two_back["Low"])
    breakout = curr["Close"] > prev["High"] and curr["Close"] > curr["Open"]
    return bool(inside and breakout)


def detect_reversal_signal(df: pd.DataFrame) -> dict:
    """Look for bullish reversal pattern on last or second-to-last bar."""
    if len(df) < 4:
        return {"found": False}

    bar_today = df.iloc[-1]
    bar_yest = df.iloc[-2]
    bar_2 = df.iloc[-3]

    patterns = []
    if is_bullish_engulfing(bar_yest, bar_today):
        patterns.append(("bullish_engulfing", 0))
    if is_hammer_or_pinbar(bar_today):
        patterns.append(("hammer/pin_bar", 0))
    if is_inside_bar_breakout(bar_yest, bar_today, bar_2):
        patterns.append(("inside_bar_break", 0))

    if len(df) >= 5:
        bar_3 = df.iloc[-4]
        if is_bullish_engulfing(bar_2, bar_yest):
            patterns.append(("bullish_engulfing", 1))
        if is_hammer_or_pinbar(bar_yest):
            patterns.append(("hammer/pin_bar", 1))
        if is_inside_bar_breakout(bar_2, bar_yest, bar_3):
            patterns.append(("inside_bar_break", 1))

    if not patterns:
        return {"found": False}

    patterns.sort(key=lambda p: (p[1], 0 if "engulfing" in p[0] else 1))
    name, bars_ago = patterns[0]
    return {"found": True, "pattern": name, "bars_ago": bars_ago}


# =============================================================================
# TREND STRUCTURE
# =============================================================================

def has_higher_highs_lows(df: pd.DataFrame, lookback: int = 30, swing_window: int = 5) -> dict:
    """Detect recent swing pivots and check HH-HL structure."""
    if len(df) < lookback + swing_window:
        return {"valid": False}
    sub = df.iloc[-(lookback + swing_window):]
    highs = sub["High"].values
    lows = sub["Low"].values

    swing_highs, swing_lows = [], []
    for i in range(swing_window, len(sub) - swing_window):
        if highs[i] == max(highs[i - swing_window:i + swing_window + 1]):
            swing_highs.append((i, float(highs[i])))
        if lows[i] == min(lows[i - swing_window:i + swing_window + 1]):
            swing_lows.append((i, float(lows[i])))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {
            "valid": True, "hh": False, "hl": False,
            "swing_high": swing_highs[-1][1] if swing_highs else None,
            "swing_low": swing_lows[-1][1] if swing_lows else None,
        }
    hh = swing_highs[-1][1] > swing_highs[-2][1]
    hl = swing_lows[-1][1] > swing_lows[-2][1]
    return {
        "valid": True, "hh": hh, "hl": hl,
        "swing_high": swing_highs[-1][1],
        "swing_low": swing_lows[-1][1],
    }


# =============================================================================
# PULLBACK ANALYSIS
# =============================================================================

def analyze_pullback(df: pd.DataFrame, ema8: pd.Series, atr14: pd.Series) -> dict:
    if len(df) < 20:
        return {"valid": False}
    closes = df["Close"]
    volumes = df["Volume"]
    last_close = float(closes.iloc[-1])
    last_ema8 = float(ema8.iloc[-1])
    last_atr = float(atr14.iloc[-1])
    dist_atr = (last_close - last_ema8) / last_atr if last_atr > 0 else 999

    lookback = min(20, len(df) - 1)
    recent = df.iloc[-lookback:]
    peak_idx = recent["High"].idxmax()
    peak_pos = df.index.get_loc(peak_idx)
    bars_since_peak = len(df) - 1 - peak_pos
    peak_high = float(df["High"].iloc[peak_pos])

    pullback_pct = (peak_high - last_close) / peak_high * 100
    pullback_atr = (peak_high - last_close) / last_atr if last_atr > 0 else 0

    if bars_since_peak >= 2 and peak_pos >= 5:
        impulse_vol = float(df["Volume"].iloc[max(0, peak_pos - 5):peak_pos + 1].mean())
        pullback_vol = float(df["Volume"].iloc[peak_pos + 1:].mean())
        vol_decline_ratio = pullback_vol / impulse_vol if impulse_vol > 0 else 1
    else:
        impulse_vol = float(volumes.iloc[-5:].mean())
        pullback_vol = float(volumes.iloc[-1])
        vol_decline_ratio = pullback_vol / impulse_vol if impulse_vol > 0 else 1

    vol20 = volumes.rolling(20).mean().iloc[-1]
    last_vol_ratio = float(volumes.iloc[-1] / vol20) if vol20 and vol20 > 0 else 1.0

    return {
        "valid": True,
        "dist_to_ema8_atr": dist_atr,
        "near_ema8": abs(dist_atr) <= 1.0,
        "above_ema8": last_close > last_ema8,
        "bars_since_peak": bars_since_peak,
        "peak_high": peak_high,
        "pullback_pct": pullback_pct,
        "pullback_atr": pullback_atr,
        "impulse_avg_vol": impulse_vol,
        "pullback_avg_vol": pullback_vol,
        "vol_decline_ratio": vol_decline_ratio,
        "last_vol_ratio_20d": last_vol_ratio,
    }


# =============================================================================
# WEEKLY TREND
# =============================================================================

def analyze_weekly_trend(df_daily: pd.DataFrame) -> dict:
    if len(df_daily) < 100:
        return {"valid": False}
    weekly = df_daily.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()
    if len(weekly) < 25:
        return {"valid": False}
    close = weekly["Close"]
    ema20_w = ema(close, 20)
    price = float(close.iloc[-1])
    ema20 = float(ema20_w.iloc[-1])
    ema20_prev = float(ema20_w.iloc[-6]) if len(ema20_w) >= 6 else ema20
    slope = (ema20 - ema20_prev) / ema20_prev * 100 if ema20_prev else 0
    return {
        "valid": True,
        "price": price,
        "ema20": ema20,
        "above_ema20": price > ema20,
        "ema20_slope_pct": slope,
        "rising": slope > 0,
    }


# =============================================================================
# MARKET LEADERSHIP (RS vs SPY, 52W high, ADR)
# Inspired by the 14 trader methodologies (Minervini, Qullamaggie, O'Neil,
# Mike Webster, Patrick Walker, etc.) — all share these filters in common.
# =============================================================================

def compute_returns(df: pd.DataFrame, periods_days: dict) -> dict:
    """Compute total returns over multiple lookback windows."""
    closes = df["Close"]
    last = float(closes.iloc[-1])
    out = {}
    for name, days in periods_days.items():
        if len(closes) > days:
            past = float(closes.iloc[-1 - days])
            out[name] = (last - past) / past * 100 if past > 0 else 0
        else:
            out[name] = None
    return out


def analyze_leadership(df_stock: pd.DataFrame, df_spy: pd.DataFrame) -> dict:
    """
    Compute relative-strength metrics, 52-week high proximity, and ADR.

    Returns:
      rs_1m / rs_3m / rs_6m / rs_12m: stock_return - spy_return for each window.
        Positive = outperforming SPY. Note: this is the raw excess-return delta,
        not a 0-100 percentile rank (which would require a full universe).
      pct_off_52w_high: how far below the 252-day high the stock is (negative %).
        e.g. -5.0 means 5% below the 52w high.
      adr_pct_20d: average daily range as a % of close, over last 20 bars.
    """
    if len(df_stock) < 60:
        return {"valid": False}

    # ---- Returns over 4 windows ----
    windows = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    stock_returns = compute_returns(df_stock, windows)

    # SPY returns over the SAME calendar dates (align via index intersection)
    if df_spy is not None and not df_spy.empty:
        # Align to stock's date range
        aligned_spy = df_spy.reindex(df_stock.index).ffill()
        spy_returns = compute_returns(aligned_spy.dropna(), windows)
    else:
        spy_returns = {k: 0 for k in windows}

    # Excess return = stock_return - SPY_return for that window
    rs = {}
    for k in windows:
        if stock_returns.get(k) is not None and spy_returns.get(k) is not None:
            rs[k] = stock_returns[k] - spy_returns[k]
        else:
            rs[k] = None

    # ---- 52-week high proximity ----
    lookback_52w = min(252, len(df_stock))
    high_52w = float(df_stock["High"].iloc[-lookback_52w:].max())
    last_close = float(df_stock["Close"].iloc[-1])
    pct_off_high = (last_close - high_52w) / high_52w * 100  # negative number

    # 52w low proximity (Minervini wants > 30% off the 52w low)
    low_52w = float(df_stock["Low"].iloc[-lookback_52w:].min())
    pct_off_low = (last_close - low_52w) / low_52w * 100

    # ---- Average Daily Range (last 20 bars) ----
    last20 = df_stock.iloc[-20:]
    daily_ranges = (last20["High"] - last20["Low"]) / last20["Close"] * 100
    adr_pct = float(daily_ranges.mean())

    return {
        "valid": True,
        "rs_1m": rs.get("1m"),
        "rs_3m": rs.get("3m"),
        "rs_6m": rs.get("6m"),
        "rs_12m": rs.get("12m"),
        "stock_return_3m": stock_returns.get("3m"),
        "stock_return_6m": stock_returns.get("6m"),
        "stock_return_12m": stock_returns.get("12m"),
        "spy_return_3m": spy_returns.get("3m"),
        "spy_return_6m": spy_returns.get("6m"),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_off_52w_high": pct_off_high,
        "pct_off_52w_low": pct_off_low,
        "adr_pct_20d": adr_pct,
    }


def fetch_spy_baseline() -> pd.DataFrame:
    """Fetch SPY once for the whole run; used as the RS benchmark."""
    spy = fetch("SPY", period="1y", interval="1d")
    if spy.empty:
        print("  ⚠ Could not fetch SPY baseline — RS scores will be zero")
    return spy


# =============================================================================
# FULL TICKER ANALYSIS
# =============================================================================

def analyze_ticker(ticker: str, min_dollar_vol: float = 20_000_000,
                   df_spy: pd.DataFrame = None) -> dict:
    df = fetch(ticker, period="1y", interval="1d")
    if df.empty or len(df) < 80:
        return {"ticker": ticker, "tier": "NO_DATA", "score": 0, "error": "insufficient data"}

    closes = df["Close"]
    volumes = df["Volume"]
    last_close = float(closes.iloc[-1])

    dollar_vol_20d = (closes * volumes).rolling(20).mean().iloc[-1]
    if pd.isna(dollar_vol_20d) or dollar_vol_20d < min_dollar_vol:
        return {
            "ticker": ticker, "tier": "ILLIQUID", "score": 0,
            "price": last_close,
            "dollar_vol_20d": float(dollar_vol_20d) if not pd.isna(dollar_vol_20d) else 0,
        }

    ema8 = ema(closes, 8)
    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    atr14 = atr(df, 14)

    last_ema8 = float(ema8.iloc[-1])
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_atr = float(atr14.iloc[-1])
    ema50_slope = (last_ema50 - float(ema50.iloc[-11])) / float(ema50.iloc[-11]) * 100 if len(ema50) >= 11 else 0

    structure = has_higher_highs_lows(df)
    pullback = analyze_pullback(df, ema8, atr14)
    signal = detect_reversal_signal(df)
    weekly = analyze_weekly_trend(df)
    leadership = analyze_leadership(df, df_spy) if df_spy is not None else {"valid": False}

    stacked = last_close > last_ema20 > last_ema50

    score = 0
    reasons = []

    # ---- Trend (35 pts) ----
    if stacked:
        score += 10
        reasons.append("price above 20-EMA & 50-EMA (stacked)")
    if last_ema20 > last_ema50:
        score += 6
        reasons.append("20-EMA > 50-EMA")
    if ema50_slope > 0.5:
        score += 8
        reasons.append(f"50-EMA rising ({ema50_slope:.2f}%)")
    elif ema50_slope > 0:
        score += 4
        reasons.append("50-EMA flattening up")

    if structure.get("hh") and structure.get("hl"):
        score += 11
        reasons.append("higher highs & higher lows")
    elif structure.get("hh") or structure.get("hl"):
        score += 5
        reasons.append("partial HH/HL structure")

    # ---- Weekly (15 pts) ----
    if weekly.get("valid"):
        if weekly["above_ema20"]:
            score += 9
            reasons.append("weekly above 20-EMA")
        if weekly["rising"]:
            score += 6
            reasons.append(f"weekly 20-EMA rising ({weekly['ema20_slope_pct']:.2f}%)")

    # ---- Pullback quality (30 pts) ----
    if pullback.get("valid"):
        d = pullback["dist_to_ema8_atr"]
        if -0.5 <= d <= 0.5:
            score += 12
            reasons.append(f"price right at 8-EMA ({d:+.2f} ATR)")
        elif -1.0 <= d <= 1.0:
            score += 8
            reasons.append(f"near 8-EMA ({d:+.2f} ATR)")
        elif -1.5 <= d <= 1.5:
            score += 3
            reasons.append(f"approaching 8-EMA ({d:+.2f} ATR)")

        pba = pullback["pullback_atr"]
        if pba < 1.5:
            score += 3
            reasons.append(f"shallow pullback ({pullback['pullback_pct']:.1f}%)")
        elif pba < 3.0:
            score += 5
            reasons.append(f"moderate pullback ({pullback['pullback_pct']:.1f}%)")
        elif pba < 5.0:
            score += 2
            reasons.append(f"deeper pullback ({pullback['pullback_pct']:.1f}%) — watch")
        else:
            score -= 5
            reasons.append(f"⚠ pullback too deep ({pullback['pullback_pct']:.1f}%)")

        vdr = pullback["vol_decline_ratio"]
        if vdr < 0.7:
            score += 10
            reasons.append(f"volume drying up on pullback ({vdr:.2f}x impulse)")
        elif vdr < 0.9:
            score += 6
            reasons.append(f"volume declining on pullback ({vdr:.2f}x)")
        elif vdr < 1.1:
            score += 2
            reasons.append("volume flat on pullback")
        else:
            score -= 4
            reasons.append(f"⚠ heavy volume on pullback ({vdr:.2f}x) — distribution risk")

        bsp = pullback["bars_since_peak"]
        if 2 <= bsp <= 7:
            score += 5
            reasons.append(f"clean pullback timing ({bsp} bars from peak)")
        elif bsp <= 1:
            reasons.append("just made new high — no pullback yet")
        elif bsp <= 12:
            score += 2
            reasons.append(f"extended pullback ({bsp} bars)")

    # ---- Entry trigger (20 pts) ----
    if signal["found"]:
        bonus = {"bullish_engulfing": 14, "hammer/pin_bar": 11, "inside_bar_break": 10}.get(signal["pattern"], 8)
        if signal["bars_ago"] == 1:
            bonus = int(bonus * 0.7)
        score += bonus
        when = "today" if signal["bars_ago"] == 0 else "yesterday"
        reasons.append(f"reversal signal: {signal['pattern']} ({when})")
        if pullback.get("above_ema8"):
            score += 6
            reasons.append("trigger bar closed above 8-EMA")

    # ---- Market leadership (up to 25 pts, can subtract for laggards) ----
    # Inspired by Minervini's Trend Template, O'Neil's RS, Qullamaggie's AS rankings.
    if leadership.get("valid"):
        # RS vs SPY over 3M and 6M — the windows traders watch most
        rs_3m = leadership.get("rs_3m") or 0
        rs_6m = leadership.get("rs_6m") or 0
        rs_12m = leadership.get("rs_12m") or 0

        # 3-month RS — recent leadership
        if rs_3m > 20:
            score += 10
            reasons.append(f"strong RS vs SPY 3M (+{rs_3m:.1f}pp)")
        elif rs_3m > 10:
            score += 6
            reasons.append(f"outperforming SPY 3M (+{rs_3m:.1f}pp)")
        elif rs_3m > 0:
            score += 2
            reasons.append(f"slight RS edge 3M (+{rs_3m:.1f}pp)")
        else:
            score -= 4
            reasons.append(f"⚠ lagging SPY 3M ({rs_3m:.1f}pp)")

        # 6-month RS — durable leadership
        if rs_6m > 25:
            score += 8
            reasons.append(f"strong RS vs SPY 6M (+{rs_6m:.1f}pp)")
        elif rs_6m > 10:
            score += 5
            reasons.append(f"outperforming SPY 6M (+{rs_6m:.1f}pp)")
        elif rs_6m > 0:
            score += 2

        # 12-month RS — long-term leader (O'Neil/Minervini classic)
        if rs_12m > 30:
            score += 4
            reasons.append(f"12M leader (+{rs_12m:.1f}pp)")

        # 52-week high proximity — hard floor at -25%
        pct_off = leadership.get("pct_off_52w_high", -100)
        if pct_off > -5:
            score += 6
            reasons.append(f"near 52w high ({pct_off:.1f}%)")
        elif pct_off > -15:
            score += 4
            reasons.append(f"close to 52w high ({pct_off:.1f}%)")
        elif pct_off > -25:
            score += 1
        else:
            score -= 6
            reasons.append(f"⚠ far below 52w high ({pct_off:.1f}%)")

        # ADR (Average Daily Range) — tradeable volatility per Qullamaggie/Moglen
        adr = leadership.get("adr_pct_20d", 0)
        if adr > 4:
            score += 4
            reasons.append(f"high-momentum ADR ({adr:.1f}%)")
        elif adr > 2.5:
            score += 2
            reasons.append(f"healthy ADR ({adr:.1f}%)")
        elif adr < 1.2:
            score -= 2
            reasons.append(f"⚠ low ADR ({adr:.1f}%) — slow mover")

    score = max(0, min(100, score))

    # ---- Tier ----
    has_trend = stacked
    has_signal = signal["found"]
    has_pullback = pullback.get("valid") and abs(pullback.get("dist_to_ema8_atr", 99)) <= 1.5

    if score >= 75 and has_trend and has_signal and has_pullback:
        tier = "A_SETUP"
    elif score >= 60 and has_trend and has_pullback:
        tier = "B_SETUP"
    elif score >= 45 and has_trend:
        tier = "WATCH"
    else:
        tier = "PASS"

    # ---- Trade plan ----
    entry = last_close
    trigger_bar_offset = -1 - (signal["bars_ago"] if signal["found"] else 0)
    trigger_low = float(df["Low"].iloc[trigger_bar_offset])
    swing_low = structure.get("swing_low") or (last_close - 2 * last_atr)
    raw_stop = min(trigger_low, swing_low) - 0.25 * last_atr
    min_stop = entry - 0.75 * last_atr  # enforce min 0.75 ATR risk
    stop = min(raw_stop, min_stop)
    risk_per_share = entry - stop
    target1 = structure.get("swing_high") or (entry + 2 * risk_per_share)
    target2 = entry + 2 * risk_per_share
    rr1 = (target1 - entry) / risk_per_share if risk_per_share > 0 else 0
    rr2 = (target2 - entry) / risk_per_share if risk_per_share > 0 else 0

    return {
        "ticker": ticker,
        "tier": tier,
        "score": score,
        "reasons": reasons,
        # Price snapshot
        "price": last_close,
        "ema8": last_ema8,
        "ema20": last_ema20,
        "ema50": last_ema50,
        "atr": last_atr,
        # Trend
        "stacked_emas": stacked,
        "ema50_slope_pct": ema50_slope,
        "higher_highs": structure.get("hh"),
        "higher_lows": structure.get("hl"),
        "swing_high": structure.get("swing_high"),
        "swing_low": structure.get("swing_low"),
        # Weekly
        "weekly_above_ema20": weekly.get("above_ema20") if weekly.get("valid") else None,
        "weekly_slope_pct": weekly.get("ema20_slope_pct") if weekly.get("valid") else None,
        # Pullback
        "dist_to_ema8_atr": pullback.get("dist_to_ema8_atr"),
        "above_ema8": pullback.get("above_ema8"),
        "bars_since_peak": pullback.get("bars_since_peak"),
        "pullback_pct": pullback.get("pullback_pct"),
        "vol_decline_ratio": pullback.get("vol_decline_ratio"),
        "last_vol_ratio_20d": pullback.get("last_vol_ratio_20d"),
        # Signal
        "signal_found": signal["found"],
        "signal_pattern": signal.get("pattern"),
        "signal_bars_ago": signal.get("bars_ago"),
        # Leadership (vs SPY)
        "rs_1m": leadership.get("rs_1m"),
        "rs_3m": leadership.get("rs_3m"),
        "rs_6m": leadership.get("rs_6m"),
        "rs_12m": leadership.get("rs_12m"),
        "stock_return_3m": leadership.get("stock_return_3m"),
        "stock_return_6m": leadership.get("stock_return_6m"),
        "high_52w": leadership.get("high_52w"),
        "pct_off_52w_high": leadership.get("pct_off_52w_high"),
        "pct_off_52w_low": leadership.get("pct_off_52w_low"),
        "adr_pct_20d": leadership.get("adr_pct_20d"),
        # Trade plan
        "entry": entry,
        "stop": stop,
        "risk_per_share": risk_per_share,
        "target1_swing_high": target1,
        "target2_2R": target2,
        "rr1": rr1,
        "rr2": rr2,
        # Liquidity
        "dollar_vol_20d": float(dollar_vol_20d),
    }


# =============================================================================
# POSITION SIZING
# =============================================================================

def position_size(account_size: float, risk_pct: float, entry: float, stop: float) -> dict:
    """1-2% account risk position sizing, per the strategy rules."""
    risk_dollars = account_size * (risk_pct / 100)
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return {"shares": 0, "position_value": 0, "risk_dollars": risk_dollars}
    shares = int(risk_dollars / risk_per_share)
    return {
        "shares": shares,
        "position_value": shares * entry,
        "risk_dollars": shares * risk_per_share,
        "risk_per_share": risk_per_share,
    }


# =============================================================================
# UNIVERSE & SCREENING
# =============================================================================

def screen(tickers, max_workers=12, min_dollar_vol=20_000_000, progress=True):
    results = []
    done = 0
    total = len(tickers)
    t_start = time.time()
    # Report every 50 for big scans, every 25 for small ones
    report_every = 50 if total > 200 else 25

    # Fetch SPY once as the RS benchmark for the whole scan
    if progress:
        print("  Fetching SPY benchmark for RS calc...")
    df_spy = fetch_spy_baseline()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(analyze_ticker, t, min_dollar_vol, df_spy): t for t in tickers}
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if progress and (done % report_every == 0 or done == total):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  scanned {done}/{total}  ({rate:.1f}/s, ETA {eta:.0f}s)")
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results


# Static fallback — S&P 500 snapshot reflecting Dec 2025 + early-2026 rebalances.
# Used only if the live Wikipedia fetch fails.
# Removed: ANSS, DAY, FI, HES, IPG, K, MMC, PARA, WBA, LKQ, MHK (mergers/deletions).
# Added:   CRH, CVNA, FIX, TTD (Dec 2025 rebalance + Ansys replacement).
SP500_STATIC = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL",
    "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK",
    "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX",
    "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "BK", "BA", "BKNG", "BSX",
    "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP", "CHRW", "CDNS",
    "CZR", "CPT", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CAT", "CBOE",
    "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX",
    "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX",
    "CME", "CMS", "KO", "CTSH", "COIN", "CL", "CMCSA", "CAG", "COP", "ED",
    "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA",
    "CRWD", "CCI", "CRH", "CSX", "CMI", "CVS", "CVNA", "DHR", "DRI", "DVA",
    "DECK", "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR", "D",
    "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "EMN", "ETN",
    "EBAY", "ECL", "EIX", "EW", "EA", "ELV", "EMR", "ENPH", "ETR", "EOG",
    "EPAM", "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG",
    "ES", "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO",
    "FAST", "FRT", "FDX", "FIS", "FITB", "FIX", "FSLR", "FE", "F", "FTNT",
    "FTV", "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV",
    "GEN", "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY",
    "GS", "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE",
    "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM",
    "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW", "INCY", "IR", "PODD", "INTC",
    "ICE", "IFF", "IP", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM",
    "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI", "JPM", "KVUE", "KDP",
    "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX",
    "LH", "LRCX", "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV",
    "LMT", "L", "LOW", "LULU", "LYB", "MTB", "MPC", "MKTX", "MAR",
    "MLM", "MAS", "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK",
    "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA",
    "MOH", "TAP", "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI",
    "NDAQ", "NTAP", "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN",
    "NSC", "NTRS", "NOC", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY",
    "ODFL", "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW",
    "PH", "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM",
    "PSX", "PNW", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD",
    "PRU", "PEG", "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "RL", "RJF",
    "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "ROK", "ROL",
    "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SRE", "NOW",
    "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV", "SWK",
    "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY", "TMUS",
    "TROW", "TTD", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA", "TXN",
    "TPL", "TXT", "TMO", "TJX", "TKO", "TSCO", "TT", "TDG", "TRV", "TRMB",
    "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP", "UAL", "UPS",
    "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK", "VZ", "VRTX",
    "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW", "WAB", "WMT",
    "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST", "WDC", "WSM",
    "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM", "ZBRA", "ZBH", "ZTS",
]


def get_sp500_universe(use_wikipedia: bool = True, verbose: bool = True) -> list:
    """
    Return current S&P 500 ticker list.

    Tries Wikipedia first (always current); falls back to the static snapshot.
    yfinance uses '-' instead of '.' for share class tickers (BRK-B not BRK.B).
    """
    if use_wikipedia:
        try:
            # Wikipedia blocks Python's default urllib user-agent; spoof a browser.
            import urllib.request
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8")
            # Newer pandas requires a file-like; passing a raw string is treated as a path.
            from io import StringIO
            tables = pd.read_html(StringIO(html))
            df = tables[0]  # first table on the page is the constituent list
            tickers = df["Symbol"].astype(str).tolist()
            # Normalize Berkshire/BF dot tickers for yfinance
            tickers = [t.replace(".", "-").strip().upper() for t in tickers]
            tickers = [t for t in tickers if t and t.isascii()]
            if verbose:
                print(f"  Fetched {len(tickers)} tickers from Wikipedia")
            return tickers
        except Exception as e:
            if verbose:
                print(f"  Wikipedia fetch failed ({e}); using static S&P 500 fallback")
    if verbose:
        print(f"  Using static S&P 500 list ({len(SP500_STATIC)} tickers)")
    return SP500_STATIC[:]


# Convenient pre-built universes
NASDAQ_100 = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO",
    "ORCL", "ADBE", "NFLX", "COST", "AMD", "PEP", "CSCO", "TMUS", "LIN",
    "INTC", "INTU", "QCOM", "AMGN", "TXN", "ISRG", "CMCSA", "HON", "AMAT",
    "BKNG", "VRTX", "MU", "ADP", "PANW", "GILD", "SBUX", "ADI", "REGN",
    "MDLZ", "LRCX", "KLAC", "PYPL", "MELI", "SNPS", "CDNS", "MAR", "ASML",
    "CRWD", "PLTR", "ABNB", "FTNT", "ORLY", "CSX", "CTAS", "MNST", "PCAR",
    "ROP", "DASH", "WDAY", "ADSK", "NXPI", "PAYX", "ROST", "FAST", "ODFL",
    "EA", "MCHP", "AEP", "KDP", "BKR", "EXC", "VRSK", "GEHC", "CTSH", "XEL",
    "IDXX", "FANG", "DDOG", "TEAM", "TTWO", "ANSS", "BIIB", "CHTR", "ON",
    "DLTR", "ZS", "WBD", "MDB", "TTD", "ARM", "MRVL", "CSGP", "CCEP", "GFS",
    "ILMN", "DXCM", "CDW", "WBA", "SMCI", "SIRI", "ENPH", "LULU", "AZN",
]


def is_us_market_day(now=None) -> tuple[bool, str]:
    """Return (is_market_day, reason). Covers weekends + major US market holidays."""
    from datetime import date as _date
    now = now or datetime.utcnow()
    if now.weekday() >= 5:
        return False, "weekend"

    # US market holidays (NYSE closed). Hardcoded for 2026-2027; extend as needed.
    # Source: NYSE official holiday calendar.
    HOLIDAYS = {
        # 2026
        _date(2026, 1, 1):   "New Year's Day",
        _date(2026, 1, 19):  "MLK Day",
        _date(2026, 2, 16):  "Presidents Day",
        _date(2026, 4, 3):   "Good Friday",
        _date(2026, 5, 25):  "Memorial Day",
        _date(2026, 6, 19):  "Juneteenth",
        _date(2026, 7, 3):   "Independence Day (observed)",
        _date(2026, 9, 7):   "Labor Day",
        _date(2026, 11, 26): "Thanksgiving",
        _date(2026, 12, 25): "Christmas",
        # 2027
        _date(2027, 1, 1):   "New Year's Day",
        _date(2027, 1, 18):  "MLK Day",
        _date(2027, 2, 15):  "Presidents Day",
        _date(2027, 3, 26):  "Good Friday",
        _date(2027, 5, 31):  "Memorial Day",
        _date(2027, 6, 18):  "Juneteenth (observed)",
        _date(2027, 7, 5):   "Independence Day (observed)",
        _date(2027, 9, 6):   "Labor Day",
        _date(2027, 11, 25): "Thanksgiving",
        _date(2027, 12, 24): "Christmas (observed)",
    }
    today = now.date()
    if today in HOLIDAYS:
        return False, f"US market closed ({HOLIDAYS[today]})"
    return True, "ok"


def run(tickers=None, top_n=None, min_dollar_vol=20_000_000,
        save=True, use_wikipedia=True, skip_non_market_days=True):
    if skip_non_market_days:
        ok, reason = is_us_market_day()
        if not ok:
            print(f"Skipping scan: {reason}")
            return []

    if tickers is None:
        print("Loading universe...")
        tickers = get_sp500_universe(use_wikipedia=use_wikipedia)
    print(f"\n8-EMA Pullback + Leadership Screener — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Universe: {len(tickers)} tickers · Min $vol: ${min_dollar_vol/1e6:.0f}M\n")

    t0 = time.time()
    results = screen(tickers, min_dollar_vol=min_dollar_vol)
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.\n")

    # Summary counts so the user sees the breakdown at a glance
    tier_counts = {}
    for r in results:
        tier_counts[r.get("tier", "?")] = tier_counts.get(r.get("tier", "?"), 0) + 1
    parts = []
    for t in ["A_SETUP", "B_SETUP", "WATCH", "PASS", "ILLIQUID", "NO_DATA"]:
        if t in tier_counts:
            parts.append(f"{t}={tier_counts[t]}")
    print("Tier breakdown: " + ", ".join(parts) + "\n")

    actionable = [r for r in results if r["tier"] in ("A_SETUP", "B_SETUP", "WATCH")]
    show = actionable[:top_n] if top_n else actionable

    print(f"{'TICKER':<7} {'TIER':<9} {'SCORE':<6} {'PRICE':<9} {'D2EMA8':<7} {'PB%':<5} {'VOL_DEC':<7} {'RS3M':<7} {'OffH':<7} {'SIGNAL':<22}")
    print("-" * 110)
    for r in show:
        d = f"{r['dist_to_ema8_atr']:+.2f}" if r.get("dist_to_ema8_atr") is not None else "-"
        pb = f"{r['pullback_pct']:.1f}%" if r.get("pullback_pct") is not None else "-"
        vd = f"{r['vol_decline_ratio']:.2f}x" if r.get("vol_decline_ratio") is not None else "-"
        rs = f"{r['rs_3m']:+.1f}" if r.get("rs_3m") is not None else "-"
        off = f"{r['pct_off_52w_high']:.1f}%" if r.get("pct_off_52w_high") is not None else "-"
        sig = f"{r['signal_pattern']}({'today' if r.get('signal_bars_ago') == 0 else 'yest'})" if r.get("signal_found") else "—"
        print(f"{r['ticker']:<7} {r['tier']:<9} {r['score']:<6} ${r.get('price', 0):<8.2f} {d:<7} {pb:<5} {vd:<7} {rs:<7} {off:<7} {sig:<22}")

    if save:
        # Save next to the script so it works on any machine (and CI)
        import os
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
        with open(out, "w") as f:
            json.dump({
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "strategy": "8-EMA Pullback + Leadership",
                "universe_size": len(tickers),
                "is_demo_data": False,
                "results": results,
            }, f, indent=2, default=str)
        print(f"\nSaved → {out}")
    return results


if __name__ == "__main__":
    run()
