# 8-EMA Pullback Swing Screener

A two-part bot that screens US stocks for the **8-EMA Pullback** swing setup: stocks in an established uptrend that have just pulled back to the 8-day EMA on declining volume, with a bullish reversal candle signaling continuation.

## The strategy in 4 steps

### 1. Trend (the prerequisite)
- Higher highs and higher lows in the recent structure
- Price > 20-EMA > 50-EMA (**stacked**)
- 50-EMA sloping up
- Weekly chart confirms — price above the weekly 20-EMA, weekly slope up

### 2. Pullback (the entry zone)
- Price has pulled back to within ~1 ATR of the 8-EMA
- **Volume declining** during the pullback vs the prior impulse move (the key tell)
- Pullback is shallow-to-moderate (didn't break the 20-EMA badly)
- Sweet spot: 2-7 bars since the impulse peak

### 3. Trigger (when to actually click buy)
A bullish reversal candle on the most recent bar (or yesterday):
- **Bullish engulfing** (strongest)
- **Hammer / pin bar**
- **Inside-bar breakout**

Per the source material: ideally enter in the last hour of the trading day once the reversal is confirmed.

### 4. Risk management
- **Stop** below the trigger bar low / recent swing low (whichever lower)
- **Position size** so risk = 1-2% of account on the trade (dashboard calculates this for you)
- **Scale out** in quarters:
  - ¼ at T1 = prior swing high
  - ¼ at T2 = 2R from entry
  - Runner trails behind the 8-EMA

## Tiers

| Tier | Score | Means |
|---|---|---|
| **A_SETUP** | ≥ 75 | Full alignment: trend + pullback + trigger candle, all in place |
| **B_SETUP** | 60–74 | Trend + pullback present, trigger may be soft/pending |
| **WATCH** | 45–59 | Trend OK but needs more pullback or no trigger yet |
| **PASS** | < 45 | Doesn't meet criteria |
| **ILLIQUID** | — | Filtered out for < $20M average daily dollar volume |

## Files

| File | What it does |
|---|---|
| `swing_screener.py` | Scans the universe, writes `results.json` |
| `dashboard.html` | Self-contained dashboard — open in any browser |
| `results.json` | Output (auto-loaded by the dashboard) |

## Run it

```bash
pip install -r requirements.txt

# Scan the full S&P 500 (~5-7 min)
python swing_screener.py
```

Then open `dashboard.html` (double-click, or serve via `python -m http.server` and visit localhost:8000/dashboard.html).

The screener fetches the current S&P 500 constituent list from Wikipedia at runtime, so the universe stays in sync as the index changes. If Wikipedia is unreachable, it falls back to a bundled 500-ticker snapshot. (`lxml` is needed for the Wikipedia HTML parsing.)

When `dashboard.html` is opened directly via `file://`, click **Load results.json** in the header to load your fresh scan; otherwise it shows the bundled demo data so the UI is functional on first open.

## Dashboard features

- **Tier filter** defaults to "Actionable" (A + B + WATCH) so you skip past PASS rows by default
- **Sector dropdown** to slice the 500-stock universe by Tech / Fin / Hlth / Cons / Ind / Engy / Util / RE / Mat / Comm
- **Signal filter** — show only setups with a trigger candle today
- **8-EMA distance filter** — at (±0.5 ATR) or near (±1.0 ATR)
- **Position sizer** built into the toolbar — punch in your account size and risk %; every expanded row shows shares to buy, capital to deploy, and dollars at risk
- **Click any row** for full signal breakdown, multi-timeframe stats, and a scaled-out trade plan
- **Showing X of Y** counter tells you how aggressively your filters are cutting

## Customizing the universe

By default, the screener fetches the live S&P 500 from Wikipedia. You can also pass any custom ticker list:

```python
from swing_screener import run, position_size, get_sp500_universe, NASDAQ_100, SP500_STATIC

# Default — fetches live S&P 500
run()

# Use NASDAQ-100 instead
run(tickers=NASDAQ_100)

# Use static fallback (no Wikipedia call)
run(use_wikipedia=False)

# Custom watchlist
run(tickers=["AAPL", "NVDA", "TSLA", "META"])

# Position sizing helper
ps = position_size(account_size=50_000, risk_pct=1.0, entry=180.50, stop=176.20)
print(ps)  # {'shares': 116, 'position_value': 20938, 'risk_dollars': 499, ...}
```

To tighten the liquidity filter (e.g. only mega-liquid names), bump `min_dollar_vol`:

```python
run(min_dollar_vol=100_000_000)  # only stocks with >$100M avg daily dollar volume
```

## Strategy notes

- **The volume-decline filter is the most discriminating signal.** A pullback on heavy volume often means real distribution (institutions selling), not a healthy pause. The screener penalizes setups with vol_decline_ratio > 1.0x and rewards those < 0.7x of the impulse volume.
- **Distance to 8-EMA is measured in ATR units** to normalize across stocks. ±0.5 ATR = "right at the 8-EMA"; ±1.0 = "near"; ±1.5 = "approaching".
- The minimum stop distance is **0.75 ATR** even if the trigger bar's low is closer, to avoid stop-hunt wicks.
- **Not financial advice.** Backtest before risking real capital. Strategy tuning happens in `score_setup()` weights and the tier thresholds.

## Automate it on GitHub (free, runs daily after market close)

The repo ships with a workflow at `.github/workflows/postmarket-scan.yml` that does three things:

1. **Scans the S&P 500 at 5:00pm ET** (1 hour after the 4pm close) Mon–Fri, using yesterday's full daily candle to find tomorrow's setups
2. **Commits the fresh `results.json`** back to your repo so the history is preserved
3. **Auto-deploys the dashboard to GitHub Pages** as your homepage, so visiting `https://YOUR-USERNAME.github.io/REPO-NAME/` shows the latest scan every morning

**Setup (5 minutes):**

1. Create a new GitHub repo and push these files (must be **public** for free Pages):
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   gh repo create swing-bot --public --source=. --push
   # or create via the GitHub web UI, then git push
   ```
2. **Settings → Actions → General → Workflow permissions** → select **Read and write permissions** and save.  
   This lets the scheduled job commit `results.json` back to the repo.
3. **Settings → Pages → Build and deployment → Source** → select **GitHub Actions** (NOT "Deploy from a branch").  
   The workflow handles the deployment itself, so you don't need to pick a branch.
4. Trigger the first deploy: **Actions → Post-market swing scan → Run workflow**.  
   Manual triggers always run regardless of time, so this gives you an immediate live dashboard.
5. After ~1 minute, visit `https://YOUR-USERNAME.github.io/REPO-NAME/` — your dashboard is live.

**How the scheduling works:**

GitHub Actions cron is in UTC and doesn't understand US daylight saving. The workflow registers two crons (21:00 UTC and 22:00 UTC) so that one always fires at 5:00pm ET year-round, then the Python timing check exits early on the cron that's wrong for the current DST regime. You get exactly one scheduled run per market afternoon.

**Why 5pm ET (post-market) instead of pre-market:**

- Daily candle is fully formed — yesterday's bullish engulfing is real, not a half-bar that could still reverse
- Yahoo Finance has had hours to ingest closing prices, so the data is reliable
- You see tomorrow's setups the night before, with time to plan position sizing and write orders into your broker

**Caveats:**

- **GitHub Actions cron is "best effort"**, not millisecond-precise. Scheduled jobs are routinely delayed 5-20 minutes during peak load; occasionally a job is skipped entirely. The data is the same regardless of whether you see it at 5:00, 5:15, or 9pm.
- **Yahoo Finance rate limits.** A 500-ticker scan with multi-timeframe fetches is ~2000 requests. yfinance occasionally returns empty/throttled responses; the screener catches these per-ticker and continues, but on a bad day you'll see more `NO_DATA` rows. If consistently bad, drop concurrency from 12 → 6 in `screen()` or switch to a paid feed.
- **Public repos have unlimited Actions minutes; private repos get 2000/month free** — way more than this needs (each run is ~5 min).
- **Free-tier GitHub Pages requires a public repo.** Your `results.json` becomes publicly readable — but it's just stock screener output, no credentials involved.
- **Holidays** are hardcoded for 2026-2027 in `is_us_market_day()`. Extend the dict when 2028 approaches.
