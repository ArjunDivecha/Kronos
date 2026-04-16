"""
=============================================================================
SCRIPT NAME: industry28_trader.py
=============================================================================

INPUT FILES:
- data/Industry28/<TICKER>.csv : OHLCV cache (updated on each run via yfinance)

OUTPUT FILES:
- trade_log.csv : Appended each rebalance with order details

VERSION: 2.0
LAST UPDATED: 2026-04-15
AUTHOR: Arjun Divecha

DESCRIPTION:
Rebalance program for the Kronos Industry28 strategy (3 long / 3 short).
Run whenever you want to rebalance — no day-of-week restrictions.

The program:
  1. Downloads the latest OHLCV for all 28 ETFs from Yahoo Finance.
  2. Loads Kronos-base and generates 20-day return forecasts (40-bar lookback).
  3. Selects the top 3 ETFs (LONG) and bottom 3 ETFs (SHORT).
  4. Connects to IBKR TWS — prompts you to start TWS if it is not running.
  5. Fetches current holdings and net liquidation value.
  6. Computes the required trades to reach the 3L / 3S target.
  7. Displays proposed trades and asks for manual confirmation.
  8. Submits orders — Market-On-Open if markets are closed, immediate if open.
  9. Writes a trade log entry to trade_log.csv.

SIZING:
  Equal weight across 6 slots: position_size = net_liq * 0.98 / 6
  (2% held in cash as a buffer for fees and slippage)

ORDER TYPE (auto-detected):
  Market hours (Mon-Fri 9:30–16:00 ET) → immediate market order
  After hours / weekend                → Market-On-Open (next open)

PARAMETERS (from sweep results):
  Lookback = 40 days, Prediction horizon = 20 days

DEPENDENCIES:
- pandas, numpy, yfinance, ib_insync, nest_asyncio
- model/ (Kronos, KronosTokenizer, KronosPredictor)

USAGE:
  source shiyu-coder-Kronos/.venv/bin/activate
  cd shiyu-coder-Kronos
  python industry28_trader.py           # live run (auto-detects order type)
  python industry28_trader.py --dry-run # simulate without placing orders

NOTES:
- TWS must be open on localhost:7496. Script will prompt if it is not running.
- Requires a margin account for short selling.
- Account: U14983106
=============================================================================
"""

import argparse
import asyncio
import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import nest_asyncio
import numpy as np
import pandas as pd
import yfinance as yf

# Fix for Python 3.14+ event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()

from ib_insync import IB, MarketOrder, Stock

# ── Add repo root to path so we can import model ──────────────────────────────
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))
from model import Kronos, KronosPredictor, KronosTokenizer

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
TICKERS = [
    "XLK",   # Technology Select Sector SPDR
    "XLF",   # Financial Select Sector SPDR
    "XLI",   # Industrial Select Sector SPDR
    "XLY",   # Consumer Discretionary Select Sector SPDR
    "XLP",   # Consumer Staples Select Sector SPDR
    "XLE",   # Energy Select Sector SPDR
    "XLV",   # Health Care Select Sector SPDR
    "XLB",   # Materials Select Sector SPDR
    "XLRE",  # Real Estate Select Sector SPDR
    "XLC",   # Communication Services Select Sector SPDR
    "SMH",   # VanEck Semiconductor ETF
    "SOXX",  # iShares Semiconductor ETF
    "IGV",   # iShares Expanded Tech-Software Sector ETF
    "FDN",   # First Trust Dow Jones Internet Index
    "ITA",   # iShares U.S. Aerospace & Defense ETF
    "IYT",   # iShares Transportation Average ETF
    "KBE",   # SPDR S&P Bank ETF
    "XHB",   # SPDR S&P Homebuilders ETF
    "ITB",   # iShares U.S. Home Construction ETF
    "MOO",   # VanEck Agribusiness ETF
    "PHO",   # Invesco Water Resources ETF
    "IBB",   # iShares Biotechnology ETF
    "IHI",   # iShares U.S. Medical Devices ETF
    "XOP",   # SPDR S&P Oil & Gas E&P ETF
    "XBI",   # SPDR S&P Biotech ETF
    "XRT",   # SPDR S&P Retail ETF
    "KRE",   # SPDR S&P Regional Banking ETF
    "OIH",   # VanEck Oil Services ETF
]

DATA_DIR   = REPO_DIR / "data" / "Industry28"
TRADE_LOG  = REPO_DIR / "trade_log.csv"

# IBKR settings
IBKR_HOST  = "127.0.0.1"
IBKR_PORT  = 7496        # Live TWS (7497 for paper trading)
IBKR_CLIENT = 6          # Different from IBKR.py (id=5) to avoid conflicts
ACCOUNT_ID = "U14983106"

# Model settings (from sweep: 40-day lookback, 20-day hold is optimal)
LOOKBACK     = 40
PRED_LEN     = 20
SAMPLE_COUNT = 10
TEMPERATURE  = 0.8

ET = ZoneInfo("America/New_York")

# NYSE holidays through 2027
NYSE_HOLIDAYS = {
    date(2025, 1, 1),   date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18),  date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4),   date(2025, 9, 1),  date(2025, 11, 27),
    date(2025, 12, 25),
    date(2026, 1, 1),   date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3),   date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3),   date(2026, 8, 31), date(2026, 11, 26),
    date(2026, 12, 25),
    date(2027, 1, 1),   date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26),  date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5),   date(2027, 9, 6),  date(2027, 11, 25),
    date(2027, 12, 24),
}


# ── MARKET HOURS DETECTION ─────────────────────────────────────────────────────

def market_is_open():
    """
    Returns True if NYSE is currently open (Mon-Fri 9:30-16:00 ET, not a holiday).
    Uses auto-detected order type: immediate if open, MOO if closed.
    """
    now_et = datetime.now(ET)
    today_et = now_et.date()

    if today_et in NYSE_HOLIDAYS:
        return False
    if now_et.weekday() >= 5:   # Saturday=5, Sunday=6
        return False

    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_et < market_close


def order_type_label(use_moo):
    if use_moo:
        # Work out the next trading day for the label
        d = date.today() + timedelta(days=1)
        while d.weekday() >= 5 or d in NYSE_HOLIDAYS:
            d += timedelta(days=1)
        return f"Market-On-Open ({d.strftime('%Y-%m-%d')} open)"
    return "Market order (immediate execution)"


# ── STEP 1: DOWNLOAD & UPDATE OHLCV ───────────────────────────────────────────

def update_ohlcv():
    """
    Downloads the latest ~3 months of OHLCV from Yahoo Finance for all 28 tickers.
    Merges with existing CSV cache (deduplicates by date).
    Returns a dict {ticker: DataFrame} for tickers with at least LOOKBACK+PRED_LEN rows.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/5] Updating OHLCV data for {len(TICKERS)} tickers...")

    stock_dfs = {}
    failed = []

    for ticker in TICKERS:
        out_path = DATA_DIR / f"{ticker}.csv"

        try:
            raw = yf.download(
                ticker,
                period="3mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            print(f"  WARNING: yfinance download failed for {ticker}: {exc}")
            failed.append(ticker)
            continue

        if raw is None or raw.empty:
            print(f"  WARNING: No data returned for {ticker}")
            failed.append(ticker)
            continue

        raw = raw.copy()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"adj close": "close"})

        if "close" not in raw.columns:
            print(f"  WARNING: No 'close' column for {ticker}")
            failed.append(ticker)
            continue

        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "timestamps"
        raw = raw.reset_index()

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in raw.columns:
                raw[col] = np.nan
        raw["volume"] = raw["volume"].fillna(0.0)
        raw["amount"] = raw["close"] * raw["volume"]
        raw["timestamps"] = raw["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S")
        new_df = raw[["timestamps", "open", "high", "low", "close", "volume", "amount"]].dropna(
            subset=["open", "high", "low", "close"]
        )

        if out_path.exists():
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined["timestamps"] = pd.to_datetime(combined["timestamps"])
        combined = combined.sort_values("timestamps").drop_duplicates("timestamps")
        combined["timestamps"] = combined["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S")
        combined.to_csv(out_path, index=False)

        df = combined.copy()
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        if len(df) >= LOOKBACK + PRED_LEN:
            stock_dfs[ticker] = df.reset_index(drop=True)
        else:
            print(f"  WARNING: {ticker} has only {len(df)} rows — skipping.")

    if failed:
        print(f"  {len(failed)} ticker(s) failed: {', '.join(failed)}")
    print(f"  {len(stock_dfs)} tickers ready for inference.")
    return stock_dfs


# ── STEP 2: GENERATE SIGNALS ───────────────────────────────────────────────────

def generate_signals(stock_dfs):
    """
    Runs Kronos-base inference on the last 40 bars of each ticker,
    predicting 20 days forward. Returns a ranked DataFrame and top3/bot3 lists.
    """
    print("\n[2/5] Loading Kronos-base model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print(f"  Generating {PRED_LEN}-day forecasts for {len(stock_dfs)} tickers...")

    batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []

    for ticker, df in stock_dfs.items():
        x_df = df.iloc[-LOOKBACK:]
        x_ts = x_df["timestamps"]

        if x_df[["open", "high", "low", "close", "volume", "amount"]].isnull().values.any():
            print(f"  WARNING: NaN in {ticker} — skipping.")
            continue

        last_date = x_ts.iloc[-1]
        y_ts = pd.Series(
            pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=PRED_LEN)
        )

        batch_dfs.append(x_df[["open", "high", "low", "close", "volume", "amount"]])
        batch_xts.append(x_ts)
        batch_yts.append(y_ts)
        batch_meta.append({
            "ticker":     ticker,
            "last_close": df.iloc[-1]["close"],
        })

    if len(batch_dfs) < 6:
        raise RuntimeError(
            f"Only {len(batch_dfs)} tickers have valid data — need at least 6. Aborting."
        )

    preds = predictor.predict_batch(
        df_list=batch_dfs,
        x_timestamp_list=batch_xts,
        y_timestamp_list=batch_yts,
        pred_len=PRED_LEN,
        sample_count=SAMPLE_COUNT,
        T=TEMPERATURE,
        verbose=False,
    )

    rows = []
    for j, pred_df in enumerate(preds):
        meta       = batch_meta[j]
        pred_close = pred_df.iloc[-1]["close"]
        rows.append({
            "ticker":      meta["ticker"],
            "last_close":  meta["last_close"],
            "pred_return": (pred_close / meta["last_close"]) - 1,
        })

    signals = pd.DataFrame(rows).sort_values("pred_return", ascending=False).reset_index(drop=True)
    top3 = signals.head(3)["ticker"].tolist()
    bot3 = signals.tail(3)["ticker"].tolist()

    print(f"  TOP 3 (LONG):  {', '.join(top3)}")
    print(f"  BOT 3 (SHORT): {', '.join(bot3)}")

    return signals, top3, bot3


# ── STEP 3: CONNECT TO IBKR (WITH RETRY) ──────────────────────────────────────

async def _connect_with_retry_async():
    """
    Tries to connect to TWS. If the connection is refused, prints a clear
    message and waits for the user to press Enter before retrying.
    Returns a connected IB instance.
    """
    ib = IB()
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"\n  Connecting to TWS at {IBKR_HOST}:{IBKR_PORT} (clientId={IBKR_CLIENT})...")
            await asyncio.wait_for(
                ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT, readonly=False),
                timeout=10,
            )
            if ib.isConnected():
                print("  Connected.")
                return ib
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            pass

        print(
            f"\n  ✗  Cannot reach TWS on port {IBKR_PORT}."
            f"\n     Please open Trader Workstation and log in, then press Enter to retry."
            f"\n     (Ctrl+C to abort)"
        )
        try:
            input("  > ")
        except KeyboardInterrupt:
            print("\n  Aborted.")
            sys.exit(0)

        if ib.isConnected():
            ib.disconnect()
        ib = IB()


async def _fetch_ibkr_state_async():
    """
    Connects to TWS (with retry), fetches NetLiquidation and current positions.
    Returns (net_liq: float, positions: dict {symbol: quantity})
    """
    ib = await _connect_with_retry_async()
    try:
        await ib.reqAccountSummaryAsync()
        await asyncio.sleep(6)
        summary = ib.accountValues()

        PRIORITY_TAGS = [
            ("NetLiquidation", "BASE"),
            ("NetLiquidation", "USD"),
            ("NetLiquidation", ""),
            ("TotalCashValue", "BASE"),
            ("TotalCashValue", "USD"),
            ("CashBalance",    "BASE"),
            ("CashBalance",    "USD"),
        ]

        net_liq = None
        for tag, currency in PRIORITY_TAGS:
            for v in summary:
                if v.tag == tag and v.account == ACCOUNT_ID and (currency == "" or v.currency == currency):
                    try:
                        net_liq = float(v.value)
                        print(f"  Account {ACCOUNT_ID} | Net Liquidation: ${net_liq:,.2f}  [{tag} {v.currency}]")
                        break
                    except ValueError:
                        pass
            if net_liq is not None:
                break

        if net_liq is None:
            available = [(v.tag, v.currency, v.value) for v in summary if v.account == ACCOUNT_ID]
            raise RuntimeError(
                f"Could not retrieve account value for {ACCOUNT_ID}. "
                f"Available tags: {available[:10]}"
            )

        raw_positions = await ib.reqPositionsAsync()
        positions = {}
        for pos in raw_positions:
            if pos.account == ACCOUNT_ID:
                positions[pos.contract.symbol] = pos.position

        print(f"  Current holdings: {len(positions)} position(s)" +
              (f"  {list(positions.keys())}" if positions else ""))

    finally:
        if ib.isConnected():
            ib.disconnect()

    return net_liq, positions


def fetch_ibkr_state():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _fetch_ibkr_state_async()).result()
        return loop.run_until_complete(_fetch_ibkr_state_async())
    except Exception as exc:
        raise RuntimeError(f"IBKR error: {exc}") from exc


# ── STEP 4: COMPUTE TRADES ─────────────────────────────────────────────────────

def compute_trades(signals, top3, bot3, net_liq, current_positions):
    """
    Diffs current_positions against the 3L/3S target.
    Each slot is sized at net_liq * 0.98 / 6 (2% cash buffer).
    Returns (trades list, position_size).
    """
    position_size = net_liq * 0.98 / 6.0
    last_prices   = {row["ticker"]: row["last_close"] for _, row in signals.iterrows()}

    target = {}
    for ticker in top3:
        price = last_prices.get(ticker, 0)
        if price <= 0:
            raise RuntimeError(f"Invalid last price for {ticker}.")
        target[ticker] = round(position_size / price)

    for ticker in bot3:
        price = last_prices.get(ticker, 0)
        if price <= 0:
            raise RuntimeError(f"Invalid last price for {ticker}.")
        target[ticker] = -round(position_size / price)

    trades = []
    for ticker in sorted(set(current_positions) | set(target)):
        current_qty = current_positions.get(ticker, 0)
        target_qty  = target.get(ticker, 0)
        delta       = target_qty - current_qty

        if delta == 0:
            continue

        price = last_prices.get(ticker, 0)

        if target_qty == 0:
            label = "CLOSE"
        elif current_qty == 0:
            label = "BUY" if delta > 0 else "SHORT"
        else:
            label = "ADJUST"

        trades.append({
            "ticker":       ticker,
            "label":        label,
            "action":       "BUY" if delta > 0 else "SELL",
            "delta_shares": abs(int(delta)),
            "current_qty":  int(current_qty),
            "target_qty":   int(target_qty),
            "last_price":   price,
            "est_value":    delta * price,
        })

    return trades, position_size


# ── STEP 5: DISPLAY & CONFIRM ──────────────────────────────────────────────────

def display_and_confirm(signals, top3, bot3, net_liq, position_size, trades, dry_run, use_moo):
    today_str = date.today().strftime("%Y-%m-%d")
    now_et    = datetime.now(ET).strftime("%H:%M ET")

    print("\n" + "═" * 62)
    print(f"  KRONOS INDUSTRY28 — Rebalance  [{today_str}  {now_et}]")
    print("═" * 62)
    print(f"  Account        : {ACCOUNT_ID}")
    print(f"  Net Liquidation: ${net_liq:>12,.2f}")
    print(f"  Position size  : ${position_size:>12,.2f}  (98% of net_liq / 6)")
    print(f"  Order type     : {order_type_label(use_moo)}")

    print(f"\n  SIGNALS  (Kronos {PRED_LEN}-day predicted return, {LOOKBACK}-bar lookback)")

    top3_rows = signals[signals["ticker"].isin(top3)].sort_values("pred_return", ascending=False)
    bot3_rows = signals[signals["ticker"].isin(bot3)].sort_values("pred_return", ascending=True)

    print(f"\n  ┌─ TOP 3 (LONG) {'─'*44}┐")
    for _, row in top3_rows.iterrows():
        print(f"  │  {row['ticker']:<6}  {row['pred_return']:+.2%}   last close: ${row['last_close']:>8.2f}           │")
    print(f"  └{'─'*59}┘")

    print(f"\n  ┌─ BOT 3 (SHORT) {'─'*43}┐")
    for _, row in bot3_rows.iterrows():
        print(f"  │  {row['ticker']:<6}  {row['pred_return']:+.2%}   last close: ${row['last_close']:>8.2f}           │")
    print(f"  └{'─'*59}┘")

    print(f"\n  PROPOSED TRADES  ({order_type_label(use_moo)})")
    if not trades:
        print("  No changes needed — portfolio already matches target.")
    else:
        header = f"  {'Ticker':<8}  {'Label':<8}  {'Action':<5}  {'Delta':>7}  {'Price':>8}  {'Est. Value':>12}"
        sep    = f"  {'-' * (len(header) - 2)}"
        print(f"\n{sep}\n{header}\n{sep}")
        for t in trades:
            sign = "+" if t["action"] == "BUY" else "-"
            print(
                f"  {t['ticker']:<8}  {t['label']:<8}  {t['action']:<5}  "
                f"{sign}{t['delta_shares']:>6}  ${t['last_price']:>7.2f}  "
                f"${t['est_value']:>+11,.0f}"
            )
        print(sep)
        print(f"  {'':>50}  Turnover: ${sum(abs(t['est_value']) for t in trades):>10,.0f}")

    if dry_run:
        print("\n  [DRY RUN] No orders will be placed.")
        return False

    if not trades:
        print("\n  Nothing to do.")
        return False

    print(f"\n  ⚠  Type YES to submit all {len(trades)} order(s), or anything else to abort:")
    return input("  > ").strip() == "YES"


# ── STEP 6: SUBMIT ORDERS ─────────────────────────────────────────────────────

async def _submit_orders_async(trades, use_moo):
    ib = await _connect_with_retry_async()
    results = []
    try:
        for t in trades:
            contract       = Stock(t["ticker"], "SMART", "USD")
            order          = MarketOrder(t["action"], t["delta_shares"])
            order.account  = ACCOUNT_ID
            if use_moo:
                order.tif = "OPG"

            trade    = ib.placeOrder(contract, order)
            await asyncio.sleep(0.5)
            order_id = trade.order.orderId
            status   = trade.orderStatus.status or "Submitted"
            results.append({
                "ticker":   t["ticker"],
                "action":   t["action"],
                "qty":      t["delta_shares"],
                "order_id": order_id,
                "status":   status,
            })
            print(f"  Placed: {t['action']} {t['delta_shares']:>5} {t['ticker']:<6}  "
                  f"orderId={order_id}  status={status}")

        await asyncio.sleep(2)
    finally:
        if ib.isConnected():
            ib.disconnect()
    return results


def submit_orders(trades, use_moo):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, _submit_orders_async(trades, use_moo)).result()
        return loop.run_until_complete(_submit_orders_async(trades, use_moo))
    except Exception as exc:
        raise RuntimeError(f"Order submission error: {exc}") from exc


# ── STEP 7: WRITE TRADE LOG ────────────────────────────────────────────────────

def write_trade_log(trades, order_results, top3, bot3, net_liq, use_moo, dry_run):
    fieldnames = ["date", "ticker", "side", "action", "qty", "order_id",
                  "status", "order_type", "account", "net_liq", "dry_run"]
    file_exists = TRADE_LOG.exists()
    with TRADE_LOG.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for t in trades:
            matched = next((r for r in order_results if r["ticker"] == t["ticker"]), {})
            writer.writerow({
                "date":       date.today().strftime("%Y-%m-%d"),
                "ticker":     t["ticker"],
                "side":       "LONG" if t["ticker"] in top3 else "SHORT",
                "action":     t["action"],
                "qty":        t["delta_shares"],
                "order_id":   matched.get("order_id", ""),
                "status":     matched.get("status", "DRY_RUN") if not dry_run else "DRY_RUN",
                "order_type": "MOO" if use_moo else "MKT",
                "account":    ACCOUNT_ID,
                "net_liq":    f"{net_liq:.2f}",
                "dry_run":    str(dry_run),
            })
    print(f"\n  Trade log updated → {TRADE_LOG}")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Kronos Industry28 rebalance trader.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show signals and trades without placing orders.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-detect order type
    is_open = market_is_open()
    use_moo = not is_open
    now_et  = datetime.now(ET)

    print("=" * 62)
    print("  KRONOS INDUSTRY28 TRADER")
    print("=" * 62)
    print(f"\n  Current time : {now_et.strftime('%A %Y-%m-%d %H:%M ET')}")
    print(f"  Market status: {'OPEN — using immediate market orders' if is_open else 'CLOSED — using Market-On-Open orders'}")

    # Update data
    stock_dfs = update_ohlcv()

    # Generate signals
    signals, top3, bot3 = generate_signals(stock_dfs)

    # Fetch IBKR state (always — dry-run still reads positions, just doesn't place orders)
    print(f"\n[3/5] Fetching IBKR account state...")
    net_liq, current_positions = fetch_ibkr_state()

    # Compute trades
    print(f"\n[4/5] Computing required trades...")
    trades, position_size = compute_trades(signals, top3, bot3, net_liq, current_positions)
    print(f"  {len(trades)} trade(s) required.")

    # Display and confirm
    confirmed = display_and_confirm(
        signals, top3, bot3, net_liq, position_size, trades, args.dry_run, use_moo
    )

    if not confirmed:
        if not args.dry_run:
            print("\n  Aborted. No orders placed.")
        if trades:
            write_trade_log(trades, [], top3, bot3, net_liq, use_moo, dry_run=True)
        sys.exit(0)

    # Submit
    print(f"\n[5/5] Submitting orders...")
    order_results = submit_orders(trades, use_moo)

    print(f"\n  ORDER CONFIRMATION")
    print(f"  {'Ticker':<8}  {'Action':<5}  {'Qty':>6}  {'OrderId':>10}  Status")
    print(f"  {'-'*52}")
    for r in order_results:
        print(f"  {r['ticker']:<8}  {r['action']:<5}  {r['qty']:>6}  {r['order_id']:>10}  {r['status']}")

    write_trade_log(trades, order_results, top3, bot3, net_liq, use_moo, dry_run=False)
    print("\n  Rebalance complete.\n")


if __name__ == "__main__":
    main()
