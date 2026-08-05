"""
tracker_daily.py — publish daily Kronos paper books for the Tracker hub
========================================================================

WHAT THIS DOES
--------------
Runs the Kronos 40/5 strategy forward as a set of PAPER books, one per universe
per construction, and publishes a Tracker-readable artifact set for each. It is
the producer side of the Tracker contract: Tracker never runs strategy code, it
only reads what this script writes.

Eleven universes x two constructions = 22 books:

  long : equal-weight Top-N, long only          (matches the published headline)
  ls   : LEG x Top-N long, LEG x Bottom-N short (matches what industry28_trader
         actually trades; LEG = 0.49, measured from the live ledger)

CADENCE — 5-day metronome with daily marks
------------------------------------------
The strategy holds 5 trading days, so the book rebalances every 5th trading day
and is marked to market every day in between. NAV is therefore a daily series
while turnover matches the backtest exactly. A book rebalances when
(trading days since last_rebalance) >= 5; the first run sets the anchor.

This is deliberately NOT overlapping daily tranches. Tranches would smooth the
series and remove start-date luck, but they are a different strategy from the
one that was backtested and the numbers would not be comparable.

DETERMINISM
-----------
Kronos SAMPLES (T=0.8). A 20-seed study on this repo measured L/S Sharpe moving
between 0.73 and 2.20 from the RNG alone on a 39-rebalance window. Signals here
are therefore pinned: RANDOM_SEED=42 and SAMPLE_COUNT=20, matching
industry28_trader.py. Raising SAMPLE_COUNT shrinks signal noise roughly as
1/sqrt(n) at linear cost — worth doing if daily runtime allows.

BLOOMBERG
---------
Nothing here needs Bloomberg. Every universe is yfinance-sourced, including UK
(.L suffix) and Japan (.T suffix). India and Australia are Bloomberg-only
(IB Equity / AT Equity) and are deliberately EXCLUDED.

SURVIVORSHIP
------------
The three US stock universes use fixed April-2026 member lists with no delisting
logic, so their BACKTESTS are survivorship-biased. Forward paper tracking is not
— you hold what the signal picked on the day. meta.json records this per book.

INPUT FILES (absolute)
----------------------
  Ticker lists come from the research caches (filenames are yfinance symbols):
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/<Universe>/<TICKER>.csv
  Prices are refreshed from Yahoo Finance into a SEPARATE light cache so the
  research caches above are never mutated:
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/tracker_data/<universe>.parquet
  Model weights: HuggingFace Hub NeoQuasar/Kronos-base + -Tokenizer-base

OUTPUT FILES (one directory per book, read by Tracker)
------------------------------------------------------
  <repo>/tracker_book/kronos-<universe>-<long|ls>/
      nav_daily.csv         date,nav,ret_1d_pct,cum_ret_pct,bench_nav,bench_cum_ret_pct,n_positions,marked_stale
      holdings_current.csv  ticker,side,shares,entry_price,entry_date,last_price,weight_pct,price_stale
      signals_latest.csv    ticker,pred_return,rank  (last rebalance)
      state.json            book state for resumption (positions, cash, anchor)
      meta.json             Tracker metadata: construction, top_n, flags, caveats
  <repo>/tracker_book/run_log/YYYY-MM-DD.log
  <repo>/tracker_book/heartbeat.txt

USAGE
-----
  .venv/bin/python tracker_daily.py                    # all books, today
  .venv/bin/python tracker_daily.py --universes industry28 sector
  .venv/bin/python tracker_daily.py --no-download      # reuse today's price cache
  .venv/bin/python tracker_daily.py --dry-run          # compute, publish nothing

DEPENDENCIES
------------
  torch (MPS), pandas, numpy, yfinance, pyarrow, tqdm; model/ (Kronos)

NOTES
-----
Every book writes its own artifacts atomically (temp file then rename) the
moment it finishes, so a crash mid-run leaves completed books intact and the
partially-run book recoverable from its previous state.json.

Version 1.0 — 2026-08-04
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings
from datetime import date, datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__))
BOOK_ROOT = os.path.join(REPO, "tracker_book")
PRICE_CACHE = os.path.join(REPO, "tracker_data")

LOOKBACK, PRED_LEN, HOLD_DAYS = 40, 5, 5
SAMPLE_COUNT, TEMPERATURE, RANDOM_SEED = 20, 0.8, 42
LEG = 0.49                 # per-side capital weight, measured from pl_tracker
START_NAV = 100_000.0
PRICE_HISTORY_DAYS = 400   # enough for a 40-bar lookback plus slack

# universe -> (research cache dir supplying the ticker list, top-N per side)
UNIVERSES = {
    "industry28":  ("Industry28",  3),
    "country34":   ("ETF",         3),
    "sector":      ("Sector",      3),
    "factor":      ("Factor",      3),
    "fixedincome": ("FixedIncome", 3),
    "commodity":   ("Commodity",   3),
    "sp500":       ("Universe500", 10),
    "smallcap600": ("SmallCap600", 10),
    "techstocks":  ("TechStocks",  5),
    "japan":       ("Japan",       10),
    "uk500":       ("UK500",       10),
}
SURVIVORSHIP_BIASED = {"sp500", "smallcap600", "techstocks", "japan", "uk500"}


def atomic_write(path, write_fn):
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def tickers_for(universe):
    d = os.path.join(REPO, "data", UNIVERSES[universe][0])
    if not os.path.isdir(d):
        raise SystemExit(f"FAIL: missing research cache {d}")
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".csv") and not f.startswith("_"))


# ── prices ────────────────────────────────────────────────────────────────────
def refresh_prices(universe, log, download=True):
    """Bulk-download recent OHLCV into a light per-universe parquet cache."""
    import yfinance as yf
    os.makedirs(PRICE_CACHE, exist_ok=True)
    path = os.path.join(PRICE_CACHE, f"{universe}.parquet")
    tickers = tickers_for(universe)

    if not download and os.path.exists(path):
        return pd.read_parquet(path)

    start = (pd.Timestamp.today() - pd.Timedelta(days=PRICE_HISTORY_DAYS)).strftime("%Y-%m-%d")
    frames, CH = [], 100
    for i in range(0, len(tickers), CH):
        chunk = tickers[i:i + CH]
        raw = yf.download(chunk, start=start, progress=False, auto_adjust=True,
                          group_by="ticker", threads=True)
        for t in chunk:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                sub = sub.dropna(subset=["Close"])
                if sub.empty:
                    continue
                frames.append(pd.DataFrame({
                    "ticker": t, "date": pd.to_datetime(sub.index).normalize(),
                    "open": sub["Open"].values, "high": sub["High"].values,
                    "low": sub["Low"].values, "close": sub["Close"].values,
                    "volume": sub["Volume"].values,
                }))
            except Exception:
                continue
        log(f"    prices {min(i+CH,len(tickers))}/{len(tickers)}")

    if not frames:
        raise RuntimeError(f"no prices downloaded for {universe}")
    px = pd.concat(frames, ignore_index=True)
    px["amount"] = px["close"] * px["volume"]
    atomic_write(path, lambda p: px.to_parquet(p, index=False))
    log(f"    {px['ticker'].nunique()} tickers, {px['date'].nunique()} dates -> {os.path.basename(path)}")
    return px


def last_price_map(px, asof):
    """Latest close at or before asof per ticker, plus staleness in days."""
    sub = px[px["date"] <= asof]
    if sub.empty:
        return {}, {}
    last = sub.sort_values("date").groupby("ticker").tail(1)
    prices = dict(zip(last["ticker"], last["close"]))
    stale = {t: (asof - d).days for t, d in zip(last["ticker"], last["date"])}
    return prices, stale


# ── signals ───────────────────────────────────────────────────────────────────
def generate_signals(px, asof, predictor, log):
    """Kronos forecast per ticker as of `asof`. Returns {ticker: pred_return}."""
    import torch
    dfs, xts, yts, names = [], [], [], []
    for t, g in px[px["date"] <= asof].groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < LOOKBACK:
            continue
        x = g.tail(LOOKBACK)
        if x[["open", "high", "low", "close", "volume", "amount"]].isnull().values.any():
            continue
        last_ts = x["date"].iloc[-1]
        y_ts = pd.Series(pd.bdate_range(last_ts + pd.Timedelta(days=1), periods=PRED_LEN))
        dfs.append(x[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True))
        xts.append(x["date"].reset_index(drop=True))
        yts.append(y_ts)
        names.append(t)

    if len(dfs) < 6:
        raise RuntimeError(f"only {len(dfs)} tickers with usable history — need >= 6")

    torch.manual_seed(RANDOM_SEED)
    out, CH = {}, 256
    for i in range(0, len(dfs), CH):
        preds = predictor.predict_batch(
            df_list=dfs[i:i + CH], x_timestamp_list=xts[i:i + CH],
            y_timestamp_list=yts[i:i + CH], pred_len=PRED_LEN,
            sample_count=SAMPLE_COUNT, T=TEMPERATURE, verbose=False)
        for j, p in enumerate(preds):
            t = names[i + j]
            entry = dfs[i + j]["close"].iloc[-1]
            out[t] = float(p.iloc[-1]["close"] / entry - 1.0)
    log(f"    signals for {len(out)} tickers")
    return out


# ── book ──────────────────────────────────────────────────────────────────────
def book_dir(bid):
    d = os.path.join(BOOK_ROOT, bid)
    os.makedirs(d, exist_ok=True)
    return d


def load_state(bid):
    p = os.path.join(book_dir(bid), "state.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"nav": START_NAV, "cash": START_NAV, "positions": [],
            "last_rebalance": None, "inception": None,
            "bench_nav": START_NAV, "bench_shares": {}, "history_dates": []}


def rebalance(state, signals, prices, universe, construction, top_n, asof):
    """Close everything, then open the new Top-N (and Bottom-N for ls)."""
    avail = {t: p for t, p in prices.items() if t in signals and p and p > 0}
    ranked = sorted(avail, key=lambda t: signals[t], reverse=True)
    if len(ranked) < top_n * 2:
        raise RuntimeError(f"{universe}: only {len(ranked)} priced tickers, need {top_n*2}")

    nav = mark(state, prices)          # liquidate at today's marks
    longs = ranked[:top_n]
    shorts = ranked[-top_n:] if construction == "ls" else []

    positions = []
    w_long = (1.0 / top_n) if construction == "long" else (LEG / top_n)
    for t in longs:
        positions.append({"ticker": t, "side": "LONG", "shares": nav * w_long / prices[t],
                          "entry_price": prices[t], "entry_date": str(asof.date())})
    for t in shorts:
        positions.append({"ticker": t, "side": "SHORT", "shares": -nav * (LEG / top_n) / prices[t],
                          "entry_price": prices[t], "entry_date": str(asof.date())})

    gross = sum(p["shares"] * p["entry_price"] for p in positions)
    state["positions"] = positions
    state["cash"] = nav - gross
    state["last_rebalance"] = str(asof.date())
    return nav


def mark(state, prices):
    """Mark-to-market: NAV = cash + sum(shares * last price)."""
    v = state["cash"]
    for p in state["positions"]:
        px = prices.get(p["ticker"])
        v += p["shares"] * (px if px else p["entry_price"])
    return v


def trading_days_since(px, last_reb, asof):
    if last_reb is None:
        return 10 ** 6
    d = sorted(px[(px["date"] > pd.Timestamp(last_reb)) & (px["date"] <= asof)]["date"].unique())
    return len(d)


def publish(bid, state, prices, stale, asof, universe, construction, top_n, signals, log):
    d = book_dir(bid)
    nav = mark(state, prices)
    prev = state.get("nav", nav)
    ret_1d = (nav / prev - 1.0) * 100.0 if prev else 0.0
    cum = (nav / START_NAV - 1.0) * 100.0

    # equal-weight benchmark over the whole universe, same inception
    if not state.get("bench_shares"):
        n = len([t for t in prices if prices[t] > 0])
        state["bench_shares"] = {t: (START_NAV / n) / prices[t] for t in prices if prices[t] > 0}
    bench_nav = sum(s * prices.get(t, 0) for t, s in state["bench_shares"].items() if prices.get(t))
    bench_cum = (bench_nav / START_NAV - 1.0) * 100.0

    n_stale = sum(1 for p in state["positions"] if stale.get(p["ticker"], 0) > 3)
    navp = os.path.join(d, "nav_daily.csv")
    hdr = not os.path.exists(navp)
    row = pd.DataFrame([{
        "date": str(asof.date()), "nav": round(nav, 2), "ret_1d_pct": round(ret_1d, 4),
        "cum_ret_pct": round(cum, 4), "bench_nav": round(bench_nav, 2),
        "bench_cum_ret_pct": round(bench_cum, 4),
        "n_positions": len(state["positions"]), "marked_stale": n_stale}])
    existing = pd.read_csv(navp) if not hdr else pd.DataFrame()
    if len(existing) and str(asof.date()) in set(existing["date"]):
        existing = existing[existing["date"] != str(asof.date())]
    out = pd.concat([existing, row], ignore_index=True) if len(existing) else row
    atomic_write(navp, lambda p: out.to_csv(p, index=False))

    hold = pd.DataFrame([{
        "ticker": p["ticker"], "side": p["side"], "shares": round(p["shares"], 4),
        "entry_price": round(p["entry_price"], 4), "entry_date": p["entry_date"],
        "last_price": round(prices.get(p["ticker"], float("nan")), 4),
        "weight_pct": round(100 * p["shares"] * prices.get(p["ticker"], p["entry_price"]) / nav, 3),
        "price_stale_days": stale.get(p["ticker"], 0),
    } for p in state["positions"]])
    atomic_write(os.path.join(d, "holdings_current.csv"), lambda p: hold.to_csv(p, index=False))

    if signals:
        s = pd.DataFrame(sorted(signals.items(), key=lambda kv: -kv[1]),
                         columns=["ticker", "pred_return"])
        s["rank"] = range(1, len(s) + 1)
        atomic_write(os.path.join(d, "signals_latest.csv"), lambda p: s.to_csv(p, index=False))

    flags = []
    if n_stale:
        flags.append("stale_marks")
    if universe in SURVIVORSHIP_BIASED:
        flags.append("backtest_survivorship_biased")
    meta = {
        "id": bid, "universe": universe, "construction": construction,
        "top_n_per_side": top_n, "leg_weight": LEG if construction == "ls" else None,
        "lookback": LOOKBACK, "pred_len": PRED_LEN, "hold_days": HOLD_DAYS,
        "sample_count": SAMPLE_COUNT, "temperature": TEMPERATURE, "seed": RANDOM_SEED,
        "start_nav_usd": START_NAV, "as_of": str(asof.date()),
        "inception": state.get("inception"), "last_rebalance": state.get("last_rebalance"),
        "nav": round(nav, 2), "cum_ret_pct": round(cum, 4),
        "bench": "equal-weight universe", "bench_cum_ret_pct": round(bench_cum, 4),
        "data_source": "yfinance", "bloomberg_required": False, "flags": flags,
        "caveat": ("Forward paper tracking is free of survivorship bias by construction; "
                   "the historical backtest for this universe is not."
                   if universe in SURVIVORSHIP_BIASED else
                   "ETF universe; panel widens honestly as funds launch."),
        "updated_utc": datetime.utcnow().isoformat() + "Z",
    }
    atomic_write(os.path.join(d, "meta.json"),
                 lambda p: json.dump(meta, open(p, "w"), indent=2))

    state["nav"] = nav
    state["bench_nav"] = bench_nav
    atomic_write(os.path.join(d, "state.json"),
                 lambda p: json.dump(state, open(p, "w"), indent=2))
    return nav, ret_1d, cum, bench_cum


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universes", nargs="*", default=None)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD (default: latest bar available)")
    args = ap.parse_args()

    os.makedirs(BOOK_ROOT, exist_ok=True)
    os.makedirs(os.path.join(BOOK_ROOT, "run_log"), exist_ok=True)
    logp = os.path.join(BOOK_ROOT, "run_log", f"{date.today()}.log")
    fh = open(logp, "a")

    def log(m=""):
        print(m, flush=True)
        fh.write(f"{datetime.now():%H:%M:%S} {m}\n")
        fh.flush()

    def beat(m):
        with open(os.path.join(BOOK_ROOT, "heartbeat.txt"), "w") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}\n")

    unis = args.universes or list(UNIVERSES)
    bad = [u for u in unis if u not in UNIVERSES]
    if bad:
        raise SystemExit(f"FAIL: unknown universes {bad}. Known: {list(UNIVERSES)}")

    log("=" * 78)
    log(f"tracker_daily.py  {datetime.now():%Y-%m-%d %H:%M:%S}  universes={len(unis)}")
    log(f"seed={RANDOM_SEED} sample_count={SAMPLE_COUNT} hold={HOLD_DAYS}d "
        f"leg={LEG}  {'DRY RUN' if args.dry_run else 'publishing'}")

    predictor = None
    summary = []
    for ui, u in enumerate(unis, 1):
        cache_dir, top_n = UNIVERSES[u]
        log(f"\n[{ui}/{len(unis)}] {u}  ({cache_dir}, top-{top_n}/side)")
        beat(f"{u}: refreshing prices ({ui}/{len(unis)})")
        try:
            px = refresh_prices(u, log, download=not args.no_download)
        except Exception as e:
            log(f"  PRICE FAIL: {type(e).__name__}: {e}")
            summary.append((u, "PRICE_FAIL", None, None))
            continue

        asof = pd.Timestamp(args.asof) if args.asof else px["date"].max()
        prices, stale = last_price_map(px, asof)
        log(f"  as_of {asof.date()}  {len(prices)} priced tickers")

        # Signals depend only on the universe and the date, never on the
        # construction, so generate them ONCE and share across both books.
        states = {c: load_state(f"kronos-{u}-{c}") for c in ("long", "ls")}
        for st in states.values():
            if st["inception"] is None:
                st["inception"] = str(asof.date())
        due_any = any(trading_days_since(px, st["last_rebalance"], asof) >= HOLD_DAYS
                      for st in states.values())

        shared_signals = None
        if due_any:
            if predictor is None:
                log("  loading Kronos-base...")
                from model import Kronos, KronosTokenizer, KronosPredictor
                tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
                mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
                predictor = KronosPredictor(mdl, tok, max_context=512)
            beat(f"{u}: generating signals ({ui}/{len(unis)})")
            t0 = time.time()
            try:
                shared_signals = generate_signals(px, asof, predictor, log)
                log(f"  signals done ({time.time()-t0:.0f}s, shared by both books)")
            except Exception as e:
                log(f"  {u} SIGNAL FAIL: {type(e).__name__}: {e}")
                for c in ("long", "ls"):
                    summary.append((f"kronos-{u}-{c}", "SIGNAL_FAIL", None, None))
                continue

        for construction in ("long", "ls"):
            bid = f"kronos-{u}-{construction}"
            state = states[construction]
            due = trading_days_since(px, state["last_rebalance"], asof) >= HOLD_DAYS
            signals = shared_signals if due else None

            if due and shared_signals:
                try:
                    rebalance(state, shared_signals, prices, u, construction, top_n, asof)
                    log(f"  {bid}: REBALANCED")
                except Exception as e:
                    log(f"  {bid} REBALANCE FAIL: {type(e).__name__}: {e}")
                    summary.append((bid, "REBAL_FAIL", None, None))
                    continue
            else:
                log(f"  {bid}: holding (marks only)")

            if args.dry_run:
                log(f"  {bid}: dry-run, nothing written")
                continue
            nav, r1, cum, bcum = publish(bid, state, prices, stale, asof,
                                         u, construction, top_n, signals, log)
            log(f"  {bid}: NAV ${nav:,.0f}  1d {r1:+.2f}%  cum {cum:+.2f}%  "
                f"(bench {bcum:+.2f}%)")
            summary.append((bid, "ok", cum, bcum))

        # Release the universe's price frame and any MPS scratch before the next
        # one. Without this an 11-universe run accumulates enough resident memory
        # that the OS SIGKILLs the process partway through — observed on the
        # first full run, which died silently on uk500 (universe 11 of 11) with
        # no Python traceback, then completed fine when run alone.
        del px, prices, stale, shared_signals, states
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    log("\n" + "=" * 78)
    ok = [s for s in summary if s[1] == "ok"]
    log(f"  {len(ok)}/{len(summary)} books published")
    for bid, st, cum, bcum in summary:
        log(f"    {bid:34s} {st:12s} " + (f"cum {cum:+7.2f}%  bench {bcum:+7.2f}%"
                                          if cum is not None else ""))
    beat("done")
    fh.close()


if __name__ == "__main__":
    main()
