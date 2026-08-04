"""
rescore_from_cache.py — recompute every Kronos backtest from cached forecasts
=============================================================================

WHAT THIS DOES
--------------
The April 2026 Kronos research sweep ran zero-shot Kronos-base inference over a
dozen universes and saved every (date, ticker, pred_return, actual_return) row
to `forecasts_<Universe>.csv` at this repo's root. Those files are the *entire*
input to the reported metrics — IC, top/bottom decile returns, Sharpe, L/S
spread, permutation p-value.

This script re-derives all of those metrics from the cached forecast files
alone. It does NOT load the Kronos model, does NOT touch HuggingFace Hub, does
NOT call yfinance or Bloomberg, and does NOT need the internet. It exists so the
published numbers can be re-audited, re-cut (different decile, different OOS
date), and reproduced without spending a single data-vendor call.

WHAT IT CANNOT DO
-----------------
It cannot fix survivorship bias. The cached forecasts only contain tickers that
were in the universe snapshot as of April 2026 and had full price history.
Names that delisted between 2015 and 2026 are absent from these files and can
never be recovered by re-scoring — that requires a fresh point-in-time universe
pull (see the Global Data Mart equities intake memo).

METHODOLOGY (mirrors large_universe_test.py:199-291 exactly)
------------------------------------------------------------
  * drop rows with null actual_return; drop |actual_return| >= 0.5 as outliers
  * per rebalance date: rank by pred_return, take top/bottom DECILE (10%) and
    top/bottom half; skip dates with fewer than 4x the bin size
  * annualise with periods_per_year = 252 / stride, where stride is INFERRED
    from the median calendar gap between rebalance dates in each file
  * Spearman IC per date, averaged
  * permutation test: 500 random-selection shuffles, seed 42, p = fraction of
    null Sharpes >= realised top-decile Sharpe
  * OOS split at 2025-01-01, matching kronos_validation_master_report.md

INPUT FILES (all absolute, all must already exist)
--------------------------------------------------
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_*.csv
      One per universe. Schema: date,ticker,pred_return,actual_return

OUTPUT FILES (written to a fresh timestamped run directory)
------------------------------------------------------------
  <repo>/runs/rescore_<YYYYMMDD_HHMMSS>/rescore_results.parquet   canonical
  <repo>/runs/rescore_<YYYYMMDD_HHMMSS>/rescore_results.xlsx      eyeball copy
  <repo>/runs/rescore_<YYYYMMDD_HHMMSS>/summary.json              run metadata
  <repo>/runs/rescore_<YYYYMMDD_HHMMSS>/rescore.log               full console log

USAGE
-----
  .venv/bin/python rescore_from_cache.py                # all cached universes
  .venv/bin/python rescore_from_cache.py --decile 0.02  # re-cut at top 2%
  .venv/bin/python rescore_from_cache.py --no-permutation   # faster

DEPENDENCIES
------------
  pandas, numpy, scipy, pyarrow, openpyxl  (all present in .venv, Python 3.9.6)

NOTES
-----
Results are written incrementally: each universe is appended to the parquet and
log as soon as it finishes, so a crash mid-run leaves completed universes intact.

Version 1.0 — 2026-08-04
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.abspath(__file__))
OOS_START = "2025-01-01"


def _stats(rets, periods_per_year):
    """Annualised return, vol, Sharpe from a series of per-period returns."""
    rets = np.asarray(rets, dtype=float)
    if len(rets) == 0:
        return 0.0, 0.0, 0.0
    ann_ret = (1 + rets.mean()) ** periods_per_year - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    return float(ann_ret), float(ann_vol), float(ann_ret / ann_vol) if ann_vol > 0 else 0.0


def infer_periods_per_year(dates):
    """
    Infer rebalance frequency as 252 / (median BUSINESS-day gap between dates).

    Business days, not calendar days: the original scripts annualise with a
    hardcoded 252/STRIDE (large_universe_test.py:205), so a stride-5 daily run
    must come back as exactly 50.4, not 365.25/7 = 52.2. Getting this wrong
    inflates both annualised return and Sharpe by ~3%.
    """
    d = pd.to_datetime(pd.Series(sorted(dates))).values.astype("datetime64[D]")
    if len(d) < 3:
        return 252.0 / 5
    gaps = np.busday_count(d[:-1], d[1:]).astype(float)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 252.0 / 5
    return 252.0 / float(np.median(gaps))


def bin_size_for(n, decile, top_n_abs):
    """Bin size at one date: absolute N if given, else a fraction of the cross-section."""
    if top_n_abs is not None:
        return top_n_abs
    return max(1, int(n * decile))


def analyze(df, decile, label, run_permutation=True, top_n_abs=None):
    """Recompute all headline metrics for one universe / one window."""
    df = df.dropna(subset=["actual_return"])
    df = df[df["actual_return"].abs() < 0.5]
    if df.empty:
        return None

    dates = sorted(df["date"].unique())
    ppy = infer_periods_per_year(dates)

    by_date = {d: g.sort_values("pred_return", ascending=False) for d, g in df.groupby("date")}

    # absolute-N mode mirrors tech_stocks_test.py:236 (n < TOP_N*2);
    # decile mode mirrors large_universe_test.py:225 (n < top_n*4)
    min_mult = 2 if top_n_abs is not None else 4

    ic_list, top10p, bot10p, top50, bot50, bench = [], [], [], [], [], []
    used_dates = []
    for d in dates:
        snap = by_date[d]
        n = len(snap)
        top_n = bin_size_for(n, decile, top_n_abs)
        if n < top_n * min_mult:
            continue
        ic, _ = spearmanr(snap["pred_return"], snap["actual_return"])
        if not np.isnan(ic):
            ic_list.append(ic)
        mid = n // 2
        top10p.append(snap.head(top_n)["actual_return"].mean())
        bot10p.append(snap.tail(top_n)["actual_return"].mean())
        top50.append(snap.head(mid)["actual_return"].mean())
        bot50.append(snap.tail(n - mid)["actual_return"].mean())
        bench.append(snap["actual_return"].mean())
        used_dates.append(d)

    if not top10p:
        return None

    t10_ann, _, t10_sr = _stats(top10p, ppy)
    b10_ann, _, b10_sr = _stats(bot10p, ppy)
    _, _, t50_sr = _stats(top50, ppy)
    _, _, b50_sr = _stats(bot50, ppy)
    bm_ann, _, bm_sr = _stats(bench, ppy)

    ls = np.array(top10p) - np.array(bot10p)
    ls_ann = float(ls.mean() * ppy)
    ls_vol = float(ls.std() * np.sqrt(ppy))
    ls_sr = ls_ann / ls_vol if ls_vol > 0 else 0.0

    p_value = np.nan
    if run_permutation:
        rng = np.random.RandomState(42)
        null_sharpes = []
        pools = [by_date[d]["actual_return"].values for d in used_dates]
        sizes = [bin_size_for(len(by_date[d]), decile, top_n_abs) for d in used_dates]
        for _ in range(500):
            shuf = [rng.choice(pool, size=k, replace=False).mean()
                    for pool, k in zip(pools, sizes)]
            _, _, s = _stats(shuf, ppy)
            null_sharpes.append(s)
        p_value = float(np.mean(np.array(null_sharpes) >= t10_sr))

    return {
        "window": label,
        "periods": len(top10p),
        "avg_names_per_date": float(df.groupby("date").size().mean()),
        "bin_size": int(df.groupby("date").size().mean() * decile),
        "periods_per_year": round(ppy, 2),
        "avg_IC": float(np.mean(ic_list)) if ic_list else 0.0,
        "top_ann_ret": t10_ann,
        "top_sharpe": t10_sr,
        "bot_ann_ret": b10_ann,
        "bot_sharpe": b10_sr,
        "ls_ann_ret": ls_ann,
        "ls_sharpe": ls_sr,
        "tophalf_sharpe": t50_sr,
        "bothalf_sharpe": b50_sr,
        "bench_ann_ret": bm_ann,
        "bench_sharpe": bm_sr,
        "p_value": p_value,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decile", type=float, default=0.10,
                    help="top/bottom fraction (default 0.10, matching large_universe_test.py)")
    ap.add_argument("--top-n", type=int, default=None,
                    help="absolute names per side (e.g. 10 for stocks, 3 for ETFs). "
                         "Overrides --decile; this is what the published "
                         "master_validation_summary.csv numbers used.")
    ap.add_argument("--no-permutation", action="store_true", help="skip the 500-shuffle null test")
    ap.add_argument("--files", nargs="*", default=None, help="specific forecasts_*.csv to score")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO, "runs", f"rescore_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "rescore.log")
    log_fh = open(log_path, "w")

    def log(msg=""):
        print(msg)
        log_fh.write(str(msg) + "\n")
        log_fh.flush()

    files = args.files or sorted(
        f for f in os.listdir(REPO)
        if f.startswith("forecasts_") and f.endswith(".csv") and "checkpoint" not in f
    )

    sel = f"top_n={args.top_n} names/side" if args.top_n else f"decile={args.decile}"
    log(f"rescore_from_cache.py  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"selection: {sel}  permutation={'off' if args.no_permutation else 'on (500x, seed 42)'}")
    log(f"run dir: {run_dir}")
    log(f"{len(files)} cached forecast files found\n")
    log("NOTE: re-scoring cached forecasts cannot correct survivorship bias — the")
    log("      delisted tail was never in these files. See MEMO-INTAKE.md.\n")

    rows = []
    parquet_path = os.path.join(run_dir, "rescore_results.parquet")

    for fname in files:
        path = os.path.join(REPO, fname)
        universe = fname.replace("forecasts_", "").replace(".csv", "")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            log(f"[SKIP] {universe}: unreadable — {type(e).__name__}: {e}")
            continue

        needed = {"date", "ticker", "pred_return", "actual_return"}
        if not needed.issubset(df.columns):
            log(f"[SKIP] {universe}: schema mismatch, has {list(df.columns)}")
            continue

        log(f"── {universe}  ({len(df):,} rows, {df['ticker'].nunique()} tickers, "
            f"{df['date'].min()} → {df['date'].max()})")

        for label, sub in (("Full", df), ("OOS_2025+", df[df["date"] >= OOS_START])):
            res = analyze(sub, args.decile, label,
                          run_permutation=not args.no_permutation, top_n_abs=args.top_n)
            if res is None:
                log(f"     {label:10s} — no scoreable periods")
                continue
            res["universe"] = universe
            res["n_tickers"] = int(sub["ticker"].nunique())
            rows.append(res)
            log(f"     {label:10s} IC={res['avg_IC']:+.4f}  "
                f"Top={res['top_ann_ret']:7.1%} (SR {res['top_sharpe']:5.2f})  "
                f"Bot={res['bot_ann_ret']:7.1%} (SR {res['bot_sharpe']:5.2f})  "
                f"L/S SR={res['ls_sharpe']:5.2f}  "
                f"Bench={res['bench_ann_ret']:7.1%} (SR {res['bench_sharpe']:5.2f})  "
                f"p={res['p_value']:.3f}" if not np.isnan(res["p_value"]) else
                f"     {label:10s} IC={res['avg_IC']:+.4f}  "
                f"Top={res['top_ann_ret']:7.1%} (SR {res['top_sharpe']:5.2f})  "
                f"Bot={res['bot_ann_ret']:7.1%} (SR {res['bot_sharpe']:5.2f})  "
                f"L/S SR={res['ls_sharpe']:5.2f}  "
                f"Bench={res['bench_ann_ret']:7.1%} (SR {res['bench_sharpe']:5.2f})")

            # incremental write — survive a crash mid-run
            tmp = parquet_path + ".tmp"
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            os.replace(tmp, parquet_path)
        log("")

    if not rows:
        log("NO RESULTS — nothing scoreable.")
        log_fh.close()
        sys.exit(1)

    out = pd.DataFrame(rows)
    cols = ["universe", "window", "n_tickers", "periods", "avg_names_per_date", "bin_size",
            "periods_per_year", "avg_IC", "top_ann_ret", "top_sharpe", "bot_ann_ret",
            "bot_sharpe", "ls_ann_ret", "ls_sharpe", "tophalf_sharpe", "bothalf_sharpe",
            "bench_ann_ret", "bench_sharpe", "p_value"]
    out = out[cols]

    tmp = parquet_path + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, parquet_path)

    xlsx_path = os.path.join(run_dir, "rescore_results.xlsx")
    out.to_excel(xlsx_path, index=False, sheet_name="rescore")

    summary = {
        "run_utc": datetime.utcnow().isoformat() + "Z",
        "script": os.path.abspath(__file__),
        "selection": ("top_n_names" if args.top_n else "decile_fraction"),
        "top_n": args.top_n,
        "decile": args.decile,
        "permutation": not args.no_permutation,
        "oos_start": OOS_START,
        "universes_scored": sorted(out["universe"].unique().tolist()),
        "n_rows": len(out),
        "network_used": False,
        "model_loaded": False,
        "survivorship_complete": False,
        "survivorship_note": (
            "Cached forecasts contain only April-2026 universe members with full "
            "price history. Delisted names are absent and cannot be recovered by "
            "re-scoring. Results are biased upward."
        ),
        "outputs": {"parquet": parquet_path, "xlsx": xlsx_path, "log": log_path},
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    log("=" * 100)
    log(f"DONE — {len(out)} rows across {out['universe'].nunique()} universes")
    log(f"  parquet : {parquet_path}")
    log(f"  xlsx    : {xlsx_path}")
    log(f"  summary : {os.path.join(run_dir, 'summary.json')}")
    log(f"  log     : {log_path}")
    log_fh.close()


if __name__ == "__main__":
    main()
