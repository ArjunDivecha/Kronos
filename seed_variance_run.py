"""
seed_variance_run.py — how much of the Industry28 headline is sampling noise?
=============================================================================

WHAT THIS DOES
--------------
Kronos is a generative model: `predict_batch` SAMPLES from the predictive
distribution at temperature T=0.8. Neither `model/kronos.py` nor
`run_industry28.py` sets a random seed, so the published Industry28 backtest
(+22.0% / 1.00 Sharpe) is ONE unseeded stochastic realisation at sample_count=10.
Re-run it and you get a different answer; nobody has ever measured how different.

This script re-runs the identical walk-forward backtest N times under N fixed
seeds and reports the DISTRIBUTION of every headline metric, converting
"22.0%" into "22.0% +/- x". It also answers whether the 40/5 parameter cell was
genuinely good or a lucky draw, which the original sweep could not do because
`industry_grid_sweep.py` measured every cell at sample_count=1.

WHAT IT DOES NOT TOUCH
----------------------
It never writes forecasts_Industry28.csv, monthly_returns_industry28.csv or
industry28_metrics.txt. Those are the published evidence and are left exactly
as they are. Every artefact goes into a fresh timestamped run directory.

METHOD
------
  * price data is read from the LOCAL cache data/Industry28/*.csv — no yfinance,
    no network beyond the one-time HuggingFace model download
  * walk-forward loop, batching and return definition are copied verbatim from
    run_industry28.py:137-213 (open[t+1] -> close[t+5], LOOKBACK=40, PRED_LEN=5,
    STRIDE=5, sample_count=10, T=0.8) so the runs are comparable to the headline
  * torch.manual_seed(seed) is set once before each full walk-forward pass
  * scoring reuses analyze() from rescore_from_cache.py at top-3 per side,
    ddof=1 — the convention run_industry28.py published with
  * batch failures are COUNTED and reported, never silently swallowed; a run
    that loses more than MAX_FAIL_FRAC of its batches is marked FAILED rather
    than reported as a result

INPUT FILES (absolute)
----------------------
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/Industry28/<TICKER>.csv
  HuggingFace Hub: NeoQuasar/Kronos-base, NeoQuasar/Kronos-Tokenizer-base (downloaded on first run)

OUTPUT FILES (timestamped run directory)
-----------------------------------------
  <repo>/runs/seedvar_<YYYYMMDD_HHMMSS>/forecasts_seed<NN>.csv   per-seed forecasts
  <repo>/runs/seedvar_<YYYYMMDD_HHMMSS>/seed_metrics.parquet     per-seed metrics (canonical)
  <repo>/runs/seedvar_<YYYYMMDD_HHMMSS>/seed_metrics.xlsx        eyeball copy
  <repo>/runs/seedvar_<YYYYMMDD_HHMMSS>/distribution.json        mean/std/min/max per metric
  <repo>/runs/seedvar_<YYYYMMDD_HHMMSS>/heartbeat.txt            live progress, inspectable mid-run
  <repo>/runs/seedvar_<YYYYMMDD_HHMMSS>/seedvar.log              full log

USAGE
-----
  .venv/bin/python seed_variance_run.py --probe 20        # timing probe, 1 seed, 20 periods
  .venv/bin/python seed_variance_run.py --seeds 10        # full: 10 seeds, all periods

DEPENDENCIES
------------
  torch (MPS), pandas, numpy, scipy, pyarrow, openpyxl, tqdm, huggingface_hub;
  imports analyze() from rescore_from_cache.py

NOTES
-----
Per-seed forecasts are flushed to disk as each seed completes, and heartbeat.txt
is rewritten every batch, so a crash or a kill loses at most the seed in flight
and progress can be watched from another shell.

Version 1.0 — 2026-08-04
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor
from rescore_from_cache import analyze

REPO = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO, "data", "Industry28")

# copied verbatim from run_industry28.py so results are comparable to the headline
LOOKBACK, PRED_LEN, STRIDE = 40, 5, 5
SAMPLE_COUNT, TEMPERATURE = 10, 0.8
MAX_FAIL_FRAC = 0.02


def load_panel(log):
    etf_dfs = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith(".csv") or f.startswith("_"):
            continue
        t = f[:-4]
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        if len(df) >= LOOKBACK + PRED_LEN:
            etf_dfs[t] = df
    if not etf_dfs:
        raise SystemExit(f"FAIL: no usable CSVs in {DATA_DIR}")
    log(f"  loaded {len(etf_dfs)} tickers from local cache (no network)")
    return etf_dfs


def build_batches(etf_dfs, max_periods=None, start_date=None):
    """
    Pre-build every walk-forward batch once; identical across seeds.

    start_date restricts the REBALANCE dates only. The 40-bar lookback still
    reaches back before it, which is correct: we are asking how stable the
    signal is over a window, not pretending the model has no history.
    """
    ref = max(etf_dfs, key=lambda t: len(etf_dfs[t]))
    ref_dates = np.sort(etf_dfs[ref]["timestamps"].unique())
    idxs = list(range(LOOKBACK, len(ref_dates) - PRED_LEN, STRIDE))
    if start_date:
        cut = np.datetime64(pd.Timestamp(start_date))
        idxs = [i for i in idxs if ref_dates[i] >= cut]
    if max_periods:
        idxs = idxs[:max_periods]

    batches = []
    for i in idxs:
        cur = ref_dates[i]
        dfs, xts, yts, meta = [], [], [], []
        for ticker, df in etf_dfs.items():
            mask = df["timestamps"] <= cur
            if not mask.any():
                continue
            idx = df[mask].index[-1]
            if idx < LOOKBACK - 1 or idx + PRED_LEN >= len(df) or idx + 1 >= len(df):
                continue
            x_df = df.iloc[idx - LOOKBACK + 1: idx + 1]
            if x_df[["open", "high", "low", "close", "volume", "amount"]].isnull().values.any():
                continue
            dfs.append(x_df[["open", "high", "low", "close", "volume", "amount"]])
            xts.append(x_df["timestamps"])
            yts.append(df.iloc[idx + 1: idx + 1 + PRED_LEN]["timestamps"])
            meta.append({
                "ticker": ticker,
                "date": pd.Timestamp(cur),
                "entry_open": df.iloc[idx + 1]["open"],
                "actual_return": (df.iloc[idx + PRED_LEN]["close"] / df.iloc[idx + 1]["open"]) - 1,
            })
        if len(dfs) >= 5:
            batches.append((dfs, xts, yts, meta))
    return batches


def run_one_seed(predictor, batches, seed, log, beat):
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows, n_fail = [], 0
    t0 = time.time()
    for k, (dfs, xts, yts, meta) in enumerate(
            tqdm(batches, desc=f"seed {seed}", leave=False, file=sys.stderr)):
        try:
            preds = predictor.predict_batch(
                df_list=dfs, x_timestamp_list=xts, y_timestamp_list=yts,
                pred_len=PRED_LEN, sample_count=SAMPLE_COUNT, T=TEMPERATURE, verbose=False)
        except Exception as e:
            n_fail += 1
            log(f"    batch {k} FAILED: {type(e).__name__}: {e}")
            continue
        for j, p in enumerate(preds):
            m = meta[j]
            rows.append({"date": m["date"], "ticker": m["ticker"],
                         "pred_return": (p.iloc[-1]["close"] / m["entry_open"]) - 1,
                         "actual_return": m["actual_return"]})
        if k % 5 == 0:
            beat(f"seed {seed}: batch {k+1}/{len(batches)}  "
                 f"({(k+1)/len(batches):.1%})  elapsed {time.time()-t0:.0f}s  fails {n_fail}")
    return pd.DataFrame(rows), n_fail, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (default 10)")
    ap.add_argument("--probe", type=int, default=None,
                    help="timing probe: run 1 seed over only this many periods")
    ap.add_argument("--start-date", default=None,
                    help="restrict rebalance dates to >= this (e.g. 2025-06-30, the date "
                         "Kronos-base weights were uploaded to HF and the first bar the "
                         "frozen model provably never saw)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO, "runs", f"seedvar_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_fh = open(os.path.join(run_dir, "seedvar.log"), "w")
    beat_path = os.path.join(run_dir, "heartbeat.txt")

    def log(m=""):
        print(m, flush=True)
        log_fh.write(str(m) + "\n")
        log_fh.flush()

    def beat(m):
        with open(beat_path, "w") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {m}\n")

    log(f"seed_variance_run.py  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"run dir: {run_dir}")
    log(f"config: LOOKBACK={LOOKBACK} PRED_LEN={PRED_LEN} STRIDE={STRIDE} "
        f"sample_count={SAMPLE_COUNT} T={TEMPERATURE}  (matches run_industry28.py)")
    log("")

    etf_dfs = load_panel(log)
    batches = build_batches(etf_dfs, max_periods=args.probe, start_date=args.start_date)
    log(f"  {len(batches)} walk-forward batches built"
        f"{' (PROBE)' if args.probe else ''}"
        f"{f'  [rebalances >= {args.start_date}]' if args.start_date else ''}")
    if not batches:
        raise SystemExit("FAIL: no batches in the requested window")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"  device: {device}")
    log("  loading Kronos-base from HuggingFace Hub...")
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(mdl, tok, max_context=512)
    log("  model ready")
    log("")

    seeds = [42] if args.probe else list(range(101, 101 + args.seeds))
    results, pq = [], os.path.join(run_dir, "seed_metrics.parquet")

    for s in seeds:
        fc, n_fail, secs = run_one_seed(predictor, batches, s, log, beat)
        if fc.empty:
            log(f"  seed {s}: FAILED — no forecasts produced")
            continue
        frac = n_fail / max(1, len(batches))
        status = "ok" if frac <= MAX_FAIL_FRAC else "FAILED"
        fc["date"] = fc["date"].dt.strftime("%Y-%m-%d")
        fc.to_csv(os.path.join(run_dir, f"forecasts_seed{s}.csv"), index=False)

        r = analyze(fc, 0.10, "Full", run_permutation=False, top_n_abs=3, ddof=1)
        rec = {"seed": s, "status": status, "batch_failures": n_fail,
               "fail_frac": round(frac, 4), "seconds": round(secs, 1),
               "rows": len(fc), "periods": r["periods"], "avg_IC": r["avg_IC"],
               "top_ann_ret": r["top_ann_ret"], "top_sharpe": r["top_sharpe"],
               "bot_ann_ret": r["bot_ann_ret"], "bot_sharpe": r["bot_sharpe"],
               "ls_sharpe": r["ls_sharpe"], "bench_ann_ret": r["bench_ann_ret"],
               "bench_sharpe": r["bench_sharpe"]}
        results.append(rec)
        pd.DataFrame(results).to_parquet(pq, index=False)   # incremental
        log(f"  seed {s:3d} [{status}]  {secs/60:5.1f} min  IC={r['avg_IC']:+.4f}  "
            f"Top={r['top_ann_ret']:+7.1%} (SR {r['top_sharpe']:.2f})  "
            f"L/S SR={r['ls_sharpe']:.2f}  fails={n_fail}")

    if not results:
        log("NO SEEDS COMPLETED — nothing to report.")
        log_fh.close()
        sys.exit(1)

    out = pd.DataFrame(results)
    out.to_parquet(pq, index=False)
    out.to_excel(os.path.join(run_dir, "seed_metrics.xlsx"), index=False)

    ok = out[out["status"] == "ok"]
    dist = {}
    log("")
    log("=" * 84)
    log(f"  DISTRIBUTION ACROSS {len(ok)} SEEDS   (published headline: "
        f"Top +22.0% / 1.00 Sharpe, IC 0.0266)")
    log("=" * 84)
    log(f"  {'metric':16s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s} {'range':>10s}")
    for m in ["avg_IC", "top_ann_ret", "top_sharpe", "ls_sharpe", "bench_sharpe"]:
        v = ok[m].astype(float)
        dist[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                   "min": float(v.min()), "max": float(v.max()),
                   "range": float(v.max() - v.min()), "n": int(len(v))}
        log(f"  {m:16s} {v.mean():>10.4f} {v.std(ddof=1) if len(v)>1 else 0:>10.4f} "
            f"{v.min():>10.4f} {v.max():>10.4f} {v.max()-v.min():>10.4f}")
    log("=" * 84)

    with open(os.path.join(run_dir, "distribution.json"), "w") as fh:
        json.dump({"run_utc": datetime.utcnow().isoformat() + "Z",
                   "seeds_requested": len(seeds), "seeds_ok": int(len(ok)),
                   "start_date": args.start_date,
                   "config": {"lookback": LOOKBACK, "pred_len": PRED_LEN,
                              "stride": STRIDE, "sample_count": SAMPLE_COUNT,
                              "T": TEMPERATURE, "device": device},
                   "published_headline": {"top_ann_ret": 0.220, "top_sharpe": 1.00,
                                          "avg_IC": 0.0266},
                   "distribution": dist,
                   "per_seed": results}, fh, indent=2)
    beat("done")
    log(f"  parquet : {pq}")
    log(f"  run dir : {run_dir}")
    log_fh.close()


if __name__ == "__main__":
    main()
