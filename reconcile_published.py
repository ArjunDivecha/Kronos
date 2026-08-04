"""
reconcile_published.py — do the published Kronos numbers still reproduce?
=========================================================================

WHAT THIS DOES
--------------
Takes the headline metrics that were published in April 2026 (in
master_validation_summary.csv and industry28_metrics.txt) and checks each one
against a fresh re-score of the cached forecast files, using the exact selection
convention each number was originally computed with. Emits a PASS/FAIL
reconciliation so it is unambiguous which published claims are still backed by
evidence on disk and which are not.

This is an audit, not a backtest. It loads no model and touches no network.

WHY IT EXISTS
-------------
`forecasts_*.csv` files are overwritten in place whenever a script is re-run
(the repo has no cache-invalidation fingerprint — see CLAUDE.md "Caches are
loaded without invalidation"). If a cache was regenerated after a number was
published, the evidence behind that number is gone. Comparing file mtimes
against the summary mtime tells you where to expect that, and this script
confirms it numerically.

SELECTION CONVENTIONS (each published number used its own — this is the trap)
------------------------------------------------------------------------------
  S&P 500 / SmallCap-600   top-10 NAMES per side   (master_validation_summary.csv)
  Tech Stocks              top-5  NAMES per side   (master_validation_summary.csv)
  Industry28               top-3  NAMES per side   (run_industry28.py:226, nlargest(3))
Note large_universe_test.py uses DECILE=0.10 (a *fraction*), which is NOT what
the published stock table reports. Annualisation is 252 / business-day stride.

INPUT FILES (absolute)
----------------------
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/master_validation_summary.csv
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/industry28_metrics.txt
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_Universe500.csv
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_SmallCap600.csv
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_TechStocks.csv
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_Industry28.csv

OUTPUT FILES
------------
  <repo>/runs/reconcile_<YYYYMMDD_HHMMSS>/reconciliation.parquet   canonical
  <repo>/runs/reconcile_<YYYYMMDD_HHMMSS>/reconciliation.xlsx      eyeball copy
  <repo>/runs/reconcile_<YYYYMMDD_HHMMSS>/reconcile.log            console log

USAGE
-----
  .venv/bin/python reconcile_published.py

DEPENDENCIES
------------
  pandas, numpy, scipy, pyarrow, openpyxl; imports analyze() from
  rescore_from_cache.py in this same directory.

Version 1.0 — 2026-08-04
"""
import json
import os
from datetime import datetime

import pandas as pd

from rescore_from_cache import analyze

REPO = os.path.dirname(os.path.abspath(__file__))

# (universe, forecasts file, top_n per side, oos_start, published values)
# Published values are transcribed from master_validation_summary.csv and
# industry28_metrics.txt — see module docstring for provenance.
CHECKS = [
    dict(universe="S&P 500 Large-Cap", source="master_validation_summary.csv", file="forecasts_Universe500.csv", top_n=10,
         oos="2025-01-01", ddof=0,
         pub={"Full.top_ann_ret": 0.467, "Full.top_sharpe": 1.426,
              "Full.bot_ann_ret": 0.198, "Full.bench_sharpe": 1.091,
              "OOS.top_ann_ret": 0.482, "OOS.top_sharpe": 1.501,
              "OOS.bot_ann_ret": 0.925, "OOS.bot_sharpe": 2.574}),
    dict(universe="SmallCap-600", source="master_validation_summary.csv", file="forecasts_SmallCap600.csv", top_n=10,
         oos="2025-01-01", ddof=0,
         pub={"Full.top_ann_ret": 0.398, "Full.top_sharpe": 1.110,
              "Full.bot_ann_ret": -0.003, "Full.bench_sharpe": 0.674,
              "OOS.top_ann_ret": 0.332, "OOS.top_sharpe": 1.013,
              "OOS.bot_ann_ret": -0.132, "OOS.bot_sharpe": -0.431}),
    dict(universe="Tech Stocks", source="master_validation_summary.csv", file="forecasts_TechStocks.csv", top_n=5,
         oos="2025-01-01", ddof=0,
         pub={"Full.top_ann_ret": 0.408, "Full.top_sharpe": 1.073,
              "Full.bot_ann_ret": 0.143, "Full.bench_sharpe": 1.041,
              "OOS.top_ann_ret": 0.147, "OOS.top_sharpe": 0.411,
              "OOS.bot_ann_ret": 0.088, "OOS.bot_sharpe": 0.237}),
    dict(universe="Industry28 (LIVE)", source="industry28_metrics.txt", file="forecasts_Industry28.csv", top_n=3,
         oos="2024-07-01", ddof=1,
         pub={"Full.top_ann_ret": 0.220, "Full.top_sharpe": 1.00,
              "Full.bot_ann_ret": 0.103, "Full.bot_sharpe": 0.47,
              "Full.bench_ann_ret": 0.156, "Full.bench_sharpe": 0.92,
              "Full.avg_IC": 0.0266,
              "OOS.top_ann_ret": 0.418, "OOS.top_sharpe": 1.96,
              "OOS.bot_ann_ret": 0.260, "OOS.bot_sharpe": 1.49,
              "OOS.bench_ann_ret": 0.184, "OOS.bench_sharpe": 1.37}),
]

# tolerance: published values are printed to 1dp (%) or 2dp (Sharpe),
# so anything inside half a printed unit is an exact match.
TOL = {"ann_ret": 0.001, "sharpe": 0.006, "IC": 0.0001}


def tol_for(metric):
    if "sharpe" in metric:
        return TOL["sharpe"]
    if "IC" in metric:
        return TOL["IC"]
    return TOL["ann_ret"]


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO, "runs", f"reconcile_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_fh = open(os.path.join(run_dir, "reconcile.log"), "w")

    def log(m=""):
        print(m)
        log_fh.write(str(m) + "\n")
        log_fh.flush()

    log(f"reconcile_published.py  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    for src in ("master_validation_summary.csv", "industry28_metrics.txt"):
        log(f"  {src} published: "
            f"{datetime.fromtimestamp(os.path.getmtime(os.path.join(REPO, src))):%Y-%m-%d %H:%M}")
    log("")

    rows = []
    for chk in CHECKS:
        path = os.path.join(REPO, chk["file"])
        if not os.path.exists(path):
            log(f"[MISSING] {chk['universe']}: {chk['file']}")
            continue
        cache_mtime = os.path.getmtime(path)
        summary_mtime = os.path.getmtime(os.path.join(REPO, chk["source"]))
        stale = cache_mtime > summary_mtime

        df = pd.read_csv(path)
        got = {}
        full = analyze(df, 0.10, "Full", run_permutation=False, top_n_abs=chk["top_n"],
                       ddof=chk["ddof"])
        oos = analyze(df[df["date"] >= chk["oos"]], 0.10, "OOS",
                      run_permutation=False, top_n_abs=chk["top_n"], ddof=chk["ddof"])
        for prefix, res in (("Full", full), ("OOS", oos)):
            if res:
                for k, v in res.items():
                    got[f"{prefix}.{k}"] = v

        log(f"── {chk['universe']}   [{chk['file']}, top-{chk['top_n']}/side, OOS {chk['oos']}, ddof={chk['ddof']}]")
        log(f"   cache written {datetime.fromtimestamp(cache_mtime):%Y-%m-%d %H:%M}"
            f"{'   ⚠️  NEWER THAN ' + chk['source'] + ' — evidence overwritten' if stale else ''}")

        for metric, pub_val in chk["pub"].items():
            new_val = got.get(metric)
            if new_val is None:
                log(f"     {metric:24s} published={pub_val:>8.3f}   rescored=  n/a  [NO DATA]")
                continue
            diff = new_val - pub_val
            ok = abs(diff) <= tol_for(metric)
            rows.append({
                "universe": chk["universe"], "metric": metric,
                "published": pub_val, "rescored": round(new_val, 4),
                "diff": round(diff, 4), "reproduces": bool(ok),
                "cache_newer_than_summary": bool(stale),
                "top_n_per_side": chk["top_n"], "oos_start": chk["oos"],
                "ddof": chk["ddof"],
            })
            log(f"     {metric:24s} published={pub_val:>8.3f}   rescored={new_val:>8.3f}   "
                f"diff={diff:+.4f}   {'PASS' if ok else 'FAIL'}")
        log("")

    out = pd.DataFrame(rows)
    pq = os.path.join(run_dir, "reconciliation.parquet")
    out.to_parquet(pq, index=False)
    xl = os.path.join(run_dir, "reconciliation.xlsx")
    out.to_excel(xl, index=False, sheet_name="reconciliation")

    log("=" * 90)
    for uni, grp in out.groupby("universe", sort=False):
        n_ok, n = int(grp["reproduces"].sum()), len(grp)
        flag = "REPRODUCES" if n_ok == n else "DOES NOT REPRODUCE"
        log(f"  {uni:22s} {n_ok}/{n} metrics match   → {flag}")
    log("=" * 90)
    log(f"  parquet : {pq}")
    log(f"  xlsx    : {xl}")

    with open(os.path.join(run_dir, "summary.json"), "w") as fh:
        json.dump({
            "run_utc": datetime.utcnow().isoformat() + "Z",
            "n_metrics": len(out),
            "n_reproduce": int(out["reproduces"].sum()),
            "universes_failing": sorted(
                out[~out["reproduces"]]["universe"].unique().tolist()),
            "outputs": {"parquet": pq, "xlsx": xl},
        }, fh, indent=2)
    log_fh.close()


if __name__ == "__main__":
    main()
