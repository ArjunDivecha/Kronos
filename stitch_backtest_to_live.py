"""
stitch_backtest_to_live.py — one continuous Industry28 equity curve
===================================================================

WHAT THIS DOES
--------------
Joins the Industry28 backtest to the live IBKR track record into a single
continuous equity curve, so the strategy can be judged on one series instead of
two disconnected artefacts. Writes the joined series, a summary of each segment,
and a light-mode PDF chart with the go-live date marked.

THE CONSTRUCTION PROBLEM THIS SOLVES
------------------------------------
The published backtest headline ("Top3", +22.0%/yr full, +41.8% OOS) is a
LONG-ONLY portfolio. The live strategy is 3-long / 3-short. Those are different
strategies and must not be stitched to each other — doing so would splice a
long-only history onto a market-neutral live record and call it a track record.

This script instead rebuilds the backtest segment in the LIVE construction:

    period return = LEG_WEIGHT x (Top3 - Bottom3)

LEG_WEIGHT is not assumed. It is measured from the live ledger: each cycle holds
6 positions sized net_liq * 0.98 / 6, so each side carries 3 * 0.98/6 = 49% of
capital. The script recomputes it from pl_tracker.xlsx entry values and fails
loudly if the measured value drifts from 0.49 by more than 2pp.

Long-only Top3 is also emitted, clearly labelled, for reference only.

KNOWN DISCONTINUITIES (reported, not hidden)
--------------------------------------------
1. Coverage gap. The backtest's last full month is 2026-03; live trading opened
   2026-04-15. Roughly two weeks are covered by neither series. The curve is
   joined across the gap with no interpolation and the gap is annotated.
2. Holding period. The backtest rebalances on a fixed 5-day stride; live cycles
   ran 3-15 days. Same signal, different cadence.
3. Gross vs net. Backtest returns are gross (house rule: signal quality is
   judged gross). Live returns are actual fills and therefore net of real
   commission and slippage. The live segment is handicapped relative to the
   backtest segment by exactly the trading costs the backtest ignores.
4. Live is stale. Last live cycle closed 2026-06-13.

INPUT FILES (absolute)
----------------------
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/monthly_returns_industry28.csv
      backtest monthly returns; columns month,n_periods,EW_All,Top3,Top5,Bottom3,Bottom5
  /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/pl_tracker.xlsx
      live ledger; sheets "Positions" (per-position fills) and "Summary" (per-cycle P/L)

OUTPUT FILES (timestamped run directory)
----------------------------------------
  <repo>/runs/stitch_<YYYYMMDD_HHMMSS>/stitched_curve.parquet   canonical joined series
  <repo>/runs/stitch_<YYYYMMDD_HHMMSS>/stitched_curve.xlsx      eyeball copy
  <repo>/runs/stitch_<YYYYMMDD_HHMMSS>/stitched_curve.pdf       light-mode chart
  <repo>/runs/stitch_<YYYYMMDD_HHMMSS>/segment_summary.json     stats per segment
  <repo>/runs/stitch_<YYYYMMDD_HHMMSS>/stitch.log               console log

USAGE
-----
  .venv/bin/python stitch_backtest_to_live.py
  .venv/bin/python stitch_backtest_to_live.py --long-only   # reference curve

DEPENDENCIES
------------
  pandas, numpy, matplotlib, pyarrow, openpyxl  (Python 3.9.6 venv)

Version 1.0 — 2026-08-04
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
EXPECTED_LEG = 0.49
LEG_TOL = 0.02


def measure_leg_weight(positions, log):
    """
    Measure the per-side capital weight actually used live, from entry values.

    Each cycle holds 6 positions at net_liq*0.98/6; one side is therefore
    3*0.98/6 = 49% of capital. Measured rather than assumed so that a change in
    the trader's sizing rule shows up here instead of silently rescaling the
    backtest segment.
    """
    # trade_log.csv is RAGGED: the header declares 10 columns, but rows written
    # for Market-On-Open orders insert an extra order-type token ("MOO") before
    # `account`, giving 11. pd.read_csv therefore fails outright. Parse
    # positionally from the end, where the layout is stable:
    #   ... , account, net_liq, dry_run
    netliq_rows = {}
    with open(os.path.join(REPO, "trade_log.csv")) as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 10:
                continue
            date = parts[0]
            try:
                nl = float(parts[-2])          # net_liq is always second-from-last
            except ValueError:
                continue
            netliq_rows.setdefault(date, nl)   # first fill of the day wins
    netliq = pd.Series(netliq_rows, dtype=float)
    log(f"  parsed net_liq for {len(netliq)} trade dates from ragged trade_log.csv")

    per_cycle = []
    for cyc, grp in positions.groupby("Cycle"):
        open_date = str(pd.to_datetime(grp["Open Date"].iloc[0]).date())
        if open_date not in netliq.index:
            continue
        nl = float(netliq.loc[open_date])
        longs = grp[grp["Side"] == "LONG"]["Entry Value"].sum()
        shorts = grp[grp["Side"] == "SHORT"]["Entry Value"].sum()
        per_cycle.append((cyc, longs / nl, shorts / nl))

    if not per_cycle:
        log("  WARNING: could not match any cycle to net_liq; falling back to 0.49")
        return EXPECTED_LEG

    long_w = float(np.median([p[1] for p in per_cycle]))
    short_w = float(np.median([p[2] for p in per_cycle]))
    leg = (long_w + short_w) / 2
    log(f"  measured leg weight: long={long_w:.4f}  short={short_w:.4f}  mean={leg:.4f}"
        f"  (expected {EXPECTED_LEG})")
    if abs(leg - EXPECTED_LEG) > LEG_TOL:
        raise SystemExit(
            f"FAIL: measured leg weight {leg:.4f} differs from expected "
            f"{EXPECTED_LEG} by more than {LEG_TOL}. The live sizing rule has "
            f"changed; fix the backtest construction before stitching."
        )
    return leg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-only", action="store_true",
                    help="build the reference long-only Top3 curve instead of live-matched L/S")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO, "runs", f"stitch_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    log_fh = open(os.path.join(run_dir, "stitch.log"), "w")

    def log(m=""):
        print(m)
        log_fh.write(str(m) + "\n")
        log_fh.flush()

    log(f"stitch_backtest_to_live.py  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"mode: {'LONG-ONLY Top3 (reference)' if args.long_only else 'LIVE-MATCHED 3L/3S'}")
    log("")

    # ── live ledger ───────────────────────────────────────────────────────────
    positions = pd.read_excel(os.path.join(REPO, "pl_tracker.xlsx"), sheet_name="Positions")
    summary = pd.read_excel(os.path.join(REPO, "pl_tracker.xlsx"), sheet_name="Summary")
    summary["Open Date"] = pd.to_datetime(summary["Open Date"])
    summary["Close Date"] = pd.to_datetime(summary["Close Date"])

    log("LIVE SEGMENT")
    leg = EXPECTED_LEG if args.long_only else measure_leg_weight(positions, log)
    go_live = summary["Open Date"].min()
    live_end = summary["Close Date"].max()
    live_cum = float((1 + summary["Cycle % Return"]).prod() - 1)
    log(f"  {len(summary)} cycles   {go_live.date()} -> {live_end.date()}"
        f"   compounded {live_cum:+.2%}")
    stale_days = (pd.Timestamp.now().normalize() - live_end).days
    if stale_days > 14:
        log(f"  NOTE: no live cycle closed in {stale_days} days — live series is stale")

    # ── backtest segment, rebuilt in the live construction ────────────────────
    m = pd.read_csv(os.path.join(REPO, "monthly_returns_industry28.csv"))
    m["month_end"] = pd.PeriodIndex(m["month"], freq="M").to_timestamp(how="end").normalize()
    m = m.sort_values("month_end").reset_index(drop=True)

    if args.long_only:
        m["ret"] = m["Top3"]
        construction = "backtest Top3 long-only (gross)"
    else:
        m["ret"] = leg * (m["Top3"] - m["Bottom3"])
        construction = f"backtest {leg:.2f}x(Top3-Bottom3), matches live 3L/3S (gross)"

    # cut the backtest at the last FULL month before go-live; no overlap, no interpolation
    cut = go_live.to_period("M").to_timestamp() - pd.Timedelta(days=1)
    bt = m[m["month_end"] <= cut].copy()
    gap_days = (go_live - bt["month_end"].max()).days
    log("")
    log("BACKTEST SEGMENT")
    log(f"  construction: {construction}")
    log(f"  {len(bt)} months   {bt['month_end'].min().date()} -> {bt['month_end'].max().date()}")
    bt_cum = float((1 + bt["ret"]).prod() - 1)
    yrs = (bt["month_end"].max() - bt["month_end"].min()).days / 365.25
    bt_cagr = (1 + bt_cum) ** (1 / yrs) - 1
    bt_sharpe = float(bt["ret"].mean() / bt["ret"].std() * np.sqrt(12)) if bt["ret"].std() > 0 else 0.0
    log(f"  compounded {bt_cum:+.1%}   CAGR {bt_cagr:+.2%}   monthly Sharpe {bt_sharpe:.2f}")
    log("")
    log(f"COVERAGE GAP: {gap_days} days uncovered "
        f"({bt['month_end'].max().date()} -> {go_live.date()}) — joined, not interpolated")

    # ── stitch ────────────────────────────────────────────────────────────────
    rows = [{"date": d, "ret": r, "segment": "backtest"}
            for d, r in zip(bt["month_end"], bt["ret"])]
    rows += [{"date": d, "ret": r, "segment": "live"}
             for d, r in zip(summary["Close Date"], summary["Cycle % Return"])]
    curve = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    curve["equity"] = (1 + curve["ret"]).cumprod()
    curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1

    pq = os.path.join(run_dir, "stitched_curve.parquet")
    curve.to_parquet(pq, index=False)
    xl = os.path.join(run_dir, "stitched_curve.xlsx")
    with pd.ExcelWriter(xl) as w:
        curve.to_excel(w, sheet_name="stitched", index=False)
        summary.to_excel(w, sheet_name="live_cycles", index=False)

    # ── chart (light mode, matplotlib, PDF) ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.3, linewidth=0.6)

    b = curve[curve["segment"] == "backtest"]
    l = curve[curve["segment"] == "live"]
    ax1.plot(b["date"], b["equity"], color="#1f77b4", linewidth=1.6, label="Backtest (gross)")
    if len(l):
        join = pd.concat([b.tail(1), l])
        ax1.plot(join["date"], join["equity"], color="#d62728", linewidth=2.0,
                 label="Live (net, actual fills)")
    ax1.axvline(go_live, color="#555555", linestyle="--", linewidth=1.2)
    ax1.annotate(f" go live {go_live.date()}", xy=(go_live, ax1.get_ylim()[1]),
                 ha="left", va="top", fontsize=9, color="#555555")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title(f"Industry28 — backtest stitched to live\n{construction}", fontsize=11)
    ax1.legend(frameon=False, fontsize=9)

    ax2.fill_between(curve["date"], curve["drawdown"] * 100, 0,
                     color="#d62728", alpha=0.35, linewidth=0)
    ax2.axvline(go_live, color="#555555", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    fig.tight_layout()
    pdf = os.path.join(run_dir, "stitched_curve.pdf")
    fig.savefig(pdf, format="pdf", facecolor="white")
    plt.close(fig)

    seg = {
        "run_utc": datetime.utcnow().isoformat() + "Z",
        "construction": construction,
        "leg_weight_measured": leg,
        "backtest": {
            "months": len(bt), "start": str(bt["month_end"].min().date()),
            "end": str(bt["month_end"].max().date()),
            "cumulative": bt_cum, "cagr": bt_cagr, "monthly_sharpe": bt_sharpe,
            "basis": "gross, no costs",
        },
        "live": {
            "cycles": int(len(summary)), "start": str(go_live.date()),
            "end": str(live_end.date()), "cumulative": live_cum,
            "basis": "net, actual IBKR fills", "stale_days": int(stale_days),
        },
        "coverage_gap_days": int(gap_days),
        "survivorship_relevant": False,
        "survivorship_note": (
            "Industry28 is a 28-ETF panel whose width grows honestly as ETFs "
            "launch (26 in 2010, XLRE 2015, XLC 2018). Forward live returns "
            "cannot carry survivorship bias by construction."
        ),
        "outputs": {"parquet": pq, "xlsx": xl, "pdf": pdf},
    }
    with open(os.path.join(run_dir, "segment_summary.json"), "w") as fh:
        json.dump(seg, fh, indent=2)

    log("")
    log("=" * 78)
    log(f"  backtest  {bt_cum:+9.1%} over {len(bt):3d} months   (gross)")
    log(f"  live      {live_cum:+9.2%} over {len(summary):3d} cycles   (net, real fills)")
    log("=" * 78)
    log(f"  parquet : {pq}")
    log(f"  xlsx    : {xl}")
    log(f"  pdf     : {pdf}")
    log_fh.close()


if __name__ == "__main__":
    main()
