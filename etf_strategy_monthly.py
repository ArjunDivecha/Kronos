"""
ETF Monthly Strategy: Kronos-base with 40-day lookback, month-end rebalancing.
1. Run month-end walk-forward forecasts across all ETFs in data/ETF.
2. At each month-end, rank ETFs by predicted next-month return and pick top-3.
3. Compare top-3 equal-weight return vs equal-weight return of all ETFs.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor


# ── 1. CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR = "data/ETF"
LOOKBACK = 40
TOP_N = 3

FORECASTS_FILE = "etf_forecasts_base40_monthly.csv"
PERF_FILE = "strategy_performance.csv"
PLOT_FILE = "strategy_results_monthly.png"
REPORT_XLSX_FILE = "strategy_performance_monthly_report.xlsx"
SAMPLE_TEMPLATE_FILE = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/Sample.xlsx"
FORECAST_PANEL_FILE = "forecast_monthly_returns.xlsx"

# Country label -> ticker mapping for Sample.xlsx-style output
COUNTRY_TO_TICKER = {
    "Singapore": "EWS",
    "Australia": "EWA",
    "Canada": "EWC",
    "Germany": "EWG",
    "Japan": "EWJ",
    "Switzerland": "EWL",
    "U.K.": "EWU",
    "NASDAQ": "QQQ",
    "U.S.": "SPY",
    "France": "EWQ",
    "Netherlands": "EWN",
    "Sweden": "EWD",
    "Italy": "EWI",
    "ChinaA": "ASHR",
    "Chile": "ECH",
    "Indonesia": "EIDO",
    "Philippines": "EPHE",
    "Poland": "EPOL",
    "US SmallCap": "IWM",
    "Malaysia": "EWM",
    "Taiwan": "EWT",
    "Mexico": "EWW",
    "Korea": "EWY",
    "Brazil": "EWZ",
    "South Africa": "EZA",
    "Denmark": "EDEN",
    "India": "INDA",
    "ChinaH": "MCHI",
    "Hong Kong": "EWH",
    "Thailand": "THD",
    "Turkey": "TUR",
    "Spain": "EWP",
    "Vietnam": "VNM",
    "Saudi Arabia": "KSA",
}


def _month_end_dates(ref_dates: pd.DatetimeIndex) -> list:
    """Return month-end trading dates based on the provided trading calendar."""
    s = pd.Series(ref_dates)
    is_month_end = s.dt.to_period("M") != s.shift(-1).dt.to_period("M")
    return s[is_month_end].tolist()


def _safe_stats(rets: pd.Series, label: str, periods_per_year: int = 12) -> dict:
    """Compute annualized stats with empty-series safety."""
    if rets is None or len(rets) == 0:
        return {
            "Label": label,
            "Ann. Return": "n/a",
            "Ann. Vol": "n/a",
            "Sharpe": "n/a",
            "Hit Rate": "n/a",
            "Periods": 0,
        }

    ann_ret = (1 + rets.mean()) ** periods_per_year - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    hit_rate = (rets > 0).mean()

    return {
        "Label": label,
        "Ann. Return": f"{ann_ret:.2%}",
        "Ann. Vol": f"{ann_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Hit Rate": f"{hit_rate:.1%}",
        "Periods": int(len(rets)),
    }


def run_sweep_monthly():
    """
    Run Kronos-base inference at month-end dates.
    For each month-end date t, forecast from t to the next month-end t+1M.
    """
    tickers = sorted([f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    print(f"Loading {len(tickers)} ETFs...")

    etf_dfs = {}
    ts_to_idx = {}
    for ticker in tickers:
        df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv")
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)
        etf_dfs[ticker] = df
        ts_to_idx[ticker] = {ts: i for i, ts in enumerate(df["timestamps"])}

    if "SPY" not in etf_dfs:
        raise ValueError("SPY.csv is required in data/ETF to define month-end reference dates.")

    ref_dates = pd.DatetimeIndex(np.sort(etf_dfs["SPY"]["timestamps"].unique()))
    ref_idx_map = {ts: i for i, ts in enumerate(ref_dates)}
    month_ends = _month_end_dates(ref_dates)
    if len(month_ends) < 2:
        raise ValueError("Not enough month-end dates found in SPY data.")

    # Model
    print("Loading Kronos-base (100M params)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    all_rows = []

    for m in tqdm(range(len(month_ends) - 1), desc="Monthly sweep"):
        current_date = pd.Timestamp(month_ends[m])
        next_month_end = pd.Timestamp(month_ends[m + 1])
        ref_cur = ref_idx_map[current_date]
        ref_next = ref_idx_map[next_month_end]
        y_ts_ref = ref_dates[ref_cur + 1 : ref_next + 1]
        pred_len_ref = len(y_ts_ref)
        if pred_len_ref <= 0:
            continue

        # One pred_len per month-end based on the reference market calendar.
        grouped = {pred_len_ref: {"dfs": [], "xts": [], "yts": [], "meta": []}}

        for ticker in tickers:
            df = etf_dfs[ticker]
            idx_map = ts_to_idx[ticker]

            idx_cur = idx_map.get(current_date)
            idx_next = idx_map.get(next_month_end)
            if idx_cur is None or idx_next is None:
                continue
            if idx_cur < LOOKBACK - 1 or idx_next <= idx_cur:
                continue

            # Strict anti-leak guard:
            # Require ticker path to match reference calendar exactly over the forecast month.
            if (idx_next - idx_cur) != pred_len_ref:
                continue

            x_df = df.iloc[idx_cur - LOOKBACK + 1 : idx_cur + 1]
            if x_df[["open", "high", "low", "close", "volume", "amount"]].isnull().values.any():
                continue

            # Leakage guard: ensure model input ends at current_date and targets start strictly after.
            if x_df["timestamps"].iloc[-1] != current_date:
                continue
            if len(y_ts_ref) == 0 or pd.Timestamp(y_ts_ref[0]) <= current_date:
                continue

            y_ts = pd.Series(y_ts_ref)
            actual_day0 = df.iloc[idx_cur]["close"]
            actual_day_next = df.iloc[idx_next]["close"]

            bucket = grouped[pred_len_ref]
            bucket["dfs"].append(x_df[["open", "high", "low", "close", "volume", "amount"]])
            bucket["xts"].append(x_df["timestamps"])
            bucket["yts"].append(y_ts)
            bucket["meta"].append(
                {
                    "ticker": ticker,
                    "date": current_date,
                    "next_month_end": next_month_end,
                    "actual_day0": actual_day0,
                    "actual_return": (actual_day_next / actual_day0) - 1,
                }
            )

        if not grouped:
            continue

        for pred_len, bucket in grouped.items():
            if len(bucket["dfs"]) == 0:
                continue
            try:
                preds = predictor.predict_batch(
                    df_list=bucket["dfs"],
                    x_timestamp_list=bucket["xts"],
                    y_timestamp_list=bucket["yts"],
                    pred_len=pred_len,
                    sample_count=10,
                    T=0.8,
                    verbose=False,
                )
            except Exception:
                continue

            for j, p_df in enumerate(preds):
                meta = bucket["meta"][j]
                pred_close = p_df.iloc[-1]["close"]
                all_rows.append(
                    {
                        "date": meta["date"],
                        "next_month_end": meta["next_month_end"],
                        "ticker": meta["ticker"],
                        "horizon_days": pred_len,
                        "pred_return": (pred_close / meta["actual_day0"]) - 1,
                        "actual_return": meta["actual_return"],
                    }
                )

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(FORECASTS_FILE, index=False)
    print(f"Saved {len(forecasts)} monthly forecasts to {FORECASTS_FILE}")
    return forecasts


def build_strategy(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    At each month-end date:
      - Benchmark: equal-weight return across all ETFs with a valid forecast.
      - Strategy: equal-weight return across TOP_N ETFs by predicted return.
    """
    if forecasts.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "next_month_end",
                "strat_ret",
                "bench_ret",
                "top_tickers",
                "n_available",
                "cum_strat",
                "cum_bench",
                "relative",
            ]
        )

    rows = []
    for d in sorted(forecasts["date"].unique()):
        snap = forecasts[forecasts["date"] == d].copy()
        n_available = len(snap)
        if n_available < TOP_N:
            continue

        bench_ret = snap["actual_return"].mean()
        top = snap.nlargest(TOP_N, "pred_return")
        strat_ret = top["actual_return"].mean()

        rows.append(
            {
                "date": d,
                "next_month_end": top["next_month_end"].iloc[0],
                "strat_ret": strat_ret,
                "bench_ret": bench_ret,
                "top_tickers": ", ".join(top["ticker"].tolist()),
                "n_available": n_available,
            }
        )

    perf = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if perf.empty:
        return perf

    perf["cum_strat"] = (1 + perf["strat_ret"]).cumprod()
    perf["cum_bench"] = (1 + perf["bench_ret"]).cumprod()
    perf["relative"] = perf["cum_strat"] / perf["cum_bench"]
    return perf


def compute_stats(perf: pd.DataFrame):
    """Compute monthly-annualized stats."""
    if perf.empty:
        empty = pd.DataFrame([_safe_stats(pd.Series(dtype=float), f"Top-{TOP_N} Strategy"),
                              _safe_stats(pd.Series(dtype=float), "EW Benchmark (All ETFs)")])
        return empty, empty.copy()

    stats = pd.DataFrame(
        [
            _safe_stats(perf["strat_ret"], f"Top-{TOP_N} Strategy"),
            _safe_stats(perf["bench_ret"], "EW Benchmark (All ETFs)"),
        ]
    )

    cutoff = pd.to_datetime("2024-07-01")
    pre = perf[perf["date"] < cutoff]
    post = perf[perf["date"] >= cutoff]

    stats_split = pd.DataFrame(
        [
            _safe_stats(pre["strat_ret"], f"Top-{TOP_N} (In-Sample)"),
            _safe_stats(pre["bench_ret"], "Bench (In-Sample)"),
            _safe_stats(post["strat_ret"], f"Top-{TOP_N} (OOS)"),
            _safe_stats(post["bench_ret"], "Bench (OOS)"),
        ]
    )
    return stats, stats_split


def plot_results(perf: pd.DataFrame, stats_text: str):
    """Generate cumulative and relative performance plots."""
    if perf.empty:
        print("No performance rows to plot; skipping chart.")
        return

    cutoff = pd.to_datetime("2024-07-01")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(perf["date"], perf["cum_strat"], label=f"Top-{TOP_N} Strategy", color="#2ecc71", linewidth=2)
    ax1.plot(perf["date"], perf["cum_bench"], label="EW All ETFs", color="#95a5a6", linewidth=1.5)
    ax1.axvline(x=cutoff, color="black", linestyle="--", alpha=0.6, label="Training Cutoff")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Kronos-base Monthly Top-3 Strategy vs Equal-Weight ETF Benchmark")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2.plot(perf["date"], perf["relative"], color="#3498db", linewidth=1.5)
    ax2.axhline(y=1.0, color="gray", linestyle="-", alpha=0.4)
    ax2.axvline(x=cutoff, color="black", linestyle="--", alpha=0.6)
    ax2.set_ylabel("Relative (Strategy / Bench)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax1.text(
        0.99,
        0.03,
        stats_text,
        transform=ax1.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"Saved {PLOT_FILE}")


def save_excel_report(stats: pd.DataFrame, stats_split: pd.DataFrame, perf: pd.DataFrame):
    """Save monthly performance outputs to a single XLSX workbook."""
    try:
        with pd.ExcelWriter(REPORT_XLSX_FILE, engine="openpyxl") as writer:
            stats.to_excel(writer, sheet_name="stats_overall", index=False)
            stats_split.to_excel(writer, sheet_name="stats_split", index=False)
            perf.to_excel(writer, sheet_name="monthly_performance", index=False)
        print(f"Saved {REPORT_XLSX_FILE}")
    except ImportError:
        print("openpyxl is not installed; skipping XLSX export.")


def save_sample_format_forecasts(forecasts: pd.DataFrame):
    """
    Save forecast monthly returns in the same tabular format as Sample.xlsx:
    sheet 1MRet with first column 'Country' and country columns in template order.
    """
    if forecasts.empty:
        print("No forecasts available; skipping Sample.xlsx-format export.")
        return

    if not os.path.exists(SAMPLE_TEMPLATE_FILE):
        print(f"Template not found: {SAMPLE_TEMPLATE_FILE}; skipping Sample.xlsx-format export.")
        return

    template = pd.read_excel(SAMPLE_TEMPLATE_FILE, sheet_name="1MRet")
    template_cols = list(template.columns)
    if not template_cols or template_cols[0] != "Country":
        print("Template sheet 1MRet does not start with 'Country'; skipping Sample.xlsx-format export.")
        return

    # Label each forecast by the forecasted month (first day) to match Sample.xlsx style.
    f = forecasts.copy()
    f["next_month_start"] = pd.to_datetime(f["next_month_end"]).dt.to_period("M").dt.to_timestamp()

    # Pivot predicted returns by ticker.
    pred_panel = (
        f.pivot_table(index="next_month_start", columns="ticker", values="pred_return", aggfunc="last")
        .sort_index()
    )

    out = pd.DataFrame(index=pred_panel.index)
    for country_col in template_cols[1:]:
        ticker = COUNTRY_TO_TICKER.get(country_col)
        if ticker is None:
            out[country_col] = np.nan
        else:
            out[country_col] = pred_panel[ticker] if ticker in pred_panel.columns else np.nan

    out = out.reset_index().rename(columns={"next_month_start": "Country"})
    out["Country"] = pd.to_datetime(out["Country"]).dt.strftime("%Y-%m-%d")

    # Enforce exact template column order.
    out = out.reindex(columns=template_cols)

    with pd.ExcelWriter(FORECAST_PANEL_FILE, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="1MRet", index=False)

    print(f"Saved {FORECAST_PANEL_FILE} in Sample.xlsx format/order.")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(FORECASTS_FILE):
        print(f"Loading cached forecasts from {FORECASTS_FILE}...")
        forecasts = pd.read_csv(FORECASTS_FILE, parse_dates=["date", "next_month_end"])
    else:
        forecasts = run_sweep_monthly()
        if not forecasts.empty:
            forecasts["date"] = pd.to_datetime(forecasts["date"])
            forecasts["next_month_end"] = pd.to_datetime(forecasts["next_month_end"])

    print(f"Total monthly forecasts: {len(forecasts)}")

    perf = build_strategy(forecasts)

    stats, stats_split = compute_stats(perf)
    print("\n" + "=" * 60)
    print("MONTHLY PERFORMANCE")
    print("=" * 60)
    print(stats.to_string(index=False))
    print("\n" + "-" * 60)
    print("IN-SAMPLE vs OUT-OF-SAMPLE")
    print("-" * 60)
    print(stats_split.to_string(index=False))

    stats_text = stats.to_string(index=False)
    plot_results(perf, stats_text)

    perf.to_csv(PERF_FILE, index=False)
    print(f"\nSaved {PERF_FILE}")
    save_excel_report(stats, stats_split, perf)
    save_sample_format_forecasts(forecasts)
