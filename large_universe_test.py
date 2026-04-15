"""
Large Universe OOS Test: 500 Large-Cap Stocks (from universe.xlsx)
==================================================================
Zero-shot Kronos-base backtest on 500 large-cap US stocks.
Compares Top-10% vs Bottom-10% and Top-50% vs Bottom-50%.

With 500 stocks: Top 10% = ~50 stocks, Bottom 10% = ~50 stocks.

Walk-forward methodology — strictly no future data used at inference.
"""
import os, sys, json
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from scipy.stats import spearmanr

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

# ── CONFIG ────────────────────────────────────────────────────────────────────
UNIVERSE_FILE  = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/BloomTest/outputs/universe.xlsx"
SMALLCAP_FILE  = "/Users/arjundivecha/Dropbox/AAA Backup/SML as of Apr 13 20261.xlsx"
LOOKBACK       = 40
PRED_LEN       = 5
STRIDE         = 5
SAMPLE_COUNT   = 5   # Reduced for speed at large scale
DECILE         = 0.10  # Top/Bottom 10%


# ── LOAD TICKERS ──────────────────────────────────────────────────────────────
def load_tickers_sp500():
    df = pd.read_excel(UNIVERSE_FILE, sheet_name="Universe")
    tickers = df["Ticker"].str.replace(" US Equity", "", regex=False).str.strip().tolist()
    print(f"Loaded {len(tickers)} tickers from universe.xlsx (SP500)")
    return tickers

def load_tickers_smallcap():
    df = pd.read_excel(SMALLCAP_FILE, sheet_name="Worksheet")
    # Strip Bloomberg exchange suffixes (UN, UW, UA, UQ, UR + 'Equity')
    tickers = (
        df["Ticker"]
        .str.replace(r"\s+U[NWARQr]\s+Equity", "", regex=True)
        .str.replace(r"\s+U[NWARQr]$", "", regex=True)
        .str.replace(r"\s+Equity", "", regex=True)
        .str.strip()
        .tolist()
    )
    print(f"Loaded {len(tickers)} tickers from SML Excel (Small-Cap 600)")
    return tickers


# ── STEP 1: DOWNLOAD (one at a time, robust) ──────────────────────────────────
def download_universe(tickers, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    to_download = [t for t in tickers
                   if not os.path.exists(os.path.join(data_dir, f"{t}.csv"))]
    already = len(tickers) - len(to_download)
    print(f"  {already} cached, downloading {len(to_download)} new tickers...")

    for ticker in tqdm(to_download, desc="Downloading"):
        out_path = os.path.join(data_dir, f"{ticker}.csv")
        try:
            data = yf.download(ticker, start="2015-01-01", progress=False, auto_adjust=True)
            if data.empty or len(data) < LOOKBACK + PRED_LEN + 10:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            pd.DataFrame({
                "timestamps": pd.to_datetime(data.index).strftime("%Y-%m-%d %H:%M:%S"),
                "open":   data["Open"].values,
                "high":   data["High"].values,
                "low":    data["Low"].values,
                "close":  data["Close"].values,
                "volume": data["Volume"].values,
                "amount": (data["Close"] * data["Volume"]).values,
            }).dropna().to_csv(out_path, index=False)
        except Exception:
            pass

    valid = [t for t in tickers
             if os.path.exists(os.path.join(data_dir, f"{t}.csv"))]
    print(f"  ✅ {len(valid)}/{len(tickers)} valid stocks")
    return valid


# ── STEP 2: WALK-FORWARD BACKTEST (with incremental checkpointing) ────────────
CHECKPOINT_EVERY = 25   # Save to CSV every N periods

def run_backtest(tickers, data_dir, forecast_csv, label, predictor):
    stock_dfs = {}
    for t in tqdm(tickers, desc=f"Loading {label} data"):
        path = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            df["timestamps"] = pd.to_datetime(df["timestamps"])
            if len(df) >= LOOKBACK + PRED_LEN:
                stock_dfs[t] = df
        except Exception:
            pass

    n_stocks = len(stock_dfs)
    top_n = max(1, int(n_stocks * DECILE))
    print(f"\n  {n_stocks} valid stocks | Top/Bottom {DECILE:.0%} = {top_n} stocks per bin")

    ref_ticker = max(stock_dfs, key=lambda t: len(stock_dfs[t]))
    ref_dates  = np.sort(stock_dfs[ref_ticker]["timestamps"].unique())
    all_periods = list(range(LOOKBACK, len(ref_dates) - PRED_LEN, STRIDE))

    # ── Resume from checkpoint if exists ─────────────────────────────────────
    checkpoint_csv = forecast_csv.replace(".csv", "_checkpoint.csv")
    done_dates = set()
    all_rows = []
    if os.path.exists(checkpoint_csv):
        ckpt = pd.read_csv(checkpoint_csv, parse_dates=["date"])
        all_rows = ckpt.to_dict("records")
        done_dates = set(ckpt["date"].astype(str).unique())
        print(f"  Resuming from checkpoint: {len(done_dates)} periods already done")

    periods_done = 0
    for i in tqdm(all_periods, desc=f"Backtesting {n_stocks} stocks"):
        current_date = ref_dates[i]
        if str(pd.Timestamp(current_date).date()) in done_dates:
            continue   # Already computed, skip

        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
        for ticker, df in stock_dfs.items():
            mask = df["timestamps"] <= current_date
            if not mask.any():
                continue
            idx = df[mask].index[-1]
            if idx < LOOKBACK - 1 or idx + PRED_LEN >= len(df) or idx + 1 >= len(df):
                continue
            x_df = df.iloc[idx - LOOKBACK + 1 : idx + 1]
            if x_df[["open","high","low","close","volume","amount"]].isnull().values.any():
                continue
            y_ts = df.iloc[idx + 1 : idx + 1 + PRED_LEN]["timestamps"]
            batch_dfs.append(x_df[["open","high","low","close","volume","amount"]])
            batch_xts.append(x_df["timestamps"])
            batch_yts.append(y_ts)
            batch_meta.append({
                "ticker":        ticker,
                "date":          pd.Timestamp(current_date),
                "actual_day0":   df.iloc[idx + 1]["open"],          # open t+1 (realistic execution)
                "actual_return": (df.iloc[idx + PRED_LEN]["close"] / df.iloc[idx + 1]["open"]) - 1,
            })

        if len(batch_dfs) < top_n * 2:
            continue

        # ── Sub-batch for MPS efficiency (large batches saturate memory bandwidth) ──
        SUB_BATCH = 32
        period_preds = []
        try:
            for sb_start in range(0, len(batch_dfs), SUB_BATCH):
                sb_end = sb_start + SUB_BATCH
                preds = predictor.predict_batch(
                    df_list=batch_dfs[sb_start:sb_end],
                    x_timestamp_list=batch_xts[sb_start:sb_end],
                    y_timestamp_list=batch_yts[sb_start:sb_end],
                    pred_len=PRED_LEN,
                    sample_count=SAMPLE_COUNT,
                    T=0.8,
                    verbose=False,
                )
                period_preds.extend(zip(preds, batch_meta[sb_start:sb_end]))

            for p_df, meta in period_preds:
                pred_close = p_df.iloc[-1]["close"]
                all_rows.append({
                    "date":          meta["date"],
                    "ticker":        meta["ticker"],
                    "pred_return":   (pred_close / meta["actual_day0"]) - 1,
                    "actual_return": meta["actual_return"],
                })
            periods_done += 1
        except Exception:
            pass

        # ── Checkpoint every N periods ────────────────────────────────────────
        if periods_done % CHECKPOINT_EVERY == 0 and periods_done > 0:
            pd.DataFrame(all_rows).to_csv(checkpoint_csv, index=False)

    if not all_rows:
        raise ValueError("No forecasts generated.")

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(forecast_csv, index=False)
    # Clean up checkpoint once complete
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)
    print(f"\n  Saved {len(forecasts):,} rows → {forecast_csv}")
    return forecasts


# ── STEP 3: ANALYSIS ──────────────────────────────────────────────────────────
def analyze(forecasts, label="Full period"):
    df = forecasts.dropna(subset=["actual_return"])
    df = df[df["actual_return"].abs() < 0.5]
    dates = sorted(df["date"].unique())

    periods_per_year = 252 / STRIDE

    def _stats(rets):
        rets = np.array(rets)
        if len(rets) == 0: return 0, 0, 0
        ann_ret = (1 + rets.mean()) ** periods_per_year - 1
        ann_vol = rets.std() * np.sqrt(periods_per_year)
        return float(ann_ret), float(ann_vol), float(ann_ret / ann_vol) if ann_vol > 0 else 0

    ic_list = []
    top10p_rets, bot10p_rets = [], []
    top50_rets,  bot50_rets  = [], []
    bench_rets = []

    for d in dates:
        snap = df[df["date"] == d].sort_values("pred_return", ascending=False)
        n = len(snap)
        top_n = max(1, int(n * DECILE))
        if n < top_n * 4:   # Need enough stocks for meaningful bins
            continue

        ic, _ = spearmanr(snap["pred_return"], snap["actual_return"])
        if not np.isnan(ic):
            ic_list.append(ic)

        mid = n // 2
        top10p_rets.append(snap.head(top_n)["actual_return"].mean())
        bot10p_rets.append(snap.tail(top_n)["actual_return"].mean())
        top50_rets.append(snap.head(mid)["actual_return"].mean())
        bot50_rets.append(snap.tail(n - mid)["actual_return"].mean())
        bench_rets.append(snap["actual_return"].mean())

    t10_ann, t10_vol, t10_sr = _stats(top10p_rets)
    b10_ann, b10_vol, b10_sr = _stats(bot10p_rets)
    t50_ann, t50_vol, t50_sr = _stats(top50_rets)
    b50_ann, b50_vol, b50_sr = _stats(bot50_rets)
    bm_ann,  bm_vol,  bm_sr  = _stats(bench_rets)
    avg_ic = float(np.mean(ic_list)) if ic_list else 0.0

    # Long-short spread (Top-10% minus Bottom-10%)
    ls_rets = np.array(top10p_rets) - np.array(bot10p_rets)
    ls_ann  = float(ls_rets.mean() * periods_per_year)
    ls_vol  = float(ls_rets.std()  * np.sqrt(periods_per_year))
    ls_sr   = ls_ann / ls_vol if ls_vol > 0 else 0

    # Permutation test (500 shuffles)
    np.random.seed(42)
    null_sharpes = []
    for _ in range(500):
        shuf = []
        for d in dates:
            snap = df[df["date"] == d]
            n = len(snap)
            top_n = max(1, int(n * DECILE))
            if n < top_n * 4: continue
            shuf.append(snap.sample(frac=1).head(top_n)["actual_return"].mean())
        if shuf:
            _, _, s = _stats(shuf)
            null_sharpes.append(s)
    p_value = float(np.mean(np.array(null_sharpes) >= t10_sr)) if null_sharpes else 1.0

    top_n_avg = int(df.groupby("date").size().mean() * DECILE)

    print(f"\n{'='*65}")
    print(f"  500-STOCK UNIVERSE — {label.upper()}")
    print(f"  Model: Kronos-base  |  Zero-shot (no fine-tuning)")
    print(f"{'='*65}")
    rows = [
        ("Stocks per period (avg)",       f"{df.groupby('date').size().mean():.0f}"),
        ("Valid periods",                 len(top10p_rets)),
        (f"Top/Bot bin size (~{DECILE:.0%})",  f"~{top_n_avg} stocks"),
        ("Avg IC (Spearman)",             f"{avg_ic:.4f}"),
        ("",                             ""),
        (f"Top-{DECILE:.0%}  Ann. Return",    f"{t10_ann:.1%}"),
        (f"Top-{DECILE:.0%}  Sharpe",         f"{t10_sr:.2f}"),
        (f"Bot-{DECILE:.0%}  Ann. Return",    f"{b10_ann:.1%}"),
        (f"Bot-{DECILE:.0%}  Sharpe",         f"{b10_sr:.2f}"),
        ("L/S Spread Ann. Return",        f"{ls_ann:.1%}"),
        ("L/S Spread Sharpe",             f"{ls_sr:.2f}"),
        ("",                             ""),
        ("Top-50%  Sharpe",               f"{t50_sr:.2f}"),
        ("Bot-50%  Sharpe",               f"{b50_sr:.2f}"),
        ("Benchmark (EW all) Sharpe",     f"{bm_sr:.2f}"),
        ("",                             ""),
        ("Permutation p-value",           f"{p_value:.3f}  ({'✅ significant' if p_value < 0.05 else '⚠️  not significant at 5%'})"),
    ]
    for k, v in rows:
        if k:
            print(f"  {k:<38} {v}")
        else:
            print()
    print(f"{'='*65}")
    return t10_sr, b10_sr, ls_sr, t50_sr, b50_sr, bm_sr, avg_ic, p_value


# ── MAIN ──────────────────────────────────────────────────────────────────────
UNIVERSES = [
    {
        "label":        "SP500 Large-Cap (500 stocks)",
        "data_dir":     "data/Universe500",
        "forecast_csv": "forecasts_Universe500.csv",
        "load_fn":      load_tickers_sp500,
    },
    {
        "label":        "SmallCap-600 (S&P 600)",
        "data_dir":     "data/SmallCap600",
        "forecast_csv": "forecasts_SmallCap600.csv",
        "load_fn":      load_tickers_smallcap,
    },
]

if __name__ == "__main__":
    print("="*65)
    print("  LARGE UNIVERSE OOS TEST")
    print("  Model: Kronos-base  |  Mode: Zero-shot (no fine-tuning)")
    print(f"  Strategy: Top {DECILE:.0%} vs Bottom {DECILE:.0%} cross-sectional ranking")
    print("="*65)

    # Load model once for all universes
    print("\nLoading Kronos-base model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    for cfg in UNIVERSES:
        label        = cfg["label"]
        data_dir     = cfg["data_dir"]
        forecast_csv = cfg["forecast_csv"]

        print(f"\n{'='*65}")
        print(f"  UNIVERSE: {label}")
        print(f"{'='*65}")

        tickers = cfg["load_fn"]()
        valid   = download_universe(tickers, data_dir)

        if os.path.exists(forecast_csv):
            print(f"  Loading cached forecasts from {forecast_csv}...")
            forecasts = pd.read_csv(forecast_csv, parse_dates=["date"])
        else:
            forecasts = run_backtest(valid, data_dir, forecast_csv, label, predictor)

        # Full period
        analyze(forecasts, label=f"{label} — Full (2015–2026)")

        # True OOS (2025+)
        oos = forecasts[forecasts["date"] >= "2025-01-01"]
        if len(oos) > 10:
            print(f"\n  [OOS: {oos['date'].nunique()} periods from 2025-01-01]")
            analyze(oos, label=f"{label} — OOS only (2025+)")

    print("\n✅ All universes complete.")
