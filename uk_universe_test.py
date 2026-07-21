"""
=============================================================================
SCRIPT NAME: uk_universe_test.py
=============================================================================

DESCRIPTION:
    Zero-shot backtest of the Kronos-base model on a large universe of UK
    stocks. Loads tickers from a UK.xlsx spreadsheet, downloads historical
    OHLCV data from Yahoo Finance, then runs a rolling backtest with 40-day
    lookback windows and 5-day forecast horizons. Uses the Kronos-base
    model with MPS sub-batching to predict future returns, computing
    predicted vs. actual returns for top/bottom decile portfolios. Saves
    forecasts to a CSV and supports checkpoint/resume via an intermediate
    checkpoint file.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/UK.xlsx
        Spreadsheet containing UK ticker symbols in a "Ticker" column.
        Tickers are cleaned and converted to Yahoo Finance .L format.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/UK500/<TICKER>.csv
        Per-ticker OHLCV CSV files downloaded from Yahoo Finance.
        Created by the download step and re-read during backtesting.
        (Intermediate output that becomes input for the backtest phase.)
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_UK500_checkpoint.csv
        Checkpoint CSV read on startup to resume a partially-completed
        backtest. Deleted after the run completes.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/UK500/<TICKER>.csv
        Per-ticker OHLCV CSV files downloaded from Yahoo Finance.
        Written during the download step.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_UK500_checkpoint.csv
        Intermediate checkpoint saved every 25 backtest periods.
        Automatically deleted when the backtest finishes successfully.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_UK500.csv
        Final output CSV containing predicted vs. actual returns for
        every ticker and date combination in the backtest.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas
    - numpy
    - yfinance
    - tqdm
    - scipy
    - torch (via model.py)

USAGE:
    python uk_universe_test.py

NOTES:
    - Requires the Kronos-base model (NeoQuasar/Kronos-base) downloaded via HuggingFace.
    - Runs on macOS MPS (Metal Performance Shaders) by default.
    - The UK.xlsx ticker file must exist at the expected path.
    - Downloading 500+ tickers from Yahoo Finance is rate-limited; the
      script caches data locally and only downloads missing tickers.
    - Checkpoint file is cleaned up on successful completion.
=============================================================================
"""
import os, sys, re, json
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from scipy.stats import spearmanr

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

# ── CONFIG ────────────────────────────────────────────────────────────────────
UK_EXCEL_FILE = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/UK.xlsx"
DATA_DIR      = "data/UK500"
FORECAST_CSV  = "forecasts_UK500.csv"
LOOKBACK      = 40
PRED_LEN      = 5
STRIDE        = 5
SAMPLE_COUNT  = 5
DECILE        = 0.10

# ── LOAD TICKERS ──────────────────────────────────────────────────────────────
def load_uk_tickers():
    df = pd.read_excel(UK_EXCEL_FILE)
    raw = df["Ticker"].dropna().tolist()
    
    cleaned = []
    for t in raw:
        t = t.replace(" LN Equity", "").strip()
        t = re.sub(r'[/]+$', '', t).strip()
        t = re.sub(r'\.', '-', t)
        t += ".L"
        cleaned.append(t)
    print(f"Loaded {len(cleaned)} UK tickers.")
    return cleaned

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
def download_universe(tickers):
    os.makedirs(DATA_DIR, exist_ok=True)
    to_download = [t for t in tickers if not os.path.exists(os.path.join(DATA_DIR, f"{t}.csv"))]
    already = len(tickers) - len(to_download)
    print(f"  {already} cached, downloading {len(to_download)} new tickers...")

    for ticker in tqdm(to_download, desc="Downloading UK Data"):
        out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
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

    valid = [t for t in tickers if os.path.exists(os.path.join(DATA_DIR, f"{t}.csv"))]
    print(f"  ✅ {len(valid)}/{len(tickers)} valid UK stocks")
    return valid

# ── BACKTEST ──────────────────────────────────────────────────────────────────
CHECKPOINT_EVERY = 25

def run_backtest(tickers, predictor):
    stock_dfs = {}
    for t in tqdm(tickers, desc="Loading data"):
        path = os.path.join(DATA_DIR, f"{t}.csv")
        try:
            df = pd.read_csv(path)
            df["timestamps"] = pd.to_datetime(df["timestamps"])
            if len(df) >= LOOKBACK + PRED_LEN:
                stock_dfs[t] = df
        except Exception:
            pass

    n_stocks = len(stock_dfs)
    top_n = max(1, int(n_stocks * DECILE))
    print(f"\n  {n_stocks} valid stocks | Top/Bottom 10% = {top_n} stocks per bin")

    ref_ticker = max(stock_dfs, key=lambda t: len(stock_dfs[t]))
    ref_dates  = np.sort(stock_dfs[ref_ticker]["timestamps"].unique())
    all_periods = list(range(LOOKBACK, len(ref_dates) - PRED_LEN, STRIDE))

    checkpoint_csv = FORECAST_CSV.replace(".csv", "_checkpoint.csv")
    done_dates = set()
    all_rows = []
    
    if os.path.exists(checkpoint_csv):
        ckpt = pd.read_csv(checkpoint_csv, parse_dates=["date"])
        all_rows = ckpt.to_dict("records")
        done_dates = set(ckpt["date"].astype(str).unique())
        print(f"  Resuming from checkpoint: {len(done_dates)} periods already done")

    periods_done = 0
    for i in tqdm(all_periods, desc="Backtesting UK"):
        current_date = ref_dates[i]
        if str(pd.Timestamp(current_date).date()) in done_dates:
            continue

        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
        for ticker, df in stock_dfs.items():
            mask = df["timestamps"] <= current_date
            if not mask.any(): continue
            idx = df[mask].index[-1]
            if idx < LOOKBACK - 1 or idx + PRED_LEN >= len(df) or idx + 1 >= len(df): continue

            x_df = df.iloc[idx - LOOKBACK + 1 : idx + 1]
            if x_df[["open","high","low","close","volume","amount"]].isnull().values.any(): continue

            y_ts = df.iloc[idx + 1 : idx + 1 + PRED_LEN]["timestamps"]
            batch_dfs.append(x_df[["open","high","low","close","volume","amount"]])
            batch_xts.append(x_df["timestamps"])
            batch_yts.append(y_ts)
            batch_meta.append({
                "ticker": ticker,
                "date": pd.Timestamp(current_date),
                "actual_day0": df.iloc[idx + 1]["open"],          # open t+1 (realistic execution)
                "actual_return": (df.iloc[idx + PRED_LEN]["close"] / df.iloc[idx + 1]["open"]) - 1,
            })

        if len(batch_dfs) < top_n * 2: continue

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
                    "date": meta["date"],
                    "ticker": meta["ticker"],
                    "pred_return": (pred_close / meta["actual_day0"]) - 1,
                    "actual_return": meta["actual_return"],
                })
            periods_done += 1
        except Exception:
            pass

        if periods_done % CHECKPOINT_EVERY == 0 and periods_done > 0:
            pd.DataFrame(all_rows).to_csv(checkpoint_csv, index=False)

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(FORECAST_CSV, index=False)
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)
    print(f"\n  Saved {len(forecasts):,} rows → {FORECAST_CSV}")
    return forecasts

if __name__ == "__main__":
    tickers = load_uk_tickers()
    valid_tickers = download_universe(tickers)
    
    print("\nLoading Kronos-base model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    run_backtest(valid_tickers, predictor)
    print("\n✅ UK Test Fully Completed. Outputs are in", FORECAST_CSV)
