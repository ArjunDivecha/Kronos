"""
Monthly Country Trial: 34 Country Assets
=========================================
Lookback: 20 months
Forecast: 1 month forward
Frequency: Monthly (1mo)

Comparing Top 3 and Top 50% (Top 17) rankings.
"""
import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
from scipy.stats import spearmanr

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS = [
    "ASHR", "ECH", "EDEN", "EIDO", "EPHE", "EPOL", "EWA", "EWC", "EWD", "EWG",
    "EWH", "EWI", "EWJ", "EWL", "EWM", "EWN", "EWP", "EWQ", "EWS", "EWT",
    "EWU", "EWW", "EWY", "EWZ", "EZA", "INDA", "IWM", "KSA", "MCHI", "QQQ",
    "SPY", "THD", "TUR", "VNM"
]
DATA_DIR      = "data/CountryMonthly"
FORECAST_CSV  = "forecasts_CountryMonthly.csv"
LOOKBACK      = 20
PRED_LEN      = 1
STRIDE        = 1

# ── STEP 1: DOWNLOAD MONTHLY DATA (from 2000) ─────────────────────────────────
def download_monthly(tickers):
    os.makedirs(DATA_DIR, exist_ok=True)
    valid_tickers = []
    for ticker in tqdm(tickers, desc="Downloading Monthly Data"):
        out_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        try:
            # Monthly data from 1995 to ensure we have data by 2000
            data = yf.download(ticker, start="1995-01-01", interval="1mo", progress=False, auto_adjust=True)
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            df = pd.DataFrame({
                "timestamps": pd.to_datetime(data.index).strftime("%Y-%m-%d %H:%M:%S"),
                "open":   data["Open"].values,
                "high":   data["High"].values,
                "low":    data["Low"].values,
                "close":  data["Close"].values,
                "volume": data["Volume"].values,
                "amount": (data["Close"] * data["Volume"]).values,
            }).dropna()
            
            if len(df) > 5:
                df.to_csv(out_path, index=False)
                valid_tickers.append(ticker)
        except Exception:
            pass
    print(f"  ✅ {len(valid_tickers)}/{len(tickers)} valid tickers")
    return valid_tickers

# ── STEP 2: BACKTEST (Expanding Lookback from 2000) ────────────────────────────
def run_backtest(tickers, predictor):
    stock_dfs = {}
    for t in tickers:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{t}.csv"))
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        stock_dfs[t] = df

    # We want all dates from 2000-01-01
    start_date = pd.Timestamp("2000-01-01")
    
    # Collect all unique monthly timestamps across ALL tickers since 2000
    all_dates = set()
    for t in tickers:
        df = stock_dfs[t]
        all_dates.update(df[df["timestamps"] >= start_date]["timestamps"])
    
    ref_dates = sorted(list(all_dates))
    print(f"  Running from {ref_dates[0]} to {ref_dates[-1]} ({len(ref_dates)} months)")
    
    all_rows = []
    for i in tqdm(range(len(ref_dates) - PRED_LEN), desc="Monthly Backtest"):
        current_date = ref_dates[i]
        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []

        for ticker in tickers:
            df = stock_dfs[ticker]
            mask = df["timestamps"] <= current_date
            if not mask.any(): continue
            
            idx = df[mask].index[-1]
            if idx < 4: continue # Need at least 5 months info
            if idx + PRED_LEN >= len(df): continue
            
            # Expanding lookback until it reaches LOOKBACK (20)
            lookback_dynamic = min(LOOKBACK, idx + 1)
            
            x_df = df.iloc[idx - lookback_dynamic + 1 : idx + 1]
            y_ts = df.iloc[idx + 1 : idx + 1 + PRED_LEN]["timestamps"]
            
            batch_dfs.append(x_df[["open","high","low","close","volume","amount"]])
            batch_xts.append(x_df["timestamps"])
            batch_yts.append(y_ts)
            batch_meta.append({
                "ticker": ticker,
                "date": pd.Timestamp(current_date),
                "actual_day0": df.iloc[idx]["close"],
                "actual_return": (df.iloc[idx + PRED_LEN]["close"] / df.iloc[idx]["close"]) - 1
            })

        if len(batch_dfs) < 6: # Need at least 6 assets for a meaningful Top-3
            continue

        # Group by length since Kronos predict_batch requires uniform length
        len_groups = {}
        for idx_in_batch, x_df in enumerate(batch_dfs):
            l = len(x_df)
            if l not in len_groups: len_groups[l] = []
            len_groups[l].append(idx_in_batch)

        period_forecasts = []
        for l, indices in len_groups.items():
            g_dfs = [batch_dfs[j] for j in indices]
            g_xts = [batch_xts[j] for j in indices]
            g_yts = [batch_yts[j] for j in indices]
            
            preds = predictor.predict_batch(
                df_list=g_dfs,
                x_timestamp_list=g_xts,
                y_timestamp_list=g_yts,
                pred_len=PRED_LEN,
                sample_count=10,
                T=0.8,
                verbose=False
            )
            for k, p_df in enumerate(preds):
                meta = batch_meta[indices[k]]
                pred_close = p_df.iloc[-1]["close"]
                period_forecasts.append({
                    "date": meta["date"],
                    "ticker": meta["ticker"],
                    "pred_return": (pred_close / meta["actual_day0"]) - 1,
                    "actual_return": meta["actual_return"]
                })
        
        all_rows.extend(period_forecasts)

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(FORECAST_CSV, index=False)
    return forecasts

# ── STEP 3: ANALYZE ───────────────────────────────────────────────────────────
def analyze(forecasts):
    df = forecasts.dropna(subset=["actual_return"])
    dates = sorted(df["date"].unique())
    
    top3_rets, bot3_rets = [], []
    top50_rets, bot50_rets = [], []
    bench_rets = []
    
    for d in dates:
        snap = df[df["date"] == d].sort_values("pred_return", ascending=False)
        n = len(snap)
        if n < 6: continue
        
        mid = n // 2
        top3_rets.append(snap.head(3)["actual_return"].mean())
        bot3_rets.append(snap.tail(3)["actual_return"].mean())
        top50_rets.append(snap.head(mid)["actual_return"].mean())
        bot50_rets.append(snap.tail(n - mid)["actual_return"].mean())
        bench_rets.append(snap["actual_return"].mean())

    def _stats(rets):
        rets = np.array(rets)
        # Monthly to Annual: 12 periods
        ann_ret = (1 + rets.mean()) ** 12 - 1
        ann_vol = rets.std() * np.sqrt(12)
        sr = ann_ret / ann_vol if ann_vol > 0 else 0
        return ann_ret, ann_vol, sr

    t3_ret, t3_vol, t3_sr = _stats(top3_rets)
    b3_ret, b3_vol, b3_sr = _stats(bot3_rets)
    t50_ret, t50_vol, t50_sr = _stats(top50_rets)
    b50_ret, b50_vol, b50_sr = _stats(bot50_rets)
    bm_ret, bm_vol, bm_sr = _stats(bench_rets)

    print("\n" + "="*60)
    print("MONTHLY COUNTRY TRIAL RESULTS")
    print("Universe: 34 Country ETFs | Lookback: 20mo | Forecast: 1mo")
    print("="*60)
    print(f"Periods: {len(dates)}")
    print(f"{'Strategy':<15} | {'Ann.Return':<10} | {'Ann.Vol':<10} | {'Sharpe':<6}")
    print("-" * 60)
    print(f"{'Top 3':<15} | {t3_ret:>10.1%} | {t3_vol:>10.1%} | {t3_sr:>6.2f}")
    print(f"{'Bottom 3':<15} | {b3_ret:>10.1%} | {b3_vol:>10.1%} | {b3_sr:>6.2f}")
    print("-" * 60)
    print(f"{'Top 50%':<15} | {t50_ret:>10.1%} | {t50_vol:>10.1%} | {t50_sr:>6.2f}")
    print(f"{'Bottom 50%':<15} | {b50_ret:>10.1%} | {b50_vol:>10.1%} | {b50_sr:>6.2f}")
    print("-" * 60)
    print(f"{'Benchmark':<15} | {bm_ret:>10.1%} | {bm_vol:>10.1%} | {bm_sr:>6.2f}")
    print("=" * 60)

if __name__ == "__main__":
    valid_tickers = download_monthly(TICKERS)
    
    print("\nLoading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    print("\nRunning backtest...")
    forecasts = run_backtest(valid_tickers, predictor)
    
    analyze(forecasts)
