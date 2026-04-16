"""
=============================================================================
SCRIPT NAME: run_industry28.py
=============================================================================

INPUT FILES:
- Yahoo Finance (downloaded automatically via yfinance, 2010-01-01 onward)

OUTPUT FILES:
- data/Industry28/<TICKER>.csv           : raw OHLCV per ETF
- forecasts_Industry28.csv              : all 5-day walk-forward forecasts
- monthly_returns_industry28.csv        : monthly compounded returns
- industry28_metrics.txt                : summary metrics table

VERSION: 1.0
LAST UPDATED: 2026-04-15

DESCRIPTION:
Walk-forward backtest of the Kronos-base model on a curated 28-ETF industry
universe. At each rebalance date (every 5 trading days), Kronos receives the
40-day OHLCV lookback for each ETF, predicts 5 days forward, and ranks ETFs
by predicted return.

Portfolios computed:
  EW_All    - equal weight all available ETFs
  Top3/Top5 - top-N by predicted return
  Bot3/Bot5 - bottom-N by predicted return

Return definition: open[t+1] → close[t+5]  (realistic: signal at close t,
execute at open t+1, exit at close t+5).

DEPENDENCIES:
- pandas, numpy, scipy, yfinance, tqdm, torch
- model/ (Kronos, KronosTokenizer, KronosPredictor)

USAGE:
  source shiyu-coder-Kronos/.venv/bin/activate
  cd shiyu-coder-Kronos
  python run_industry28.py

NOTES:
- Model inference runs on MPS (Apple Silicon) if available.
- First run downloads ~10 MB of data; subsequent runs use the cache.
- Existing forecasts_Industry28.csv is used as a cache; delete it to re-run.
=============================================================================
"""

import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

# ── CONFIG ────────────────────────────────────────────────────────────────────
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

DATA_DIR        = "data/Industry28"
FORECASTS_FILE  = "forecasts_Industry28.csv"
MONTHLY_FILE    = "monthly_returns_industry28.csv"
METRICS_FILE    = "industry28_metrics.txt"

LOOKBACK   = 40
PRED_LEN   = 5
STRIDE     = 5
START_DATE = "2010-01-01"    # enough history for most ETFs
PPY        = 252 / 5         # non-overlapping 5-day periods per year


# ── STEP 1: DOWNLOAD DATA ─────────────────────────────────────────────────────
def download_data():
    """Download OHLCV from Yahoo Finance and save in Kronos format."""
    os.makedirs(DATA_DIR, exist_ok=True)
    valid = []
    for ticker in tqdm(TICKERS, desc="Downloading"):
        out = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(out):
            valid.append(ticker)
            continue
        try:
            raw = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
            if raw.empty or len(raw) < LOOKBACK + PRED_LEN + 10:
                print(f"  Skipping {ticker}: only {len(raw)} rows")
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = pd.DataFrame({
                'timestamps': pd.to_datetime(raw.index).strftime('%Y-%m-%d %H:%M:%S'),
                'open':   raw['Open'].values,
                'high':   raw['High'].values,
                'low':    raw['Low'].values,
                'close':  raw['Close'].values,
                'volume': raw['Volume'].values,
                'amount': (raw['Close'] * raw['Volume']).values,
            }).dropna()
            df.to_csv(out, index=False)
            valid.append(ticker)
        except Exception as e:
            print(f"  Error {ticker}: {e}")
    print(f"Downloaded {len(valid)}/{len(TICKERS)} tickers.")
    return valid


# ── STEP 2: WALK-FORWARD INFERENCE ───────────────────────────────────────────
def run_backtest(valid_tickers, predictor):
    """Walk-forward Kronos inference. Returns a DataFrame of per-period forecasts."""
    # Load all data into memory
    etf_dfs = {}
    for t in valid_tickers:
        path = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        if len(df) >= LOOKBACK + PRED_LEN:
            etf_dfs[t] = df

    if not etf_dfs:
        raise RuntimeError("No valid ETF data found.")

    # Use the ticker with the most data as the reference date grid
    ref_ticker = max(etf_dfs, key=lambda t: len(etf_dfs[t]))
    ref_dates  = np.sort(etf_dfs[ref_ticker]['timestamps'].unique())
    print(f"Reference ticker: {ref_ticker}  |  {len(ref_dates)} trading days")

    all_rows = []
    for i in tqdm(range(LOOKBACK, len(ref_dates) - PRED_LEN, STRIDE), desc="Walk-forward"):
        current_date = ref_dates[i]
        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []

        for ticker, df in etf_dfs.items():
            mask = df['timestamps'] <= current_date
            if not mask.any():
                continue
            idx = df[mask].index[-1]
            if idx < LOOKBACK - 1 or idx + PRED_LEN >= len(df) or idx + 1 >= len(df):
                continue

            x_df = df.iloc[idx - LOOKBACK + 1 : idx + 1]
            if x_df[['open','high','low','close','volume','amount']].isnull().values.any():
                continue

            y_ts = df.iloc[idx + 1 : idx + 1 + PRED_LEN]['timestamps']
            batch_dfs.append(x_df[['open','high','low','close','volume','amount']])
            batch_xts.append(x_df['timestamps'])
            batch_yts.append(y_ts)
            batch_meta.append({
                'ticker':        ticker,
                'date':          pd.Timestamp(current_date),
                'actual_day0':   df.iloc[idx + 1]['open'],       # open t+1 (entry)
                'actual_return': (df.iloc[idx + PRED_LEN]['close'] / df.iloc[idx + 1]['open']) - 1,
            })

        if len(batch_dfs) < 5:
            continue

        try:
            preds = predictor.predict_batch(
                df_list=batch_dfs,
                x_timestamp_list=batch_xts,
                y_timestamp_list=batch_yts,
                pred_len=PRED_LEN,
                sample_count=10,
                T=0.8,
                verbose=False,
            )
            for j, p_df in enumerate(preds):
                meta = batch_meta[j]
                pred_close = p_df.iloc[-1]['close']
                all_rows.append({
                    'date':          meta['date'],
                    'ticker':        meta['ticker'],
                    'pred_return':   (pred_close / meta['actual_day0']) - 1,
                    'actual_return': meta['actual_return'],
                })
        except Exception as e:
            print(f"  Batch error at {current_date}: {e}")
            continue

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(FORECASTS_FILE, index=False)
    print(f"Saved {len(forecasts)} forecasts → {FORECASTS_FILE}")
    return forecasts


# ── STEP 3: MONTHLY RETURNS ───────────────────────────────────────────────────
def build_monthly(forecasts):
    """Compound 5-day period returns within each calendar month."""
    df = forecasts[forecasts['actual_return'].abs() < 0.5].copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')

    portfolios = {
        'EW_All':  lambda s: s['actual_return'].mean(),
        'Top3':    lambda s: s.nlargest(3, 'pred_return')['actual_return'].mean(),
        'Top5':    lambda s: s.nlargest(5, 'pred_return')['actual_return'].mean(),
        'Bottom3': lambda s: s.nsmallest(3, 'pred_return')['actual_return'].mean(),
        'Bottom5': lambda s: s.nsmallest(5, 'pred_return')['actual_return'].mean(),
    }

    # Period-level returns
    period_rows = []
    for d, snap in df.groupby('date'):
        if len(snap) < 5:
            continue
        row = {'date': d, 'month': pd.Period(d, 'M')}
        for name, fn in portfolios.items():
            row[name] = fn(snap)
        period_rows.append(row)

    periods = pd.DataFrame(period_rows).sort_values('date').reset_index(drop=True)

    # Compound within month
    monthly_rows = []
    for month, grp in periods.groupby('month'):
        row = {'month': str(month), 'n_periods': len(grp)}
        for name in portfolios:
            row[name] = (1 + grp[name]).prod() - 1
        monthly_rows.append(row)

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(MONTHLY_FILE, index=False)
    print(f"Saved {len(monthly)} months → {MONTHLY_FILE}")
    return monthly


# ── STEP 4: METRICS TABLE ────────────────────────────────────────────────────
def compute_metrics(forecasts, monthly, cutoff='2024-07-01'):
    """Print and save the full metrics table (full-history + OOS split)."""
    cutoff_dt = pd.Timestamp(cutoff)
    cutoff_m  = pd.Period(cutoff, 'M')

    df = forecasts[forecasts['actual_return'].abs() < 0.5].copy()

    def period_stats(fcast, top_pct=0.1, label=''):
        """IC and rank-decile stats from 5-day periods."""
        ic_list, top_rets, bot_rets, bench_rets = [], [], [], []
        for d, snap in fcast.groupby('date'):
            n = len(snap)
            if n < 5:
                continue
            top_n = max(1, int(np.ceil(n * top_pct)))
            ic, _ = spearmanr(snap['pred_return'], snap['actual_return'])
            ic_list.append(ic)
            snap_s = snap.sort_values('pred_return', ascending=False)
            top_rets.append(snap_s.head(top_n)['actual_return'].mean())
            bot_rets.append(snap_s.tail(top_n)['actual_return'].mean())
            bench_rets.append(snap['actual_return'].mean())
        ic_arr = np.array(ic_list)
        return ic_arr, np.array(top_rets), np.array(bot_rets), np.array(bench_rets)

    def ann(r):
        return (1 + np.mean(r)) ** PPY - 1

    def sh(r):
        ar = ann(r)
        av = np.std(r, ddof=1) * np.sqrt(PPY)
        return ar / av if av > 0 else np.nan

    def fmt_block(ic_arr, top_r, bot_r, bench_r):
        ls_r = top_r - bot_r
        return {
            'Periods':           len(ic_arr),
            'Top 10% Return':    f'{ann(top_r):+.1%}',
            'Top 10% Sharpe':    f'{sh(top_r):.2f}',
            'Bot 10% Return':    f'{ann(bot_r):+.1%}',
            'Bot 10% Sharpe':    f'{sh(bot_r):.2f}',
            'L/S Spread Return': f'{ann(ls_r):+.1%}',
            'L/S Spread Sharpe': f'{sh(ls_r):.2f}',
            'Benchmark Return':  f'{ann(bench_r):+.1%}',
            'Benchmark Sharpe':  f'{sh(bench_r):.2f}',
            'Mean IC':           f'{np.nanmean(ic_arr):.4f}',
            'Median IC':         f'{np.nanmedian(ic_arr):.4f}',
            'Positive IC %':     f'{(ic_arr > 0).mean():.1%}',
        }

    def monthly_stats(mon, label=''):
        """Sharpe from monthly compounded returns."""
        results = {}
        for col in ['EW_All', 'Top3', 'Top5', 'Bottom3', 'Bottom5']:
            r = mon[col]
            ar = (1 + r.mean()) ** 12 - 1
            av = r.std(ddof=1) * np.sqrt(12)
            sh_v = ar / av if av > 0 else np.nan
            hit = (r > 0).mean()
            cum = (1 + r).prod() - 1
            results[col] = {
                'Ann Return': f'{ar:+.1%}',
                'Ann Vol':    f'{av:.1%}',
                'Sharpe':     f'{sh_v:.2f}',
                'Hit Rate':   f'{hit:.0%}',
                'Cumulative': f'{cum:+.0%}',
            }
        return results

    # Full history
    ic_all, top_all, bot_all, bench_all = period_stats(df)
    full_block = fmt_block(ic_all, top_all, bot_all, bench_all)

    # OOS only
    df_oos = df[pd.to_datetime(df['date']) >= cutoff_dt]
    if len(df_oos) > 0:
        ic_oos, top_oos, bot_oos, bench_oos = period_stats(df_oos)
        oos_block = fmt_block(ic_oos, top_oos, bot_oos, bench_oos)
    else:
        oos_block = {'Periods': 0}

    # Monthly portfolio stats
    mon_full = monthly
    mon_oos  = monthly[monthly['month'] >= str(cutoff_m)]
    mon_full_stats = monthly_stats(mon_full)
    mon_oos_stats  = monthly_stats(mon_oos) if len(mon_oos) > 0 else {}

    # ── Format output ──────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("INDUSTRY 28-ETF UNIVERSE  —  Kronos-base  (open[t+1] → close[t+5])")
    lines.append("=" * 80)
    lines.append(f"Tickers: {', '.join(TICKERS)}")
    lines.append(f"Period:  {df['date'].min().date()} → {df['date'].max().date()}")
    lines.append(f"OOS cutoff: {cutoff}  (post = {len(df_oos['date'].unique())} periods)")
    lines.append("")

    # Rank-decile table
    metric_order = [
        'Periods','Top 10% Return','Top 10% Sharpe',
        'Bot 10% Return','Bot 10% Sharpe',
        'L/S Spread Return','L/S Spread Sharpe',
        'Benchmark Return','Benchmark Sharpe',
        'Mean IC','Median IC','Positive IC %',
    ]
    lines.append("── PERIOD-LEVEL RANK STATS ──────────────────────────────────")
    col_w = 16
    hdr = f"{'Metric':<24}{'Full History':>{col_w}}{'OOS (post-Jul24)':>{col_w}}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in metric_order:
        v_full = str(full_block.get(m, 'n/a'))
        v_oos  = str(oos_block.get(m, 'n/a'))
        lines.append(f"{m:<24}{v_full:>{col_w}}{v_oos:>{col_w}}")
    lines.append("")

    # Monthly portfolio table
    lines.append("── MONTHLY COMPOUNDED PORTFOLIO RETURNS ─────────────────────")
    pfols = ['EW_All','Top3','Top5','Bottom3','Bottom5']
    stat_keys = ['Ann Return','Ann Vol','Sharpe','Hit Rate','Cumulative']
    for sk in stat_keys:
        row_full = f"{'Full '+sk:<24}" + "".join(
            f"{mon_full_stats.get(p,{}).get(sk,'n/a'):>{col_w}}" for p in pfols)
        lines.append(row_full)
    lines.append("")
    for sk in stat_keys:
        row_oos = f"{'OOS '+sk:<24}" + "".join(
            f"{mon_oos_stats.get(p,{}).get(sk,'n/a'):>{col_w}}" for p in pfols)
        lines.append(row_oos)
    lines.append("")
    lines.append(f"{'Portfolio':<24}" + "".join(f"{p:>{col_w}}" for p in pfols))
    lines.append("=" * 80)

    output = "\n".join(lines)
    print(output)
    with open(METRICS_FILE, 'w') as f:
        f.write(output)
    print(f"\nSaved metrics → {METRICS_FILE}")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Download
    print("\n── Step 1: Download data ────────────────────────────────────────")
    valid_tickers = download_data()

    # Step 2: Inference (or load cache)
    print("\n── Step 2: Walk-forward inference ───────────────────────────────")
    if os.path.exists(FORECASTS_FILE):
        print(f"Loading cached forecasts from {FORECASTS_FILE}")
        forecasts = pd.read_csv(FORECASTS_FILE, parse_dates=['date'])
    else:
        print("Loading Kronos-base model...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        forecasts = run_backtest(valid_tickers, predictor)

    print(f"Total forecast rows: {len(forecasts):,}")

    # Step 3: Monthly returns
    print("\n── Step 3: Monthly returns ───────────────────────────────────────")
    monthly = build_monthly(forecasts)

    # Step 4: Metrics
    print("\n── Step 4: Metrics ──────────────────────────────────────────────")
    compute_metrics(forecasts, monthly)
