"""
ETF Strategy: Kronos-base 40-day lookback, 5-day horizon.
1. Run walk-forward sweep across all 34 ETFs, saving every individual forecast.
2. Build a top-3 selection strategy vs equal-weight benchmark.
3. Calculate stats and plot results.
"""
import os, sys, json
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor


# ── 1. CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR   = "data/ETF"
LOOKBACK   = 40
PRED_LEN   = 5
STRIDE     = 5
BATCH_SIZE = 200
FORECASTS_FILE = "etf_forecasts_base40.csv"
TOP_N      = 3


def run_sweep():
    """Run Kronos-base inference across all ETFs and save every prediction."""

    tickers = sorted([f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    print(f"Loading {len(tickers)} ETFs...")

    etf_dfs = {}
    for ticker in tickers:
        df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv")
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        etf_dfs[ticker] = df

    # Model
    print("Loading Kronos-base (100M params)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    ref_dates = np.sort(etf_dfs['SPY']['timestamps'].unique())
    all_rows  = []

    for i in tqdm(range(LOOKBACK, len(ref_dates) - PRED_LEN, STRIDE), desc="Sweep"):
        current_date = ref_dates[i]

        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []

        for ticker in tickers:
            df = etf_dfs[ticker]
            mask = df['timestamps'] <= current_date
            if not mask.any():
                continue
            idx = df[mask].index[-1]
            if idx < LOOKBACK - 1 or idx + PRED_LEN >= len(df) or idx + 1 >= len(df):
                continue

            x_df  = df.iloc[idx - LOOKBACK + 1 : idx + 1]
            x_ts  = x_df['timestamps']
            y_ts  = df.iloc[idx + 1 : idx + 1 + PRED_LEN]['timestamps']

            if x_df[['open','high','low','close','volume','amount']].isnull().values.any():
                continue

            batch_dfs.append(x_df[['open','high','low','close','volume','amount']])
            batch_xts.append(x_ts)
            batch_yts.append(y_ts)

            actual_entry = df.iloc[idx + 1]['open']    # open on t+1 (realistic execution)
            actual_exit  = df.iloc[idx + PRED_LEN]['close']  # close on t+5
            batch_meta.append({
                'ticker': ticker,
                'date': pd.Timestamp(current_date),
                'actual_day0': actual_entry,
                'actual_return': (actual_exit / actual_entry) - 1
            })

        if not batch_dfs:
            continue

        try:
            preds = predictor.predict_batch(
                df_list=batch_dfs,
                x_timestamp_list=batch_xts,
                y_timestamp_list=batch_yts,
                pred_len=PRED_LEN,
                sample_count=10,
                T=0.8,
                verbose=False
            )
            for j, p_df in enumerate(preds):
                meta = batch_meta[j]
                pred_close = p_df.iloc[-1]['close']
                all_rows.append({
                    'date':          meta['date'],
                    'ticker':        meta['ticker'],
                    'pred_return':   (pred_close / meta['actual_day0']) - 1,
                    'actual_return': meta['actual_return']
                })
        except Exception as e:
            pass   # skip malformed batches silently

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(FORECASTS_FILE, index=False)
    print(f"Saved {len(forecasts)} individual forecasts to {FORECASTS_FILE}")
    return forecasts


def build_strategy(forecasts):
    """
    At each date pick the TOP_N highest predicted-return ETFs.
    Compare their equal-weight actual return to the equal-weight
    return of ALL ETFs available on that date.
    """
    dates = sorted(forecasts['date'].unique())
    rows  = []

    for d in dates:
        snap = forecasts[forecasts['date'] == d].copy()
        n_available = len(snap)
        if n_available < TOP_N:
            continue

        # Benchmark: equal-weight of everything available
        bench_ret = snap['actual_return'].mean()

        # Strategy: equal-weight of top-N by predicted return
        top = snap.nlargest(TOP_N, 'pred_return')
        strat_ret = top['actual_return'].mean()

        rows.append({
            'date':        d,
            'strat_ret':   strat_ret,
            'bench_ret':   bench_ret,
            'top_tickers': ', '.join(top['ticker'].tolist()),
            'n_available': n_available
        })

    perf = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)

    # Cumulative returns
    perf['cum_strat'] = (1 + perf['strat_ret']).cumprod()
    perf['cum_bench'] = (1 + perf['bench_ret']).cumprod()
    perf['relative']  = perf['cum_strat'] / perf['cum_bench']

    return perf


def compute_stats(perf):
    """Compute annualised stats for strategy and benchmark."""
    periods_per_year = 252 / STRIDE   # ~50 non-overlapping 5-day periods / year

    def _stats(rets, label):
        ann_ret  = (1 + rets.mean()) ** periods_per_year - 1
        ann_vol  = rets.std() * np.sqrt(periods_per_year)
        sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
        hit_rate = (rets > 0).mean()
        return {
            'Label':         label,
            'Ann. Return':   f"{ann_ret:.2%}",
            'Ann. Vol':      f"{ann_vol:.2%}",
            'Sharpe':        f"{sharpe:.2f}",
            'Hit Rate':      f"{hit_rate:.1%}",
            'Periods':       len(rets)
        }

    stats = pd.DataFrame([
        _stats(perf['strat_ret'], f'Top-{TOP_N} Strategy'),
        _stats(perf['bench_ret'], 'EW Benchmark (All ETFs)')
    ])

    # In-sample vs OOS split
    cutoff = pd.to_datetime("2024-07-01")
    pre  = perf[perf['date'] < cutoff]
    post = perf[perf['date'] >= cutoff]

    stats_split = pd.DataFrame([
        _stats(pre['strat_ret'],  f'Top-{TOP_N} (In-Sample)'),
        _stats(pre['bench_ret'],  'Bench (In-Sample)'),
        _stats(post['strat_ret'], f'Top-{TOP_N} (OOS)'),
        _stats(post['bench_ret'], 'Bench (OOS)')
    ])

    return stats, stats_split


def plot_results(perf, stats_text):
    """Generate a 2-panel chart: cumulative returns + relative performance."""
    cutoff = pd.to_datetime("2024-07-01")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Panel 1: Cumulative returns
    ax1.plot(perf['date'], perf['cum_strat'], label=f'Top-{TOP_N} Strategy', color='#2ecc71', linewidth=2)
    ax1.plot(perf['date'], perf['cum_bench'], label='EW All ETFs', color='#95a5a6', linewidth=1.5)
    ax1.axvline(x=cutoff, color='black', linestyle='--', alpha=0.6, label='Training Cutoff')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Kronos-base Top-3 Selection Strategy vs Equal-Weight Benchmark')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Panel 2: Relative performance (strategy / benchmark)
    ax2.plot(perf['date'], perf['relative'], color='#3498db', linewidth=1.5)
    ax2.axhline(y=1.0, color='gray', linestyle='-', alpha=0.4)
    ax2.axvline(x=cutoff, color='black', linestyle='--', alpha=0.6)
    ax2.set_ylabel('Relative (Strategy / Bench)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    # Add stats text
    ax1.text(0.99, 0.03, stats_text, transform=ax1.transAxes,
             fontsize=8, ha='right', va='bottom', family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig("strategy_results.png", dpi=150)
    print("Saved strategy_results.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: Run sweep (or load cached)
    if os.path.exists(FORECASTS_FILE):
        print(f"Loading cached forecasts from {FORECASTS_FILE}...")
        forecasts = pd.read_csv(FORECASTS_FILE, parse_dates=['date'])
    else:
        forecasts = run_sweep()
        forecasts['date'] = pd.to_datetime(forecasts['date'])

    print(f"Total forecasts: {len(forecasts)}")

    # Step 2: Build strategy
    perf = build_strategy(forecasts)

    # Step 3: Stats
    stats, stats_split = compute_stats(perf)
    print("\n" + "="*60)
    print("OVERALL PERFORMANCE")
    print("="*60)
    print(stats.to_string(index=False))
    print("\n" + "-"*60)
    print("IN-SAMPLE vs OUT-OF-SAMPLE")
    print("-"*60)
    print(stats_split.to_string(index=False))

    # Step 4: Plot
    stats_text = stats.to_string(index=False)
    plot_results(perf, stats_text)

    # Step 5: Save strategy returns
    perf.to_csv("strategy_performance.csv", index=False)
    print("\nSaved strategy_performance.csv")
