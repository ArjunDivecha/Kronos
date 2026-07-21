"""
=============================================================================
SCRIPT NAME: etf_sweep.py
=============================================================================

DESCRIPTION:
    Loads historical ETF price/volume data from CSV files and evaluates the
    Kronos time-series model across a configurable lookback window. For each
    ETF ticker and each prediction date (stepping by stride), the script
    constructs a batch of price windows, runs Kronos inference to predict
    forward returns (pred_len days ahead), and records predicted vs. actual
    returns. After the sweep, it computes per-ticker metrics (Spearman IC
    in-sample pre-2024-07-01 and out-of-sample post-2024-07-01, directional
    accuracy) and saves them to CSV. Two heatmap PNGs (in-sample IC and
    out-of-sample IC) are generated for visual inspection.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/ETF/{ticker}.csv
        One CSV per ETF ticker containing columns: timestamps, open, high,
        low, close, volume, amount. The script lists all .csv files in the
        data/ETF/ directory and reads each as a DataFrame.

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/sweep_metrics_progress.csv
        Intermediate per-lookback metrics saved during the sweep loop.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/sweep_metrics_final.csv
        Final consolidated metrics (ticker, lookback, IC pre/post, accuracy).
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/sweep_heatmap_oos.png
        Heatmap of out-of-sample IC (post-June 2024) across tickers and
        lookback periods.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/sweep_heatmap_is.png
        Heatmap of in-sample IC (training period) across tickers and lookback
        periods.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - pandas
    - numpy
    - torch
    - matplotlib
    - seaborn
    - scipy
    - tqdm
    - model.py (local module providing Kronos, KronosTokenizer, KronosPredictor)

USAGE:
    python etf_sweep.py

NOTES:
    - Requires a Kronos model checkpoint downloaded via HuggingFace
      (from_pretrained calls to "NeoQuasar/Kronos-small" and
      "NeoQuasar/Kronos-Tokenizer-base").
    - The script uses SPY as the reference timeline for coordinating
      batch inference across assets.
    - Only tickers with a CSV file in data/ETF/ are processed; ETFs
      that start trading later (e.g. ASHR in 2013) are handled
      gracefully.
    - Heatmap generation may fail silently if metrics data is
      insufficient for pivoting.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm
from datetime import datetime

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

def calculate_metrics(results_df, cutoff_date="2024-07-01"):
    # Split into Pre and Post
    cutoff = pd.to_datetime(cutoff_date)
    pre_df = results_df[results_df['date'] < cutoff]
    post_df = results_df[results_df['date'] >= cutoff]
    
    metrics = {}
    
    # In-Sample (Pre)
    if len(pre_df) > 10:
        ic_pre, _ = spearmanr(pre_df['pred_return'], pre_df['actual_return'])
        metrics['ic_pre'] = ic_pre
    else:
        metrics['ic_pre'] = np.nan
        
    # Out-of-Sample (Post)
    if len(post_df) > 5:
        ic_post, p_val = spearmanr(post_df['pred_return'], post_df['actual_return'])
        metrics['ic_post'] = ic_post
        metrics['ic_post_p'] = p_val
        metrics['acc_post'] = (np.sign(post_df['pred_return']) == np.sign(post_df['actual_return'])).mean()
    else:
        metrics['ic_post'] = np.nan
        metrics['ic_post_p'] = np.nan
        metrics['acc_post'] = np.nan
        
    return metrics

def main():
    # 1. Config
    data_dir = "data/ETF"
    lookbacks = [40]
    pred_len = 5
    stride = 5
    batch_size = 200 # Total batch size across assets
    
    tickers = [f.replace(".csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    print(f"Loading {len(tickers)} ETFs...")
    etf_dfs = {}
    for ticker in tickers:
        df = pd.read_csv(f"{data_dir}/{ticker}.csv")
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        etf_dfs[ticker] = df
        
    # 2. Setup Model
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    all_metrics = []
    
    # 3. Sweep Loop
    for lb in lookbacks:
        print(f"\n===== TESTING LOOKBACK: {lb} DAYS =====")
        
        # We need a master timeline to coordinate asset batching
        # Using SPY as the reference timeline
        ref_dates = etf_dfs['SPY']['timestamps'].sort_values().unique()
        
        # Results storage for current lookback
        results_memory = {t: [] for t in tickers}
        
        # Iterate over the timeline with stride
        # We start from the maximum possible initial offset
        # Note: Some ETFs start later (e.g. ASHR starts in 2013)
        for i in tqdm(range(lb, len(ref_dates) - pred_len, stride), desc=f"Lookback {lb}"):
            current_date = ref_dates[i]
            x_end_date = current_date
            
            # Prepare a batch of ALL ETFs that are active at this time
            batch_dfs = []
            batch_xts = []
            batch_yts = []
            batch_meta = []
            
            for ticker in tickers:
                df = etf_dfs[ticker]
                
                # Check if this ETF exists and has enough history at this time point
                # We find the index of the date closest to our reference date
                mask = (df['timestamps'] <= x_end_date)
                if not mask.any():
                    continue
                
                idx = df[mask].index[-1]
                
                # Eligibility check
                if idx < lb - 1:
                    continue
                if idx + pred_len >= len(df):
                    continue
                
                # Pull windows
                x_df = df.iloc[idx - lb + 1 : idx + 1]
                x_ts = x_df['timestamps']
                y_ts = df.iloc[idx + 1 : idx + 1 + pred_len]['timestamps']
                
                # Check for NaNs
                if x_df[['open', 'high', 'low', 'close', 'volume', 'amount']].isnull().values.any():
                    continue

                batch_dfs.append(x_df[['open', 'high', 'low', 'close', 'volume', 'amount']])
                batch_xts.append(x_ts)
                batch_yts.append(y_ts)
                
                actual_day0 = df.iloc[idx + 1]['open']      # open t+1 (realistic execution)
                actual_day5 = df.iloc[idx + pred_len]['close']

                batch_meta.append({
                    'ticker': ticker,
                    'date': current_date,
                    'actual_day0': actual_day0,
                    'actual_return': (actual_day5 / actual_day0) - 1
                })
                
            # Run Batch Inference if we have candidates
            if not batch_dfs:
                continue
                
            try:
                # Predict for all eligible ETFs simultaneously
                pred_outputs = predictor.predict_batch(
                    df_list=batch_dfs,
                    x_timestamp_list=batch_xts,
                    y_timestamp_list=batch_yts,
                    pred_len=pred_len,
                    sample_count=3, # Reduced sample count for speed in large sweep
                    T=0.8,
                    verbose=False
                )
                
                for j, p_df in enumerate(pred_outputs):
                    meta = batch_meta[j]
                    pred_day5 = p_df.iloc[-1]['close']
                    
                    results_memory[meta['ticker']].append({
                        'date': meta['date'],
                        'actual_return': meta['actual_return'],
                        'pred_return': (pred_day5 / meta['actual_day0']) - 1
                    })
            except Exception as e:
                # Silently catch errors for specific malformed windows locally
                pass
                
        # Calculate Metrics for this Lookback
        for ticker in tickers:
            ticker_results = pd.DataFrame(results_memory[ticker])
            if not ticker_results.empty:
                m = calculate_metrics(ticker_results)
                m['ticker'] = ticker
                m['lookback'] = lb
                all_metrics.append(m)
                
        # Intermediate Save
        pd.DataFrame(all_metrics).to_csv("sweep_metrics_progress.csv", index=False)
        
    # 4. Final Processing
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("sweep_metrics_final.csv", index=False)
    
    print("\n" + "="*40)
    print("ETFS SWEEP FINAL SUMMARY")
    print("="*40)
    top_post = metrics_df.sort_values('ic_post', ascending=False).head(10)
    print("\nTop 10 Out-of-Sample Performers:")
    print(top_post[['ticker', 'lookback', 'ic_post', 'acc_post']])
    
    # 5. Visualizations
    try:
        # Heatmap of IC Post
        pivot_df = metrics_df.pivot(index='ticker', columns='lookback', values='ic_post')
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
        plt.title("Out-of-Sample IC Sweep (Post-June 2024)")
        plt.tight_layout()
        plt.savefig("sweep_heatmap_oos.png")
        
        # Heatmap of IC Pre
        pivot_df_pre = metrics_df.pivot(index='ticker', columns='lookback', values='ic_pre')
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_df_pre, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
        plt.title("In-Sample IC Sweep (Training Period)")
        plt.tight_layout()
        plt.savefig("sweep_heatmap_is.png")
    except Exception as e:
        print(f"Error generating heatmap: {e}")

if __name__ == "__main__":
    main()
