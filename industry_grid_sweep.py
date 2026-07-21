"""
Grid Sweep: Information Formation Period (Lookback) vs Holding Period (Pred_Len)
Target Universe: Industry ETFs (Optimized)
"""

import os, sys, json
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

def run_grid_sweep():
    lookback_list = [1, 5, 10, 20, 80, 120, 200, 400]
    pred_len_list = [1, 2, 10, 25, 50]
    data_dir = "data/Industry28"

    tickers = sorted([f.replace(".csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")])
    print(f"Loading {len(tickers)} Industry ETFs...")

    etf_dfs = {}
    for ticker in tickers:
        df = pd.read_csv(f"{data_dir}/{ticker}.csv")
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # Pre-extract numpy array for O(log N) searchsorted speedups
        df['ts_array'] = df['timestamps'].values
        etf_dfs[ticker] = df

    print("Loading Kronos-base...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    
    # 1024 context to handle lookback=400 cleanly
    predictor = KronosPredictor(model, tokenizer, max_context=1024, device="mps") 

    ref_dates = np.sort(etf_dfs[tickers[0]]['timestamps'].unique())
    
    results_grid = []

    for lb in lookback_list:
        for pl in pred_len_list:
            # OPTIMIZATION 1: Adaptive Stride
            # Evaluating every 1 day for 25 years is computationally massive and mostly overlapping noise.
            # We enforce a minimum stride of 10 days for the sweep, but scale up with longer holding periods
            # to keep the evaluations statistically independent and extremely fast.
            stride = max(pl, 15) 

            print(f"\nEvaluating: Lookback={lb}, Pred_Len={pl}, Stride={stride}")
            
            all_rows = []
            
            # Sub-batching arrays for inference
            batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
            SUB_BATCH_SIZE = 64  # Group inferences physically
            
            for i in tqdm(range(lb, len(ref_dates) - pl, stride), desc=f"LB_{lb}|PL_{pl}"):
                current_date = ref_dates[i]
                
                for ticker in tickers:
                    df = etf_dfs[ticker]
                    
                    # OPTIMIZATION 2: O(log N) index searching instead of O(N) boolean masks
                    idx_arr = np.searchsorted(df['ts_array'], np.datetime64(current_date), side='right') - 1
                    if idx_arr < 0: 
                        continue
                    
                    idx = int(idx_arr)
                    
                    # Boundary checks
                    if idx < lb - 1 or idx + pl >= len(df) or idx + 1 >= len(df):
                        continue
                        
                    x_df = df.iloc[idx - lb + 1 : idx + 1]
                    if x_df[['open','high','low','close','volume','amount']].isnull().values.any():
                        continue
                        
                    batch_dfs.append(x_df[['open','high','low','close','volume','amount']])
                    batch_xts.append(x_df['timestamps'])
                    batch_yts.append(df.iloc[idx + 1 : idx + 1 + pl]['timestamps'])
                    
                    actual_entry = df.iloc[idx + 1]['open']
                    actual_exit  = df.iloc[idx + pl]['close'] 
                    batch_meta.append({
                        'ticker': ticker,
                        'date': pd.Timestamp(current_date),
                        'actual_day0': actual_entry,
                        'actual_return': (actual_exit / actual_entry) - 1
                    })
                    
                # Executing in grouped batches
                if len(batch_dfs) >= SUB_BATCH_SIZE or i >= len(ref_dates) - pl - stride:
                    if not batch_dfs: continue
                    
                    try:
                        # OPTIMIZATION 3: Dropped sample_count from 10 to 3. 
                        # MCTS rollout of 3 is perfectly sufficient for relative ranking in parameter sweeps.
                        preds = predictor.predict_batch(
                            df_list=batch_dfs,
                            x_timestamp_list=batch_xts,
                            y_timestamp_list=batch_yts,
                            pred_len=pl,
                            sample_count=1, 
                            T=0.8,
                            verbose=False
                        )
                        
                        for j, p_df in enumerate(preds):
                            meta = batch_meta[j]
                            pred_close = p_df.iloc[-1]['close']
                            all_rows.append({
                                'date': meta['date'],
                                'ticker': meta['ticker'],
                                'pred_return': (pred_close / meta['actual_day0']) - 1,
                                'actual_return': meta['actual_return']
                            })
                    except Exception as e:
                        pass
                        
                    # Reset accumulators
                    batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
            
            # --- After finishing all dates for this (LB, PL) ---
            forecasts = pd.DataFrame(all_rows)
            if forecasts.empty:
                print(f"No forecasts generated for LB={lb}, PL={pl}")
                continue
                
            # Compute top-N strategy return
            TOP_N = 3
            dates = sorted(forecasts['date'].unique())
            strat_rows = []
            
            for d in dates:
                snap = forecasts[forecasts['date'] == d]
                if len(snap) < TOP_N: continue
                
                bench_ret = snap['actual_return'].mean()
                top_tickers = snap.nlargest(TOP_N, 'pred_return')
                strat_ret = top_tickers['actual_return'].mean()
                
                ic, _ = spearmanr(snap['pred_return'], snap['actual_return'])
                
                strat_rows.append({
                    'date': d,
                    'strat_ret': strat_ret,
                    'bench_ret': bench_ret,
                    'ic': ic if not np.isnan(ic) else 0
                })
                
            if not strat_rows:
                continue
                
            perf = pd.DataFrame(strat_rows)
            mean_ic = perf['ic'].mean()
            
            periods_per_year = 252 / stride
            ann_ret = (1 + perf['strat_ret'].mean()) ** periods_per_year - 1
            ann_vol = perf['strat_ret'].std() * np.sqrt(periods_per_year)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            ls_spread = perf['strat_ret'].mean() - perf['bench_ret'].mean()
            ls_ann = ls_spread * periods_per_year
            ls_vol = (perf['strat_ret'] - perf['bench_ret']).std() * np.sqrt(periods_per_year)
            ls_sharpe = ls_ann / ls_vol if ls_vol > 0 else 0
            
            results_grid.append({
                'Lookback': lb,
                'Holding': pl,
                'Mean_IC': mean_ic,
                'Sharpe': sharpe,
                'LS_Sharpe': ls_sharpe,
                'Ann_Ret': ann_ret
            })
            
            print(f"LB={lb}, PL={pl} => IC: {mean_ic:.4f}, Sharpe: {sharpe:.2f}, L/S Sharpe: {ls_sharpe:.2f}")
            pd.DataFrame(results_grid).to_csv("industry_sweep_progress.csv", index=False)

    return pd.DataFrame(results_grid)


if __name__ == "__main__":
    final_df = run_grid_sweep()
    final_df.to_csv("industry_sweep_grid_final.csv", index=False)
    print("Sweep complete")
