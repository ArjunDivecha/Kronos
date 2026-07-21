"""
Targeted Sweep for Specific (Lookback, Holding) Combinations
Target Universe: Industry ETFs (Optimized)
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

def run_targeted_sweep():
    # Explicit combinations requested:
    # 40-day lookback holding: 1, 2, 5, 10, 25
    # 5-day holding lookbacks: 5, 10, 20, 80
    # 25-day holding lookbacks: 5, 10, 20, 80
    combos = [
        (40, 5)
    ]
    
    # Optional: ensure uniqueness and sort
    combos = sorted(list(set(combos)))

    data_dir = "data/Industry28"

    tickers = sorted([f.replace(".csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")])
    print(f"Loading {len(tickers)} Industry ETFs...")

    etf_dfs = {}
    for ticker in tickers:
        df = pd.read_csv(f"{data_dir}/{ticker}.csv")
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df['ts_array'] = df['timestamps'].values
        etf_dfs[ticker] = df

    print("Loading Kronos-base...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=1024, device="mps") 

    ref_dates = np.sort(etf_dfs[tickers[0]]['timestamps'].unique())
    results_grid = []

    for lb, pl in combos:
        stride = max(pl, 15) 

        print(f"\nEvaluating: Lookback={lb}, Pred_Len={pl}, Stride={stride}")
        all_rows = []
        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
        SUB_BATCH_SIZE = 64
        
        for i in tqdm(range(lb, len(ref_dates) - pl, stride), desc=f"LB_{lb}|PL_{pl}"):
            current_date = ref_dates[i]
            
            for ticker in tickers:
                df = etf_dfs[ticker]
                idx_arr = np.searchsorted(df['ts_array'], np.datetime64(current_date), side='right') - 1
                if idx_arr < 0: continue
                idx = int(idx_arr)
                
                if idx < lb - 1 or idx + pl >= len(df) or idx + 1 >= len(df):
                    continue
                    
                x_df = df.iloc[idx - lb + 1 : idx + 1]
                if x_df[['open','high','low','close','volume','amount']].isnull().values.any(): continue
                    
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
                
            if len(batch_dfs) >= SUB_BATCH_SIZE or i >= len(ref_dates) - pl - stride:
                if not batch_dfs: continue
                try:
                    preds = predictor.predict_batch(
                        df_list=batch_dfs, x_timestamp_list=batch_xts, y_timestamp_list=batch_yts,
                        pred_len=pl, sample_count=20, T=0.8, verbose=False
                    )
                    
                    for j, p_df in enumerate(preds):
                        meta = batch_meta[j]
                        pred_close = p_df.iloc[-1]['close']
                        all_rows.append({
                            'date': meta['date'], 'ticker': meta['ticker'],
                            'pred_return': (pred_close / meta['actual_day0']) - 1,
                            'actual_return': meta['actual_return']
                        })
                except Exception as e:
                    pass
                batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
        
        forecasts = pd.DataFrame(all_rows)
        if forecasts.empty: continue
            
        TOP_N = 3
        dates = sorted(forecasts['date'].unique())
        strat_rows = []
        
        for d in dates:
            snap = forecasts[forecasts['date'] == d]
            if len(snap) < TOP_N * 2: continue
            
            bench_ret = snap['actual_return'].mean()
            top_tickers = snap.nlargest(TOP_N, 'pred_return')
            bot_tickers = snap.nsmallest(TOP_N, 'pred_return')
            
            long_ret = top_tickers['actual_return'].mean()
            bot_ret  = bot_tickers['actual_return'].mean()
            short_ret = -bot_ret # Shorting the losers
            
            ic, _ = spearmanr(snap['pred_return'], snap['actual_return'])
            
            strat_rows.append({
                'date': d, 
                'long_ret': long_ret, 
                'bot_ret': bot_ret,
                'short_ret': short_ret,
                'bench_ret': bench_ret,
                'ic': ic if not np.isnan(ic) else 0
            })
            
        if not strat_rows: continue
            
        perf = pd.DataFrame(strat_rows)
        mean_ic = perf['ic'].mean()
        
        periods_per_year = 252 / stride
        
        # Long Stats
        long_ann = (1 + perf['long_ret'].mean()) ** periods_per_year - 1
        long_vol = perf['long_ret'].std() * np.sqrt(periods_per_year)
        long_sharpe = long_ann / long_vol if long_vol > 0 else 0
        
        # Short Stats (Returns on the short position)
        short_ann = perf['short_ret'].mean() * periods_per_year  # Approx simple annualization for shorts
        short_vol = perf['short_ret'].std() * np.sqrt(periods_per_year)
        short_sharpe = short_ann / short_vol if short_vol > 0 else 0
        
        # L/S Spread (Long - Bot)
        ls_spread = perf['long_ret'].mean() - perf['bot_ret'].mean()
        ls_ann_spread = ls_spread * periods_per_year
        ls_vol_spread = (perf['long_ret'] - perf['bot_ret']).std() * np.sqrt(periods_per_year)
        ls_sharpe = ls_ann_spread / ls_vol_spread if ls_vol_spread > 0 else 0
        
        results_grid.append({
            'Lookback': lb, 'Holding': pl, 'Mean_IC': mean_ic,
            'Long_Sharpe': long_sharpe, 'Short_Sharpe': short_sharpe, 'LS_Sharpe': ls_sharpe, 
            'Long_Ann': long_ann, 'Short_Ann': short_ann
        })
        
        pd.DataFrame(results_grid).to_csv("industry_targeted_sweep_progress.csv", index=False)

    return pd.DataFrame(results_grid)

if __name__ == "__main__":
    final_df = run_targeted_sweep()
    final_df.to_csv("industry_targeted_sweep_final.csv", index=False)
    print("Targeted sweep complete")
