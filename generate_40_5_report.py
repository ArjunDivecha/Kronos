"""
Script to generate the Cumulative Return Chart and Monthly Spreadsheet
for the 40/5 configuration.
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

def generate_report():
    lb = 40
    pl = 5
    stride = max(pl, 15)  # 15
    sample_count = 10 # Sufficiently stable for plotting but fast (15 mins)

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

    all_rows = []
    batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
    SUB_BATCH_SIZE = 64
    
    for i in tqdm(range(lb, len(ref_dates) - pl, stride), desc="Running Inference"):
        current_date = ref_dates[i]
        
        for ticker in tickers:
            df = etf_dfs[ticker]
            idx_arr = np.searchsorted(df['ts_array'], np.datetime64(current_date), side='right') - 1
            if idx_arr < 0: continue
            idx = int(idx_arr)
            
            if idx < lb - 1 or idx + pl >= len(df) or idx + 1 >= len(df): continue
                
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
                    pred_len=pl, sample_count=sample_count, T=0.8, verbose=False
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
    # Save the granular asset-level forecasts explicitly
    forecasts.to_csv("forecasts_industry_40_5_raw.csv", index=False)
    
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
        short_ret = -bot_ret 
        net_ret = long_ret + short_ret
        
        strat_rows.append({
            'Date': d, 
            'Long_Return': long_ret, 
            'Short_Return': short_ret,
            'Net_Return': net_ret,
            'Benchmark': bench_ret
        })
        
    perf = pd.DataFrame(strat_rows)
    perf['Date'] = pd.to_datetime(perf['Date'])
    perf = perf.sort_values('Date').set_index('Date')
    
    # Calculate cumulatives
    perf['Long_Cum'] = (1 + perf['Long_Return']).cumprod()
    perf['Short_Cum'] = (1 + perf['Short_Return']).cumprod()
    perf['Net_Cum'] = (1 + perf['Net_Return']).cumprod()

    # Save to CSV
    perf.to_csv("industry28_40_5_daily.csv")
    
    # Resample Monthly for spreadsheet
    monthly = perf[['Long_Return', 'Short_Return', 'Net_Return']].resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly.to_csv("industry28_40_5_monthly.csv")
    monthly.to_excel("industry28_40_5_monthly.xlsx")

    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(perf.index, perf['Long_Cum'], label='Long Portfolio (Top 3)', color='green', linewidth=2)
    plt.plot(perf.index, perf['Short_Cum'], label='Short Portfolio (Bottom 3)', color='red', linewidth=2)
    plt.plot(perf.index, perf['Net_Cum'], label='Net Portfolio (Long + Short)', color='blue', linewidth=3)
    
    plt.title('Kronos Industry28 Edge (40-day Lookback, 5-day Holding)', fontsize=16, fontweight='bold', color='#333333')
    plt.ylabel('Cumulative Return (1.0 = Base)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('industry28_40_5_chart.png', dpi=300, facecolor='white')
    print("Report generated: industry28_40_5_chart.png and industry28_40_5_monthly.xlsx")

if __name__ == "__main__":
    generate_report()
