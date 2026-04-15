import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

def plot_results(results_df, ticker, output_path):
    # Plot 1: Scatter of Predicted vs Actual
    # Plot 2: Cumulative Return (Directional)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pre- vs Post- June 2024
    cutoff_date = pd.to_datetime("2024-07-01")
    pre_mask = results_df['date'] < cutoff_date
    post_mask = ~pre_mask
    
    # Scatter plot
    ax1.scatter(results_df.loc[pre_mask, 'pred_return'], results_df.loc[pre_mask, 'actual_return'], 
                alpha=0.3, label='Training Period (In-Sample)', color='blue')
    ax1.scatter(results_df.loc[post_mask, 'pred_return'], results_df.loc[post_mask, 'actual_return'], 
                alpha=0.6, label='Out-of-Sample (Post-June 2024)', color='red')
    
    ax1.set_xlabel('Predicted 5-Day Return')
    ax1.set_ylabel('Actual 5-Day Return')
    ax1.set_title(f'Predicted vs Actual Returns - {ticker}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative Performance (Theoretical Directional Strategy)
    # Simple strategy: If pred > 0, long. If pred < 0, short.
    results_df['strategy_return'] = np.sign(results_df['pred_return']) * results_df['actual_return']
    results_df['cum_strategy'] = (1 + results_df['strategy_return']).cumprod()
    results_df['cum_buy_hold'] = (1 + results_df['actual_return']).cumprod()
    
    ax2.plot(results_df['date'], results_df['cum_buy_hold'], label='Buy & Hold (Every 5-day step)', color='gray', alpha=0.7)
    ax2.plot(results_df['date'], results_df['cum_strategy'], label='Kronos Directional Strategy', color='green', linewidth=2)
    
    # Add vertical line for cutoff
    ax2.axvline(x=cutoff_date, color='black', linestyle='--', label='Training Cutoff (June 2024)')
    
    ax2.set_title(f'Cumulative Strategy Performance - {ticker}')
    ax2.set_ylabel('Equity Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Backtest results plot saved to {output_path}")

def main():
    # 1. Setup
    ticker = "SPY"
    lookback = 20
    pred_len = 5
    stride = 5
    batch_size = 100
    
    csv_path = f"data/ETF/{ticker}.csv"
    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    print(f"Loading {ticker} with {len(df)} rows...")
    
    # 2. Models
    print(f"Loading {ticker} with {len(df)} rows...")
    print("Loading Kronos-base model (100M parameters)...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    # 3. Window Generation
    # Create windows starting from lookback, ending 5 days before the end
    start_indices = range(0, len(df) - lookback - pred_len, stride)
    
    dfs_batch = []
    xts_batch = []
    yts_batch = []
    metadata = [] # To keep track of returns later
    
    all_results = []
    
    print(f"Generated {len(start_indices)} test windows. Running in batches of {batch_size}...")
    
    # Run in batches
    for i in tqdm(range(0, len(start_indices), batch_size), desc="Processing Batches"):
        batch_idxs = start_indices[i : i + batch_size]
        
        current_batch_dfs = []
        current_batch_xts = []
        current_batch_yts = []
        current_batch_meta = []
        
        for idx in batch_idxs:
            # Inputs
            x_df = df.iloc[idx : idx + lookback].copy()
            x_ts = x_df['timestamps']
            y_ts = df.iloc[idx + lookback : idx + lookback + pred_len]['timestamps']
            
            current_batch_dfs.append(x_df[['open', 'high', 'low', 'close', 'volume', 'amount']])
            current_batch_xts.append(x_ts)
            current_batch_yts.append(y_ts)
            
            # Metadata for actual results
            actual_day0_open  = df.iloc[idx + lookback]['open']                    # open t+1 (realistic execution)
            actual_day5_close = df.iloc[idx + lookback + pred_len - 1]['close']    # close t+5

            current_batch_meta.append({
                'date': df.iloc[idx + lookback]['timestamps'],  # Date prediction was made
                'actual_day0': actual_day0_open,
                'actual_day5': actual_day5_close,
                'actual_return': (actual_day5_close / actual_day0_open) - 1
            })
            
        # Run Batch Inference
        try:
            pred_dfs = predictor.predict_batch(
                df_list=current_batch_dfs,
                x_timestamp_list=current_batch_xts,
                y_timestamp_list=current_batch_yts,
                pred_len=pred_len,
                sample_count=5, # Using 5 for better stability
                T=0.8,
                verbose=False
            )
            
            # Post-process results
            for j, p_df in enumerate(pred_dfs):
                pred_day5_close = p_df.iloc[-1]['close']
                meta = current_batch_meta[j]
                
                all_results.append({
                    'date': meta['date'],
                    'actual_return': meta['actual_return'],
                    'pred_return': (pred_day5_close / meta['actual_day0']) - 1
                })
                
        except Exception as e:
            print(f"Error in batch {i//batch_size}: {e}")
            
    # 4. Final Analysis
    results_df = pd.DataFrame(all_results).sort_values('date')
    
    # Calculate Metrics
    cutoff_date = pd.to_datetime("2024-07-01")
    pre_df = results_df[results_df['date'] < cutoff_date]
    post_df = results_df[results_df['date'] >= cutoff_date]
    
    # Total IC
    ic_total, _ = spearmanr(results_df['pred_return'], results_df['actual_return'])
    ic_pre, _ = spearmanr(pre_df['pred_return'], pre_df['actual_return'])
    
    # Out of Sample metrics
    if not post_df.empty:
        ic_post, p_val = spearmanr(post_df['pred_return'], post_df['actual_return'])
        acc_post = (np.sign(post_df['pred_return']) == np.sign(post_df['actual_return'])).mean()
    else:
        ic_post, p_val, acc_post = 0, 1, 0
        
    print("\n" + "="*40)
    print("WALK-FORWARD BACKTEST RESULTS (KRONOS-BASE)")
    print("="*40)
    print(f"Ticker:               {ticker}")
    print(f"Windows Tested:       {len(results_df)}")
    print(f"Lookback:             {lookback} Days")
    print(f"Horizon:              {pred_len} Days (Stride={stride})")
    print("-"*40)
    print(f"TOTAL IC (Spearman):  {ic_total:.4f}")
    print(f"IN-SAMPLE IC:         {ic_pre:.4f}")
    print(f"OUT-OF-SAMPLE IC:     {ic_post:.4f} (p-value: {p_val:.4f})")
    print(f"OOS DIR. ACCURACY:    {acc_post:.2%}")
    print("="*40)
    
    plot_results(results_df, ticker, "walkforward_results_base.png")

if __name__ == "__main__":
    main()
