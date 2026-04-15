import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved forecasts
FORECASTS_FILE = "etf_forecasts_base40.csv"
print(f"Loading forecasts from {FORECASTS_FILE}...")
df = pd.read_csv(FORECASTS_FILE, parse_dates=['date'])

# Drop rows without actual returns (latest forecasts)
df = df.dropna(subset=['actual_return'])

def calculate_group_returns(df):
    results = []
    
    # Group by date to perform ranking
    for date, group in df.groupby('date'):
        if len(group) < 10: # Skip dates with too few ETFs
             continue
             
        # Sort by prediction
        group = group.sort_values('pred_return', ascending=False)
        num_etfs = len(group)
        
        # Top 3 / Bottom 3
        top3_ret = group.head(3)['actual_return'].mean()
        bot3_ret = group.tail(3)['actual_return'].mean()
        
        # Top 50% / Bottom 50%
        mid = num_etfs // 2
        top50_ret = group.head(mid)['actual_return'].mean()
        bot50_ret = group.tail(num_etfs - mid)['actual_return'].mean()
        
        # All (Benchmark)
        bench_ret = group['actual_return'].mean()
        
        results.append({
            'date': date,
            'Top 3': top3_ret,
            'Bottom 3': bot3_ret,
            'Top 50%': top50_ret,
            'Bottom 50%': bot50_ret,
            'Benchmark': bench_ret
        })
        
    return pd.DataFrame(results).set_index('date')

def get_metrics(returns_df):
    metrics = []
    for col in returns_df.columns:
        rets = returns_df[col]
        # Since these are 5-day returns, we scale to annual assuming we rebalance every 5 days
        # 252 trading days / 5 = ~50 periods
        ann_ret = (rets.mean() / 5) * 252 
        ann_vol = (rets.std() / np.sqrt(5)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        metrics.append({
            'Group': col,
            'Ann. Return': f"{ann_ret:.2%}",
            'Ann. Vol': f"{ann_vol:.2%}",
            'Sharpe': round(sharpe, 2)
        })
    return pd.DataFrame(metrics)

# Run analysis
returns = calculate_group_returns(df)
metrics = get_metrics(returns)

print("\n" + "="*50)
print("RANK-ORDERING ANALYSIS (Top vs Bottom)")
print("="*50)
print(metrics.to_string(index=False))

# Calculate cumulative returns
cum_returns = (1 + returns).cumprod()

# Plotting
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
for col in cum_returns.columns:
    lw = 3 if col in ['Top 3', 'Bottom 3'] else 1.5
    plt.plot(cum_returns.index, cum_returns[col], label=col, linewidth=lw)

plt.title("Kronos ETF Strategy: Top vs Bottom Rank Comparison", fontsize=15)
plt.ylabel("Cumulative Growth (5-Day Compounded)")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.savefig("rank_analysis_results.png")
print("\nSaved comparison plot to rank_analysis_results.png")

# Calculate Relative Performance (Spread)
spreads = pd.DataFrame()
spreads['Top3 - Bot3'] = (1 + returns['Top 3']).cumprod() - (1 + returns['Bottom 3']).cumprod()
spreads['Top50 - Bot50'] = (1 + returns['Top 50%']).cumprod() - (1 + returns['Bottom 50%']).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(spreads.index, spreads['Top3 - Bot3'], label='Top 3 vs Bottom 3 Spread', color='blue')
plt.plot(spreads.index, spreads['Top50 - Bot50'], label='Top 50% vs Bottom 50% Spread', color='green')
plt.axhline(0, color='black', linestyle='--')
plt.title("Strategy Alpha Spread (Relative Return)", fontsize=15)
plt.ylabel("Alpha (Return Difference)")
plt.legend()
plt.tight_layout()
plt.savefig("rank_spread_results.png")
print("Saved spread analysis to rank_spread_results.png")
