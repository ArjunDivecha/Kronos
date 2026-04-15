"""
Plotter for Large Universe Backtests (SP500 and SmallCap 600)
Creates professional dual-pane charts for Top 10 vs Benchmark.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_report(forecast_file, output_prefix, title_label, top_n=10, stride=5):
    df = pd.read_csv(forecast_file, parse_dates=["date"])
    df = df.dropna(subset=["actual_return"])
    df = df[df["actual_return"].abs() < 0.5]
    dates = sorted(df["date"].unique())

    summary_rows = []
    for d in dates:
        snap = df[df["date"] == d].sort_values("pred_return", ascending=False)
        if len(snap) < top_n * 2: continue
        
        top10 = snap.head(top_n)
        strat_ret = top10["actual_return"].mean()
        bench_ret = snap["actual_return"].mean()
        
        summary_rows.append({
            "date": d,
            "strat_ret": strat_ret,
            "bench_ret": bench_ret
        })

    perf = pd.DataFrame(summary_rows)
    perf["cum_strat"] = (1 + perf["strat_ret"]).cumprod()
    perf["cum_bench"] = (1 + perf["bench_ret"]).cumprod()
    perf["relative"] = perf["cum_strat"] / perf["cum_bench"]

    # Plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Top Subplot: Cumulative Returns
    ax1.plot(perf["date"], perf["cum_strat"], color='#00ff99', linewidth=2, label=f'Kronos Top {top_n}')
    ax1.plot(perf["date"], perf["cum_bench"], color='#ff3366', linewidth=1.5, alpha=0.7, label='Equal Weight Benchmark')
    ax1.set_title(f"{title_label} Strategy Performance", fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel("Cumulative Multiplier", fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Bottom Subplot: Relative Performance
    ax2.fill_between(perf["date"], perf["relative"], 1.0, where=(perf["relative"] >= 1), color='#00ff99', alpha=0.3)
    ax2.fill_between(perf["date"], perf["relative"], 1.0, where=(perf["relative"] < 1), color='#ff3366', alpha=0.3)
    ax2.plot(perf["date"], perf["relative"], color='white', linewidth=1, alpha=0.8)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Relative (Strat/Bench)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.2)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.tight_layout()

    plt.savefig(f"{output_prefix}_results.png", dpi=300)
    print(f"Saved {output_prefix}_results.png")

if __name__ == "__main__":
    # SP500 Plot
    generate_report("forecasts_Universe500.csv", "sp500", "S&P 500 Large-Cap", top_n=10)
    # SmallCap Plot
    generate_report("forecasts_SmallCap600.csv", "smallcap600", "S&P 600 Small-Cap", top_n=10)
