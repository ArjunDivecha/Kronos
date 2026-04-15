"""
Monthly Performance Report Generator
Matches strategy_performance.csv and strategy_results.png style.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
df = pd.read_csv("forecasts_CountryMonthly.csv", parse_dates=["date"])
df = df.dropna(subset=["actual_return"])
dates = sorted(df["date"].unique())

# ── 2. COMPUTE RETURNS ────────────────────────────────────────────────────────
summary_rows = []
for d in dates:
    snap = df[df["date"] == d].sort_values("pred_return", ascending=False)
    if len(snap) < 6: continue
    
    top3 = snap.head(3)
    strat_ret = top3["actual_return"].mean()
    bench_ret = snap["actual_return"].mean()
    top_tickers = ", ".join(top3["ticker"].tolist())
    
    summary_rows.append({
        "date": d,
        "strat_ret": strat_ret,
        "bench_ret": bench_ret,
        "top_tickers": top_tickers,
        "n_available": len(snap)
    })

perf = pd.DataFrame(summary_rows)
# Cumulative
perf["cum_strat"] = (1 + perf["strat_ret"]).cumprod()
perf["cum_bench"] = (1 + perf["bench_ret"]).cumprod()
perf["relative"] = perf["cum_strat"] / perf["cum_bench"]

# Save CSV
perf.to_csv("monthly_country_performance.csv", index=False)
print(f"Saved monthly_country_performance.csv ({len(perf)} rows)")

# ── 3. PLOT ──────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top Subplot: Cumulative Returns
ax1.plot(perf["date"], perf["cum_strat"], color='#00ff99', linewidth=2, label='Kronos Top 3 Strategy')
ax1.plot(perf["date"], perf["cum_bench"], color='#ff3366', linewidth=1.5, alpha=0.7, label='Equal Weight Benchmark')
ax1.set_title("Long-Term Monthly Country Strategy (2000-2026)", fontsize=16, fontweight='bold', pad=20)
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

# Formatting
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
plt.tight_layout()

plt.savefig("monthly_country_results.png", dpi=300)
print("Saved monthly_country_results.png")

# Final Stats Print
total_ret = perf["cum_strat"].iloc[-1] - 1
ann_ret = (1 + total_ret) ** (12/len(perf)) - 1
total_bench = perf["cum_bench"].iloc[-1] - 1
ann_bench = (1 + total_bench) ** (12/len(perf)) - 1

print(f"\nSummary Stats:")
print(f"  Total Return (Strat): {total_ret:.1%}")
print(f"  Ann. Return (Strat): {ann_ret:.1%}")
print(f"  Total Return (Bench): {total_bench:.1%}")
print(f"  Ann. Return (Bench): {ann_bench:.1%}")
