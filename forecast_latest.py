"""
Generate the latest forecast for all ETFs using the last 40 days of data.
Appends to the existing forecasts file with actual_return = NaN.
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

DATA_DIR = "data/ETF"
LOOKBACK = 40
PRED_LEN = 5
FORECASTS_FILE = "etf_forecasts_base40.csv"

tickers = sorted([f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
print(f"Loading {len(tickers)} ETFs...")

etf_dfs = {}
for t in tickers:
    df = pd.read_csv(f"{DATA_DIR}/{t}.csv")
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    etf_dfs[t] = df

# Load model
print("Loading Kronos-base...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Build batch from the tail of each ETF
batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []

for t in tickers:
    df = etf_dfs[t]
    if len(df) < LOOKBACK:
        continue

    x_df = df.iloc[-LOOKBACK:]
    x_ts = x_df['timestamps']

    # Generate future business-day timestamps for the prediction window
    last_date = x_ts.iloc[-1]
    y_ts = pd.Series(pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=PRED_LEN))

    if x_df[['open','high','low','close','volume','amount']].isnull().values.any():
        continue

    batch_dfs.append(x_df[['open','high','low','close','volume','amount']])
    batch_xts.append(x_ts)
    batch_yts.append(y_ts)
    batch_meta.append({
        'ticker': t,
        'date': last_date,
        'actual_day0': df.iloc[-1]['close']
    })

print(f"Forecasting {len(batch_dfs)} ETFs from their latest date...")

preds = predictor.predict_batch(
    df_list=batch_dfs,
    x_timestamp_list=batch_xts,
    y_timestamp_list=batch_yts,
    pred_len=PRED_LEN,
    sample_count=10,
    T=0.8,
    verbose=True
)

new_rows = []
for j, p_df in enumerate(preds):
    meta = batch_meta[j]
    pred_close = p_df.iloc[-1]['close']
    new_rows.append({
        'date':          meta['date'],
        'ticker':        meta['ticker'],
        'pred_return':   (pred_close / meta['actual_day0']) - 1,
        'actual_return': np.nan
    })

latest = pd.DataFrame(new_rows)

# Print the latest forecasts ranked
print("\n" + "="*50)
print(f"LATEST FORECASTS (as of {latest['date'].iloc[0].strftime('%Y-%m-%d')})")
print("="*50)
print(latest.sort_values('pred_return', ascending=False).to_string(index=False))

# Append to existing file
existing = pd.read_csv(FORECASTS_FILE, parse_dates=['date'])
combined = pd.concat([existing, latest], ignore_index=True)
combined.to_csv(FORECASTS_FILE, index=False)
print(f"\nAppended {len(latest)} forecasts → {FORECASTS_FILE} now has {len(combined)} rows.")

# Write standalone xlsx in ETF.xlsx ticker order
ETF_ORDER_FILE = "/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/ETF.xlsx"
etf_order = pd.read_excel(ETF_ORDER_FILE)
ordered_tickers = [t.replace(" Equity", "") for t in etf_order['Ticker']]

latest_ordered = latest.set_index('ticker').reindex(ordered_tickers).reset_index()
latest_ordered.rename(columns={'index': 'ticker'}, inplace=True)

forecast_date = latest['date'].iloc[0].strftime('%Y-%m-%d')
xlsx_path = f"latest_forecast_{forecast_date}.xlsx"
latest_ordered.to_excel(xlsx_path, index=False, sheet_name='Forecast')
print(f"Saved standalone forecast to {xlsx_path}")
