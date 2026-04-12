import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import torch

# Ensure we can import from the root 'model' package
sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

def plot_prediction(historical_df, true_df, pred_df, ticker, output_path):
    # Set index to timestamps for better plotting
    historical_df = historical_df.copy()
    true_df = true_df.copy()
    pred_df = pred_df.copy()
    
    historical_df['timestamps'] = pd.to_datetime(historical_df['timestamps'])
    true_df['timestamps'] = pd.to_datetime(true_df['timestamps'])
    
    # pred_df returned by Kronos has timestamps as the index
    if 'timestamps' not in pred_df.columns:
        pred_df = pred_df.reset_index()
    pred_df['timestamps'] = pd.to_datetime(pred_df['timestamps'])
    
    plt.figure(figsize=(12, 6))
    
    # Plot historical
    plt.plot(historical_df['timestamps'], historical_df['close'], label='Historical (Lookback)', color='gray', alpha=0.6)
    
    # Plot ground truth (future)
    plt.plot(true_df['timestamps'], true_df['close'], label='Ground Truth (Actual)', color='blue', linewidth=2)
    
    # Plot prediction
    plt.plot(pred_df['timestamps'], pred_df['close'], label='Kronos Prediction', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Kronos Price Prediction Test - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def main():
    # 1. Load Model and Tokenizer
    print("Loading models...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    
    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    model = model.to(device)
    
    # 2. Instantiate Predictor
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    
    # 3. Prepare Data
    ticker = "SPY"
    csv_path = f"data/ETF/{ticker}.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # Let's pick a recent point in history (but not the very end, so we have ground truth)
    # Total rows: let's see
    total_len = len(df)
    lookback = 400
    pred_len = 30
    
    # Start lookback 430 days before the end so we have a 30-day "future" to test against
    start_idx = total_len - lookback - pred_len - 100 # Go back a bit further to see a nice chart
    
    x_df = df.iloc[start_idx : start_idx + lookback].copy()
    x_timestamp = x_df['timestamps']
    
    # These are the timestamps we WANT to predict for (the "ground truth" period)
    y_true_df = df.iloc[start_idx + lookback : start_idx + lookback + pred_len].copy()
    y_timestamp = y_true_df['timestamps']
    
    print(f"Running prediction for {ticker} starting from {y_timestamp.iloc[0]}...")
    
    # 4. Make Prediction
    pred_df = predictor.predict(
        df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=5, # Average 5 samples for more stability
        verbose=True
    )
    
    # 5. Visualize Results
    # We'll plot a bit of history + the future
    output_plot = "kronos_test_result.png"
    # Show last 100 days of history + 30 days of prediction
    plot_prediction(x_df.tail(100), y_true_df, pred_df, ticker, output_plot)
    
    # Print some metrics
    mse = ((pred_df['close'].values - y_true_df['close'].values) ** 2).mean()
    print(f"Test Result for {ticker}:")
    print(f"Mean Squared Error: {mse:.4f}")

if __name__ == "__main__":
    main()
