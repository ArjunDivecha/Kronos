import os
import pandas as pd
from tqdm import tqdm

def convert_file(input_path, output_path):
    # Read the yfinance CSV which has multi-header (Price, Ticker) and index (Date)
    # The structure we saw was:
    # Price,Close,High,Low,Open,Volume
    # Ticker,EWS,EWS,EWS,EWS,EWS
    # Date,,,,,
    # 2000-01-03,...
    
    try:
        # Read with multi-index header
        df = pd.read_csv(input_path, header=[0, 1], index_col=0)
        
        # Flatten columns or just select the first level (Price)
        # We know the Ticker is the same for all columns in one file
        df.columns = df.columns.get_level_values(0)
        
        # Rename columns to Kronos format
        # Current: Close, High, Low, Open, Volume
        # Target: timestamps, open, high, low, close, volume, amount
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Reset index to get Date as a column
        df = df.reset_index()
        df = df.rename(columns={'Date': 'timestamps'})
        
        # Convert timestamps to string format YYYY-MM-DD HH:MM:SS (standard for Kronos)
        df['timestamps'] = pd.to_datetime(df['timestamps']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate amount (close * volume) if not present
        df['amount'] = df['close'] * df['volume']
        
        # Reorder columns to match example: timestamps,open,high,low,close,volume,amount
        cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
        df = df[cols]
        
        # Save to new location
        df.to_csv(output_path, index=False)
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def main():
    input_dir = '/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/ETF_Data'
    output_dir = '/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/ETF'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    for filename in tqdm(files, desc="Converting to Kronos format"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        convert_file(input_path, output_path)

if __name__ == "__main__":
    main()
