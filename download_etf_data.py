import os
import pandas as pd
import yfinance as yf
from tqdm import tqdm

def main():
    excel_path = '/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/ETF.xlsx'
    output_dir = '/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/ETF_Data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Read the tickers
    try:
        df_tickers = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    tickers = df_tickers['Ticker'].tolist()
    
    # Process each ticker
    for ticker_raw in tqdm(tickers, desc="Downloading ETF data"):
        ticker = ticker_raw.replace(' Equity', '').strip()
        
        # Download data
        try:
            data = yf.download(ticker, start='2000-01-01', progress=False)
            if data.empty:
                print(f"No data found for {ticker}")
                continue
                
            # Save to CSV
            file_path = os.path.join(output_dir, f"{ticker}.csv")
            data.to_csv(file_path)
            
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

if __name__ == "__main__":
    main()
