"""
Broad OOS Test: Download, convert, and backtest multiple ETF universes.
Reuses the same Kronos-base 40-day/5-day pipeline from etf_strategy.py.
"""
import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.curdir))
from model import Kronos, KronosTokenizer, KronosPredictor

# ── UNIVERSE DEFINITIONS ─────────────────────────────────────────────────────
UNIVERSES = {
    "Sector": [
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"
    ],
    "Industry": [
        "XBI", "XHB", "KBE", "XME", "XOP", "OIH", "ITB", "IBB", "SMH", "JETS",
        "XRT", "KRE", "XAR", "HACK", "TAN", "SKYY", "IGV", "SOXX", "GDX", "GDXJ",
        "XSD", "CIBR", "FINX", "PAVE", "MOO", "MJ", "BETZ", "PBW", "BLOK", "CLOU",
        "ARKK", "ARKG", "IHI", "ITA", "IYT", "PHO", "FDN", "KWEB", "CQQQ", "ASHR"
    ],
    "Factor": [
        "MTUM", "VLUE", "QUAL", "USMV", "SIZE", "VTV", "VUG", "IWD", "IWF",
        "SPYG", "SPYV", "RPV", "SPHQ", "MOAT", "COWZ"
    ],
    "FixedIncome": [
        "TLT", "IEF", "SHY", "HYG", "LQD", "EMB", "AGG", "BND", "TIPS",
        "MUB", "VCSH", "VCIT", "JNK", "BNDX", "IGIB"
    ],
    "Commodity": [
        "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "PPLT", "WEAT", "CPER"
    ]
}

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_BASE   = "data"
LOOKBACK    = 40
PRED_LEN    = 5
STRIDE      = 5
TOP_N       = 3

# ── STEP 1: DOWNLOAD & CONVERT ───────────────────────────────────────────────
def download_and_convert(universe_name, tickers):
    """Download from yfinance and convert to Kronos OHLCV format."""
    out_dir = os.path.join(DATA_BASE, universe_name)
    os.makedirs(out_dir, exist_ok=True)

    valid_tickers = []
    for ticker in tqdm(tickers, desc=f"Downloading {universe_name}"):
        out_path = os.path.join(out_dir, f"{ticker}.csv")
        if os.path.exists(out_path):
            valid_tickers.append(ticker)
            continue

        try:
            data = yf.download(ticker, start='2015-01-01', progress=False)
            if data.empty or len(data) < LOOKBACK + PRED_LEN + 10:
                print(f"  Skipping {ticker}: insufficient data ({len(data)} rows)")
                continue

            # Flatten multi-index columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            df = pd.DataFrame()
            df['timestamps'] = pd.to_datetime(data.index).strftime('%Y-%m-%d %H:%M:%S')
            df['open']   = data['Open'].values
            df['high']   = data['High'].values
            df['low']    = data['Low'].values
            df['close']  = data['Close'].values
            df['volume'] = data['Volume'].values
            df['amount'] = (data['Close'] * data['Volume']).values

            # Drop NaN rows
            df = df.dropna()
            df.to_csv(out_path, index=False)
            valid_tickers.append(ticker)
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")

    return valid_tickers


# ── STEP 2: RUN BACKTEST FOR ONE UNIVERSE ─────────────────────────────────────
def run_universe_backtest(universe_name, tickers, predictor):
    """Run the walk-forward backtest for a single universe."""
    data_dir = os.path.join(DATA_BASE, universe_name)

    etf_dfs = {}
    for t in tickers:
        path = os.path.join(data_dir, f"{t}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        if len(df) >= LOOKBACK + PRED_LEN:
            etf_dfs[t] = df

    if len(etf_dfs) < 3:
        print(f"  Skipping {universe_name}: only {len(etf_dfs)} valid tickers")
        return None

    # Use the ticker with most data as the date reference
    ref_ticker = max(etf_dfs, key=lambda t: len(etf_dfs[t]))
    ref_dates = np.sort(etf_dfs[ref_ticker]['timestamps'].unique())

    all_rows = []
    for i in tqdm(range(LOOKBACK, len(ref_dates) - PRED_LEN, STRIDE), 
                  desc=f"Backtesting {universe_name} ({len(etf_dfs)} tickers)"):
        current_date = ref_dates[i]
        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []

        for ticker, df in etf_dfs.items():
            mask = df['timestamps'] <= current_date
            if not mask.any():
                continue
            idx = df[mask].index[-1]
            if idx < LOOKBACK - 1 or idx + PRED_LEN >= len(df) or idx + 1 >= len(df):
                continue

            x_df = df.iloc[idx - LOOKBACK + 1 : idx + 1]
            x_ts = x_df['timestamps']
            y_ts = df.iloc[idx + 1 : idx + 1 + PRED_LEN]['timestamps']

            if x_df[['open','high','low','close','volume','amount']].isnull().values.any():
                continue

            batch_dfs.append(x_df[['open','high','low','close','volume','amount']])
            batch_xts.append(x_ts)
            batch_yts.append(y_ts)

            actual_entry = df.iloc[idx + 1]['open']       # open on t+1 (realistic execution)
            actual_exit  = df.iloc[idx + PRED_LEN]['close']  # close on t+5
            batch_meta.append({
                'ticker': ticker,
                'date': pd.Timestamp(current_date),
                'actual_day0': actual_entry,
                'actual_return': (actual_exit / actual_entry) - 1
            })

        if len(batch_dfs) < 3:
            continue

        try:
            preds = predictor.predict_batch(
                df_list=batch_dfs,
                x_timestamp_list=batch_xts,
                y_timestamp_list=batch_yts,
                pred_len=PRED_LEN,
                sample_count=10,
                T=0.8,
                verbose=False
            )
            for j, p_df in enumerate(preds):
                meta = batch_meta[j]
                pred_close = p_df.iloc[-1]['close']
                all_rows.append({
                    'date':          meta['date'],
                    'ticker':        meta['ticker'],
                    'pred_return':   (pred_close / meta['actual_day0']) - 1,
                    'actual_return': meta['actual_return']
                })
        except Exception as e:
            pass

    if not all_rows:
        return None

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(f"forecasts_{universe_name}.csv", index=False)
    return forecasts


# ── STEP 3: ANALYSIS ──────────────────────────────────────────────────────────
def analyze_universe(forecasts, universe_name):
    """Compute IC, Top-3 vs Bottom-3, Top-50% vs Bottom-50%."""
    from scipy.stats import spearmanr

    df = forecasts.dropna(subset=['actual_return'])
    dates = sorted(df['date'].unique())

    ic_list = []
    top3_rets, bot3_rets, top50_rets, bot50_rets, bench_rets = [], [], [], [], []

    for d in dates:
        snap = df[df['date'] == d].copy()

        # IC
        ic, _ = spearmanr(snap['pred_return'], snap['actual_return'])
        if not np.isnan(ic):
            ic_list.append(ic)

        # Sort by Kronos prediction
        snap_sorted = snap.sort_values('pred_return', ascending=False)
        n = len(snap_sorted)
        if n < 6: continue
        mid = n // 2

        top3_rets.append(snap_sorted.head(TOP_N)['actual_return'].mean())
        bot3_rets.append(snap_sorted.tail(TOP_N)['actual_return'].mean())
        top50_rets.append(snap_sorted.head(mid)['actual_return'].mean())
        bot50_rets.append(snap_sorted.tail(n - mid)['actual_return'].mean())
        bench_rets.append(snap_sorted['actual_return'].mean())

    periods_per_year = 252 / STRIDE

    def _sharpe(rets):
        rets = np.array(rets)
        if len(rets) == 0: return 0, 0, 0
        mean_ret = rets.mean()
        std_ret = rets.std()
        ann_ret = (1 + mean_ret) ** periods_per_year - 1
        ann_vol = std_ret * np.sqrt(periods_per_year)
        return ann_ret, ann_vol, ann_ret / ann_vol if ann_vol > 0 else 0

    top3_ann, top3_vol, top3_sr   = _sharpe(top3_rets)
    bot3_ann, bot3_vol, bot3_sr   = _sharpe(bot3_rets)
    top50_ann, top50_vol, top50_sr = _sharpe(top50_rets)
    bot50_ann, bot50_vol, bot50_sr = _sharpe(bot50_rets)
    bench_ann, bench_vol, bench_sr = _sharpe(bench_rets)

    avg_ic = np.mean(ic_list) if ic_list else 0

    # Permutation test (100 shuffles for speed)
    np.random.seed(42)
    null_sharpes = []
    for _ in range(100):
        shuf_top3_rets = []
        for d in dates:
            snap = df[df['date'] == d].copy()
            if len(snap) < 6: continue
            shuffled = snap.sample(frac=1)
            shuf_top3_rets.append(shuffled.head(TOP_N)['actual_return'].mean())
        if shuf_top3_rets:
            _, _, shuf_sr = _sharpe(shuf_top3_rets)
            null_sharpes.append(shuf_sr)

    p_value = np.mean(np.array(null_sharpes) >= top3_sr) if null_sharpes else 1.0

    result = {
        'Universe':       universe_name,
        '# Assets':       df['ticker'].nunique(),
        '# Periods':      len(dates),
        'Avg IC':         f"{avg_ic:.3f}",
        'Top-3 Ann.Ret':  f"{top3_ann:.1%}",
        'Top-3 Sharpe':   round(float(top3_sr), 2),
        'Bot-3 Ann.Ret':  f"{bot3_ann:.1%}",
        'Bot-3 Sharpe':   round(float(bot3_sr), 2),
        'Top50% Sharpe':  round(float(top50_sr), 2),
        'Bot50% Sharpe':  round(float(bot50_sr), 2),
        'Bench Sharpe':   round(float(bench_sr), 2),
        'p-value':        f"{p_value:.3f}"
    }
    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load model once
    print("Loading Kronos-base model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    summary_rows = []

    for universe_name, tickers in UNIVERSES.items():
        print(f"\n{'='*60}")
        print(f"UNIVERSE: {universe_name} ({len(tickers)} tickers)")
        print(f"{'='*60}")

        # Check for cached forecasts
        cache_file = f"forecasts_{universe_name}.csv"
        if os.path.exists(cache_file):
            print(f"  Loading cached forecasts from {cache_file}...")
            forecasts = pd.read_csv(cache_file, parse_dates=['date'])
        else:
            # Download & convert
            valid_tickers = download_and_convert(universe_name, tickers)
            if len(valid_tickers) < 3:
                print(f"  Skipping {universe_name}: too few valid tickers")
                continue

            # Run backtest
            forecasts = run_universe_backtest(universe_name, valid_tickers, predictor)

        if forecasts is None or len(forecasts) == 0:
            print(f"  No forecasts generated for {universe_name}")
            continue

        # Analyze
        result = analyze_universe(forecasts, universe_name)
        summary_rows.append(result)
        print(f"  → {result}")

    # Also include the existing Country ETF results
    country_file = "etf_forecasts_base40.csv"
    if os.path.exists(country_file):
        print(f"\n{'='*60}")
        print("UNIVERSE: Country ETFs (existing)")
        print(f"{'='*60}")
        country_fc = pd.read_csv(country_file, parse_dates=['date'])
        country_result = analyze_universe(country_fc, "Country")
        summary_rows.insert(0, country_result)

    # Final summary table
    summary = pd.DataFrame(summary_rows)
    print("\n" + "="*80)
    print("CROSS-UNIVERSE SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    summary.to_csv("cross_universe_summary.csv", index=False)
    print("\nSaved cross_universe_summary.csv")
