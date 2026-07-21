"""
=============================================================================
SCRIPT NAME: india_extended_run.py
=============================================================================

DESCRIPTION:
    Loads a universe of India equities from an Excel file, downloads
    historical OHLCV data for each security from Bloomberg (via the
    OpusBloomberg library), then runs the Kronos neural-network backtest
    over a user-configurable historical window.

    The backtest evaluates Kronos-base predictions against actual returns,
    computing top/bottom decile bins for each period. Supports resumption
    from partial checkpoint files. The Bloomberg download can be run as a
    separate subprocess pass in the OpusBloomberg conda environment.

INPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/India.xlsx
        India securities universe (columns: Ticker, Name). Loaded at
        startup to determine which securities to download and backtest.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/India/{file_stem}.csv
        Individual stock OHLCV CSVs (one per security), produced by the
        Bloomberg download pass and consumed by the backtest loop. Each
        contains columns: timestamps, open, high, low, close, volume.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_India_{start_year}_{end_year}[_limitN]_checkpoint.csv
        Optional checkpoint file read on startup to resume a partially
        completed backtest run (written every CHECKPOINT_EVERY periods).

OUTPUT FILES:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/India/{file_stem}.csv
        Historical OHLCV CSV for each India security, downloaded from
        Bloomberg and named by a sanitised ticker file_stem.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/data/India/_download_manifest.csv
        Download-status manifest recording each security, its CSV filename,
        download status (cached/downloaded/too_short/error), and row count.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_India_{start_year}_{end_year}[_limitN].csv
        Final backtest predictions: per-period, per-ticker predicted return
        and actual return. Written once the backtest loop completes.
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/forecasts_India_{start_year}_{end_year}[_limitN]_checkpoint.csv
        Intermediate backtest checkpoint, saved every CHECKPOINT_EVERY
        periods. Deleted when the final forecast CSV is written.

VERSION: 1.0
LAST UPDATED: 2026-06-05
AUTHOR: Arjun Divecha

DEPENDENCIES:
    - blpapi (via OpusBloomberg)
    - pandas
    - numpy
    - kronos (local model package from this repo)
    - tqdm (optional; no-op fallback provided)

USAGE:
    python india_extended_run.py [--download-only] [--skip-download]
        [--force-refresh] [--limit N] [--start-date YYYYMMDD]
        [--end-date YYYYMMDD] [--analysis-end-date YYYY-MM-DD]

NOTES:
    - Bloomberg Terminal must be open in Parallels before running.
    - The download pass for Bloomberg runs in the OpusBloomberg conda
      environment (auto-spawned as a subprocess).
    - The script loads Kronos-base and Kronos-Tokenizer from HuggingFace
      (pretrained "NeoQuasar/Kronos-base" and "NeoQuasar/Kronos-Tokenizer-base").
=============================================================================
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - download env may not include tqdm
    def tqdm(iterable=None, **_kwargs):
        return iterable


REPO_DIR = Path(__file__).resolve().parent
INDIA_EXCEL_FILE = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/India.xlsx")
DATA_DIR = REPO_DIR / "data" / "India"
MANIFEST_CSV = DATA_DIR / "_download_manifest.csv"

OPUS_BLOOMBERG_DIR = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/OpusBloomberg")
OPUS_BLOOMBERG_ENV = OPUS_BLOOMBERG_DIR / ".venv"

BLOOMBERG_FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
LOOKBACK = 40
PRED_LEN = 5
STRIDE = 5
SAMPLE_COUNT = 5
DECILE = 0.10
CHECKPOINT_EVERY = 50
DEFAULT_START_DATE = "20000101"
DEFAULT_END_DATE = "20201231"
DEFAULT_ANALYSIS_END_DATE = "2020-12-31"


def parse_args():
    parser = argparse.ArgumentParser(description="India extended Kronos run with Bloomberg data.")
    parser.add_argument("--download-only", action="store_true", help="Only fetch Bloomberg history.")
    parser.add_argument("--skip-download", action="store_true", help="Use already-downloaded CSVs.")
    parser.add_argument("--force-refresh", action="store_true", help="Re-download Bloomberg history.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the run to the first N securities.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Bloomberg history start date (YYYYMMDD).")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Bloomberg history end date (YYYYMMDD).")
    parser.add_argument(
        "--analysis-end-date",
        default=DEFAULT_ANALYSIS_END_DATE,
        help="Inclusive backtest end date (YYYY-MM-DD).",
    )
    return parser.parse_args()


def security_to_file_stem(security):
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", security.strip())
    stem = re.sub(r"-{2,}", "-", stem).strip("-")
    return stem or "unknown-security"


def load_india_universe(limit=None):
    df = pd.read_excel(INDIA_EXCEL_FILE)
    records = []
    seen = set()

    for _, row in df.iterrows():
        security = str(row.get("Ticker", "")).strip()
        if not security or security in seen:
            continue

        seen.add(security)
        records.append({
            "security": security,
            "name": str(row.get("Name", "")).strip(),
            "file_stem": security_to_file_stem(security),
        })

    if limit is not None:
        records = records[:limit]

    print(f"Loaded {len(records)} India securities from {INDIA_EXCEL_FILE.name}.")
    return records


def rows_to_price_frame(rows):
    if not rows:
        return pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume", "amount"])

    df = pd.DataFrame(rows)
    rename_map = {
        "date": "timestamps",
        "PX_OPEN": "open",
        "PX_HIGH": "high",
        "PX_LOW": "low",
        "PX_LAST": "close",
        "PX_VOLUME": "volume",
    }
    df = df.rename(columns=rename_map)

    df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df["volume"] = df["volume"].fillna(0.0)
    df["amount"] = df["close"] * df["volume"]
    df = df.dropna(subset=["timestamps", "open", "high", "low", "close"])
    df = df.sort_values("timestamps").drop_duplicates("timestamps")
    df["timestamps"] = df["timestamps"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]


def analysis_end_timestamp(analysis_end_date):
    return pd.Timestamp(analysis_end_date)


def forecast_csv_path(start_date, analysis_end_date, limit=None):
    end_year = analysis_end_timestamp(analysis_end_date).year
    base_name = f"forecasts_India_{start_date[:4]}_{end_year}"
    if limit is not None:
        base_name += f"_limit{limit}"
    return REPO_DIR / f"{base_name}.csv"


def download_bloomberg_history(records, start_date, end_date, force_refresh=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if str(OPUS_BLOOMBERG_DIR) not in sys.path:
        sys.path.insert(0, str(OPUS_BLOOMBERG_DIR))

    from bbg import BBG, bloomberg_setup

    vm_ip = bloomberg_setup(verbose=False)
    print(f"Bloomberg connected via {vm_ip}.")

    manifest_rows = []
    with BBG() as bbg:
        if not bbg.ping():
            raise RuntimeError("Bloomberg ping failed after successful setup.")

        for record in tqdm(records, desc="Downloading India Bloomberg history"):
            out_path = DATA_DIR / f"{record['file_stem']}.csv"
            if out_path.exists() and not force_refresh:
                with out_path.open("r", encoding="utf-8") as handle:
                    row_count = sum(1 for _ in handle) - 1
                manifest_rows.append({
                    "security": record["security"],
                    "name": record["name"],
                    "file": out_path.name,
                    "status": "cached",
                    "rows": row_count,
                })
                continue

            try:
                rows = bbg.hist(record["security"], BLOOMBERG_FIELDS, start_date, end_date)
                price_df = rows_to_price_frame(rows)
                if len(price_df) < LOOKBACK + PRED_LEN:
                    if out_path.exists():
                        out_path.unlink()
                    manifest_rows.append({
                        "security": record["security"],
                        "name": record["name"],
                        "file": out_path.name,
                        "status": "too_short",
                        "rows": len(price_df),
                    })
                    continue

                price_df.to_csv(out_path, index=False)
                manifest_rows.append({
                    "security": record["security"],
                    "name": record["name"],
                    "file": out_path.name,
                    "status": "downloaded",
                    "rows": len(price_df),
                })
            except Exception as exc:
                if out_path.exists():
                    out_path.unlink()
                manifest_rows.append({
                    "security": record["security"],
                    "name": record["name"],
                    "file": out_path.name,
                    "status": f"error: {exc}",
                    "rows": 0,
                })
                print(f"Warning: failed on {record['security']}: {exc}")

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False)
    valid_count = sum(row["status"] in {"cached", "downloaded"} for row in manifest_rows)
    print(f"Saved Bloomberg history for {valid_count}/{len(records)} India securities.")


def ensure_bloomberg_data(args):
    if args.skip_download:
        print("Skipping Bloomberg download stage.")
        return

    cmd = [
        "conda",
        "run",
        "-p",
        str(OPUS_BLOOMBERG_ENV),
        "python",
        str(Path(__file__).resolve()),
        "--download-only",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
    ]
    if args.force_refresh:
        cmd.append("--force-refresh")
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("Running Bloomberg download pass in shared OpusBloomberg env...")
    subprocess.run(cmd, cwd=str(REPO_DIR), check=True)


def build_predictor():
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))

    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    return KronosPredictor(model, tokenizer, max_context=512)


def load_stock_frames(records, analysis_end_ts):
    stock_dfs = {}
    for record in tqdm(records, desc="Loading India data"):
        path = DATA_DIR / f"{record['file_stem']}.csv"
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
            df["timestamps"] = pd.to_datetime(df["timestamps"])
            df = df[df["timestamps"] <= analysis_end_ts].sort_values("timestamps").reset_index(drop=True)
            if len(df) >= LOOKBACK + PRED_LEN:
                stock_dfs[record["security"]] = df
        except Exception:
            continue

    return stock_dfs


def run_backtest(records, predictor, forecast_csv, analysis_end_ts):
    stock_dfs = load_stock_frames(records, analysis_end_ts)
    if not stock_dfs:
        print(f"No India data found through {analysis_end_ts.date()}.")
        return pd.DataFrame()

    n_stocks = len(stock_dfs)
    top_n = max(1, int(n_stocks * DECILE))
    print(f"\n  {n_stocks} valid India stocks through {analysis_end_ts.date()} | Top/Bottom 10% = {top_n} stocks per bin")

    ref_ticker = max(stock_dfs, key=lambda ticker: len(stock_dfs[ticker]))
    ref_dates = np.sort(stock_dfs[ref_ticker]["timestamps"].unique())
    all_periods = list(range(LOOKBACK, len(ref_dates) - PRED_LEN, STRIDE))

    checkpoint_csv = forecast_csv.with_name(f"{forecast_csv.stem}_checkpoint.csv")
    done_dates = set()
    all_rows = []

    if checkpoint_csv.exists():
        checkpoint = pd.read_csv(checkpoint_csv, parse_dates=["date"])
        all_rows = checkpoint.to_dict("records")
        done_dates = set(checkpoint["date"].astype(str).unique())
        print(f"  Resuming from checkpoint: {len(done_dates)} periods already done")

    periods_done = 0
    for idx in tqdm(all_periods, desc=f"Backtesting India ({analysis_end_ts.year})"):
        current_date = ref_dates[idx]
        if str(pd.Timestamp(current_date).date()) in done_dates:
            continue

        batch_dfs, batch_xts, batch_yts, batch_meta = [], [], [], []
        for security, df in stock_dfs.items():
            mask = df["timestamps"] <= current_date
            if not mask.any():
                continue

            last_idx = df[mask].index[-1]
            if last_idx < LOOKBACK - 1 or last_idx + PRED_LEN >= len(df) or last_idx + 1 >= len(df):
                continue

            x_df = df.iloc[last_idx - LOOKBACK + 1:last_idx + 1]
            if x_df[["open", "high", "low", "close", "volume", "amount"]].isnull().values.any():
                continue

            y_ts = df.iloc[last_idx + 1:last_idx + 1 + PRED_LEN]["timestamps"]
            batch_dfs.append(x_df[["open", "high", "low", "close", "volume", "amount"]])
            batch_xts.append(x_df["timestamps"])
            batch_yts.append(y_ts)
            batch_meta.append({
                "ticker": security,
                "date": pd.Timestamp(current_date),
                "actual_day0": df.iloc[last_idx + 1]["open"],          # open t+1 (realistic execution)
                "actual_return": (df.iloc[last_idx + PRED_LEN]["close"] / df.iloc[last_idx + 1]["open"]) - 1,
            })

        if len(batch_dfs) < top_n * 2:
            continue

        period_preds = []
        try:
            for sb_start in range(0, len(batch_dfs), 32):
                sb_end = sb_start + 32
                preds = predictor.predict_batch(
                    df_list=batch_dfs[sb_start:sb_end],
                    x_timestamp_list=batch_xts[sb_start:sb_end],
                    y_timestamp_list=batch_yts[sb_start:sb_end],
                    pred_len=PRED_LEN,
                    sample_count=SAMPLE_COUNT,
                    T=0.8,
                    verbose=False,
                )
                period_preds.extend(zip(preds, batch_meta[sb_start:sb_end]))

            for prediction_df, meta in period_preds:
                pred_close = prediction_df.iloc[-1]["close"]
                all_rows.append({
                    "date": meta["date"],
                    "ticker": meta["ticker"],
                    "pred_return": (pred_close / meta["actual_day0"]) - 1,
                    "actual_return": meta["actual_return"],
                })
            periods_done += 1
        except Exception as exc:
            print(f"Warning: failed on backtest date {current_date}: {exc}")

        if periods_done % CHECKPOINT_EVERY == 0 and periods_done > 0:
            pd.DataFrame(all_rows).to_csv(checkpoint_csv, index=False)

    forecasts = pd.DataFrame(all_rows)
    forecasts.to_csv(forecast_csv, index=False)
    if checkpoint_csv.exists():
        checkpoint_csv.unlink()
    print(f"\nSaved {len(forecasts):,} rows -> {forecast_csv.name}")
    return forecasts


def main():
    args = parse_args()
    records = load_india_universe(limit=args.limit)
    analysis_end_ts = analysis_end_timestamp(args.analysis_end_date)
    download_end_ts = pd.Timestamp(args.end_date)
    if analysis_end_ts > download_end_ts:
        raise ValueError(
            f"analysis-end-date {args.analysis_end_date} exceeds download end-date {args.end_date}."
        )
    forecast_csv = forecast_csv_path(args.start_date, args.analysis_end_date, limit=args.limit)

    if args.download_only:
        download_bloomberg_history(records, args.start_date, args.end_date, force_refresh=args.force_refresh)
        return

    ensure_bloomberg_data(args)

    print("\nLoading Kronos-base model...")
    predictor = build_predictor()
    run_backtest(records, predictor, forecast_csv, analysis_end_ts)
    print("\nIndia extended run completed.")


if __name__ == "__main__":
    main()
