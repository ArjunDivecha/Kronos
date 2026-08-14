# AGENTS.md — shiyu-coder-Kronos (operator's manual for coding agents)

Global rules (light mode, doc-header template, `file://` links, FAIL-IS-FAIL) live in
`~/AGENTS.md` and `~/Dropbox/AAA Backup/AGENTS.md` — not repeated here.

## Purpose
This is Arjun's fork of the open-source **Kronos** foundation model for financial candlesticks
(`shiyu-coder/Kronos`, remote `github.com/ArjunDivecha/Kronos.git`). Upstream provides the model
(`model/`), finetuning, and a web UI. **Arjun's ~9 commits since 2026-04-11 (fork point `d5ffd46`)
added an entire quant research + live-trading layer on top**: cross-universe walk-forward backtests
(US/UK/Japan/India ETFs and single stocks), parameter sweeps, and a live IBKR rebalancer. When you
are asked to change strategy/backtest/trading behavior, it is almost always Arjun's layer you touch,
not the upstream model internals. Diff Arjun's work with `git diff d5ffd46 HEAD -- <file>`.

## Architecture map (load-bearing files, absolute paths under this repo root)
Root = `/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos/`
- `model/kronos.py` — upstream model + `KronosPredictor`. **Arjun's one functional change**: temporal-leakage
  guards in `predict`/`predict_batch` (`kronos.py:561-578`, `:654-676`) — raises if `x_timestamp` end is not
  strictly before `y_timestamp` start, plus length/monotonicity checks. Everything else he touched here is docstrings.
- `model/module.py` — upstream BSQ tokenizer / RoPE / embeddings. Arjun added docstrings only; no logic change.
- `industry28_trader.py` — **LIVE IBKR REBALANCER (places real orders).** 28-ETF, 3-long/3-short, MOO-when-closed.
  This is the safety-critical file. See "Live trading is real" below.
- `run_industry28.py` — walk-forward backtest for the 28-ETF universe (40-bar lookback → 5-day hold).
- `broad_oos_test.py` — cross-universe validation (Sector/Industry/Factor/FixedIncome/Commodity + permutation test).
- `etf_strategy.py` / `etf_sweep.py` — 34-country-ETF top-3 strategy and lookback/IC sweep. Note: `etf_sweep.py`
  and the tests use `Kronos-small`; everything else uses `Kronos-base` — results are not directly comparable.
- `forecast_latest.py` — one-shot "as of today" forecast for all ETFs → `latest_forecast_<date>.xlsx`.
- `convert_to_kronos.py` — reformats raw yfinance CSVs into Kronos schema (hardcoded absolute paths, see gotchas).
- `tests/test_kronos_regression.py` — the only test file; two allclose/MSE regressions vs pinned HF revisions.
- `pl_tracker.xlsx`, `trade_log.csv` — live P/L ledger and filled-trade log (real money, real fills).

## Commands that work
Run from the repo root with the venv active. Python **3.9.6** (`/.venv/bin/python`, verified).
```bash
source .venv/bin/activate          # or: .venv/bin/python <script>
python industry28_trader.py --dry-run   # SAFE: computes signals + trades, places nothing (verified: dry-run returns before order submit, industry28_trader.py:555-557)
python run_industry28.py                 # backtest (unverified — downloads yfinance data + loads Kronos from HF Hub)
python etf_strategy.py                    # 34-ETF strategy (unverified — network + model download)
pytest tests/test_kronos_regression.py -v # (unverified — REQUIRES live HuggingFace Hub access; no offline skip)
```
- Live trades run via the parent wrapper's `../Kronos Rebalance.command` launcher, not from here directly.
- Model weights are pulled from the **HuggingFace Hub** at runtime (`NeoQuasar/Kronos-base`, `-small`,
  `-Tokenizer-base`), not a local checkpoint — first run needs internet.

## Data locations (absolute)
- OHLCV caches: `.../shiyu-coder-Kronos/data/<Universe>/<TICKER>.csv` (yfinance, refreshed on each run).
- Backtest outputs: `.../shiyu-coder-Kronos/forecasts_<Universe>.csv` (**~153 MB of these are committed** — see flags),
  `strategy_performance.csv`, `monthly_returns_*.csv`, `*_metrics.txt`, `*.png` charts, `*.xlsx` reports — all at repo root.
- Live-trading state: `.../shiyu-coder-Kronos/trade_log.csv`, `.../shiyu-coder-Kronos/pl_tracker.xlsx`.
- Hardcoded absolute paths (break if repo moves): `convert_to_kronos.py:54-55`, `forecast_latest.py:97`.

## Conventions & gotchas (repo-specific, non-obvious)
- **Live trading is real.** `industry28_trader.py` connects to IBKR TWS on `localhost:7496` (live port; 7497 = paper),
  account `U14983106`, and submits `MarketOrder`s after a literal `YES` prompt (`:563-564`). `--dry-run` is safe
  (returns False before submit). Abort-safety was fixed in `fdd5bca` (nothing written unless you type `YES`). Treat any
  edit to this file as safety-critical; never automate past the confirm prompt.
- **Silent failure violates the repo's own FAIL-IS-FAIL rule.** The batch-inference loops swallow errors:
  `etf_strategy.py:165-166` (`except Exception: pass`), and print-and-continue at `run_industry28.py:208`,
  `broad_oos_test.py:166`, `etf_sweep.py:222`. A malformed batch silently shrinks the cross-section and corrupts
  reported IC/Sharpe. See FABLE.md P0.
- **Caches are loaded without invalidation.** Every walk-forward script does `if os.path.exists(FORECASTS_FILE): load`
  with no check that the cached file's tickers/lookback/pred_len/model match the current config
  (`run_industry28.py:406-409`, `etf_strategy.py:290-292`, `broad_oos_test.py:281-283`). Delete the stale
  `forecasts_*.csv` by hand after changing any parameter, or you will silently backtest old results. See FABLE.md P1.
- **Model inconsistency:** `etf_sweep.py` + tests use `Kronos-small`; all other scripts use `Kronos-base`. Do not
  compare Sharpe/IC numbers across the two groups.
- **OOS cutoff `2024-07-01` is hardcoded** in several scripts (`run_industry28.py:259`, `etf_sweep.py`). Confirm it
  post-dates Kronos's pretraining window before citing any "OOS" number as leakage-free.
- **Tests are not hermetic.** No `conftest.py` / pytest config exists; `test_kronos_regression.py` calls
  `from_pretrained` live and will error (not skip) without HF Hub access.
- **Live trader targets Python 3.14** (`industry28_trader.py:73-79` event-loop shim) but the committed venv is 3.9.6 —
  reconcile before assuming a modern-Python feature is available at runtime.

## Current state
Active and in live use (last live fill 2026-04-15; `pl_tracker.xlsx`/`trade_log.csv` touched 2026-06-13). The research
sweep phase (Apr 2026) is done; the live 40/5 Industry28 strategy is the surviving product. **Working tree is dirty:**
~79 uncommitted changes — modified scripts (`etf_strategy.py`, `etf_sweep.py`, `tech_stocks_test.py`, `uk_universe_test.py`,
`tests/test_kronos_regression.py`), regenerated `data/Industry28/*.csv`, and many untracked `data/<Universe>/` dirs. Commit
or stash before starting work so you can tell your changes from the pre-existing drift.


## Cross-session messaging

Codex sessions can message each other directly. `ListAgents` (or `/list-agents`, `/peers`)
lists reachable sessions; `SendMessage` delivers plain text to one by name. Same-machine delivery
uses a local socket; cross-machine is reply-only via Remote Control. Use it to hand off a finding
to a session working elsewhere instead of relaying it through the user. A message is text only —
never conversation history or files; to share full context, resume the session instead.
