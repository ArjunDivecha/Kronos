#!/bin/zsh
#
# run_tracker_daily.sh — launchd wrapper for the Kronos Tracker publisher
# =======================================================================
#
# Activates the repo venv and runs tracker_daily.py, which publishes the 22
# Kronos paper books that Tracker's kronos-* adapters ingest.
#
# Called by ~/Library/LaunchAgents/com.arjun.kronos-tracker.plist on weekdays
# at 13:05 PT, 40 minutes before Tracker consolidates at 13:45 PT.
#
# INPUT  : data/<Universe>/*.csv (ticker lists), Yahoo Finance, HuggingFace Hub
# OUTPUT : tracker_book/kronos-*/{nav_daily.csv,holdings_current.csv,
#          signals_latest.csv,meta.json,state.json}
#          tracker_book/run_log/YYYY-MM-DD.log, tracker_book/heartbeat.txt
#
# Places NO orders. Paper books only.
#
set -uo pipefail

REPO="/Users/arjundivecha/Dropbox/AAA Backup/A Working/Kronos/shiyu-coder-Kronos"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

mkdir -p tracker_book/run_log
LOG="tracker_book/run_log/$(date +%Y-%m-%d).log"

echo "=== run_tracker_daily.sh start $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"

if [[ ! -x .venv/bin/python ]]; then
  echo "FATAL: .venv/bin/python missing or not executable" | tee -a "$LOG"
  exit 1
fi

.venv/bin/python tracker_daily.py 2>&1 | tee -a "$LOG"
rc=${pipestatus[1]}

echo "=== exit rc=$rc $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"

# Surface a hard failure in the heartbeat so Tracker/NightWatch can see it
if [[ $rc -ne 0 ]]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S')  FAILED rc=$rc — see $LOG" > tracker_book/heartbeat.txt
fi

exit $rc
