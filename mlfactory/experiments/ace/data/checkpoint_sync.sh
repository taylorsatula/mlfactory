#!/bin/bash
# Automated checkpoint sync from Vast boxes to local
# Runs via cron every 2 hours

set -euo pipefail

LOCAL_DIR="/home/admin/mlfactory/mlfactory/experiments/ace/data"
LOG_FILE="/home/admin/mlfactory/mlfactory/experiments/ace/data/checkpoint_sync.log"

echo "=== Checkpoint sync: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ===" >> "$LOG_FILE"

# Sync from box A
echo "Syncing r4fork-a..." >> "$LOG_FILE"
mkdir -p "$LOCAL_DIR/r4fork-a"
if rsync -az --timeout=30 r4fork-a:/root/mlfactory/mlfactory/experiments/ace/data/fork_r4_results_*.jsonl "$LOCAL_DIR/r4fork-a/" 2>>"$LOG_FILE"; then
    echo "  r4fork-a: OK ($(wc -l < "$LOCAL_DIR/r4fork-a/fork_r4_results_1.jsonl" 2>/dev/null || echo 0) + $(wc -l < "$LOCAL_DIR/r4fork-a/fork_r4_results_2.jsonl" 2>/dev/null || echo 0) rows)" >> "$LOG_FILE"
else
    echo "  r4fork-a: FAILED" >> "$LOG_FILE"
fi

# Sync from box B
echo "Syncing r4fork-b..." >> "$LOG_FILE"
mkdir -p "$LOCAL_DIR/r4fork-b"
if rsync -az --timeout=30 r4fork-b:/root/mlfactory/mlfactory/experiments/ace/data/fork_r4_results_*.jsonl "$LOCAL_DIR/r4fork-b/" 2>>"$LOG_FILE"; then
    echo "  r4fork-b: OK ($(wc -l < "$LOCAL_DIR/r4fork-b/fork_r4_results_1.jsonl" 2>/dev/null || echo 0) + $(wc -l < "$LOCAL_DIR/r4fork-b/fork_r4_results_2.jsonl" 2>/dev/null || echo 0) rows)" >> "$LOG_FILE"
else
    echo "  r4fork-b: FAILED" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"
