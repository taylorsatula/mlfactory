#!/bin/bash
# Pilot generation for the r4v2 judge hillclimb (2026-08-28).
# Fills the arms/seeds missing from the v1-derived pilot file:
#   toward_diverge  seeds 0-7 x 3 states  (24 rows)
#   toward_healthy  cycle_02 seeds 2-7, cycle_03 seeds 5-7  (9 rows)
# Resume-safe: done rows in fork_r4v2_pilot.jsonl are skipped.
cd /home/admin/mlfactory
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/admin/mlfactory/mlfactory/experiments/ace/.venv/bin/python
OUT=/home/admin/mlfactory/mlfactory/experiments/ace/data/fork_r4v2_pilot.jsonl

$PY -m mlfactory.experiments.ace.annotate.fork_r4v2 --run \
  --out "$OUT" --only r4_cycle_00,r4_cycle_02,r4_cycle_03 \
  --arms toward_diverge --seeds 0,1,2,3,4,5,6,7

$PY -m mlfactory.experiments.ace.annotate.fork_r4v2 --run \
  --out "$OUT" --only r4_cycle_02 --arms toward_healthy \
  --seeds 2,3,4,5,6,7

$PY -m mlfactory.experiments.ace.annotate.fork_r4v2 --run \
  --out "$OUT" --only r4_cycle_03 --arms toward_healthy \
  --seeds 5,6,7

echo "PILOT GEN DONE"
