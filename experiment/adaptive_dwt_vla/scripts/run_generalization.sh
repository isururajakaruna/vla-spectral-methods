#!/usr/bin/env bash
# Zero-shot task generalization: step-2000 checkpoints trained only on libero_object #0, evaluated on the other
# nine libero_object tasks (#1-#9), 5 initial states per task sampled from 0-19 (same states for every model).
# Usage: scripts/run_generalization.sh [seed] [variants...]   (default: 0 dwt_fixed dwt_adaptive dct_fixed)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
s=${1:-0}; shift $(( $# < 1 ? $# : 1 ))
VARIANTS=("$@"); [ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(dwt_fixed dwt_adaptive dct_fixed)
LOG=results/logs/generalization.log
mkdir -p results/logs
for v in "${VARIANTS[@]}"; do
  ck="lc_s${s}_${v}/ckpt/step_02000"
  echo "[$(date)] generalization eval $ck" | tee -a "$LOG"
  "$PY" scripts/evaluate_policy.py --models "ckpt:$ck" --suite libero_object --task-ids 1 2 3 4 5 6 7 8 9 \
    --episodes 20 --first-init-state 0 --sample 5 --sample-seed 0 --videos 0 >> results/logs/eval_generalization.log 2>&1 \
    || echo "[$(date)] FAILED $ck" | tee -a "$LOG"
done
echo "[$(date)] finished generalization seed $s" | tee -a "$LOG"
