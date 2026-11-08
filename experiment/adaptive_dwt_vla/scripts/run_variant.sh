#!/usr/bin/env bash
# One seed of one variant with the learning-curve recipe (5k cosine, stop at 2k), every milestone evaluated on
# libero_object #0 init states 0-19, then the reports are regenerated. Extra arguments go to train.py.
# Usage: scripts/run_variant.sh [variant] [seed] [extra train.py args...]   (default: dwt_fixed_bpe 0)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
v=${1:-dwt_fixed_bpe}; s=${2:-0}; shift $(( $# < 2 ? $# : 2 )); EXTRA=("$@")
STEPS=(250 500 1000 1500 2000)
COMMON=(--tasks 24 --horizon 10 --steps 5000 --stop-at 2000 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 100
        --val-every 250 --val-windows 64 --ckpt-every 500 --milestones "${STEPS[@]}" --num-workers 6)
EVAL=(--suite libero_object --task-ids 0 --episodes 20 --first-init-state 0 --videos 0)
LOG=results/logs/run_variant.log
name="lc_s${s}_${v}"
mkdir -p runs results/logs

resume=(); [ -f "runs/$name/ckpt/latest/meta.json" ] && resume=(--resume)
echo "[$(date)] train $name" | tee -a "$LOG"
"$PY" scripts/train.py --variant "$v" --name "$name" --seed "$s" "${COMMON[@]}" "${EXTRA[@]}" "${resume[@]}" \
  >> "runs/$name.log" 2>&1 || echo "[$(date)] FAILED train $name" | tee -a "$LOG"
for st in "${STEPS[@]}"; do
  ck=$(printf "%s/ckpt/step_%05d" "$name" "$st")
  [ -d "runs/$ck" ] || continue
  echo "[$(date)] eval $ck" | tee -a "$LOG"
  "$PY" scripts/evaluate_policy.py --models "ckpt:$ck" "${EVAL[@]}" >> results/logs/eval_bpe.log 2>&1 \
    || echo "[$(date)] FAILED eval $ck" | tee -a "$LOG"
done
"$PY" scripts/write_reports.py >> "$LOG" 2>&1
"$PY" scripts/write_experiment_trail.py >> "$LOG" 2>&1
echo "[$(date)] finished: $name" | tee -a "$LOG"
