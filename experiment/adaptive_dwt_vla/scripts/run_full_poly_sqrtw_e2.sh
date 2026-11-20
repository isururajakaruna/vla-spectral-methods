#!/usr/bin/env bash
# Second epoch of √step-weighted orthogonal-polynomial tokens from scratch: resumes runs/full_s0_poly_fixed_sqrtw
# (adapters + optimizer at step 14,500) under the 43,500-step cosine to step 29,000, exactly like DCT epoch 2 (plain and
# √step-weighted), then the 400-episode evaluation. Started by scripts/run_full_poly_sqrtw.sh; hands over to
# scripts/run_full_dct_control.sh (skips finished work, trains plain DCT epoch 4).
# Usage: scripts/run_full_poly_sqrtw_e2.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
BACKUP=/media/isuru/BC5EC6405EC5F2E6/smooth-operator-backup
v=poly_fixed
name=full_s0_poly_fixed_sqrtw
SUITES=(libero_spatial libero_object libero_goal libero_10)
EVAL=(--task-ids 0 1 2 3 4 5 6 7 8 9 --episodes 10 --first-init-state 0 --videos 0 --label-prefix full40_)

evaluate() {  # $1 = run/ckpt/step_XXXXX, $2 = model label as written into results/eval
  for suite in "${SUITES[@]}"; do
    [ -f "results/eval/full40_${2}__${suite}_t9.json" ] && continue
    echo "[$(date)] eval $1 $suite" | tee -a "$LOG"
    "$PY" scripts/evaluate_policy.py --models "ckpt:$1" --suite "$suite" "${EVAL[@]}" \
      >> results/logs/eval_full_libero.log 2>&1 || echo "[$(date)] FAILED eval $1 $suite" | tee -a "$LOG"
  done
}

reports() {
  "$PY" scripts/write_reports.py >> "$LOG" 2>&1
  "$PY" scripts/write_training_report.py >> "$LOG" 2>&1
  "$PY" scripts/write_experiment_trail.py >> "$LOG" 2>&1
  [ -d "$(dirname "$BACKUP")" ] && mkdir -p "$BACKUP" && rsync -rt runs/ "$BACKUP/runs/"
}

ck=$name/ckpt/step_29000
if [ ! -d "runs/$ck" ] && [ -d "runs/$name/ckpt/step_14500" ]; then
  echo "[$(date)] train $name (polynomial tokens, sqrt-step weighting, epoch 2) to step 29000 of 43500" | tee -a "$LOG"
  "$PY" scripts/train.py --variant $v --weight-power 0.5 --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 \
    --steps 43500 --stop-at 29000 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 \
    --ckpt-every 500 --milestones 21750 29000 --num-workers 6 --resume >> "runs/$name.log" 2>&1 \
    || echo "[$(date)] FAILED train $name epoch 2; skipping to the DCT control" | tee -a "$LOG"
fi
if [ -d "runs/$ck" ]; then
  evaluate "$ck" "${v}@${name}_step_29000"
  [ -d "weights/${name}_step_29000" ] || cp -r "runs/$ck" "weights/${name}_step_29000"
  reports
fi
echo "[$(date)] finished $name epoch 2; continuing the DCT control" | tee -a "$LOG"
exec bash scripts/run_full_dct_control.sh
