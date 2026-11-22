#!/usr/bin/env bash
# Third epoch of √step-weighted orthogonal-polynomial tokens from scratch: resumes runs/full_s0_poly_fixed_sqrtw
# (adapters + optimizer at step 29,000) under the same 43,500-step cosine to step 43,500, matching plain DCT epoch 3,
# then runs the 400-episode evaluation, copies adapter weights, and regenerates the reports.
# Usage: scripts/run_full_poly_sqrtw_e3.sh
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

ck=$name/ckpt/step_43500
if [ ! -d "runs/$ck" ] && [ -d "runs/$name/ckpt/step_29000" ]; then
  echo "[$(date)] train $name (polynomial tokens, sqrt-step weighting, epoch 3) to step 43500 of 43500" | tee -a "$LOG"
  "$PY" scripts/train.py --variant $v --weight-power 0.5 --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 \
    --steps 43500 --stop-at 43500 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 \
    --ckpt-every 500 --milestones 36250 43500 --num-workers 6 --resume >> "runs/$name.log" 2>&1 \
    || echo "[$(date)] FAILED train $name epoch 3" | tee -a "$LOG"
fi
if [ -d "runs/$ck" ]; then
  evaluate "$ck" "${v}@${name}_step_43500"
  [ -d "weights/${name}_step_43500" ] || cp -r "runs/$ck" "weights/${name}_step_43500"
  reports
fi
echo "[$(date)] finished $name epoch 3" | tee -a "$LOG"
