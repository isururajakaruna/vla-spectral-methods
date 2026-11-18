#!/usr/bin/env bash
# √step-weighted DCT from scratch: one epoch of DCT without BPE with coefficient-token loss × quantizer step ** 0.5
# (mean 1), under the same 29,000-step cosine as plain DCT epoch 1, then the 400-episode evaluation.
# Takes over the queue from scripts/run_full_dct_control.sh: waits for the running plain DCT epoch 3, evaluates it,
# runs this epoch, then hands over to scripts/run_full_poly_sqrtw.sh (which ends with plain DCT epoch 4).
# Usage: scripts/run_full_dct_sqrtw_scratch.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
BACKUP=/media/isuru/BC5EC6405EC5F2E6/smooth-operator-backup
name=full_s0_dct_fixed_sqrtw
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

while pgrep -f "train.py --variant dct_fixed --name full_s0_dct_fixed " > /dev/null; do sleep 60; done
dct=full_s0_dct_fixed/ckpt/step_43500
if [ -d "runs/$dct" ]; then
  evaluate "$dct" dct_fixed@full_s0_dct_fixed_step_43500
  cp -r --update=none "runs/$dct" weights/full_s0_dct_fixed_step_43500
  reports
else
  echo "[$(date)] plain DCT step 43500 checkpoint missing; stopping" | tee -a "$LOG"
  exit 1
fi

ck=$name/ckpt/step_14500
if [ ! -d "runs/$ck" ]; then
  resume=(); [ -f "runs/$name/ckpt/latest/meta.json" ] && resume=(--resume)
  echo "[$(date)] train $name (DCT, sqrt-step weighting, from scratch) to step 14500 of 29000" | tee -a "$LOG"
  "$PY" scripts/train.py --variant dct_fixed --weight-power 0.5 --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 \
    --steps 29000 --stop-at 14500 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 \
    --ckpt-every 500 --milestones 3625 7250 10875 14500 --num-workers 6 "${resume[@]}" >> "runs/$name.log" 2>&1 \
    || { echo "[$(date)] FAILED train $name" | tee -a "$LOG"; exit 1; }
fi
evaluate "$ck" "dct_fixed@${name}_step_14500"
cp -r --update=none "runs/$ck" "weights/${name}_step_14500"
reports
echo "[$(date)] finished $name; next the polynomial tokenizer" | tee -a "$LOG"
exec bash scripts/run_full_poly_sqrtw.sh
