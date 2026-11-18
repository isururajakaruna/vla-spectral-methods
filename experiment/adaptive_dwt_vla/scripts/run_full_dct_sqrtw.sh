#!/usr/bin/env bash
# DCT loss branch: one epoch of DCT without BPE with √step weighting (coefficient-token loss × quantizer step ** 0.5,
# mean 1) from the DCT epoch-2 checkpoint (step 29,000, adapters + optimizer, copied; the plain DCT run is untouched),
# under the same 43,500-step cosine as plain DCT epoch 3, then the 400-episode evaluation.
# Takes over the queue from scripts/run_full_plain_haar.sh: waits for the running plain-Haar epoch 2, evaluates it,
# runs this branch, then hands over to scripts/run_full_dct_control.sh (plain DCT epochs 3-4).
# Usage: scripts/run_full_dct_sqrtw.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
BACKUP=/media/isuru/BC5EC6405EC5F2E6/smooth-operator-backup
SRC=runs/full_s0_dct_fixed
name=full_e3_dct_fixed_sqrtw
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

if [ ! -f "runs/$name/ckpt/latest/meta.json" ]; then
  step=$("$PY" -c "import json; print(json.load(open('$SRC/ckpt/latest/meta.json'))['step'])")
  [ "$step" = 29000 ] || { echo "[$(date)] $SRC latest is at step $step, not 29000; stopping" | tee -a "$LOG"; exit 1; }
  mkdir -p "runs/$name/ckpt"
  cp -r "$SRC/ckpt/latest" "runs/$name/ckpt/latest"
  cp "$SRC/metrics.jsonl" "runs/$name/metrics.jsonl"
fi

while pgrep -f "train.py --variant dwt_fixed --name full_s0_dwt_fixed " > /dev/null; do sleep 60; done
haar=full_s0_dwt_fixed/ckpt/step_29000
if [ -d "runs/$haar" ]; then
  evaluate "$haar" dwt_fixed@full_s0_dwt_fixed_step_29000
  cp -r --update=none "runs/$haar" weights/full_s0_dwt_fixed_step_29000
  reports
else
  echo "[$(date)] plain Haar step 29000 checkpoint missing; skipping its evaluation" | tee -a "$LOG"
fi

ck=$name/ckpt/step_43500
if [ ! -d "runs/$ck" ]; then
  echo "[$(date)] train $name (DCT, sqrt-step weighting, from DCT step 29000) to step 43500" | tee -a "$LOG"
  "$PY" scripts/train.py --variant dct_fixed --weight-power 0.5 --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 \
    --steps 43500 --stop-at 43500 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 \
    --ckpt-every 500 --milestones 36250 43500 --num-workers 6 --resume >> "runs/$name.log" 2>&1 \
    || { echo "[$(date)] FAILED train $name" | tee -a "$LOG"; exit 1; }
fi
evaluate "$ck" "dct_fixed@${name}_step_43500"
cp -r --update=none "runs/$ck" "weights/${name}_step_43500"
reports
echo "[$(date)] finished $name; continuing the DCT control" | tee -a "$LOG"
exec bash scripts/run_full_dct_control.sh
