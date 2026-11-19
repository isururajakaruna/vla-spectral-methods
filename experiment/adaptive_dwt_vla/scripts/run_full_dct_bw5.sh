#!/usr/bin/env bash
# DCT loss branch: one epoch of DCT without BPE with 5-level stepped coefficient-token weights from the plain DCT
# epoch-3 checkpoint (step 43,500, adapters + optimizer copied into runs/full_e4_dct_fixed_bw5; the plain run is
# untouched). The codec's step range is cut into 5 equal log-scale segments; tokens in them weigh 0.3, 0.6, 0.9, 1.2, 1.5
# (smallest step first), not normalized, no soft labels. Trained to step 58,000 under the 58,000-step cosine, exactly
# like plain DCT epoch 4, then the 400-episode evaluation.
# Takes over the queue from scripts/run_full_dct_sqrtw_scratch.sh: waits for the running √step-weighted DCT epoch
# from scratch, evaluates it, runs this branch, then hands over to scripts/run_full_poly_sqrtw.sh.
# Usage: scripts/run_full_dct_bw5.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
BACKUP=/media/isuru/BC5EC6405EC5F2E6/smooth-operator-backup
SRC=runs/full_s0_dct_fixed
name=full_e4_dct_fixed_bw5
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
  [ "$step" = 43500 ] || { echo "[$(date)] $SRC latest is at step $step, not 43500; stopping" | tee -a "$LOG"; exit 1; }
  mkdir -p "runs/$name/ckpt"
  cp -r "$SRC/ckpt/latest" "runs/$name/ckpt/latest"
  cp "$SRC/metrics.jsonl" "runs/$name/metrics.jsonl"
fi

while pgrep -f "train.py --variant dct_fixed --weight-power 0.5 --name full_s0_dct_fixed_sqrtw " > /dev/null; do sleep 60; done
prev=full_s0_dct_fixed_sqrtw/ckpt/step_14500
if [ -d "runs/$prev" ]; then
  evaluate "$prev" dct_fixed@full_s0_dct_fixed_sqrtw_step_14500
  cp -r --update=none "runs/$prev" weights/full_s0_dct_fixed_sqrtw_step_14500
  reports
else
  echo "[$(date)] sqrt-weighted DCT step 14500 checkpoint missing; skipping its evaluation" | tee -a "$LOG"
fi

ck=$name/ckpt/step_58000
if [ ! -d "runs/$ck" ]; then
  echo "[$(date)] train $name (DCT, 5-level stepped weights, from DCT step 43500) to step 58000" | tee -a "$LOG"
  "$PY" scripts/train.py --variant dct_fixed --weight-levels 0.3 0.6 0.9 1.2 1.5 --weight-bins log --name "$name" --seed 0 \
    --tasks $(seq 0 39) --horizon 10 --steps 58000 --stop-at 58000 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 \
    --val-every 1000 --val-windows 256 --ckpt-every 500 --milestones 47125 50750 54375 58000 --num-workers 6 --resume \
    >> "runs/$name.log" 2>&1 || echo "[$(date)] FAILED train $name; skipping to the polynomial run" | tee -a "$LOG"
fi
if [ -d "runs/$ck" ]; then
  evaluate "$ck" "dct_fixed@${name}_step_58000"
  cp -r --update=none "runs/$ck" "weights/${name}_step_58000"
  reports
fi
echo "[$(date)] finished $name; next the polynomial tokenizer" | tee -a "$LOG"
exec bash scripts/run_full_poly_sqrtw.sh
