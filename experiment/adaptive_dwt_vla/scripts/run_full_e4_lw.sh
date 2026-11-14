#!/usr/bin/env bash
# Epoch 4 of the full-LIBERO policy: starts from the epoch-3 checkpoint of runs/full_s0_dwt_fixed_slw (step 43,500,
# adapters + optimizer copied, the source run is left untouched) and trains one more epoch (to step 58,000) with the
# given loss variant: dwt_fixed_lw (soft labels removed, step weighting kept; the default) or dwt_fixed_slw (both kept,
# the matched control). The cosine schedule is stretched to 58,000 steps, so the lr steps up from the 2e-5 floor to
# ~4.6e-5 and decays back. Then 10 episodes on every task of the four suites, and the adapters are copied to weights/.
# Usage: scripts/run_full_e4_lw.sh [dwt_fixed_lw|dwt_fixed_slw]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
SRC=runs/full_s0_dwt_fixed_slw
VARIANT=${1:-dwt_fixed_lw}
name=full_e4_$VARIANT
STEPS=58000
SUITES=(libero_spatial libero_object libero_goal libero_10)
EVAL=(--task-ids 0 1 2 3 4 5 6 7 8 9 --episodes 10 --first-init-state 0 --videos 0 --label-prefix full40_)
mkdir -p results/logs weights

if [ ! -f "runs/$name/ckpt/latest/meta.json" ]; then
  mkdir -p "runs/$name/ckpt"
  cp -r "$SRC/ckpt/latest" "runs/$name/ckpt/latest"
  cp "$SRC/metrics.jsonl" "runs/$name/metrics.jsonl"
fi
echo "[$(date)] train $name (from slw step 43500) to step $STEPS" | tee -a "$LOG"
"$PY" scripts/train.py --variant "$VARIANT" --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 --steps $STEPS \
  --stop-at $STEPS --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 \
  --ckpt-every 500 --milestones 47125 50750 54375 58000 --num-workers 6 --resume >> "runs/$name.log" 2>&1 \
  || { echo "[$(date)] FAILED train $name" | tee -a "$LOG"; exit 1; }

ck=$(printf "%s/ckpt/step_%05d" "$name" "$STEPS")
for suite in "${SUITES[@]}"; do
  echo "[$(date)] eval $ck $suite" | tee -a "$LOG"
  "$PY" scripts/evaluate_policy.py --models "ckpt:$ck" --suite "$suite" "${EVAL[@]}" \
    >> results/logs/eval_full_libero.log 2>&1 || echo "[$(date)] FAILED eval $ck $suite" | tee -a "$LOG"
done
cp -rn "$SRC/ckpt/step_43500" weights/full_s0_dwt_fixed_slw_step_43500
cp -rn "runs/$ck" "weights/${name}_step_$STEPS"
"$PY" scripts/write_reports.py >> "$LOG" 2>&1
echo "[$(date)] finished $name" | tee -a "$LOG"
