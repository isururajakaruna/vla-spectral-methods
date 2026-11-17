#!/usr/bin/env bash
# Full-LIBERO loss control: Haar DWT fixed with plain cross-entropy (dwt_fixed) trained from scratch for 2 epochs with
# the same learning-rate history as the other full runs (epoch 1 under a 29,000-step cosine, epoch 2 resumed under
# 43,500 steps), the 400-episode evaluation after each epoch. Slots in between epochs 2 and 3 of the DCT control: waits
# for the running DCT training to reach step 29,000, evaluates it, runs the plain Haar epochs, then hands over to
# scripts/run_full_dct_control.sh, which skips everything already done and continues with DCT epochs 3-4.
# Usage: scripts/run_full_plain_haar.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
BACKUP=/media/isuru/BC5EC6405EC5F2E6/smooth-operator-backup
name=full_s0_dwt_fixed
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

while pgrep -f "train.py --variant dct_fixed" > /dev/null; do sleep 60; done
dct=full_s0_dct_fixed/ckpt/step_29000
[ -d "runs/$dct" ] || { echo "[$(date)] DCT step 29000 checkpoint missing; stopping" | tee -a "$LOG"; exit 1; }
evaluate "$dct" dct_fixed@full_s0_dct_fixed_step_29000
cp -r --update=none "runs/$dct" weights/full_s0_dct_fixed_step_29000
reports

for plan in "29000 14500" "43500 29000"; do
  read -r STEPS STOP <<< "$plan"
  ck=$(printf "%s/ckpt/step_%05d" "$name" "$STOP")
  if [ ! -d "runs/$ck" ]; then
    resume=(); [ -f "runs/$name/ckpt/latest/meta.json" ] && resume=(--resume)
    echo "[$(date)] train $name (plain cross-entropy, from scratch) to step $STOP of $STEPS" | tee -a "$LOG"
    "$PY" scripts/train.py --variant dwt_fixed --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 --steps "$STEPS" \
      --stop-at "$STOP" --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 \
      --ckpt-every 500 --milestones 3625 7250 10875 14500 21750 29000 \
      --num-workers 6 "${resume[@]}" >> "runs/$name.log" 2>&1 \
      || { echo "[$(date)] FAILED train $name" | tee -a "$LOG"; exit 1; }
  fi
  evaluate "$ck" "$(printf "dwt_fixed@%s_step_%05d" "$name" "$STOP")"
  cp -r --update=none "runs/$ck" "weights/$(printf "%s_step_%05d" "$name" "$STOP")"
  reports
done
echo "[$(date)] finished plain Haar; continuing the DCT control" | tee -a "$LOG"
exec bash scripts/run_full_dct_control.sh
