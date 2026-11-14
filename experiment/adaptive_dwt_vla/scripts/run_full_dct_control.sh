#!/usr/bin/env bash
# Full-LIBERO control: DCT tokens without BPE (dct_fixed, plain cross-entropy) trained for 4 epochs on all 40 tasks with
# exactly the learning-rate history of the Haar run (epoch 1 under a 29,000-step cosine, epochs 2-3 under 43,500 steps,
# epoch 4 under 58,000 steps; each change resumes adapters + optimizer), and the 400-episode evaluation after every
# epoch. Starts with the missing epoch-2 evaluation of the Haar run so both have all four epochs. Every finished epoch
# is copied to weights/, runs/ is mirrored to the data partition, and the reports are regenerated.
# Usage: scripts/run_full_dct_control.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
BACKUP=/media/isuru/BC5EC6405EC5F2E6/smooth-operator-backup
v=dct_fixed
name=full_s0_$v
SUITES=(libero_spatial libero_object libero_goal libero_10)
EVAL=(--task-ids 0 1 2 3 4 5 6 7 8 9 --episodes 10 --first-init-state 0 --videos 0 --label-prefix full40_)
mkdir -p results/logs weights

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

evaluate full_s0_dwt_fixed_slw/ckpt/step_29000 dwt_fixed_slw@full_s0_dwt_fixed_slw_step_29000
reports

for plan in "29000 14500" "43500 29000" "43500 43500" "58000 58000"; do
  read -r STEPS STOP <<< "$plan"
  ck=$(printf "%s/ckpt/step_%05d" "$name" "$STOP")
  if [ ! -d "runs/$ck" ]; then
    resume=(); [ -f "runs/$name/ckpt/latest/meta.json" ] && resume=(--resume)
    echo "[$(date)] train $name to step $STOP of $STEPS" | tee -a "$LOG"
    "$PY" scripts/train.py --variant $v --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 --steps "$STEPS" \
      --stop-at "$STOP" --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 \
      --ckpt-every 500 --milestones 3625 7250 10875 14500 21750 29000 36250 43500 47125 50750 54375 58000 \
      --num-workers 6 "${resume[@]}" >> "runs/$name.log" 2>&1 \
      || { echo "[$(date)] FAILED train $name" | tee -a "$LOG"; exit 1; }
  fi
  evaluate "$ck" "$(printf "%s@%s_step_%05d" $v "$name" "$STOP")"
  cp -r --update=none "runs/$ck" "weights/$(printf "%s_step_%05d" "$name" "$STOP")"
  reports
done
echo "[$(date)] finished DCT control" | tee -a "$LOG"
