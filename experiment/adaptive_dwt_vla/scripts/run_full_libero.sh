#!/usr/bin/env bash
# Full LIBERO: one Haar fixed policy trained on all 40 tasks with a 2-epoch cosine schedule (29,000 steps at effective
# batch 16), paused after epoch 1 (STOP=14500); then 10 episodes (initial states 0-9) on every task of the four suites
# for it and for lerobot/pi0fast-libero (the reference only once). Re-run with STOP=29000 to train epoch 2 and evaluate it.
# Waits for scripts/run_slw_seeds.sh first. The variant comes from the argument, else from the only existing
# runs/full_s0_* run, else from scripts/pick_full_variant.py.
# Usage: [STOP=14500|29000] scripts/run_full_libero.sh [variant]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
LOG=results/logs/full_libero.log
mkdir -p runs results/logs
while pgrep -f run_slw_seeds.sh > /dev/null; do sleep 300; done
n_runs=$(ls -d runs/full_s0_* 2>/dev/null | wc -l)
if [ $# -ge 1 ]; then v=$1
elif [ "$n_runs" -eq 1 ]; then v=$(basename "$(ls -d runs/full_s0_*)"); v=${v#full_s0_}
elif [ "$n_runs" -gt 1 ]; then echo "several full_s0_* runs exist; give the variant" | tee -a "$LOG"; exit 1
else v=$("$PY" scripts/pick_full_variant.py 2>> "$LOG"); fi
name="full_s0_$v"
STEPS=29000
STOP=${STOP:-14500}
SUITES=(libero_spatial libero_object libero_goal libero_10)
EVAL=(--task-ids 0 1 2 3 4 5 6 7 8 9 --episodes 10 --first-init-state 0 --videos 0 --label-prefix full40_)

resume=(); [ -f "runs/$name/ckpt/latest/meta.json" ] && resume=(--resume)
echo "[$(date)] train $name to step $STOP of $STEPS" | tee -a "$LOG"
"$PY" scripts/train.py --variant "$v" --name "$name" --seed 0 --tasks $(seq 0 39) --horizon 10 --steps $STEPS --stop-at "$STOP" \
  --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 300 --val-every 1000 --val-windows 256 --ckpt-every 500 \
  --milestones 3625 7250 10875 14500 21750 29000 --num-workers 6 "${resume[@]}" >> "runs/$name.log" 2>&1 \
  || echo "[$(date)] FAILED train $name" | tee -a "$LOG"

ck=$(printf "%s/ckpt/step_%05d" "$name" "$STOP")
models=()
[ -d "runs/$ck" ] && models+=("ckpt:$ck")
[ -f results/eval/full40_ref_pi0fast_libero__libero_10_t9.json ] || models+=("ref:pi0fast_libero")
for model in "${models[@]}"; do
  for suite in "${SUITES[@]}"; do
    echo "[$(date)] eval $model $suite" | tee -a "$LOG"
    "$PY" scripts/evaluate_policy.py --models "$model" --suite "$suite" "${EVAL[@]}" \
      >> results/logs/eval_full_libero.log 2>&1 || echo "[$(date)] FAILED eval $model $suite" | tee -a "$LOG"
  done
done
"$PY" scripts/write_reports.py >> "$LOG" 2>&1
"$PY" scripts/write_training_report.py >> "$LOG" 2>&1
"$PY" scripts/write_experiment_trail.py >> "$LOG" 2>&1
echo "[$(date)] finished full LIBERO ($name, step $STOP)" | tee -a "$LOG"
