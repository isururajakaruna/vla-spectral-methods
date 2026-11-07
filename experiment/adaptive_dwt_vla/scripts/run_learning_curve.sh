#!/usr/bin/env bash
# Learning-curve experiment: closed-loop success vs optimizer step, 3 seeds x 3 variants.
# Same recipe as run_pilot.sh (5k-step cosine schedule, effective batch 16), stopped at 2k with
# adapter checkpoints at 250/500/1000/1500/2000; every checkpoint is evaluated on init states 0-19.
# Usage: scripts/run_learning_curve.sh [seed...]   (default: 1 2 0)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
VARIANTS=(fast_matched dwt_fixed dwt_adaptive)
STEPS=(250 500 1000 1500 2000)
COMMON=(--tasks 24 --horizon 10 --steps 5000 --stop-at 2000 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 100
        --val-every 250 --val-windows 64 --ckpt-every 500 --milestones "${STEPS[@]}" --num-workers 6)
EVAL=(--suite libero_object --task-ids 0 --episodes 20 --first-init-state 0 --videos 0)
LOG=results/logs/learning_curve.log
mkdir -p runs results/logs

seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(1 2 0)
for s in "${seeds[@]}"; do
  for v in "${VARIANTS[@]}"; do
    name="lc_s${s}_${v}"
    if grep -q '"state": "paused"' "runs/$name/status.json" 2>/dev/null; then continue; fi
    resume=(); [ -f "runs/$name/ckpt/latest/meta.json" ] && resume=(--resume)
    echo "[$(date)] train $name" | tee -a "$LOG"
    "$PY" scripts/train.py --variant "$v" --name "$name" --seed "$s" "${COMMON[@]}" "${resume[@]}" \
      >> "runs/$name.log" 2>&1 || echo "[$(date)] FAILED train $name" | tee -a "$LOG"
  done
  for v in "${VARIANTS[@]}"; do
    for st in "${STEPS[@]}"; do
      ck=$(printf "lc_s%s_%s/ckpt/step_%05d" "$s" "$v" "$st")
      out=$(printf "results/eval/%s@lc_s%s_%s_step_%05d__libero_object_t0.json" "$v" "$s" "$v" "$st")
      [ -d "runs/$ck" ] || continue
      if [ -f "$out" ] && python3 -c "import json,sys; sys.exit(len(json.load(open('$out'))['episodes']) < 20)"; then continue; fi
      echo "[$(date)] eval $ck" | tee -a "$LOG"
      "$PY" scripts/evaluate_policy.py --models "ckpt:$ck" "${EVAL[@]}" >> results/logs/eval_lc.log 2>&1 \
        || echo "[$(date)] FAILED eval $ck" | tee -a "$LOG"
    done
  done
  "$PY" scripts/write_reports.py >> "$LOG" 2>&1
done
echo "[$(date)] learning curve finished: seeds ${seeds[*]}" | tee -a "$LOG"
