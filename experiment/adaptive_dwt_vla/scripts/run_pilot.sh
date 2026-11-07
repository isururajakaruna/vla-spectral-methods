#!/usr/bin/env bash
# Stage C + D pilot on LIBERO task 24 (= libero_object #0), H=10, effective batch 16, seed 0.
# One 5k-step cosine schedule per variant, paused at 2k for evaluation, then resumed to 5k.
# Usage: scripts/run_pilot.sh [phase...]   phases: train2k eval2k train5k eval5k (default: all)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/isuru/miniconda3/envs/smooth-operator-vla/bin/python}
VARIANTS=(fast_matched dwt_fixed dwt_adaptive)
COMMON=(--tasks 24 --horizon 10 --steps 5000 --micro-batch 8 --grad-accum 2 --lr 2e-4 --warmup 100
        --val-every 200 --val-windows 64 --ckpt-every 500 --milestones 2000 5000 --seed 0 --num-workers 6)
EVAL=(--suite libero_object --task-ids 0 --episodes 20 --first-init-state 0 --videos 2)
mkdir -p runs results/logs

train() {  # $1 = stop step
  for v in "${VARIANTS[@]}"; do
    name="pilot_${v}"
    resume=(); [ -f "runs/$name/ckpt/latest/meta.json" ] && resume=(--resume)
    echo "[$(date)] train $name to $1" | tee -a results/logs/pilot.log
    "$PY" scripts/train.py --variant "$v" --name "$name" "${COMMON[@]}" --stop-at "$1" "${resume[@]}" \
      >> "runs/$name.log" 2>&1 || echo "[$(date)] FAILED train $name" | tee -a results/logs/pilot.log
  done
}

evaluate() {  # $1 = milestone dir name, $2.. extra model specs
  local ck=$1; shift
  for spec in "${VARIANTS[@]/#/ckpt:pilot_}" "$@"; do
    [[ $spec == ckpt:* ]] && spec="$spec/ckpt/$ck"
    echo "[$(date)] eval $spec" | tee -a results/logs/pilot.log
    "$PY" scripts/evaluate_policy.py --models "$spec" "${EVAL[@]}" >> results/logs/eval.log 2>&1 \
      || echo "[$(date)] FAILED eval $spec" | tee -a results/logs/pilot.log
  done
}

phases=("$@"); [ ${#phases[@]} -eq 0 ] && phases=(train2k eval2k train5k eval5k)
for p in "${phases[@]}"; do
  case $p in
    train2k) train 2000 ;;
    eval2k) evaluate step_02000 ref:pi0fast_libero ref:pi0_libero ;;
    train5k) train 5000 ;;
    eval5k) evaluate step_05000 ;;
    *) echo "unknown phase $p"; exit 1 ;;
  esac
done
echo "[$(date)] pilot finished: ${phases[*]}" | tee -a results/logs/pilot.log
