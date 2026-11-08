#!/usr/bin/env bash
# sym4→haar fixed plan, seed 0: learning-curve recipe + task-0 evals, then the zero-shot generalization eval.
set -u
cd "$(dirname "$0")/.."
bash scripts/run_variant.sh dwt_fixed_s4h 0 "$@"
bash scripts/run_generalization.sh 0 dwt_fixed_s4h
