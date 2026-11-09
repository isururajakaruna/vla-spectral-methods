#!/usr/bin/env bash
# Soft labels + step weighting on Haar fixed, seeds 1 and 2 (seed 0 already done): learning-curve recipe + task-0 evals.
set -u
cd "$(dirname "$0")/.."
bash scripts/run_variant.sh dwt_fixed_slw 1 "$@"
bash scripts/run_variant.sh dwt_fixed_slw 2 "$@"
