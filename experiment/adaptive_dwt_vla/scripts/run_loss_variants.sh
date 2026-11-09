#!/usr/bin/env bash
# Haar fixed with the step-weighted loss, then with soft labels + step weighting; seed 0, learning-curve recipe and
# task-0 evals for each (see the LW/SLW nodes of the experiment trail).
set -u
cd "$(dirname "$0")/.."
bash scripts/run_variant.sh dwt_fixed_lw 0 "$@"
bash scripts/run_variant.sh dwt_fixed_slw 0 "$@"
