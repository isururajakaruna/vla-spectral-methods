#!/usr/bin/env bash
# Waits for the running DCT seed-0 job, then runs DCT seeds 1 and 2 with the same learning-curve recipe.
# Usage: scripts/queue_dct_seeds.sh <pid-to-wait-for>
set -u
cd "$(dirname "$0")/.."
pid=${1:-}
[ -n "$pid" ] && while kill -0 "$pid" 2>/dev/null; do sleep 60; done
for s in 1 2; do
  bash scripts/run_variant.sh dct_fixed "$s"
done
