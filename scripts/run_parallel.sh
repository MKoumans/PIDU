#!/bin/bash
# Submit the paper's training runs (paper/train.sh) as separate Slurm jobs:
# CNN, DU and PIDU x N_REPEATS seeds.
#
#   bash scripts/run_parallel.sh <1|2> [pidu.train args...]      # env: MODELS, N_REPEATS
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paper/common.sh"

USECASE="${1:?usage: bash scripts/run_parallel.sh <1|2> [pidu.train args...]}"
shift
usecase_settings "${USECASE}"

for MODEL in ${MODELS}; do
    for ((i = 0; i < N_REPEATS; i++)); do
        sbatch --job-name="case${USECASE}-${MODEL}-${i}" scripts/slurm.sh -m pidu.train \
            --config "configs/case${USECASE}.yaml" --sm --usecase "${USECASE}" --model "${MODEL}" \
            --output-name "loss_${MODEL}_${i}" --trail-id "${TRAIL}" --model-id "${i}" --seed "$((1000 * TRAIL + i))" "$@"
    done
done
