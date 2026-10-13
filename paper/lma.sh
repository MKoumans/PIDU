#!/usr/bin/env bash
# Levenberg-Marquardt baseline (baseline / noisy / random initialisation) on the test set.
#
#   bash paper/lma.sh 1 [research/lma/run_all_schemes.py args, e.g. --samples 16]
#
# Writes outputs/case<N>/outputs/loss_lma_{baseline,noisy,random}/. The full test set takes hours on a CPU.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

USECASE="${1:?usage: bash paper/lma.sh <1|2> [run_all_schemes.py args...]}"
shift
usecase_settings "${USECASE}"

step "LMA, use case ${USECASE}, dataset ${LMA_DATASET}"
${PYTHON} research/lma/run_all_schemes.py --dataset "${LMA_DATASET}" "$@"
