#!/usr/bin/env bash
# Tables and figure data of one use case, from the checkpoints, run exports and LMA results.
#
#   bash paper/figures.sh 1 [--device cpu]
#
# Writes outputs/case<N>/figures/ (see docs/reproduce.md#outputs); compare with paper/reference/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

USECASE="${1:?usage: bash paper/figures.sh <1|2> [--device DEV]}"
DEVICE="$(default_device)"
[ "${2:-}" = --device ] && DEVICE="${3:?--device needs a value}"
usecase_settings "${USECASE}"

step "Tables and figures, use case ${USECASE} (${DEVICE})"
${PYTHON} -m research.analysis --chapter "case${USECASE}" --usecase "${USECASE}" \
    --sizes "${DS_SIZES[@]}" --device "${DEVICE}" --export
${PYTHON} -m research.analysis.loss_comparison --usecase "${USECASE}"
