#!/usr/bin/env bash
# Export the val/test predictions of the checkpoints in models/usecase<N>/ on your own device.
#
#   bash paper/evaluate.sh 1 [--device cpu]
#
# Writes outputs/case<N>/outputs/loss_<TYPE>_<i>_001/, as paper/train.sh does after training.
# Needs the checkpoints and test sets: pidu-hub download --what models data --usecase <N>.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

USECASE="${1:?usage: bash paper/evaluate.sh <1|2> [--device DEV]}"
DEVICE="$(default_device)"
[ "${2:-}" = --device ] && DEVICE="${3:?--device needs a value}"
[ "${DEVICE}" = cuda ] && DEVICE=cuda:0   # the datasets are placed on this exact device
usecase_settings "${USECASE}"

LOGS="${CASE_DIR}/logs"
mkdir -p "${LOGS}"
shopt -s nullglob
for PT in models/usecase${USECASE}/{CNN,DU,PIDU}-t01-*.pt; do
    NAME="$(basename "${PT}" .pt)"   # <TYPE>-t01-<i>
    MODEL="${NAME%%-*}"
    I="$((10#${NAME##*-}))"
    step "${PT} (${DEVICE}) -> ${LOGS}/evaluate_${MODEL}_${I}.log"
    ${PYTHON} -m pidu.train --evaluate --config "configs/case${USECASE}.yaml" --sm \
        --device "${DEVICE}" --usecase "${USECASE}" --model "${MODEL}" \
        --output-name "loss_${MODEL}_${I}" --trail-id "${TRAIL}" --model-id "${I}" --seed "$((1000 * TRAIL + I))" > "${LOGS}/evaluate_${MODEL}_${I}.log" 2>&1
    FOUND=1
done
[ -n "${FOUND:-}" ] || { echo "No checkpoints in models/usecase${USECASE}/ (pidu-hub download --what models)" >&2; exit 1; }
