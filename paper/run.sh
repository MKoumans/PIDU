#!/usr/bin/env bash
# Reproduce the paper's tables and figures (docs/reproduce.md).
#
#   bash paper/run.sh                                 # import: download released runs, rebuild figures
#   bash paper/run.sh --mode execute --gpus 0,1,2,3   # execute: generate data, train, run LMA, rebuild figures
#
# Options: --mode import|execute, --usecase "1 2", --gpus LIST|cpu (default: all visible GPUs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MODE=import
USECASES="1 2"
GPUS=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode) MODE="$2" ;;
        --usecase) USECASES="$2" ;;
        --gpus) GPUS="$2" ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift 2
done
# dataset generation and timings run on the first device of --gpus (train.sh uses all of them)
[ -n "${GPUS}" ] && FIRST="${GPUS%%,*}" && export CUDA_VISIBLE_DEVICES="${FIRST/cpu/}"

if [ "${MODE}" = import ]; then
    for USECASE in ${USECASES}; do
        step "Downloading artifacts, use case ${USECASE}"
        ${PYTHON} -m pidu.hub download --usecase "${USECASE}"
    done
elif [ "${MODE}" = execute ]; then
    for USECASE in ${USECASES}; do bash paper/data.sh "${USECASE}"; done
    bash paper/train.sh "${USECASES}" ${GPUS:+--gpus "${GPUS}"}
    # after training and one use case at a time, so the LMA wall-clock timings are not inflated
    for USECASE in ${USECASES}; do
        LOG="${PIDU_CASE_DIR:-outputs/case${USECASE}}/logs/lma.log"
        mkdir -p "$(dirname "${LOG}")"
        step "LMA, use case ${USECASE} -> ${LOG}"
        bash paper/lma.sh "${USECASE}" > "${LOG}" 2>&1
    done
else
    echo "--mode must be import or execute" >&2; exit 2
fi

for USECASE in ${USECASES}; do bash paper/figures.sh "${USECASE}"; done
step "Done. Figures: outputs/case<N>/figures/; reference: paper/reference/"
