#!/usr/bin/env bash
# Train CNN, DU and PIDU (N_REPEATS seeds each) for one or more use cases, in parallel over GPUs.
#
#   bash paper/train.sh 1 [--gpus 0,1,2] [pidu.train args...]
#   bash paper/train.sh "1 2" --gpus 0,0,1,1     # two jobs per GPU
#
# One worker per --gpus entry (default: all visible GPUs, else cpu). Run i of model M uses seed
# 1000*TRAIL+i and writes models/usecase<N>/M-t01-<i>.pt, outputs/case<N>/outputs/loss_M_<i>_001/
# and the log outputs/case<N>/logs/train_M_<i>.log. Settings come from configs/case<N>.yaml.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

USECASES="${1:?usage: bash paper/train.sh <1|2|\"1 2\"> [--gpus LIST] [pidu.train args...]}"
shift
GPUS=""
if [ "${1:-}" = --gpus ]; then GPUS="$2"; shift 2; fi
[ -n "${GPUS}" ] || GPUS="$(${PYTHON} -c 'import torch; print(",".join(map(str, range(torch.cuda.device_count()))) or "cpu")')"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"

JOBS=()  # interleaved over models so each worker gets a similar mix
for ((i = 0; i < N_REPEATS; i++)); do
    for USECASE in ${USECASES}; do
        for MODEL in ${MODELS}; do JOBS+=("${USECASE} ${MODEL} ${i}"); done
    done
done

train_one() {  # train_one GPU USECASE MODEL I
    local gpu="$1" usecase="$2" model="$3" i="$4" device=cuda:0
    usecase_settings "${usecase}"
    local log="${CASE_DIR}/logs/train_${model}_${i}.log"
    mkdir -p "$(dirname "${log}")"
    [ "${gpu}" = cpu ] && gpu="" && device=cpu
    echo "[$(date +%H:%M:%S)] ${model} ${i}, use case ${usecase} on ${device} ${gpu} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" ${PYTHON} -m pidu.train --config "configs/case${usecase}.yaml" --sm \
        --device "${device}" --usecase "${usecase}" --model "${model}" \
        --output-name "loss_${model}_${i}" --trail-id "${TRAIL}" --model-id "${i}" --seed "$((1000 * TRAIL + i))" "${@:5}" > "${log}" 2>&1
}

worker() {  # jobs k, k+W, k+2W, ... on GPU_LIST[k]
    local k="$1" failed=0 j
    for ((j = k; j < ${#JOBS[@]}; j += ${#GPU_LIST[@]})); do
        # shellcheck disable=SC2086
        train_one "${GPU_LIST[$k]}" ${JOBS[$j]} "${EXTRA[@]}" || { failed=1; echo "FAILED: ${JOBS[$j]}" >&2; }
    done
    return "${failed}"
}

EXTRA=("$@")
step "Training ${#JOBS[@]} runs (use cases ${USECASES}; ${MODELS}; ${N_REPEATS} seeds) on: ${GPU_LIST[*]}"
PIDS=()
for ((k = 0; k < ${#GPU_LIST[@]}; k++)); do worker "${k}" & PIDS+=("$!"); done
STATUS=0
for pid in "${PIDS[@]}"; do wait "${pid}" || STATUS=1; done
[ "${STATUS}" = 0 ] || { echo "Some runs failed; see outputs/case<N>/logs/" >&2; exit 1; }
step "All training runs finished"
