#!/usr/bin/env bash
# Generate the datasets of one use case into data/datasets/usecase<N>/.
#
#   bash paper/data.sh 1 [--force]   # case1_{train_n2_12,val_n2_10,test_n2_10}_nwsn
#   bash paper/data.sh 2 [--force]   # case2_{train_n2_14,val_n2_10,test_n2_10}_{nwsn,wsn} + case2_experimental (CUDA)
#
# Existing datasets are kept (new draws would differ from the released ones) unless --force is given.
# Use case 2 needs the SrTiO3 measurements in data/experimental_data/STO (not released, available on
# request) for its structured noise and experimental set; without them the released datasets are downloaded.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

USECASE="${1:?usage: bash paper/data.sh <1|2> [--force]}"
usecase_settings "${USECASE}"

DIR="data/datasets/usecase${USECASE}"
FILES=("${DIR}/case${USECASE}_"{train_n2_${DS_SIZES[0]},val_n2_${DS_SIZES[1]},test_n2_${DS_SIZES[2]}}"_nwsn.npz")
[ "${USECASE}" = 2 ] && FILES+=("${FILES[@]/_nwsn/_wsn}")
if [ "${2:-}" != --force ] && ls "${FILES[@]}" > /dev/null 2>&1; then
    step "Datasets of use case ${USECASE} already in ${DIR}/ (--force to regenerate)"
    exit 0
fi

if [ "${USECASE}" = 2 ] && [ ! -d data/experimental_data/STO ]; then
    step "No SrTiO3 measurements in data/experimental_data/STO; downloading the released datasets"
    ${PYTHON} -m pidu.hub download --what data --usecase 2
    exit 0
fi

step "Simulated datasets, use case ${USECASE}"
${PYTHON} -m pidu.data.simulation --n_train "${DS_SIZES[0]}" --n_val "${DS_SIZES[1]}" --n_test "${DS_SIZES[2]}"
if [ "${USECASE}" = 2 ]; then
    step "Experimental dataset (SrTiO3)"
    ${PYTHON} -m pidu.data.importer
fi
