#!/usr/bin/env bash
# Timings of Fig. 2 and Table 1 on one machine, strictly sequential so TRF has the CPU to itself.
#
#   bash paper/timings.sh
#
# Writes outputs/timings_final/case<N>/ and copies the timings to paper/timings/ (scripts/export_timings.py).
# Needs the released checkpoints (seed 0) and test sets: pidu-hub download --what models data.
# TRF in use case 2 runs on the SrTiO3 measurements, which are not released.
set -euo pipefail
cd "$(dirname "$0")/.."
source paper/common.sh
T=outputs/timings_final

# 1. TRF, use case 2: the 75 measured spectra
export PIDU_USECASE=2 PIDU_CASE_DIR="$PWD/$T/case2"
step "TRF, use case 2, experimental set"
${PYTHON} research/lma/run_all_schemes.py --dataset usecase2/case2_experimental

# 2. TRF, use case 1: the first 128 test spectra
export PIDU_USECASE=1 PIDU_CASE_DIR="$PWD/$T/case1"
step "TRF, use case 1, 128 test spectra"
${PYTHON} research/lma/run_all_schemes.py --dataset usecase1/case1_test_n2_10_nwsn --samples 128
for s in baseline noisy random; do
    rm -rf "$T/case1/outputs/loss_lma_$s"
    cp -a "$T/case1/outputs/loss_lma_${s}_n128" "$T/case1/outputs/loss_lma_$s"
done

# 3. networks (GPU): forward-pass timings of the checkpoints, exported with the tables
PIDU_CASE_DIR="$PWD/$T/case1" bash paper/evaluate.sh 1
PIDU_CASE_DIR="$PWD/$T/case1" bash paper/figures.sh 1
PIDU_CASE_DIR="$PWD/$T/case2" bash paper/figures.sh 2

${PYTHON} scripts/export_timings.py --run "$T" --out paper/timings
step "Done: paper/timings/"
