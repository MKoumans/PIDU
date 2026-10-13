# Shared settings for the paper scripts (sourced).

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg PYTHONIOENCODING=utf-8
PYTHON="${PYTHON:-python}"

MODELS="${MODELS:-CNN DU PIDU}"
N_REPEATS="${N_REPEATS:-8}"   # seeds per model
TRAIL=1                       # checkpoints models/usecase<N>/<TYPE>-t01-<i>.pt

usecase_settings() {  # dataset sizes as in configs/case<N>.yaml
    case "$1" in
        1) DS_SIZES=(12 10 10); LMA_DATASET="usecase1/case1_test_n2_10_nwsn" ;;
        2) DS_SIZES=(14 10 10); LMA_DATASET="usecase2/case2_test_n2_10_wsn" ;;
        *) echo "Unknown use case: $1 (expected 1 or 2)" >&2; return 2 ;;
    esac
    export PIDU_USECASE="$1"
    CASE_DIR="${PIDU_CASE_DIR:-outputs/case$1}"
}

default_device() { ${PYTHON} -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")'; }

step() { echo; echo "==== [$(date +%H:%M:%S)] $*"; }
