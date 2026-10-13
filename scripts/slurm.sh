#!/bin/bash
#SBATCH --job-name=pidu
#SBATCH --output=./slurm-out/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
# Run a Python command inside the Apptainer image ./pidu-env.sif, from the repository root:
#
#   sbatch scripts/slurm.sh -m pidu.train --config configs/case2.yaml --sm --model PIDU
#   PIDU_USECASE=2 sbatch scripts/slurm.sh -m pidu.data.simulation --n_train 14 --n_val 10 --n_test 10
#   sbatch --time=0:15:00 scripts/slurm.sh -m research.analysis -c case2 -u 2 -e
mkdir -p slurm-out
srun apptainer exec --nv --env PYTHONPATH="$PWD/src:$PWD" ./pidu-env.sif python "$@"
