# Scripts

| Script | Purpose |
| --- | --- |
| `benchmark_timings.py` | Forward-pass time of the CNN/DU/PIDU checkpoints (Fig. 2) |
| `collect_figure_summary.py` | MAPE, precision/recall (`research/analysis/metrics.py`) and timings of all methods → `figure_summary_data.csv` |
| `paper_numbers.py` | Numbers of the paper text, recomputed from `figure_summary_data.csv` and checked |
| `figure_summary.py` | Fig. 2 |
| `figure_p_init.py` | Spread of the TRF initialisations |
| `fit_evolution.py`, `fit_evolution_dataset.py` | Animations of the fit over the unfolding stages |
| `slurm.sh` | `sbatch scripts/slurm.sh <python args>`: run a Python command in `./pidu-env.sif` |
| `run_parallel.sh` | Submit the paper's training runs (`paper/train.sh`) as separate Slurm jobs |
| `setup/pulljob.sh` | Pull `ghcr.io/mkoumans/pidu-env:latest` into `pidu-env.sif` |

See [docs/reproduce.md](../docs/reproduce.md) for the order in which the Fig. 2 scripts run.
