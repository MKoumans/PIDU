# PIDU

**Physics-infused Deep Unfolding for Initialisation-free Material Parameter Extraction from Optical Spectra**

Milan Koumans¹, Tristan S.W. Stevens¹, Ruud J.G. van Sloun¹ and J.L.M. Van Mechelen¹

¹ Signal Processing, Eindhoven University of Technology, Eindhoven, The Netherlands

<!-- [![Paper](https://img.shields.io/badge/paper-coming%20soon-b31b1b.svg)](#citation) -->
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c.svg)](https://pytorch.org)
[![Project page](https://img.shields.io/badge/project-page-0b7285.svg)](https://mkoumans.github.io/PIDU/)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20model-pidu--optical--parameter--extraction-ffd21e.svg)](https://huggingface.co/MKoumans/pidu-optical-parameter-extraction)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-pidu--data-ffd21e.svg)](https://huggingface.co/datasets/MKoumans/pidu-data)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reference implementation for the paper, in PyTorch.

---

## Overview

Parameter estimation from optical data is a highly non-linear inverse problem which typically requires
expert knowledge for proper convergence. Conventional optimisation methods remain the predominant approach
for inferring model parameters from observational data; however, many existing solvers are computationally
expensive, exhibit strong sensitivity to initialisation, or require expert tuning. We introduce
*physics-infused deep unfolding* (PIDU), a learned optimisation framework that enables fixed-latency,
real-time parameter extraction from optical data. PIDU leverages the paradigm of deep unfolding while
explicitly incorporating the physical forward model at each iteration stage, enabling end-to-end learning of
update rules that are driven by physics-based residuals. We demonstrate that, without any initialisation,
PIDU matches the accuracy of least-squares fitting initialised near the ground truth on numerical data, and
outperforms it by 45% when extracting the optical material parameters of SrTiO₃ from experimental
terahertz spectra. Compared with randomly initialised fitting, PIDU reduces the error by 62% to 69%, while
accelerating inference by one to four orders of magnitude. On experimental data, the physics-based inductive
bias yields a 26% improvement in accuracy over the same unfolded network without it.

| Method | MAPE (%), use case 1 | MAPE (%), use case 2 (SrTiO₃) | Time per spectrum, use case 2 (ms, b = 1 / 256) |
| --- | --- | --- | --- |
| TRF (baseline), all parameters one | 88.4 | 734 | 38 100 |
| TRF (random), random initialisation | 19.8 | 21.2 | 28 300 |
| TRF (noisy), reference ± 20% | 7.4 | 12.2 | 17 600 |
| CNN | 11.1 ± 0.4 | 20.9 ± 1.7 | 6.8 / 0.024 |
| DU (no physics) | 8.2 ± 0.8 | 9.0 ± 0.9 | 187 / 0.71 |
| **PIDU** | **7.6 ± 0.5** | **6.7 ± 0.6** | 208 / 1.8 |

Values of Fig. 2, from [`scripts/figure_summary_data.csv`](scripts/figure_summary_data.csv). TRF is SciPy's
trust-region reflective least squares; network errors are mean ± std over 8 seeds. Use case 1 is simulated
data against the ground truth; use case 2 is 75 measured SrTiO₃ spectra (4–300 K) against an expert fit.
These measurements are not released. All methods were timed on one machine
([`paper/timings/`](paper/timings/README.md)).

## Installation

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/); a CUDA GPU is needed for training at paper scale.

```bash
git clone https://github.com/MKoumans/PIDU.git && cd PIDU
uv sync
```

Run commands with `uv run`, or activate the environment with `source .venv/bin/activate`. Alternatively,
`pip install -e .` into an environment with PyTorch ≥ 2.8. Docker and Apptainer images are described in
[docs/installation.md](docs/installation.md).

## Quick start

Extract the parameters of simulated spectra with a released model and plot the reconstruction:

```bash
uv run pidu-hub download --what models --usecase 1   # released checkpoints (optional)
uv run python examples/quickstart.py                 # outputs/examples/quickstart.png
```

Without the released checkpoints, the quickstart first trains a small model.

To train:

```bash
uv run pidu-train --config configs/case1.yaml --model PIDU
```

See [docs/training.md](docs/training.md) for all options.

### Reproducing the paper

```bash
bash paper/run.sh                                 # download the released runs, rebuild tables and figures
bash paper/run.sh --mode execute --gpus 0,1,2,3   # regenerate data, train, run TRF, rebuild
```

See [docs/reproduce.md](docs/reproduce.md) for the individual steps, the released artifacts and how each
number in the paper is computed.

## Method

PIDU estimates the Drude-Lorentz parameters of a sample (background permittivity and, per oscillator,
resonance frequency `w0`, plasma frequency `wp` and damping `g`) from its optical spectra.

- **Forward model** — a differentiable transfer-matrix method simulates reflection and transmission spectra
  from the current parameter estimate.
- **Unfolding** — the iterative fit is unrolled into a fixed number of stages. Each stage receives the
  measured spectrum together with the spectrum simulated from its current estimate, and a network predicts
  the update, so inference costs a fixed number of forward-model evaluations.
- **Detection** — oscillators are detected per spectral cell and regressed within it, so the number of
  resonances does not need to be known.

The paper compares PIDU with the same unfolded network without the physics residual (DU), a direct CNN
regressor and least-squares fitting (TRF) under three initialisations.

## Documentation

| Document | Contents |
| --- | --- |
| [Installation](docs/installation.md) | uv, pip, Docker, Apptainer |
| [Architecture](docs/architecture.md) | Package layout, forward model, unfolding, losses |
| [Data](docs/data.md) | Use cases, datasets, experimental data |
| [Training](docs/training.md) | Options, outputs, TRF baseline |
| [Reproduce](docs/reproduce.md) | Paper tables and figures, released artifacts, timings |
| [Development](docs/development.md) | Tests, environment variables, publishing |
| [`paper/timings/`](paper/timings/README.md) | Timing set of Fig. 2 and Table 1 |

## Repository layout

```
src/pidu/        # Library
├── data/        #   TMM forward model, simulation, datasets
├── model/       #   Unfolding framework, networks, trainer
├── baselines/   #   Least-squares (TRF) baseline
├── train.py     #   Training and evaluation entrypoint (pidu-train)
└── hub.py       #   Hugging Face download and upload (pidu-hub)

research/        # Analysis (tables, timings, tracking), TRF runners, notebooks
paper/           # One-command reproduction, timing set, reference outputs
scripts/         # Fig. 2 numbers, fit animations, Slurm launchers
configs/         # Run configuration per use case
examples/        # Quickstart
docker/          # Development and environment images
docs/            # This documentation and the project page
tests/           # pytest suite
```

## Citation

See [CITATION.cff](CITATION.cff).

<!--
```bibtex
@article{koumans2026pidu,
  title   = {Physics-infused Deep Unfolding for Initialisation-free Material
             Parameter Extraction from Optical Spectra},
  author  = {Koumans, Milan and Stevens, Tristan S. W. and
             van Sloun, Ruud J. G. and Van Mechelen, J. L. M.},
  year    = {2026}
}
```
-->

## References

Built on [PyTorch](https://pytorch.org), [NumPy](https://numpy.org), [SciPy](https://scipy.org) and the
[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub). The transfer-matrix core is ported from
[sbyrnes321/tmm](https://github.com/sbyrnes321/tmm).

## License

MIT — see [LICENSE](LICENSE). Contact: Milan Koumans, [m.koumans@tue.nl](mailto:m.koumans@tue.nl).
