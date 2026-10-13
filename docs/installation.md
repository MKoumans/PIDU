# Installation

Python 3.13 and PyTorch ≥ 2.8. A CUDA GPU is needed for paper-scale training; the quickstart, tests and evaluation run on a CPU.

## uv (recommended)

```bash
uv sync                # .venv with pidu (editable) and the dev group
uv run pidu-train --help
```

On Linux and Windows PyTorch comes from the CUDA 12.6 wheel index (`[tool.uv.sources]` in
`pyproject.toml`); change `[[tool.uv.index]]` for another CUDA version or CPU-only wheels.

## pip

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or .../whl/cpu
pip install -e .
```

Without installing, set `PYTHONPATH=src:.`. Run commands from the repository root: `research` is
imported as a top-level package and configs are resolved relative to the root.

## Docker

```bash
docker compose -f docker/docker-compose.yml up -d app
docker compose -f docker/docker-compose.yml exec app bash
```

The image mounts the checkout at `/app` and requests all GPUs.

## Apptainer / Slurm

`docker/Dockerfile.env` builds a dependency-only image (`ghcr.io/mkoumans/pidu-env`); the checkout
is bind-mounted at run time.

```bash
sbatch scripts/setup/pulljob.sh                    # -> ./pidu-env.sif
sbatch scripts/slurm.sh -m pidu.train --config configs/case2.yaml --sm --model PIDU
bash scripts/run_parallel.sh 2                     # all paper training runs as separate jobs
```

Adjust the `#SBATCH` partition and resources in `scripts/slurm.sh` to your cluster.
