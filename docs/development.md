# Development

## Layout

- `src/pidu/`: the library. Tensors carry `dt` (dtype) and `dv` (device); convert with
  `pt2np` / `np2pt` (`pidu.data.utils`).
- `research/`: analysis and experiment scripts; run from the repository root
  (`python -m research.analysis`, `python research/lma/run_all_schemes.py`).
- `paper/`: reproduction only; calls `pidu` and `research`.

## Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `PIDU_USECASE` | 2 | Use case initialised by `pidu.data.simulation` at import |
| `PIDU_ROOT` | repository root | Base of `data/`, `models/`, `outputs/` |
| `PIDU_CASE_DIR` | `outputs/case<N>` | Output directory of one use case |

## Tests

```bash
uv run pytest
uv run pytest -m "not slow"   # skip tests that simulate a dataset
```

Tests in `test_optimizer_combinations.py` and `test_initializers_forward.py` are skipped unless
`data/datasets/usecase1/case1_{train,val}_n2_6.npz` / `..._n2_10.npz` exist.

## Checkpoints

Checkpoints are pickled modules (`torch.save(model)`). `load_checkpoint` also maps the module paths
of checkpoints saved before the `src/pidu` layout onto the current package. Only load checkpoints
from sources you trust.

## Publishing artifacts

```bash
hf auth login
pidu-hub upload                            # data, lma, runs, loss, logs, checkpoints, models
pidu-hub upload --what models --usecase 2
```

An upload mirrors the local files ([reproduce.md](reproduce.md#released-artifacts)): Hub files of an
artifact that no longer exist locally are deleted. The experimental dataset is never uploaded.

Pushes to the `publish` branch build `docker/Dockerfile` and push it to `ghcr.io/<repo>`
(`.github/workflows/publish.yml`).
