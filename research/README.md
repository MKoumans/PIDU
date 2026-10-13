# Research

Analysis and experiment code built on `pidu`. Run from the repository root with `pidu`
importable (`uv sync`, or `PYTHONPATH=src:.`).

| Path | Contents |
| --- | --- |
| `analysis/` | Paper tables and figure data: `python -m research.analysis -c case<N> -u <N> [--sizes 14 10 10] -e` (see `--summary`). Also `analyse_outputs.py` (residual distributions, re-validation of runs) and `analyze_loss_run.py` (loss-curve comparison) |
| `lma/` | TRF baseline over all initialisation schemes (`run_all_schemes.py`), residual recomputation, initialisation plots |
| `timings/` | Stand-alone inference benchmark of the network vs. TRF (`benchmark.py`), earlier results in `outputs/` |
| `experimental/` | Export of RefFIT layer fits to transmission spectra |
| `notebooks/` | Structured-noise derivation (SrTiO₃, metastructure) and loss-run exploration |
| `legacy/` | Earlier analysis scripts (hyper-parameter sweeps, residuals, activation ranges, architecture tables); kept for reference, not maintained |
