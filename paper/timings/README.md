# Timings of Fig. 2 and Table 1

All methods were timed on one laptop by [`paper/timings.sh`](../timings.sh) (run of 9 October 2026),
strictly sequentially so TRF had the CPU to itself:

- CPU: Intel Core i7-14650HX (TRF, single process, one spectrum at a time, each spectrum timed once).
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, PyTorch 2.14.1 (CUDA 12.6) (CNN, DU, PIDU).

| File | Content |
| --- | --- |
| `trf_case<N>_<scheme>.csv` | Wall-clock time per TRF fit (`sample_idx`, `total_time_ms`), schemes `baseline`, `noisy`, `random` |
| `nn_case<N>.csv` | Forward-pass time per spectrum of the released checkpoints `<TYPE>-t01-00` at batch sizes 1–256: mean, std, median, min and max over 20 repetitions after 3 warm-up runs, excluding data transfer to the GPU |

Spectra: use case 1, the first 128 test spectra of `case1_test_n2_10_nwsn`; use case 2, the 75 measured
spectra (not released).

`scripts/collect_figure_summary.py` reads these files for the time columns of
`scripts/figure_summary_data.csv` (TRF: mean and std over spectra; networks: mean and std over
repetitions, b = 1 and b = 256). `--timings-only` refreshes just those columns.
