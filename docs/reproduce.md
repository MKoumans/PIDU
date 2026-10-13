# Reproducing the paper

```bash
bash paper/run.sh                                 # import: ~1.5 GB download, minutes, CPU is enough
bash paper/run.sh --mode execute --gpus 0,1,2,3   # execute: 48 training runs + TRF, hours to days
```

`--usecase 1` or `--usecase 2` restricts to one use case. `--gpus` takes comma-separated GPU ids or
`cpu`; training runs one job per entry in parallel (repeat an id, `0,0,1,1`, for two jobs per GPU).
Run inside the project environment (`uv run bash paper/run.sh`) or set `PYTHON=/path/to/python`.
Everything is written to `outputs/case<N>/` (override with `PIDU_CASE_DIR`, one use case at a time).

## Steps

`paper/run.sh` chains these scripts; each can be run on its own.

| Step | Script | Writes |
| --- | --- | --- |
| Datasets | `bash paper/data.sh N` | `data/datasets/usecaseN/` |
| Training | `bash paper/train.sh N --gpus 0,1` | `models/usecaseN/`, `outputs/caseN/outputs/loss_<TYPE>_<i>_001/` |
| TRF baseline | `bash paper/lma.sh N` | `outputs/caseN/outputs/loss_lma_{baseline,noisy,random}/` |
| Tables, figures | `bash paper/figures.sh N` | `outputs/caseN/figures/` |

The import route replaces the first three steps with `pidu-hub download`. Dataset sizes and the
TRF test set are in [`paper/common.sh`](../paper/common.sh), training settings in
`configs/case<N>.yaml`. Run `i` of each model uses `--trail-id 1 --model-id i --seed 1000+i`,
`i = 0..7`. GPU training is not bit-reproducible, so retrained models match the paper statistically.

The SrTiO₃ measurements and the experimental dataset built from them are not released, so for use case
2 `paper/data.sh 2` downloads the released simulated datasets instead of generating them, and training
skips the evaluation on the experimental set.

## Released artifacts

| Artifact | Repository | Local path |
| --- | --- | --- |
| Simulated datasets, label maps | [`MKoumans/pidu-data`](https://huggingface.co/datasets/MKoumans/pidu-data) | `data/datasets/`, `data/labels/` |
| TRF results | `MKoumans/pidu-data` | `outputs/case<N>/outputs/loss_lma_*/` |
| Test exports, loss curves and logs of all runs | `MKoumans/pidu-data` | `outputs/case<N>/{outputs,loss,logs}/` |
| Checkpoints of all runs | `MKoumans/pidu-data` | `models/usecase<N>/<TYPE>-t01-<i>.pt` |
| Released checkpoints (seed 0) | [`MKoumans/pidu-optical-parameter-extraction`](https://huggingface.co/MKoumans/pidu-optical-parameter-extraction) | `models/usecase<N>/<TYPE>-t01-00.pt` |
| Timings of Fig. 2 and Table 1 (all methods, one machine) | this repository | [`paper/timings/`](../paper/timings/README.md) |
| Fig. 2 data: MAPE, precision, recall and time per method | this repository | [`scripts/figure_summary_data.csv`](../scripts/figure_summary_data.csv) |

`pidu-hub download [--what data lma runs loss logs checkpoints models] [--usecase 1 2]`; files keep
their repository paths.

## Evaluating the released models on your own device

```bash
pidu-hub download --what models data --usecase 1
bash paper/evaluate.sh 1 --device cpu      # writes outputs/case1/outputs/loss_<TYPE>_<i>_001/
python scripts/fit_evolution_dataset.py --usecase 1 --n-samples 4 --device cpu   # animate fits of test spectra
```

## Fig. 2 and the numbers in the text

```bash
bash paper/timings.sh                        # all methods timed on one machine (optional)    -> paper/timings/
python scripts/collect_figure_summary.py     # MAPE (runs) + times (paper/timings/)           -> scripts/figure_summary_data.csv
python scripts/paper_numbers.py              # abstract / text numbers from that CSV          -> outputs/fig-summary/paper_numbers.{md,csv}
python scripts/figure_summary.py             # Fig. 2                                         -> outputs/fig-summary/figure-summary.{pdf,png}
```

The metric is the same for all methods ([`research/analysis/metrics.py`](../research/analysis/metrics.py)):

- System parameters: `|pred - GT| / |GT|` over the parameters with non-zero ground truth.
- Oscillators: a prediction is paired with a ground-truth oscillator (TRF by slot, networks by rank on
  the cell grid) and counts if `|w0_pred - w0_GT| <= 0.2 w0_GT`. Only these pairs enter the MAPE;
  missed and spurious oscillators show up in precision and recall.
- Per-spectrum MAPE, averaged over spectra; networks: mean and std over the seeds shared by CNN, DU
  and PIDU (8 per use case).
- Use case 1: 1024 simulated test spectra. Use case 2: the 75 measured spectra against the expert fit.

All times come from one machine ([`paper/timings/`](../paper/timings/README.md): Intel Core i7-14650HX for TRF,
RTX 4070 Laptop GPU for the networks); on other hardware the speed-ups differ.
`python scripts/collect_figure_summary.py --timings runs` takes the times from your own runs instead
(`timing_per_sample.csv` of the LMA runs and `scripts/benchmark_timings.py`), and `--timings-only` refreshes
only the time columns of the CSV.

## Outputs (`outputs/case<N>/figures/`, compare with `paper/reference/`)

| File | Content |
| --- | --- |
| `usecase<N>_timings_table.tex` | Table 1: MAPE, oscillator precision and inference time per method (see below) |
| `usecase<N>_data_figures.xlsx` | Figure data: example spectra, parameter distributions, timings, parameter tracking (use case 2) |
| `usecase<N>_loss_comparison.{png,pdf,csv}` | Loss per epoch of CNN, DU, PIDU: median and min–max over seeds |
| `usecase<N>_training_dynamics.xlsx` | The same curves as mean and std per model and split |
| `usecase<N>_loss_dynamics_origin.xlsx` | The same curves per seed, with mean, std, median, min, max |
| `usecase2_parameter_tracking.png` | SrTiO₃ oscillator parameters over temperature: expert fit vs. CNN/DU/PIDU |

Table 1 evaluates one run per network (the released seed 0) and uses its own MAPE: the mean over the
parameter columns of `res_dis_percentage.xlsx` (system 1) of the mean relative error over all spectra,
where an oscillator without a prediction counts as 0 % error. Oscillator precision (TP / predicted,
same matching as Fig. 2) shows the detections this MAPE does not penalise.

`paper/reference/usecase0_data_figures.xlsx` (loss vs. number of unfolds) comes from
`python -m research.analysis -c case0` and needs unfold-sweep runs that are not released.
