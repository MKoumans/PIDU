# Training

```bash
pidu-train --config configs/case2.yaml --model PIDU --seed 1000   # = python -m pidu.train ...
```

Values come from the defaults in `src/pidu/train.py`, then the YAML config, then the command line.
Without `--config`, `--usecase` must equal `PIDU_USECASE`.

| Option | Default (configs) | Meaning |
| --- | --- | --- |
| `--config` | — | `configs/case<N>.yaml`; keys are the option names |
| `--usecase` | `PIDU_USECASE` | 1 or 2 (3: defined, not in the paper) |
| `--model` | `PIDU` | `PIDU`, `DU` (no physics) or `CNN` |
| `--epochs` | 100 (200) | |
| `--device` | `cuda:0` if available | |
| `--load-ds`, `--ds-sizes` | see configs | Dataset name template (`{ds_name}` = train/val/test, `{number}` from `--ds-sizes`, log2 of train/val/test size). `load_ds: null` generates new sets |
| `--nwsn` | off (case 1: on) | Datasets without structured noise (`_nwsn` suffix) |
| `--exp-ds` | off (case 2: on) | Also evaluate on `case2_experimental` |
| `--wgn` | 0 | Std of extra white Gaussian noise added to the input spectra while training |
| `--unfolds` | 3 | Unfolding stages (PIDU, DU) |
| `--optimisation-mode` | `scale-optics-type` | Grouping of the losses into tasks ([architecture](architecture.md#training-objective)) |
| `--optimiser-algo` | `base` (`nash`) | `base`, `pcgrad`, `nash` |
| `--lr`, `--weight-decay` | 1e-4, 1.0 | AdamW |
| `--initialiser` | `xavier` | Weight initialisation (`pidu.model.initializer`) |
| `--cnn-size` | 1.0 | Width factor of the CNN baseline |
| `--mixed-precision` | off | bfloat16 autocast |
| `--seed` | 199 | Python, NumPy and PyTorch seed |
| `--trail-id`, `--model-id`, `--output-name` | —, —, `loss` | Run identifiers used in the output names |
| `--sm` | off | Headless: no intermediate plots |
| `--evaluate` | off | Skip training; export the predictions of `models/usecase<N>/<MODEL>-t<trail>-<id>.pt` |

YAML-only keys (`batch_size`, `betas`, `eps`, `clip_grad_norm`, `loss_alpha`, `scheduler_*`,
`shuffle`, `track_gradient`) are listed in `TRAINING_SYSTEM_DEFAULTS` in `src/pidu/train.py`.

## Outputs

For `--output-name loss_PIDU_3 --trail-id 1 --model-id 3`, use case 2:

| Path | Content |
| --- | --- |
| `models/usecase2/PIDU-t01-03.pt` | Checkpoint (pickled model; load with `pidu.model.networks.io.load_checkpoint`) |
| `outputs/case2/runs/loss_PIDU_3_001.txt` | Per-batch loss log |
| `outputs/case2/loss/loss_PIDU_3_001/` | Loss curves and loss matrices |
| `outputs/case2/outputs/loss_PIDU_3_001/test_{val,test,exp}/` | Predictions (`last_unfold.xlsx`), residuals (`res_dis*.xlsx`), plots |

Without `--model-id` the next free id is used; without `--trail-id` the run index increments.

## TRF baseline

```bash
PIDU_USECASE=1 python research/lma/run_all_schemes.py --dataset usecase1/case1_test_n2_10_nwsn
```

Fits every spectrum with SciPy's trust-region reflective least squares (`pidu.baselines.lma`) from
three initialisations: `baseline` (zeros), `noisy` (ground truth + 20 % noise) and `random`
(uniform within the simulation bounds). Each writes `outputs/case<N>/outputs/loss_lma_<scheme>/`
with per-sample timings, fitted parameters and residuals. `--samples N` limits the run, `--seed`
(default 0) seeds the initialisation.
