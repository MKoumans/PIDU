# Architecture

## Package layout (`src/pidu`)

| Module | Contents |
| --- | --- |
| `config` | Global use case (`PIDU_USECASE`, default 2) and its active models |
| `paths` | Repository-rooted locations of datasets, labels, checkpoints, outputs (`PIDU_ROOT`) |
| `data.physics` | TMM forward model: `TmMethod` (NumPy), `TmMethodPyTorch` (differentiable), Drude-Lorentz `OscillatorModel` |
| `data.tmm_core` | Coherent/incoherent TMM core, ported from [sbyrnes321/tmm](https://github.com/sbyrnes321/tmm) |
| `data.optical_parameters` | `OpticalParameters`: per-layer parameters packed as `[samples, layers, attributes]` |
| `data.simulation` | Use-case definitions, parameter sampling, label generation, `SimulationDatasetBatch`, `create_dataset`, `load_sim_ds` |
| `data.importer` | Builds the experimental dataset from measured spectra and reference fits |
| `model.networks.framework` | `MultiModelFramework`: multi-system, multi-unfold model (PIDU, DU) |
| `model.networks.unfold_model` | `SSDModel`, `CNNModel`: one stage of one system |
| `model.networks.heads`, `blocks` | Detection/regression heads, backbone and residual blocks |
| `model.networks.activations`, `param_activations` | Bounded activations that map outputs to valid parameter ranges |
| `model.networks.cnn_basic` | `CNNBasic`: direct-regression CNN baseline |
| `model.networks.io` | Checkpoint save/load, including checkpoints from the pre-`src/` layout |
| `model.networks.experimental` | Archived architecture experiments (not used) |
| `model.trainer` | `Trainer` (data loading, loop, schedulers, test export) and `History` |
| `model.optimisation` | `LossFunction`, `BaseOptimiser`, `PCGrad`, `NashMTL` |
| `model.initializer` | `NetworkInitializer` (Xavier, Kaiming, orthogonal, LSUV, Fixup, ...) |
| `model.utils` | Parameter/wavenumber masks, checkpoint naming, plotting helpers |
| `baselines.lma` | TRF baseline: trust-region reflective fit (`scipy.optimize.least_squares`) with `baseline`/`noisy`/`random` initialisation |
| `train` | Training entry point (`pidu-train`) |
| `hub` | Hugging Face artifacts (`pidu-hub`) |

## Forward model

A sample is a stack of layers; each layer has thickness `d`, `eps_inf`, a Drude term and
Lorentz oscillators `(w0, wp, g)`. `TmMethodPyTorch` computes the requested response
(`fd_T` transmittance, `fd_R` reflectance, `fd_t` complex transmission) on the system's
wavenumber grid. It is differentiable and batched.

Every layer's permittivity also contains a fixed Drude term with `wp = g = 1 cm⁻¹`
(`OscillatorModel.add_correcting_term`). It only matters at the lowest wavenumbers and is part of the
forward model the datasets and released models were generated with.

## Systems and active models

A use case has one or two measurement *systems* (e.g. reflection and complex
transmission of the same film). Each system is split into *models* per real-valued
channel: use case 2 has `system1` (reflectance), `system2_0` and `system2_1` (real and
imaginary transmission).

## Parameter layout

- `x_sys`: system parameters per layer —
  `th0, d, eps_inf, drude_p, drude_g, w0/wp/g_before, w0/wp/g_after`.
- `x_osc`: oscillator parameters per layer, scale and cell — `(confidence, w0, wp, g)`.

Datasets return `(init, true)` pairs of both. `create_masks_sys`/`create_masks_osc` select
which entries the model predicts; the rest are taken from the known values.

## Unfolding (PIDU / DU)

`MultiModelFramework.forward(wn, y, x_sys, x_osc)`:

1. Initial estimate from the masked initial parameters; simulate `ŷ₀`.
2. For each unfold `k`: the stage-`k` model of each system receives `[ŷₖ, y]`
   (two channels, wavenumber-masked) and the current estimate. It outputs updated system
   and oscillator parameters. These are activated into valid ranges, masked, and gated by
   confidence (`conf_threshold`, 0.75), and `ŷₖ₊₁` is simulated.
3. Returns `y`, `p_sys`, `p_osc` (stacked over unfolds) and the raw outputs `z_sys`, `z_osc`.

PIDU (`physics=True`) uses the TMM to simulate `ŷ`. DU (`physics=False`) runs the same
network without the physics simulation. CNN (`CNNBasic`) regresses all parameters in a
single pass.

Stage models: `SSDModel` for systems with oscillator detection (a shared backbone and
multi-scale cell heads that predict confidence and `(w0, wp, g)` per cell), and `CNNModel`
for systems with only system parameters.

## Training objective

`LossFunction` combines per-parameter regression losses (hybrid log/MSE, `loss_alpha`)
and a confidence loss. Per-unfold weights decay towards the last unfold over the epochs.
`optimisation_mode` groups these losses into tasks (`complete`, `unfold`, `scale`,
`optics-type`, `system`, `scale-optics-type`). `optimiser_algo` combines the tasks:
`base` (sum), `pcgrad` or `nash` (Nash-MTL).
