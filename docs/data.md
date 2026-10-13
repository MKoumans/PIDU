# Data

## Use cases

| | Use case 1 (numerical) | Use case 2 (SrTiO₃) |
| --- | --- | --- |
| Layers | 1 (d = 50 µm) | 1 (d fixed per system: 60.6 µm, 27.5 µm) |
| Systems / models | `system1`: transmittance, 1–2001 cm⁻¹, 2048 points | `system1`: reflectance, 1–1025 cm⁻¹, 2048 points; `system2_{0,1}`: complex transmission, 1–91 cm⁻¹, 512 points |
| Oscillators | 0–4 (each present with p = 0.75), detected in 250 cm⁻¹ cells (`w0` 100–1925, `wp`, `g` 50–125 cm⁻¹) | 2 in `system1` (cells of 256 and 16 cm⁻¹) + a soft mode below the first cell |
| Noise | none (`nwsn`) | white Gaussian (`wgn_factor` 0.005) + structured noise from measured residuals (`wsn`) |
| Experimental set | — | 75 spectra, 4 K – 300 K |

Use case 3 (two-layer metastructure) is defined but not part of the paper.

`PIDU_USECASE` selects the use case that `pidu.data.simulation` initialises at import; the
`paper/` scripts set it.

## Datasets

Stored as `data/datasets/usecase<N>/<name>.npz`, loaded with
`pidu.data.simulation.load_sim_ds('usecase<N>/<name>')`.

Name: `case<N>_<split>_n2_<k>[_<noise>]`, with `2^k` samples. The noise suffix is
`nwsn` (no structured noise) or `wsn` (with structured noise).

| Use case | Train | Val | Test | Other |
| --- | --- | --- | --- | --- |
| 1 | `case1_train_n2_12_nwsn` | `case1_val_n2_10_nwsn` | `case1_test_n2_10_nwsn` | |
| 2 | `case2_train_n2_14_wsn` | `case2_val_n2_10_wsn` | `case2_test_n2_10_wsn` | `case2_experimental` |

## Generation

```bash
PIDU_USECASE=1 python -m pidu.data.simulation --n_train 12 --n_val 10 --n_test 10
PIDU_USECASE=2 python -m pidu.data.simulation --n_train 14 --n_val 10 --n_test 10   # nwsn and wsn
PIDU_USECASE=2 python -m pidu.data.importer                                         # case2_experimental (CUDA, needs the measurements)
```

or `bash paper/data.sh <N>`. Generation also writes the label mappings to `data/labels/`,
which the importer needs. Parameter sampling is seeded (NumPy seed 199 at import) but the noise
is not, so regenerated sets match the released ones in distribution.

In code:

```python
from pidu.data.simulation import init_usecase, create_dataset
p_gen, sys_pars = init_usecase(usecase=1, device='cpu')
ds = create_dataset(usecase=1, setname='my_set', n_samples=2**8, system_params=sys_pars,
                    p_gen_params=p_gen, verbose=False, export=False)
wn, x_sys, x_osc, y, idx = ds[0]
```

`create_dataset` always saves the set under `data/datasets/usecase<N>/<setname>.npz`.
`export` controls only the parameter/label exports.

## Experimental data

The SrTiO₃ measurements are not released (available on request). With them in
`data/experimental_data/STO/` (spectra per system, the expert RefFIT fits `fit_sto_transmission_*.xlsx`
and the fit residuals `difference_*.xlsx`), `pidu.data.simulation` adds structured noise to the use case
2 `wsn` sets and `pidu.data.importer` builds `case2_experimental`. Without them, use the released
simulated datasets (`pidu-hub download --what data --usecase 2`); `paper/data.sh 2` does so.

The use case 3 measurements (`data/experimental_data/metastructure/`) and RefFIT layer models
(`data/reffit_fit_files/`) are not released.
