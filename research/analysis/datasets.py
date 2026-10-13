"""Dataset loading plus per-sample and parameter-distribution extraction."""

import numpy as np
import pandas as pd

from pidu.data.simulation import SimulationDatasetBatch, load_sim_ds
from pidu.data.utils import pt2np

from .utils import PROJECT_DIR, as_dataframe

SYS_COL_NAMES = [
    'theta', 'd', 'eps_infty',
    'drude_wp', 'drude_gamma',
    'lorentz_wp1', 'lorentz_gamma1', 'lorentz_sigma1',
    'lorentz_wp2', 'lorentz_gamma2', 'lorentz_sigma2',
]
OSC_COL_NAMES = ['has_obj', 'w0', 'wp', 'g']


def load_dataset(
    filename: str | None = None,
    usecase: int | None = None,
    device: str | None = None,
    y_return_mode: str = 'noised',
    split: str = 'test',
) -> SimulationDatasetBatch:
    """Load a simulation dataset by name, or the latest one for a usecase/split.

    A `filename` that does not exist exactly is resolved to the latest dataset
    whose name starts with it (e.g. 'case1_test_n2_10' -> 'case1_test_n2_10_nwsn').
    """
    if filename is None:
        ds_dir = PROJECT_DIR / 'data' / 'datasets' / f'usecase{usecase}'
        names = sorted(p.stem for p in ds_dir.iterdir() if split in p.name)
        filename = f'usecase{usecase}/{names[-1]}'
    else:
        subdir, _, name = filename.rpartition('/')
        ds_dir = PROJECT_DIR / 'data' / 'datasets' / subdir
        if not (ds_dir / f'{name}.npz').exists():
            matches = sorted(p.stem for p in ds_dir.glob(f'{name}*.npz'))
            if matches:
                filename = f'{subdir}/{matches[-1]}'
    return load_sim_ds(filename, y_return_mode=y_return_mode, device=device)


def flatten_oscillator_params(osc_array: np.ndarray, param_names: list[str] | None = None) -> pd.DataFrame:
    """Flatten a (n_layers, n_scales, n_cells, n_params) oscillator array into a
    DataFrame with columns: layer_id, scale_id, cell_id, <param_names>."""
    n_layers, n_scales, n_cells, n_params = osc_array.shape
    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_params)]

    layers, scales, cells = np.indices((n_layers, n_scales, n_cells)).reshape(3, -1)
    df = pd.DataFrame(osc_array.reshape(-1, n_params), columns=param_names)
    df.insert(0, 'cell_id', cells)
    df.insert(0, 'scale_id', scales)
    df.insert(0, 'layer_id', layers)
    return df


def get_sample_info(
    ds: SimulationDatasetBatch,
    sample_idx: int,
    systems_to_save: tuple[int, ...] = (1, 2),  # (thickness, eps_infty)
) -> dict[str, pd.DataFrame]:
    """Extract one sample from the dataset.

    For each system, returns the signal (y_<sys>), region boundaries
    (x_boundaries_<sys>_s<i>), system parameters (sys_<sys>) and, for SSD
    models, oscillator parameters (osc_<sys>).
    """
    sample = ds[sample_idx]
    # sample = (wn, (x_sys_init, x_sys_tr), (x_osc_init, x_osc_tr), y, idx)
    out: dict[str, pd.DataFrame] = {}

    for system_name in ds.data_y:
        general_name = system_name.split('_')[0]  # e.g. 'system2' from 'system2_1'
        system_dict = ds.system_dicts[general_name]

        # -- Signal: columns [wn, noise-free, noisy] or [wn, signal] --
        wn = pt2np(system_dict['wn'])
        sample_y = pt2np(sample[3][system_name])
        y = np.zeros((wn.shape[0], 3 if ds.y_return_mode == 'both' else 2))
        y[:, 0] = wn
        y[:, 1:] = sample_y if sample_y.ndim == 2 else sample_y.reshape(-1, 1)
        out[f'y_{system_name}'] = y

        # -- Region boundaries --
        for i_region, region_width in enumerate(system_dict.get('region_width', [])):
            bounds = np.arange(*system_dict['wn_ran'][:2], region_width)
            boundaries = np.ones((len(bounds), 2))
            boundaries[:, 0] = bounds
            out[f'x_boundaries_{system_name}_s{i_region}'] = boundaries

        # -- System parameters --
        sys_data = pt2np(sample[1][1][general_name])[:, list(systems_to_save)]
        out[f'sys_{system_name}'] = {
            SYS_COL_NAMES[i]: value for i, value in zip(systems_to_save, sys_data[0, :])
        }

        # -- Oscillator parameters --
        if system_dict['model'] == 'SSD':
            osc_data = pt2np(sample[2][1][general_name])
            out[f'osc_{system_name}'] = flatten_oscillator_params(osc_data, param_names=OSC_COL_NAMES)

    return {key: as_dataframe(value) for key, value in out.items()}


def param_column_names(ds: SimulationDatasetBatch, system_name: str, n_params: int) -> list[str]:
    """System parameter names followed by (w0, wp, g) triplets per oscillator."""
    sys_cols = list(ds.col_names_sys[system_name])
    n_osc = (n_params - len(sys_cols)) // 3
    for osc_idx in range(n_osc):
        sys_cols += [f'w0_{osc_idx}', f'wp_{osc_idx}', f'g_{osc_idx}']
    return sys_cols


def get_p_distribution(ds: dict[str, SimulationDatasetBatch]) -> dict[str, pd.DataFrame]:
    """True-parameter distributions per dataset, system and layer.

    Keys: '<ds_name>_s<sys_idx>_l<layer_idx>' -> DataFrame of all samples with
    columns <sys_param_names>, w0_<i>, wp_<i>, g_<i>.
    """
    out: dict[str, pd.DataFrame] = {}

    for ds_name, _ds in ds.items():
        first_sys_name = next(iter(_ds.data_tr))

        for idx_sys, system_name in enumerate(_ds.data_y):
            general_name = system_name.split('_')[0]
            p_true = pt2np(_ds.data_tr[general_name])
            col_names = param_column_names(_ds, first_sys_name, p_true.shape[-1])

            for idx_layer in range(p_true.shape[1]):
                out[f'{ds_name}_s{idx_sys}_l{idx_layer}'] = pd.DataFrame(
                    p_true[:, idx_layer, :], columns=col_names
                )

    return out


def extract_gt_params(ds: SimulationDatasetBatch, system_name: str = 'system1') -> pd.DataFrame:
    """Ground-truth parameters of layer 0 as a DataFrame (oscillators numbered from 1)."""
    p_true = pt2np(ds.data_tr[system_name])[:, 0, :]

    cols = list(ds.col_names_sys[system_name])
    n_osc = (p_true.shape[1] - len(cols)) // 3
    for osc_idx in range(1, n_osc + 1):
        cols += [f'w0_{osc_idx}', f'wp_{osc_idx}', f'g_{osc_idx}']

    return pd.DataFrame(p_true, columns=cols)
