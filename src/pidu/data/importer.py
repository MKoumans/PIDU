"""
Import experimental measurement data (.dat signals + fitted parameters from
Excel) into a dataset compatible with a trained simulation dataset.

Run as a script to build and save the experimental dataset for the usecase
selected by PIDU_USECASE (see pidu.config). Pass --debug (or set DEBUG=1) for verbose logging.
"""
import logging
import os
import sys
from os.path import join as path_join
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from torch import nn

from pidu.config import usecase
from pidu.data.simulation import LMIDataset, LabelGenerator, load_sim_ds, pt2np
from pidu.paths import EXPERIMENTAL_DIR

log = logging.getLogger(__name__)


class DataImporter(LMIDataset):
  def __init__(
    self,
    settype:str,
    trained_ds:str,
    device:str|torch.device=torch.device('cuda'),
    dtype:torch.dtype=torch.float32
  ):
    super().__init__(settype, device, dtype)
    self.relu = nn.ReLU()
    self.tmm = None
    self.oscs_mapping = None

    self._init_trained_ds(trained_ds)

    self.system_columns = ['theta', 'thickness', 'eps_inf', 'wp_drude', 'g_drude'] + \
      ['w0_before', 'wp_before', 'g_before', 'w0_after', 'wp_after', 'g_after']

  def _init_trained_ds(self, trained_ds_name:str):
    """
    Initialize dataset based on a trained dataset. This includes initial x and
    some metadata.
    """
    log.info(f"Loading trained dataset '{trained_ds_name}' as template.")
    trained_ds = load_sim_ds(trained_ds_name)

    self.wn = trained_ds.wn
    self.data_x_sys_init = {k: v[0].unsqueeze(0) if v is not None else None for k, v in trained_ds.data_x_sys_init.items()}
    self.data_x_osc_init = {k: v[0].unsqueeze(0) if v is not None else None for k, v in trained_ds.data_x_osc_init.items()}
    self.data_x_sys_true = {k: torch.zeros_like(v, dtype=self.dtype, device=self.device)[0] if v is not None else None for k, v in trained_ds.data_x_sys_tr.items()}
    self.data_x_osc_true = {k: torch.zeros_like(v, dtype=self.dtype, device=self.device)[0] if v is not None else None for k, v in trained_ds.data_x_osc_tr.items()}

    # -- items necessary for plotting --
    self.system_dicts = trained_ds.system_dicts

    # -- items necessary for saving/loading dataset --
    self.n_layers = trained_ds.n_layers
    self.wgn_factor = trained_ds.wgn_factor
    self.return_mode = trained_ds.return_mode
    self.params_msd = trained_ds.params_msd
    self.col_names_sys = trained_ds.col_names_sys

    self.stepsize = trained_ds.stepsize
    self.anchors = trained_ds.anchors
    self.region_widths = trained_ds.region_widths
    self.n_regions_per_scale = trained_ds.n_regions_per_scale
    self.n_scales = trained_ds.n_scales

    self.n = {k: v.shape[0] for k, v in self.data_x_sys_init.items()}
    log.debug(f"Template systems: {list(self.system_dicts.keys())}, wn keys: {list(self.wn.keys())}, n_layers: {self.n_layers}")

  def __getitem__(self, idx) -> Tuple[Dict, Tuple[Dict, Dict], Tuple[Dict, Dict], Dict, int]:
    """Return (wn, (x_sys_init, x_sys_tr), (x_osc_init, x_osc_tr), y, idx) for a sample."""
    wn = dict(self.wn)
    x_sys_init = {k: v[idx] for k, v in self.data_x_sys_init.items()}
    x_sys_tr = {k: v[idx] for k, v in self.data_x_sys_tr.items()}
    x_osc_init = {k: v[idx] for k, v in self.data_x_osc_init.items()}
    x_osc_tr = {k: v[idx] for k, v in self.data_x_osc_tr.items()}
    y = {k: v[idx] for k, v in self.data_y.items()}

    return wn, (x_sys_init, x_sys_tr), (x_osc_init, x_osc_tr), y, idx

  def _check_monotonicity(self, data:np.ndarray) -> np.ndarray:
    """Drop rows whose x-value does not strictly increase (required by CubicSpline)."""
    keep = np.ones(data.shape[0], dtype=bool)
    last_x = data[0, 0]
    for i in range(1, data.shape[0]):
      if data[i, 0] <= last_x:
        keep[i] = False
      else:
        last_x = data[i, 0]

    n_removed = int((~keep).sum())
    if n_removed:
      log.debug(f"Removed {n_removed} non-monotonic data point(s) at indices {np.flatnonzero(~keep).tolist()}.")

    return data[keep]

  def _load_dat(self, path:str) -> np.ndarray:
    return np.loadtxt(path, skiprows=0)

  def _interpolate(self, data:np.ndarray, system:str) -> np.ndarray:
    """Cubic-spline interpolation of measured data onto the system's wavenumber grid."""
    matching_wn = [v for k, v in self.wn.items() if system.startswith(k)]
    if not matching_wn:
      raise KeyError(f"No wavenumber grid found for system '{system}'. Available: {list(self.wn.keys())}.")
    wn = matching_wn[0]

    data = self._check_monotonicity(data)
    spline = CubicSpline(data[:, 0], data[:, 1])
    return spline(pt2np(wn))

  def _add_data(self, system_name:str, signal:np.ndarray, i_signal:int, to_clamp:bool=False):
    signal_ = self._interpolate(signal, system=system_name)
    if to_clamp:
      signal_ = np.clip(signal_, 0.0, 1.0)

    if system_name not in self.data_y:
      self.data_y[system_name] = np.zeros((self.n_files_to_load, signal_.shape[0]))

    if i_signal >= self.data_y[system_name].shape[0]:
      raise IndexError(
        f"File index {i_signal} out of range for system '{system_name}' "
        f"(allocated for {self.data_y[system_name].shape[0]} files). "
        "Do all subsystem directories contain the same number of files?"
      )

    self.data_y[system_name][i_signal, :] = signal_

  def _get_filenames(self, path:str, file_ext:str='.dat', filter_substrings:List[str]=[]) -> List[str]:
    """List files in `path` with `file_ext`, sorted by their trailing number (e.g. 'm_12.dat' -> 12)."""
    filenames = [
      path_join(path, item) for item in os.listdir(path)
      if item.endswith(file_ext) and not any(s in item for s in filter_substrings)
    ]

    try:
      filenames_sorted = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except ValueError as e:
      raise ValueError(f"Could not sort files in '{path}' by trailing number: {e}. Files: {filenames}") from e

    log.debug(f"Found {len(filenames_sorted)} '{file_ext}' file(s) in '{path}'.")
    return filenames_sorted

  def _np_to_pt(self):
    for system_name, values in self.data_y.items():
      self.data_y[system_name] = torch.tensor(values, device=self.device, dtype=self.dtype)

  def load_measurements(
    self,
    systems:Dict[str, Dict[str, Any]],
    dir_path:str,
    fitted_params:Dict[str, str]|None=None,
    mappings:Dict[str, str]|None=None,
    **kwargs
  ):
    """
    Load measured signals (.dat) and fitted parameters (Excel) for the given systems.

    Args:
      systems: Per-system settings, e.g. {'system1': {'to_clamp': True}}.
      dir_path: Directory containing '<system>_*' subdirectories with .dat files.
      fitted_params: Per-system Excel filename (relative to dir_path) with fitted parameters.
      mappings: Per-system oscillator mapping (LabelGenerator name or instance); required for pred_osc systems.
      kwargs: 'forced_scale' (per-system anchor scale overrides), 'max_opt_params' (truncate oscillator params).
    """
    base_path = dir_path
    if not os.path.isdir(base_path):
      raise FileNotFoundError(f"Measurement directory '{os.path.abspath(base_path)}' does not exist.")

    files_in_base_path = os.listdir(base_path)
    system_names = set(systems.keys())
    log.info(f"Loading measurements for systems {sorted(system_names)} from '{os.path.abspath(base_path)}'.")

    # Drop template systems that are not requested
    for key in list(self.data_x_sys_init.keys()):
      if key not in system_names:
        log.debug(f"Dropping unused template system '{key}'.")
        del self.data_x_sys_init[key]
        del self.data_x_sys_true[key]
        if key in self.data_x_osc_init:
          del self.data_x_osc_init[key]
          del self.data_x_osc_true[key]

    # -- Load measured signals --
    for system_name in system_names:
      system_dirs = [p for p in files_in_base_path if p.startswith(system_name)]
      if not system_dirs:
        log.warning(f"No subdirectories starting with '{system_name}' found in '{base_path}'. "
                    f"Directory contains: {files_in_base_path}")

      for item in system_dirs:
        path_ = path_join(base_path, item)
        ds_name = item.split('_')[-1]
        subsystem = item[:item.index(f"_{ds_name}")]

        filenames = self._get_filenames(path_, filter_substrings=['no_var'])
        self.n_files_to_load = len(filenames)
        log.info(f"Loading {self.n_files_to_load} file(s) for subsystem '{subsystem}' from '{item}'.")

        for index, dat_filepath in enumerate(filenames):
          signal = self._load_dat(dat_filepath)
          self._add_data(subsystem, signal, index, to_clamp=systems[system_name]['to_clamp'])

    if not self.data_y:
      raise RuntimeError(f"No measurement data loaded from '{base_path}' for systems {sorted(system_names)}.")

    self._np_to_pt()

    loaded_system_names = set(key.split('_')[0] for key in self.data_y.keys())
    self.system_dicts = {k: v for k, v in self.system_dicts.items() if k in loaded_system_names}
    log.debug(f"Loaded signal keys: {list(self.data_y.keys())} -> systems {sorted(loaded_system_names)}.")

    # -- Load fitted (true) parameters from Excel --
    if fitted_params is not None:
      for system, fitted_params_file in fitted_params.items():
        if system not in loaded_system_names:
          raise ValueError(f"System '{system}' in fitted_params not found in loaded systems {sorted(loaded_system_names)}.")

        full_path = path_join(dir_path, fitted_params_file)
        log.info(f"Reading fitted parameters for '{system}' from '{full_path}'.")
        df = pd.read_excel(full_path, engine='openpyxl', sheet_name=None)
        log.debug(f"Excel sheets (layers): {list(df.keys())}")

        for count_layer, (layer_name, layer_df) in enumerate(df.items()):
          self.non_sys_columns = [col for col in layer_df.columns if col not in self.system_columns + ['Sample']]
          log.debug(f"Layer '{layer_name}': {layer_df.shape[0]} rows, oscillator columns: {self.non_sys_columns}")

          if layer_df.shape[0] != self.n_files_to_load:
            raise ValueError(
              f"Number of fitted parameters ({layer_df.shape[0]}) in sheet '{layer_name}' of '{fitted_params_file}' "
              f"does not match number of loaded files ({self.n_files_to_load})."
            )

          sys_layer = torch.tensor(layer_df[self.system_columns].values[:, None, :], device=self.device, dtype=self.dtype)
          if count_layer == 0:
            self.data_x_sys_init[system] = self.data_x_sys_init[system].repeat(self.n_files_to_load, 1, 1)
            self.data_x_sys_tr[system] = sys_layer
          else:
            self.data_x_sys_tr[system] = torch.cat((self.data_x_sys_tr[system], sys_layer), dim=1)

          if self.system_dicts[system]['pred_osc']:
            raw_osc_layer = torch.tensor(layer_df[self.non_sys_columns].values[:, None, :], device=self.device, dtype=self.dtype)
            if raw_osc_layer.shape[-1] % 3 != 0:
              raise ValueError(
                f"Oscillator parameter columns for system '{system}', layer '{layer_name}' must be divisible by 3 (w0, wp, g triplets). "
                f"Found {raw_osc_layer.shape[-1]} columns: {self.non_sys_columns}."
              )

            if count_layer == 0:
              self.data_x_osc_init[system] = self.data_x_osc_init[system].repeat(self.n_files_to_load, 1, 1, 1, 1)
              self.data_x_osc_tr[system] = raw_osc_layer
            else:
              self.data_x_osc_tr[system] = torch.cat((self.data_x_osc_tr[system], raw_osc_layer), dim=1)

    self.n = {k: v.shape[0] for k, v in self.data_x_sys_tr.items()}
    log.debug(f"Samples per system: {self.n}")

    pred_osc_systems = [name for name, config in self.system_dicts.items() if config.get('pred_osc', False)]

    # -- Oscillator mappings (required for pred_osc systems) --
    if mappings is not None:
      self.assign_oscs_mapping(mappings, forced_scale=kwargs.get('forced_scale', None))
    elif pred_osc_systems:
      raise ValueError(
        "Oscillator mapping is required when 'pred_osc=True' in DataImporter.load_measurements. "
        f"Missing mappings for systems: {pred_osc_systems}."
      )

    # -- Optionally truncate oscillator parameters --
    max_params = kwargs.get('max_opt_params', None)
    if max_params:
      log.info(f"Truncating oscillator parameters to the first {max_params} oscillator(s).")
      for name, p_osc in self.data_x_osc_tr.items():
        if p_osc is not None:
          self.data_x_osc_tr[name] = p_osc[..., :max_params * 3]

    # -- Concatenated true parameters (system + oscillators) per system --
    self.data_tr = {}
    for key in self.data_x_sys_true.keys():
      if self.data_x_sys_tr[key] is not None and key in self.data_x_osc_tr:
        self.data_tr[key] = torch.concat((self.data_x_sys_tr[key], self.data_x_osc_tr[key]), dim=-1)
      else:
        self.data_tr[key] = self.data_x_sys_tr[key]

    # -- Generate detection labels for pred_osc systems --
    for name in pred_osc_systems:
      p_osc = self.data_x_osc_tr.get(name)
      if p_osc is None:
        log.warning(f"System '{name}' has pred_osc=True but no oscillator parameters were loaded; skipping label generation.")
        continue
      if p_osc.ndim != 3:
        raise ValueError(
          f"Expected raw oscillator tensor with shape [n_samples, n_layers, n_params] for system '{name}', got {tuple(p_osc.shape)}."
        )
      if not torch.isfinite(p_osc).all():
        raise ValueError(f"Raw oscillator tensor contains non-finite values for system '{name}'.")

      log.info(f"Generating detection labels for system '{name}'.")
      p_osc_init, p_osc_true = self._generate_labels(name, p_osc[..., :6])

      if p_osc_init is not None:
        self.data_x_osc_init[name] = p_osc_init
        self.data_x_osc_tr[name] = p_osc_true

    log.info(f"Finished loading measurements: {sum(self.n.values())} sample(s) across {len(self.n)} system(s).")

  def get_statistics_p(self):
    for system in self.data_x_sys_tr.keys():
      log.info(f"Statistics for system '{system}':")
      min_sys = self.data_x_sys_tr[system].min(dim=0).values
      max_sys = self.data_x_sys_tr[system].max(dim=0).values

      for il in range(self.n_layers):
        log.info(f" - Layer {il + 1}:")
        for ic, column in enumerate(self.system_columns):
          log.info(f"  - {column}: min={min_sys[il][ic]}, max={max_sys[il][ic]}")

  def assign_oscs_mapping(self, label_classes:Dict[str, Any], **kwargs):
    """Assign per-system oscillator mappings (parameter -> bounding box) from LabelGenerators."""
    self.oscs_mapping = {}
    forced_scale = kwargs.get('forced_scale') or {}

    for systemname, label_class in label_classes.items():
      if isinstance(label_class, str):
        log.debug(f"Loading oscillator mapping for '{systemname}' from '{label_class}'.")
        label_gen = LabelGenerator(self.device, self.settype)
        label_gen.load(label_class)
        mapping_system = label_gen.mapping
      elif isinstance(label_class, LabelGenerator):
        mapping_system = dict(label_class.mapping)
      else:
        raise TypeError(f"Label class {label_class} not supported. Only (LabelGenerator|str).")

      self.oscs_mapping[systemname] = mapping_system

      if (forced_scale.get(systemname) is None) and (self.system_dicts[systemname]['model'] == 'SSD'):
        forced_scale[systemname] = -1 * np.ones((self.n_layers, self.n_scales[systemname]), dtype=int)

    self.forced_scale = forced_scale
    log.debug(f"Assigned oscillator mappings for {list(self.oscs_mapping.keys())}, forced_scale: { {k: np.asarray(v).tolist() for k, v in forced_scale.items()} }")

  # -- LABEL GENERATION (multi-scale detector) --

  def _iou_shape(self, w1:torch.Tensor, w2:torch.Tensor) -> torch.Tensor:
    left = torch.max(-w1 / 2, -w2 / 2)
    right = torch.min(w1 / 2, w2 / 2)
    intersection = self.relu(right - left)
    union = w1 + w2 - intersection
    return intersection / union

  def _bounding_box_to_label_v2(
    self,
    bbox:torch.Tensor,
    region_sizes:torch.Tensor,
    anchors:torch.Tensor,
    params:torch.Tensor,
    sample_id:int,
    layer_id:int,
    osc_id:int|None,
    systemname:str,
    targets:torch.Tensor,
    wn:torch.Tensor
  ) -> Tuple[torch.Tensor, bool]:
    """
    Maps a bounding box to the appropriate anchor based on Intersection Over Union (IOU) with predefined anchors.

    Args:
      bbox: Bounding box in wavenumbers: (center position, width).
      params: Parameters associated with the bounding box: [w0, wp, g].
      sample_id: Sample index within the batch, used to index into targets.
      targets: Target tensor of shape [n_samples, n_layers, n_scales, n_regions, n_attributes].

    Returns:
      (updated targets, whether an anchor was assigned)
    """
    middle, width = bbox
    iou_anchors = self._iou_shape(width, anchors)

    if osc_id is not None and self.forced_scale[systemname][layer_id][osc_id] >= 0:
      # Force this oscillator onto a specific scale by making its IOU the largest
      iou_anchors[self.forced_scale[systemname][layer_id][osc_id]] = torch.max(iou_anchors) + 0.1
    sorted_anchors = torch.flipud(torch.argsort(iou_anchors))

    for anchor_idx in sorted_anchors:
      scale = anchor_idx
      n_regs_of_scale = int(torch.ceil((wn[-1] - wn[0]) / region_sizes[scale]))
      window_length = len(wn) / n_regs_of_scale
      idx_ = torch.argmin(torch.abs(wn - middle))  # Index of the wavenumber closest to the bbox center
      region = int(idx_ / window_length)
      anchor_taken = targets[sample_id, layer_id, scale, region, 0]

      if not anchor_taken:
        targets[sample_id, layer_id, scale, region, 0] = 1  # Attribute: has_obj
        targets[sample_id, layer_id, scale, region, 1:] = params  # Attribute: classes + (w0,wp,g)
        return targets, True

    log.warning(
      f"No free anchor for oscillator (system={systemname}, sample={sample_id}, layer={layer_id}, osc={osc_id}): "
      f"bbox={bbox.tolist()}, anchor preference={sorted_anchors.tolist()}. Oscillator dropped from labels."
    )
    return targets, False

  def _generate_labels(
    self,
    systemname:str,
    p_osc:torch.Tensor  # [n_samples, n_layers, n_params (typically n_oscs*3)]
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Init variables
    n_samples, n_layers, n_params = p_osc.shape
    n_scales = self.n_scales[systemname]
    domain = self.system_dicts[systemname]['domain']
    prop_dir = self.system_dicts[systemname]['prop_dir']
    n_oscs_max = n_params // 3
    df_map_p2mwh = self.oscs_mapping[systemname][f"{domain}_{prop_dir}"]
    col_names = ['sample_id', 'scale_id', 'pc_1', 'w0', 'wp', 'g', 'middle', 'width', 'height']
    class_names = ['pc_1']
    wn = self.wn[systemname]
    n_regions_per_scale = self.n_regions_per_scale[systemname]
    targets = torch.zeros((n_samples, n_layers, n_scales, int(max(n_regions_per_scale)), len(self.params_msd['columns_target'])), dtype=self.dtype, device=self.device)

    log.debug(
      f"Generating labels for '{systemname}': n_samples={n_samples}, n_layers={n_layers}, "
      f"n_oscs_max={n_oscs_max}, n_scales={n_scales}, mapping key='{domain}_{prop_dir}'."
    )

    # Assign sample id, osc id, probability class1, and parameters
    data_df = np.zeros((n_samples * n_layers * n_oscs_max, len(col_names)))
    data_df[:, 0] = np.repeat(np.arange(n_samples), n_oscs_max * n_layers)
    data_df[:, 1] = np.tile(np.arange(n_oscs_max), n_samples * n_layers)  # Oscillator ID
    data_df[:, 2] = 1
    data_df[:, 3:6] = pt2np(p_osc).reshape(-1, 3)  # [w0, wp, g] per sample/layer/oscillator

    # 2. Interpolate (w0, wp, g) -> (middle, width, height) via the mapping
    map_points = df_map_p2mwh.loc[:, ['w0', 'wp', 'g']].to_numpy()
    middle_mapping = LinearNDInterpolator(map_points, df_map_p2mwh.loc[:, 'middle'].to_numpy())
    width_mapping = LinearNDInterpolator(map_points, df_map_p2mwh.loc[:, 'width'].to_numpy())
    height_mapping = LinearNDInterpolator(map_points, df_map_p2mwh.loc[:, 'height'].to_numpy())

    idx_nonzero = data_df[:, 3:6].sum(axis=1) != 0
    params_nonzero = data_df[idx_nonzero, 3:6]

    data_df[idx_nonzero, 6] = middle_mapping(params_nonzero)
    data_df[idx_nonzero, 7] = width_mapping(params_nonzero)
    data_df[idx_nonzero, 8] = height_mapping(params_nonzero)

    df = pd.DataFrame(data_df, columns=col_names)
    if np.any(df.isna()):
      nan_rows = df[df.isna().any(axis=1)]
      log.error(
        f"NaN values after interpolating oscillator parameters for '{systemname}' — "
        f"parameters likely outside the mapping's (w0, wp, g) range. Offending rows:\n{nan_rows}"
      )
      raise ValueError(f"NaN values in mapping dataframe ({systemname}). Please check the ranges of the mappings.")

    # 3. Iterate over the rows and create targets
    for i, row in df.iterrows():
      if np.all([row[class_name] for class_name in class_names] == 0):
        continue

      sample_id = i // (n_oscs_max * n_layers)
      layer_id = ((i - sample_id * (n_oscs_max * n_layers)) // n_oscs_max) % n_layers
      scale_id = row['scale_id']
      # TODO: Fix scale_id
      assert row['sample_id'] == sample_id, f"Sample ID {row['sample_id']} does not match index {i // (n_oscs_max * n_layers)}."

      bbox_oscs = torch.tensor([row['middle'], row['width']], dtype=self.dtype, device=self.device)
      params_oscs = torch.tensor([row['w0'], row['wp'], row['g']], dtype=self.dtype, device=self.device)
      if bbox_oscs.sum() == 0:
        continue

      targets, passed = self._bounding_box_to_label_v2(
        bbox_oscs,
        self.region_widths[systemname],
        self.anchors[systemname],
        params_oscs,
        sample_id,
        layer_id,
        int(scale_id),  # used to determine 'forced_scale'
        systemname,
        targets,
        wn
      )

      if not passed:
        df.loc[i, ['pc_1', 'w0', 'wp', 'g', 'middle', 'width', 'height']] = 0

    p_osc_true = targets
    p_osc_init = torch.zeros_like(p_osc_true, dtype=self.dtype, device=self.device)

    return p_osc_init, p_osc_true


USECASE_CONFIGS = {
  'usecase2': {
    'trained_ds': 'usecase2/case2_train_n2_14_wsn',
    'systems': {'system1': {'to_clamp': True}, 'system2': {'to_clamp': False}},
    'dir_path': str(EXPERIMENTAL_DIR / 'STO'),
    'fitted_params': {'system1': 'fit_sto_transmission_dook1.xlsx', 'system2': 'fit_sto_transmission_dook2.xlsx'},
    'mappings': {
      'system1': 'mapping_system1_w0[125.0_600.0]_wp[50.0_700.0]_g[0.5_30.0].xlsx',
      'system2': 'mapping_system1_w0[125.0_600.0]_wp[50.0_700.0]_g[0.5_30.0].xlsx',
    },
    'load_kwargs': {'forced_scale': {'system1': [[1, 0]]}},
    'save_as': 'usecase2/case2_experimental',
  },
  # usecase3 is deadend trail
  'usecase3': {
    'trained_ds': 'usecase3/case3_train_n2_14',
    'systems': {'system1': {'to_clamp': True}, 'system2': {'to_clamp': False}},
    'dir_path': str(EXPERIMENTAL_DIR / 'metastructure'),
    'fitted_params': {'system1': 'fit_t.xlsx', 'system2': 'fit_t.xlsx'},
    'mappings': {
      'system1': 'mapping_system1_w0[0.0_110.0]_wp[0.0_400.0]_g[0.0_30.0].xlsx',
      'system2': 'mapping_system2_w0[0.0_110.0]_wp[0.0_400.0]_g[0.0_30.0].xlsx',
    },
    'load_kwargs': {},
    'save_as': 'usecase3/case3_experimental',
  },
}


def load_exp_ds(current_usecase:str, device:str='cuda') -> DataImporter:
  """Build, save, and return the experimental dataset for the given usecase."""
  from pidu.data.simulation import measured_file
  measured_file(USECASE_CONFIGS[current_usecase]['dir_path'])
  config = USECASE_CONFIGS.get(current_usecase)
  if config is None:
    raise ValueError(
      f"No experimental dataset configuration for '{current_usecase}'. "
      f"Available: {sorted(USECASE_CONFIGS.keys())}."
    )

  if device == 'cuda' and not torch.cuda.is_available():
    log.warning(
      "CUDA requested but not available (running on a login node?). "
      "Loading will fail when tensors are moved to GPU; run on a GPU node or pass device='cpu'."
    )

  log.info(f"Building experimental dataset for '{current_usecase}' (trained_ds='{config['trained_ds']}', device='{device}').")

  data_importer = DataImporter(
    settype=f'experimental_{current_usecase}',
    trained_ds=config['trained_ds'],
    device=device,
    dtype=torch.float32
  )
  data_importer.load_measurements(
    systems=config['systems'],
    dir_path=config['dir_path'],
    fitted_params=config['fitted_params'],
    mappings=config['mappings'],
    **config['load_kwargs']
  )
  data_importer.y_return_mode = 'noised'
  data_importer.save_dataset(config['save_as'])
  log.info(f"Saved experimental dataset as '{config['save_as']}'.")

  return data_importer


if __name__ == '__main__':
  debug = '--debug' in sys.argv or os.environ.get('DEBUG')
  logging.basicConfig(
    level=logging.DEBUG if debug else logging.INFO,
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
  )

  data_importer = load_exp_ds(f'usecase{usecase}')

  data_importer.sample_info(
    idxs=[0, 9, 19, 29, 39, 49, 59, 69],
    print_info=True,
    plot_info=True,
    save_plots=True,
    show_plots=False,
    bounding_boxes=True,
  )

  data_importer.get_statistics_p()
  data_importer.get_info_for_model(print_info=1)
