import time

import os
import sys
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from os.path import join as path_join

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from scipy.interpolate import LinearNDInterpolator
from openpyxl import load_workbook
from typing import List, Dict, Any, Tuple
from numpy.typing import NDArray

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pidu.config import usecase
from pidu.paths import ROOT, DATASETS_DIR, LABELS_DIR, EXPERIMENTAL_DIR, OUTPUTS_DIR
from pidu.data.physics import TmMethodPyTorch
from pidu.data.optical_parameters import OpticalParameters
from pidu.data.utils import generate_noise_profile_uc2_sys1, pt2np, np2pt,  get_matches, create_wn, generate_noise_profile_uc2, \
  generate_noise_profile_uc3

def measured_file(*parts) -> str:
  """Path of a file under data/experimental_data/. The SrTiO3 measurements are not part of the public
  repository, so their absence ends with a pointer to the released datasets."""
  path = EXPERIMENTAL_DIR.joinpath(*parts)
  if not path.exists():
    raise FileNotFoundError(
      f"{path} not found. The SrTiO3 measurements (and the residuals used as structured noise) are not "
      "released; they are available on request. The simulated datasets: pidu-hub download --what data --usecase 2")
  return str(path)


# --- COSTANTS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
dtype = torch.float32
random_seed = 199 #
np.random.seed(random_seed) # For reproducibility

# Usecase 1
n_system_parameters = 5 + 2*3  # d, eps_inf, w0, wp, g + osc_before + osc_after
system_params_case1 = {
  'model': 'SSD',               # Model type: SingleShotDetector|ConvolutionalNeuralNetwork
  'domain': 'fd',               # Domain: fd|td
  'prop_dir': 't',              # Propagation direction
  'wn_spacing': 'linear',       # Wavenumber spacing
  'wn_ran': (1, 2000+1, 2048),  # Wavenumber range
  'region_width': [250],        # Region widths (given in wavenumbers): should be from large to small in width
  'anchors': [250*(5/4)],       # Anchor size (given in wavenumbers)
  'thickness': 0.005,      # Thickness (given in cm)
  'return_mode': 'fd_T',            # Return mode (see utils.py for more info)
  'p_init': torch.zeros(n_system_parameters, dtype=dtype, device=device), # Initial parameters for the system
  'signal_th': 0.8,   # Threshold for object detection with bbox
  'pred_sys': True,   
  'pred_osc': True,  # 
}

thickness = system_params_case1['thickness']

p_gen_usecase1 = {
  'case':'main',      #
  'n_samples': 2**12, # Number of samples
  'n_layers': 1,      # Number of layers
  'num_sys_params': 5 + 2*3,  # Number of system parameters (d, eps_inf, w0, wp, g) + osc_before + osc_after
  'sampling': 'uniform',      # Sampling method
  'wgn_factor': 0.000,        # Factor for white Gaussian noise
  'y_return_mode': 'both',    # Mode for returning y: normal, noised, both
  
  'th0':([(0,0),]),
  'd':([(thickness, thickness),]),
  'e_inf':([(1,8),]),
  'drude_p': ([(0,0),]), #
  'drude_g': ([(0,0),]), #
  'oscs_start': [((0,0),), ((0,0),), ((0,0),)], # oscs before first w0
  # 'oscs_end': ([[(100,130)], [(500,1000)], [(60,120)]], [[(0,0)], [(0,0)], [(0,0)]]), # oscs after last w0
  'oscs_end': [((0,0),), ((0,0),), ((0,0),)], # oscs after last w0
  'w0':[
    [(100,425),(575,925),(1075,1425),(1575,1925)], # ,(0,0)
  ], # Wavenumbers (cm^-1)
  'wp':[
    [(50,125),(50,125),(50,125),(50,125)], # ,(0,0)
  ], # Wavenumbers (cm^-1)
  'g':[
    [(50,125),(50,125),(50,125),(50,125)], # ,(0,0)
  ], # Drude damping (cm^-1)],
  'forced_scale': {'system1': [[-1,-1,-1,-1]]}, # ,0
  'oscs_chance': 0.75,
}

# Usecase 2
thickness = 0.006061516148 # 
thickness2 = 0.00275 # 
n_system_parameters = 5 + 2*3  # d, eps_inf, w0, wp, g + osc_before + osc_after
p_gen_usecase2 = {
  'case':'main',      #
  'n_layers': 1,      #
  'num_sys_params': 5 + 2*3,  # Number of system parameters (d, eps_inf, w0, wp, g) + osc_before + osc_after
  'sampling': 'uniform',      # Sampling method
  'wgn_factor': 0.005,        # Factor for white Gaussian noise
  'y_return_mode': 'both',  # Mode for returning y: normal, noised, both
  
  'th0':([(0,0),]),
  'd':{0: ([(thickness, thickness),]), 1: ([(thickness2, thickness2),])}, # cm
  'e_inf':([(5,6),]),
  'drude_p': ([(0,0)]), #
  'drude_g': ([(0,0)]), #
  'oscs_start': [((10,100),), ((1400,1700),), ((4,20),)], # oscs before first w0
  'oscs_end': [((0,0),), ((0,0),), ((0,0),)], # oscs after last w0
  'w0':[
    [(125,200),(400,600)], # ,(1200,1500)
  ], # Wavenumbers (cm^-1)
  'wp':[
    [(50,400),(500,700)], #
  ], # Wavenumbers (cm^-1)
  'g':[
    [(0.5,3.5),(10,30)], #
  ], # Drude damping (cm^-1)],
  'forced_scale': {'system1': [[1,0]]}, # ,0
  'oscs_chance': 0.95, # 0.95, #
}
system_params_case2_1 = {
  'model': 'SSD',                   # Model type: SingleShotDetector|ConvolutionalNeuralNetwork
  'domain': 'fd',                   # Domain: fd|td
  'prop_dir': 'r',                  # Propagation direction
  'wn_spacing': 'linear',           # Wavenumber spacing
  'wn_ran': (1, 1024+1, 2048),      # Wavenumber range
  'region_width': [256, 16],        # Region widths (given in wavenumbers): should be from large to small in width
  'anchors': [256*(5/4), 16*(5/4)], # Anchor size (given in wavenumbers)
  'thickness': 0.006061516148,      # Thickness (given in cm)
  'return_mode': 'fd_R',            # Return mode (see utils.py for more info)
  'p_init': torch.zeros(n_system_parameters, dtype=dtype, device=device), # Initial parameters for the system
  'signal_th': 0.8,   # Threshold for object detection with bbox
  'pred_sys': True,   # 
  'pred_osc': True,  # 
}
system_params_case2_2 = {
  'model': 'CNN',                   # Model type
  'domain': 'fd',                   # Domain: fd|td
  'prop_dir': 't',                  # Propagation direction
  'wn_spacing': 'linear',           # Wavenumber spacing
  'wn_ran': (1, 1+90, 512),         # Wavenumber range
  'thickness': 0.00275,             # Thickness (given in cm)
  'return_mode': 'fd_t',            # Return mode (see utils.py for more info)
  'p_init': torch.zeros(n_system_parameters, dtype=dtype, device=device), # Initial parameters for the system
  'signal_th': 0.8,           # Threshold for object detection with bbox
  'pred_sys': True,
  'pred_osc': False,
}


# Usecase 3
thickness = 0.03940425802
n_system_parameters = 5 + 2*3  # d, eps_inf, w0, wp, g + osc_before + osc_after
p_gen_usecase3 = {
  'case':'main',      #
  'n_samples': 2**12, # Number of samples
  'n_layers': 2,      # Number of layers
  'num_sys_params': 5 + 2*3,  # Number of system parameters (d, eps_inf, w0, wp, g) + osc_before + osc_after
  'sampling': 'uniform',      # Sampling method
  'wgn_factor': 0.000,        # Factor for white Gaussian noise
  'y_return_mode': 'both',    # Mode for returning y: normal, noised, both
  
  'th0':([(0,0),(0,0)]),
  'd':([(thickness, thickness),(1e-4, 2e-4)]),
  'e_inf':([(10,14),(20,20)]),
  'drude_p': ([(0,0),(0,0)]), #
  'drude_g': ([(0,0),(0,0)]), #
  'oscs_start': [((0,0), (0,0)), ((0,0), (0,0)), ((0,0), (0,0))], # oscs before first w0
  # 'oscs_end': ([[(100,130)], [(500,1000)], [(60,120)]], [[(0,0)], [(0,0)], [(0,0)]]), # oscs after last w0
  'oscs_end': [((0,0), (120,200)), ((0,0), (300,1000)), ((0,0), (80,200))], # oscs after last w0
  'w0':[
    [(0,0)], # ,(0,0)
    [(20,110)], # ,(20,105)
  ], # Wavenumbers (cm^-1)
  'wp':[
    [(0,0)], # ,(0,0)
    [(150,400)], # ,(200,900)
  ], # Wavenumbers (cm^-1)
  'g':[
    [(0,0)], # ,(0,0)
    [(1,30)], # ,(5,150)
  ], # Drude damping (cm^-1)],
  'oscs_chance': 0.85,
}
system_params_case3_1 = {
  'model': 'SSD',                   # Model type: SingleShotDetector|ConvolutionalNeuralNetwork
  'domain': 'fd',                   # Domain: fd|td
  'prop_dir': 't',                  # Propagation direction
  'wn_spacing': 'linear',           # Wavenumber spacing
  'wn_ran': (1, 128+1, 512),        # Wavenumber range
  'region_width': [32],             # Region widths (given in wavenumbers): should be from large to small in width
  'anchors': [32*(5/4)],            # Anchor size (given in wavenumbers)
  'thickness': 0.039404,            # Thickness (given in cm)
  'return_mode': 'fd_T',            # Return mode (see utils.py for more info)
  'p_init': torch.zeros(n_system_parameters, dtype=dtype, device=device), # Initial parameters for the system
  'signal_th': 0.8,   # Threshold for object detection with bbox
  'pred_sys': True,   # 
  'pred_osc': True,  # 
}
system_params_case3_2 = {
  'model': 'SSD',                   # Model type
  'domain': 'fd',                   # Domain: fd|td
  'prop_dir': 't',                  # Propagation direction
  'wn_spacing': 'linear',           # Wavenumber spacing
  'wn_ran': (1, 128+1, 512),        # Wavenumber range
  'region_width': [32],             # Region widths (given in wavenumbers): should be from large to small in width
  'anchors': [32*(5/4)],            # Anchor size (given in wavenumbers)
  'thickness': 0.039404,            # Thickness (given in cm)
  'return_mode': 'fd_t',            # Return mode (see utils.py for more info)
  'p_init': torch.zeros(n_system_parameters, dtype=dtype, device=device), # Initial parameters for the system
  'signal_th': 0.8,                   # Threshold for object detection with bbox
  'pred_sys': True,
  'pred_osc': True,
}


def init_systems_params(sys1, sys2:None=None, device:str='cpu') -> Tuple[Dict[str, Any], Dict[str, Any]]:
  sys1['wn'] = create_wn(sys1['wn_spacing'], sys1['wn_ran'], device, dtype) # Wavenumber
  sys1['p_init'][:3] = torch.tensor([0, sys1['thickness'], 0], dtype=dtype, device=device) # angle, thickness, e_inf
  if 'region_width' in sys1.keys():
    sys1['n_regions_per_scale'] = [int((sys1['wn_ran'][1] - sys1['wn_ran'][0]) / i) for i in sys1['region_width']]
  
  if sys2 is not None:
    sys2['wn'] = create_wn(sys2['wn_spacing'], sys2['wn_ran'], device, dtype)            # Wavenumber
    sys2['p_init'][:3] = torch.tensor([0, sys2['thickness'], 0], dtype=dtype, device=device)  # angle, thickness, e_inf
    if 'region_width' in sys2.keys():
      sys2['n_regions_per_scale'] = [int((sys2['wn_ran'][1] - sys2['wn_ran'][0]) / i) for i in sys2['region_width']]

  return {'system1': sys1, 'system2': sys2} 


class ParametersSetGenerator():
  """
    SAVE AND LOAD ONLY IMPLEMENTED FOR SINGLE LAYER
  """
  def __init__(self, case:str='main', mode:str='torch', device:str='cpu', dtype:Any=torch.float64) -> None:
    self.device = device            #
    self.dtype = dtype              #
    self.data_mode = mode           # torch|numpy
    self.parameters:Dict[str, OpticalParameters] = {}
    self.parameters_ranges:Dict[str, Dict[str, Tuple[float, float]]] = {}
    self.case = case      #
    self.last_case = ''   # Last case used: main|r3_0|
    self.cases = {
      'main': self._case_main
    }

    self.data_directory = str(DATASETS_DIR)

  def __call__(self, p_gen:Dict[str, Any], case:str='main', set_name:str='all', **kwds) -> Dict[str, OpticalParameters]|OpticalParameters|None:
    n_samples = kwds.get('n_samples', None)
    if n_samples is not None:
      p_gen['n_samples'] = n_samples
    
    return self.create(p_gen, case, set_name, **kwds)  # Call the create function with the case name

  def __str__(self):
    return f"ParametersSetGenerator(case={self.case}"

  # --- TESTCASES ---
  def _case_main(self, p_gen, **kwargs) -> OpticalParameters:
    n_samples = p_gen.get('n_samples', 64)     #
    n_layers = p_gen.get('n_layers', 1)        #
    oscs_chance = p_gen.get('oscs_chance', 1)  #
    self.sys_pars_names = [
      'th0', 'd', 'e_inf', 'drude_p', 'drude_g',
      'w0_before', 'wp_before', 'g_before',
      'w0_after', 'wp_after', 'g_after'
    ]

    # --OPTICAL PARAMETERS--
    opt_params = self._get_optical_params(p_gen, n_layers)
    th0_r, d_r, e_inf_r, drude_p_r, drude_g_r, w0_b, wp_b, g_b, w0_a, wp_a, g_a, w0_r, wp_r, g_r = opt_params
    # ----------------------

    self.n_sys_pars = self.n_sys_pars # th0, d, e_inf, drude(wp), drude(g), w0_before, wp_before, g_before, w0_after, wp_after, g_after
    th0_r = np.array(th0_r)

    max_oscs = max([len(x) for x in w0_r])                      #
    data = np.zeros((n_samples, n_layers, self.n_sys_pars + 3*max_oscs))  #
    data[:, :, 0] = np.random.uniform(th0_r[0,0], th0_r[0,1], (n_samples,n_layers))

    for i_layer in range(n_layers):
      data[:, i_layer, 1] = np.random.uniform(d_r[i_layer,0], d_r[i_layer,1], (n_samples,))             # 
      data[:, i_layer, 2] = np.random.uniform(e_inf_r[i_layer,0], e_inf_r[i_layer,1], (n_samples,))     #
      data[:, i_layer, 3] = np.random.uniform(drude_p_r[i_layer,0], drude_p_r[i_layer,1], (n_samples,)) #
      data[:, i_layer, 4] = np.random.uniform(drude_g_r[i_layer,0], drude_g_r[i_layer,1], (n_samples,)) #
      
      data[:, i_layer, 5] = np.random.uniform(w0_b[i_layer,0], w0_b[i_layer,1], (n_samples,)) # w0_before
      data[:, i_layer, 6] = np.random.uniform(wp_b[i_layer,0], wp_b[i_layer,1], (n_samples,)) # wp_before
      data[:, i_layer, 7] = np.random.uniform(g_b[i_layer,0], g_b[i_layer,1], (n_samples,))   # g_before
      oscs_change_before = np.random.choice([0,1], n_samples, p=[1-oscs_chance, oscs_chance]) # 
      data[:, i_layer, 5:8] = data[:, i_layer, 5:8] * oscs_change_before[:,None]              # 
      
      data[:, i_layer, 8] = np.random.uniform(w0_a[i_layer,0], w0_a[i_layer,1], (n_samples,)) # w0_after
      data[:, i_layer, 9] = np.random.uniform(wp_a[i_layer,0], wp_a[i_layer,1], (n_samples,)) # wp_after
      data[:, i_layer, 10] = np.random.uniform(g_a[i_layer,0], g_a[i_layer,1], (n_samples,))  # g_after
      oscs_change_after = np.random.choice([0,1], n_samples, p=[1-oscs_chance, oscs_chance])  # 
      data[:, i_layer, 8:11] = data[:, i_layer, 8:11] * oscs_change_after[:,None]

      for i_oscs in range(len(w0_r[i_layer])):
        w0 = np.random.uniform(*w0_r[i_layer][i_oscs], (n_samples,))
        wp = np.random.uniform(*wp_r[i_layer][i_oscs], (n_samples,))
        g = np.random.uniform(*g_r[i_layer][i_oscs], (n_samples,))
        activate_oscs = np.random.choice([0,1], n_samples, p=[1-oscs_chance, oscs_chance])
        data_oscs = np.stack([w0, wp, g], axis=-1) * activate_oscs[:,None]
        data[:, i_layer, self.n_sys_pars+3*i_oscs:self.n_sys_pars+3*(i_oscs+1)] = data_oscs

    optical_params = OpticalParameters(
      th0=data[..., 0],
      d=data[..., 1],
      eps_inf=data[..., 2],
      drude_p=data[..., 3],
      drude_g=data[..., 4],
      w0_before=data[..., 5],
      wp_before=data[..., 6],
      g_before=data[..., 7],
      w0_after=data[..., 8],
      wp_after=data[..., 9],
      g_after=data[..., 10],
      opt_params=data[..., self.n_sys_pars:],
      dtype=self.dtype,
      device=self.device
    )

    return optical_params
  # -----------------

  # -- HELP FUNCTIONS --
  def _get_optical_params(
    self,
    user_values:Dict[str, List[Tuple[float]]],
    n_layers:int
    ) -> Tuple[NDArray[np.float64], ...]:
    # Use self.sys_pars_names to fetch parameter values
    params = []
    for name in self.sys_pars_names:
      val = user_values.get(name)
      if val is None:
        raise ValueError(f"Missing required parameter '{name}' in user_values.")
      # Repeat for n_layers if not already a list of length n_layers
      if isinstance(val, (list, tuple)) and len(val) == n_layers:
        params.append(np.array(val))
      else:
        params.append(np.array([val]*n_layers))
    # Handle oscillator parameters (w0, wp, g) if present
    w0 = user_values.get('w0', [])
    wp = user_values.get('wp', [])
    g = user_values.get('g', [])
    params.extend([np.array(w0), np.array(wp), np.array(g)])
    return tuple(params)

  def _get_min_max(self, params:OpticalParameters|Dict[str, List[Tuple[float, float]]]) -> Dict[str, Tuple[float]]:
      if isinstance(params, OpticalParameters):
          data = params[None]

          min_func = lambda x: torch.min(x).item()
          max_func = lambda x: torch.max(x).item()

          minmax = np.zeros((self.n_sys_pars,2))  #
          minmax_osc = np.zeros((3, 2))           # w0, wp, g
          minmax_osc[:, 0] = 1e10

          for i in range(self.n_sys_pars):
            minmax[i, 0] = max(0, min_func(data[..., i]))
            minmax[i, 1] = max_func(data[..., i])

          for i in range((data.shape[-1]-self.n_sys_pars)//3):
            # drop zeros from data
            w0 = data[data[..., self.n_sys_pars + 3*i] > 0][:, self.n_sys_pars + 3*i]
            wp = data[data[..., self.n_sys_pars + 3*i+1] > 0][:, self.n_sys_pars + 3*i+1]
            g = data[data[...,self.n_sys_pars + 3*i+2] > 0][:, self.n_sys_pars + 3*i+2]

            minmax_osc[0, 0] = min(minmax_osc[0, 0], min_func(w0))
            minmax_osc[0, 1] = max(minmax_osc[0, 1], max_func(w0))
            minmax_osc[1, 0] = min(minmax_osc[1, 0], min_func(wp))
            minmax_osc[1, 1] = max(minmax_osc[1, 1], max_func(wp))
            minmax_osc[2, 0] = min(minmax_osc[2, 0], min_func(g))
            minmax_osc[2, 1] = max(minmax_osc[2, 1], max_func(g))

          dict_result = {
            self.sys_pars_names[i]: (minmax[i, 0], minmax[i, 1]) for i in range(self.n_sys_pars)
          }
          dict_result.update({
            'w0': (minmax_osc[0, 0], minmax_osc[0, 1]),
            'wp': (minmax_osc[1, 0], minmax_osc[1, 1]),
            'g': (minmax_osc[2, 0], minmax_osc[2, 1]),
          })

      elif isinstance(params, dict):
          raise Exception(f"Type {type(params)} not yet supported. Only (OpticalParameters).")

      elif isinstance(params, tuple):
          raise Exception(f"Type {type(params)} not yet supported. Only (OpticalParameters).")

      else:
          raise Exception(f"Type {type(params)} not supported. Only (OpticalParameters).")

      return dict_result

  def _check_and_return_min_max(self, min_max:Dict[str, Tuple[float]], user_values:Dict[str, Any]) -> None:
      used_ranges = self._get_optical_params(user_values, self.std_values_cases[self.last_case], 1)
      to_compare = self._get_min_max(used_ranges)

      for key, gen_ranges in min_max.items():
          sim_ranges = to_compare[key]
          assert gen_ranges[0] >= sim_ranges[0], f"[{key}] Generated min {gen_ranges[0]} not within limit {sim_ranges[0]}."
          assert gen_ranges[1] <= sim_ranges[1], f"[{key}] Generated max {gen_ranges[1]} not within limit {sim_ranges[1]}."

      return to_compare

  def _rewrite_p_case(self, p_gen) -> Dict[str, Any]:
    p_gen = p_gen.copy()
    if 'oscs_start' in p_gen:
      oscs_start = p_gen['oscs_start']
      p_gen['w0_before'] = oscs_start[0]
      p_gen['wp_before'] = oscs_start[1]
      p_gen['g_before'] = oscs_start[2]

    if 'oscs_end' in p_gen:
      oscs_end = p_gen['oscs_end']    #
      p_gen['w0_after'] = oscs_end[0] #
      p_gen['wp_after'] = oscs_end[1] #
      p_gen['g_after'] = oscs_end[2]  #

    return p_gen

  def _extract_min_max_short(self, params):
    to_return = {}
    for key, value in params.items():
      min_, max_ = np.inf, 0
      for n_layer in range(len(value)):
        if not isinstance(value[n_layer], list):
          min_ = min(min_, value[n_layer][0]) # [item]
          max_ = max(max_, value[n_layer][1]) # [item]
        else:
          for item in range(len(value[n_layer])):
            min_ = min(min_, value[n_layer][item][0])
            max_ = max(max_, value[n_layer][item][1])
      to_return[key] = (min_, max_)

    return to_return

  # --- CREATE FUNCTIONS ---
  def create(self, p_gen:Dict[str, Any], case:str='main', set_name:str='all', **kwargs) -> Dict[str, OpticalParameters]|OpticalParameters|None:
    return_data = kwargs.get('return_data', False)
    self.last_case = case

    self.n_sys_pars = p_gen['num_sys_params'] # Number of system parameters (th0, d, e_inf, drude_p, drude_g, w0_before, wp_before, g_before, w0_after, wp_after, g_after)

    if set_name == 'all':
      set_name = ['train', 'val', 'test']
    else:
      set_name = [set_name]

    p_gen = self._rewrite_p_case(p_gen) #

    for i_s, current_set in enumerate(set_name):
      optical_params = self.cases[case](p_gen)      # sample optical parameters
      self.parameters[current_set] = optical_params # assign optical parameters

      min_max = self._get_min_max(optical_params)   #
      ranges = self._extract_min_max_short({key: value for key, value in p_gen.items() if key in min_max.keys()})  # Remove w0, wp, g from ranges

      # for key in min_max.keys():

      self.parameters_ranges[current_set] = {
        'resolution': 50,
        'n_layers': range((optical_params.shape[2]-2)//3),
      }
      self.parameters_ranges[current_set].update(ranges)
      # ---

    if return_data:
      return self.parameters

  def copy(self) -> 'ParametersSetGenerator':
    """
    Create a copy of the ParametersSetGenerator instance.

    Returns:
        ParametersSetGenerator: A new ParametersSetGenerator instance with copied parameters and parameters_ranges.
    """
    new_instance = ParametersSetGenerator(
        self.case,
        self.data_mode,
        self.device,
        self.dtype
    )

    # Deep copy of parameters dictionary
    for key, value in self.parameters.items():
        if isinstance(value, OpticalParameters):
            new_instance.parameters[key] = value

    # Deep copy of parameters_ranges dictionary
    for key, value in self.parameters_ranges.items():
        new_instance.parameters_ranges[key] = {}
        for subkey, subvalue in value.items():
            new_instance.parameters_ranges[key][subkey] = subvalue

    # Copy other necessary attributes
    new_instance.last_case = self.last_case
    new_instance.sys_pars_names = self.sys_pars_names.copy() if hasattr(self, 'sys_pars_names') else None

    return new_instance

  # -- GET FUNCTIONS --
  def get_cases(self):
    return self.cases.keys()

  def get_parameters(self, set_name:str='all', **kwargs):
    if set_name == 'all': return self.parameters
    return self.parameters[set_name]

  def set_parameters(self, change:Dict[str, OpticalParameters], set_name:str='all', **kwargs):
    if set_name == 'all':
      set_name = list(self.parameters.keys())

    for set_name in self.parameters.keys():
      for param_key, param_value in change.items():
        self.parameters[set_name].set_parameter(param_key, param_value)

  def get_ranges(self, set_name:str='all', **kwargs) -> Dict[str, Tuple[float, float]|int|range]:
    if set_name == 'all':
      return self.parameters_ranges
    return self.parameters_ranges[set_name]

  # -- LOAD/SAVE FUNCTIONS --
  def save(self, filename:str, **kwargs):
    # EXCEL WRITER
    full_path = os.path.join(self.data_directory, filename)

    with pd.ExcelWriter(full_path) as writer:
      for key in self.parameters.keys():
        data_np = pt2np(self.parameters[key])               #
        data_np = data_np.reshape(-1, data_np.shape[-1])    # RESHAPE DATA TO 2D

        pd.DataFrame(data_np).to_excel(writer, sheet_name=key, index=False)

  def load(self, filename:str):
      full_path = os.path.join(self.data_directory, filename)

      if not os.path.exists(full_path):
          print(f"File {full_path} does not exist.")
          return

      wb = load_workbook(full_path)

      for sheet in wb.sheetnames:
          data_sheet = pd.read_excel(full_path, sheet_name=sheet)
          if self.data_mode == 'torch':
              data_sheet = torch.tensor(data_sheet.values, dtype=self.dtype, device=self.device)
          elif self.data_mode == 'numpy':
              data_sheet = data_sheet.to_numpy()
          else:
              raise Exception(f"Data mode {self.data_mode} not supported. Only (torch/numpy).")

          if data_sheet.ndim == 2:
              data_sheet = data_sheet.reshape(data_sheet.shape[0], 1, data_sheet.shape[1])

          self.parameters[sheet] = data_sheet
          min_max = self._get_min_max(data_sheet)

          self.parameters_ranges[sheet] = {
              'resolution': 50,
              'n_layers': range((data_sheet.shape[-1]-2)//3),
              'w0': 500, # (min_w0, max_w0),
              'd': min_max['d'],
              'e_inf': min_max['e_inf'],
              'drude': min_max['drude'],
              'wp': min_max['wp'],
              'g': min_max['g'],
          }

  def copy(self, p_generator: 'ParametersSetGenerator', params_updates: Dict[str, OpticalParameters]|None=None) -> None:
      self.parameters = {k: v.copy() for k, v in p_generator.parameters.items()}
      self.parameters_ranges = p_generator.parameters_ranges.copy()
      self.last_case = p_generator.last_case
      self.sys_pars_names = p_generator.sys_pars_names.copy()
      
      if params_updates is not None:
          self.set_parameters(params_updates)

  # -- PRINT FUNCTIONS --
  def print_parameters(self):
      print(f"Parameter generator: {','.join(list(self.parameters.keys()))}")
      for key in self.parameters.keys():
          if isinstance(self.parameters[key], OpticalParameters):
              print(f" - ({key}) {self.parameters[key].shape}")
              data_p = self.parameters[key].get_np()

              print(f"  - System parameters:")
              for i in range(data_p.shape[-1]):
                  if i < self.n_sys_pars:
                      print(f"   - {self.sys_pars_names[i]}: {data_p[..., i].min():.2f} - {data_p[..., i].max():.2f}")
              print(f"  - Oscillator parameters:")
              for i in range((data_p.shape[-1]-self.n_sys_pars)//3):
                  print(f"   - Oscillator {i}: w0: {data_p[..., self.n_sys_pars + 3*i].min():.2f} - {data_p[..., self.n_sys_pars + 3*i].max():.2f}, "
                        f"wp: {data_p[..., self.n_sys_pars + 3*i + 1].min():.2f} - {data_p[..., self.n_sys_pars + 3*i + 1].max():.2f}, "
                        f"g: {data_p[..., self.n_sys_pars + 3*i + 2].min():.2f} - {data_p[..., self.n_sys_pars + 3*i + 2].max():.2f}")

  def plot_parameters(self):
      for key, item in self.parameters.items():
          data_set = item
          n_items = self.n_sys_pars
          fig, ax = plt.subplots(n_items, 1, figsize=(10, 10))

          for i in range(n_items):
              # PLOT DISTRIBUTION
              ax[i].hist(pt2np(data_set[..., i]), bins=50)
              ax[i].set_title(f"Parameter {i}")
              ax[i].set_xlabel("Value")
              ax[i].set_ylabel("Frequency")

          plt.show()
      # -------------------

  def _print_cell_info(self, bbox, n_spaces:int=0):
    print(f"{' '*n_spaces}Bbox:")
    print(f"{' '*n_spaces} - has_obj: {bbox[0]:.2f}")
    print(f"{' '*n_spaces} - offset_center: {bbox[1]:.2f}")
    print(f"{' '*n_spaces} - width: {bbox[2]:.2f}")
    print(f"{' '*n_spaces} - iou_score: {bbox[3]:.2f}")
    print(f"{' '*n_spaces} - class_scores: {bbox[4:4+1]}")
    print(f"{' '*n_spaces} - optical_params: {bbox[4+1:]}")


class LabelGenerator():
    def __init__(
        self,
        name:str='',
        device:str='cpu',
        dtype:Any=torch.float32,
        verbose:bool=False
    ) -> None:
        self.name = name
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.tmm = TmMethodPyTorch(device=self.device, dtype=self.dtype, return_mode='all')
        self.wn:torch.Tensor

        self.threshold_value:float
        self.export_dir = str(LABELS_DIR)

    def __str__(self) -> str:
        return f"LabelGenerator({self.name},{self.device},{self.dtype})"

    def __call__(self, *args, **kwds):
        """
        Call the create_mapping function with the given parameters.
        """
        return self.create(*args, **kwds)

    def create_params(self, param_ranges:Dict, resolution:int):
        """

        """
        wp  = np.linspace(param_ranges['wp'][0], param_ranges['wp'][1], resolution)
        g   = np.linspace(param_ranges['g'][0], param_ranges['g'][1], resolution)
        w0  = np.linspace(param_ranges['w0'][0], param_ranges['w0'][1], resolution)

        return np.stack(np.meshgrid(w0, wp, g), -1).reshape(-1, 3)

    def create_mapping(self, opt_params:torch.Tensor|np.ndarray, prop_dir:str, **kwargs) -> Dict[str, pd.DataFrame]:
        """
          opt_params: nx3 tensor with [w0, wp, g]
        """
        self.threshold_value = threshold_value = kwargs.get('threshold_value', 0.25)
        batch_size = kwargs.get('batch_size', 128)
        wn = self.wn
        stepsize = wn[1] - wn[0]

        n_sets  = opt_params.shape[0]
        self.col_names = col_names = {
          'th0': 0,
          'd': 1,
          'e_inf': 2,
          'drude_p': 3,
          'drude_g': 4,
          'w0': 5,
          'wp': 6,
          'g': 7,
          'middle': 8,
          'width': 9,
          'height': 10
        }
        th0, d, eps_inf, drude_p, drude_g = 0, thickness, 1, 0, 0

        mapping_domain  = {
          "fd_t": torch.zeros((n_sets, len(col_names.keys())), dtype=self.dtype, device=self.device),
          "fd_r": torch.zeros((n_sets, len(col_names.keys())), dtype=self.dtype, device=self.device),
        }
        opt_params_pt = torch.tensor(opt_params, dtype=self.dtype, device=self.device)
        sys_params_pt = torch.tensor([th0, d, eps_inf, drude_p, drude_g], dtype=self.dtype, device=self.device).repeat(n_sets, 1)
        p:torch.Tensor = torch.concat((sys_params_pt, opt_params_pt), axis=1)
        p = p.unsqueeze(1)

        for batch_idx in range(0, n_sets, batch_size):
          if self.verbose: print(f"[{self}] - Mapping: {batch_idx}/{n_sets}")          #
          p_batch = p[batch_idx:batch_idx+batch_size]                 #
          if prop_dir == 'r': p_batch[...,1] = 1e30  # Set thickness d to infinity for reflection

          ((td_t, td_r), (fd_t, fd_r), (fd_t_real, fd_t_imag)) = self.tmm(p_batch, wn)    #

          min_, arg_min = torch.min(fd_t,dim=1,keepdim=True)  #
          max_, arg_max = torch.max(fd_r,dim=1,keepdim=True)  #
          slope_r = torch.gradient(fd_r, dim=1)[0]            #
          curvature_r = torch.gradient(slope_r, dim=1)[0]     #
          max_slope_r, arg_max_slope_r = torch.max(slope_r, dim=1, keepdim=True) #
          curvature_peaks, arg_curvature_peaks = torch.max(curvature_r, dim=1, keepdim=True)
          th_val_signal_t = (min_ + (1-min_)*threshold_value) #
          th_val_signal_r = (max_*threshold_value)            #

          middle_t  = self.wn[arg_min.squeeze()] # Do not forget offset
          middle_r  = self.wn[arg_max_slope_r.squeeze()] # Do not forget offset; argmin*stepsize does not include the beginnein
          width_t   = torch.abs(torch.argmin(torch.abs(fd_t-th_val_signal_t),1)*stepsize + wn[0] - middle_t) * 2
          width_r   = torch.abs(self.wn[arg_max_slope_r.squeeze()] - self.wn[arg_curvature_peaks.squeeze()])*2
          height_t  = 1 - min_.squeeze()
          height_r  = max_.squeeze()

          # bound middle_t/middle_r between [0, max(wn)-1]  #
          middle_t = torch.clamp(middle_t, min(wn)+1, max(wn)-1)  #
          middle_r = torch.clamp(middle_r, min(wn)+1, max(wn)-1)  #

          assert height_t.isnan().sum() == 0, "Nan values in mapping table."

          mapping_domain['fd_t'][batch_idx:batch_idx+batch_size, col_names['th0']]    = th0       #
          mapping_domain['fd_t'][batch_idx:batch_idx+batch_size, col_names['d']]      = d         #
          mapping_domain['fd_t'][batch_idx:batch_idx+batch_size, col_names['e_inf']]  = eps_inf   #
          mapping_domain['fd_t'][batch_idx:batch_idx+batch_size, col_names['w0']]     = opt_params_pt[batch_idx:batch_idx+batch_size,0] #
          mapping_domain['fd_t'][batch_idx:batch_idx+batch_size, col_names['wp']]     = opt_params_pt[batch_idx:batch_idx+batch_size,1] #
          mapping_domain['fd_t'][batch_idx:batch_idx+batch_size, col_names['g']]      = opt_params_pt[batch_idx:batch_idx+batch_size,2] #
          mapping_domain["fd_t"][batch_idx:batch_idx+batch_size, col_names['middle']] = middle_t  #
          mapping_domain["fd_t"][batch_idx:batch_idx+batch_size, col_names['width']]  = width_t   #
          mapping_domain["fd_t"][batch_idx:batch_idx+batch_size, col_names['height']] = height_t  #

          mapping_domain['fd_r'][batch_idx:batch_idx+batch_size, col_names['th0']]    = th0       #
          mapping_domain['fd_r'][batch_idx:batch_idx+batch_size, col_names['d']]      = d         #
          mapping_domain['fd_r'][batch_idx:batch_idx+batch_size, col_names['e_inf']]  = eps_inf   #
          mapping_domain['fd_r'][batch_idx:batch_idx+batch_size, col_names['w0']]     = opt_params_pt[batch_idx:batch_idx+batch_size,0] #
          mapping_domain['fd_r'][batch_idx:batch_idx+batch_size, col_names['wp']]     = opt_params_pt[batch_idx:batch_idx+batch_size,1] #
          mapping_domain['fd_r'][batch_idx:batch_idx+batch_size, col_names['g']]      = opt_params_pt[batch_idx:batch_idx+batch_size,2] #
          mapping_domain["fd_r"][batch_idx:batch_idx+batch_size, col_names['middle']] = middle_r  #
          mapping_domain["fd_r"][batch_idx:batch_idx+batch_size, col_names['width']]  = width_r   #
          mapping_domain["fd_r"][batch_idx:batch_idx+batch_size, col_names['height']] = height_r  #

        table_mapping = {}
        table_mapping['fd_t'] = pd.DataFrame(pt2np(mapping_domain['fd_t']), columns=list(col_names.keys()))
        table_mapping['fd_r'] = pd.DataFrame(pt2np(mapping_domain['fd_r']), columns=list(col_names.keys()))

        assert table_mapping['fd_t'].isna().sum().sum() == 0, "Nan values in mapping table."
        assert table_mapping['fd_r'].isna().sum().sum() == 0, "Nan values in mapping table."

        return table_mapping

    def create(self, param_ranges:Dict[str, Tuple[int]]|None, system_params:Dict[str, float|str], filename:str|None=None, export:bool=False, **kwargs):
      if self.verbose: print(f"[{self}] Generate mapping")
      self.wn = system_params.get('wn')
      self.resolution = kwargs.get('resolution', int(1e3**(1/3)))
      str_w0  = f"w0[{param_ranges['w0'][0]:.1f}_{param_ranges['w0'][1]:.1f}]"  #
      str_wp  = f"wp[{param_ranges['wp'][0]:.1f}_{param_ranges['wp'][1]:.1f}]"  #
      str_g   = f"g[{param_ranges['g'][0]:.1f}_{param_ranges['g'][1]:.1f}]"     #
      filename = filename or f'{self.name}_{str_w0}_{str_wp}_{str_g}.xlsx'      #

      # 1. Create params
      if self.verbose: print(f"[{self}] - Creating parameters. (resolution:{self.resolution})")  #
      opt_params = self.create_params(param_ranges, self.resolution)            #

      # 2. Create mapping
      if self.verbose: print(f"[{self}] - Creating mapping.")
      tables = self.create_mapping(opt_params, system_params['prop_dir'], **kwargs)

      # 3. Export mapping
      if export:
        if self.verbose: print(f"[{self}] - Exporting mapping: {filename}.")
        self.save(tables, filename)

      self.mapping = tables

      return filename

    def load(self, filename:str):
        full_path = path_join(self.export_dir, filename)

        if not os.path.exists(full_path):
            if self.verbose: print(f"File {full_path} does not exist.")
            return

        wb = load_workbook(full_path)
        mapping = {}
        for sheet in wb.sheetnames:
            mapping[sheet] = pd.read_excel(full_path, sheet_name=sheet)

        self.mapping = mapping

    def save(self, mapping:Dict[str, pd.DataFrame], filename:str):
        full_path = path_join(self.export_dir, filename)
        os.makedirs(self.export_dir, exist_ok=True)

        with pd.ExcelWriter(full_path) as writer:
            for key in mapping.keys():
                mapping[key].to_excel(writer, sheet_name=key)


class _NpzDict(dict):
  """Eagerly loaded npz contents with the `files` attribute of `np.lib.npyio.NpzFile`."""
  @property
  def files(self):
    return list(self)


def _load_npz(full_path:str):
  """np.load a dataset; without CUDA, tensors that were pickled on a GPU are mapped to the CPU."""
  data = np.load(full_path, allow_pickle=True)
  if torch.cuda.is_available():
    return data
  import io
  import torch.storage as torch_storage
  load_from_bytes = torch_storage._load_from_bytes
  torch_storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), weights_only=False, map_location='cpu')
  try:
    return _NpzDict({key: data[key] for key in data.files})
  finally:
    torch_storage._load_from_bytes = load_from_bytes


class LMIDataset(Dataset):
  def __init__(self, settype:str, device:str|torch.device=torch.device('cpu'), dtype:str|torch.dtype=torch.float32):
    #
    self.dtype  = dtype   # Data type (float32)
    self.device = device  # Device (cpu/gpu)

    # META DATA CLASS
    self.settype = settype      # Type of dataset (train|val|test)
    # self.systems = systems      # List of systems (e.g. ['sys_t', 'sys_r']|['sys1', 'sys2'])
    self.n = 0                  # Number of samples
    self.return_mode = 'data'   # In which modus to infer: Model(within initial guess)-Data(baseline)
    self.col_names_sys = {}     # Column names for system parameters
    self.y_return_mode = 'both'  # Mode for returning y: normal, noised, both

    self.data_tr:Dict[str, torch.Tensor] = {}           # True data for system parameters and oscillator parameters, as a single line
    self.data_x_sys_init:Dict[str, torch.Tensor] = {}   # Initial guess for system parameters
    self.data_x_sys_tr:Dict[str, torch.Tensor] = {}     # True data for system parameters
    self.data_x_osc_init:Dict[str, torch.Tensor] = {}   # Initial guess for oscillator parameters
    self.data_x_osc_tr:Dict[str, torch.Tensor] = {}     # True data for oscillator parameters
    self.data_y:Dict[str, torch.Tensor] = {}            # Labels for the dataset
    self.data_y_noised:Dict[str, torch.Tensor] = {}     # Noised labels for the dataset
    self.wn:Dict[str, torch.Tensor] = {}                # Wavenumbers for each system

    # Multi-scale detector (MSD)
    self.params_msd = {
      'columns_target': ['has_obj', 'w0', 'wp', 'g'], # Columns of the target
      # 'labels': {},         # Labels for the target [sample, scale, region, params] params=(has_obj, offset_m, bbox_w, iou_score, classes)-(w0,wp,g)
    }

    # META DATA DIRECTORY
    self.project_directory  = str(ROOT)
    self.data_directory     = str(DATASETS_DIR)
    self.labels_directory   = str(LABELS_DIR)

    # print(f"- Initialized dataset {self.settype}.")

  def __len__(self):
    self.n = self.data_y[list(self.data_y.keys())[0]].shape[0] if self.data_y else 0
    return self.n

  def __getitem__(self, idx) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], int]:
    """
    
    """

    to_return_wn = {}
    for key, value in self.wn.items():
      to_return_wn[key] = value # return single array

    to_return_x_sys_init = {}
    for key, value in self.data_x_sys_init.items():
      to_return_x_sys_init[key] = value[idx]
    
    to_return_x_sys_tr = {}
    for key, value in self.data_x_sys_tr.items():
      to_return_x_sys_tr[key] = value[idx]
    
    to_return_x_osc_init = {}
    for key, value in self.data_x_osc_init.items():
      to_return_x_osc_init[key] = value[idx]
    
    to_return_x_osc_tr = {}
    for key, value in self.data_x_osc_tr.items():
      to_return_x_osc_tr[key] = value[idx]

    if self.y_return_mode == 'normal':
      y = self.data_y
    elif self.y_return_mode == 'noised':
      y = {key: self.data_y_noised.get(key, self.data_y[key]) for key in self.data_y.keys()}
    elif self.y_return_mode == 'both':
      y = None

    to_return_y = {}
    if self.y_return_mode == 'both':
      for key in self.data_y.keys():
        to_return_y[key] = torch.stack((self.data_y[key][idx], self.data_y_noised.get(key, self.data_y[key])[idx]), dim=-1)
    else:
      for key, value in y.items():
        to_return_y[key] = value[idx]

    return (to_return_wn), (to_return_x_sys_init, to_return_x_sys_tr), (to_return_x_osc_init, to_return_x_osc_tr), to_return_y, idx

  def __str__(self) -> str:
    return f"CustomDataset(settype={self.settype}, n={self.n})"

  def init_tmm(self, system_params:Dict[str, Any]) -> None:
    self.system_dicts = system_params
    self.tmm = {}                                     # TMM method
    self.wn = {}                                      # Wavenumbers
    self.stepsize = {}                                # Stepsize
    self.anchors:Dict[str, torch.Tensor] = {}         # Anchors
    self.region_widths:Dict[str, torch.Tensor] = {}   # Region widths
    self.n_regions_per_scale:Dict[str, torch.Tensor] = {} # Number of regions per scale
    self.n_scales:Dict[str, int] = {}                  # Number of scales

    for system_name, system_params in system_params.items():
      wn_ = system_params.get('wn', None)                                       # Wavenumbers for each system
      self.wn[system_name] = np2pt(wn_, self.dtype, self.device)                # Get wavenumbers for each system
      self.stepsize[system_name] = wn_[1] - wn_[0]                    # Get stepsize for each system
      wn_diff = wn_[-1] - wn_[0]

      if system_params.get('model') == 'SSD':
        self.anchors[system_name] = anchors = np2pt(system_params.get('anchors'), self.dtype, self.device) # Anchors for each system
        self.region_widths[system_name] = rw = system_params.get('region_width')  # Region widths for each system
        self.n_regions_per_scale[system_name] = [int(np.ceil(wn_diff.item() / i)) for i in rw]
        self.n_scales[system_name] = len(rw)

        assert len(anchors) == len(rw), f"Number of anchors ({len(anchors)}) should be equal to the number of region widths ({len(self.region_widths[system_name])})."

      self.tmm[system_name] = TmMethodPyTorch(device=self.device, dtype=self.dtype, return_mode=system_params['return_mode'])

    self.sys_var = system_params

  # -- ANALYSE FUNCTIONS: PRINT/PLOT --
  def info(self):
    if self.wn is None:
      print(f"Initialize wn first.")
      return

    print(f"Information about dataset.")
    print(f" - Total number of samples: {self.n}.")
    print(f" - Systems: {list(self.system_dicts.keys())}.")

    # assert len(self.columns_target) == self.label_t.shape[-1] == self.label_r.shape[-1], f"Number of columns in label ({len(self.columns_target)}) should be equal to the number of columns in label ({self.label.shape[-1]})."
  
  def get_system_shapes(self) -> Dict[str, Dict[str, Tuple[int]]]:
    """
      Return {system_name: {'sys': shape_sys, 'osc': shape_osc}}
    """
    shapes = {
      'sys': {key: value.shape for key, value in self.data_x_sys_tr.items()},
      'osc': {key: value.shape for key, value in self.data_x_osc_tr.items()},
      'y': {key: list(value.shape) for key, value in self.data_y.items()},
    }

    return shapes

  # -----------------------------------
  # -- LOAD/SAVE FUNCTIONS --
  def save_dataset(self, filename:str):
    os.makedirs(self.data_directory, exist_ok=True)
    os.makedirs(os.path.dirname(f"{self.data_directory}/{filename}"), exist_ok=True)  # Ensure directory exists
    
    def convert_dict(data:tuple|torch.Tensor, string:str):
      data_dict = {}
      for key, value in data.items():
        if isinstance(value, (torch.Tensor, int, float, np.ndarray, list)):
          data_dict[f'{string}_{key}'] = pt2np(value)
        elif isinstance(value, tuple):
          for i, v in enumerate(value):
            data_dict[f'{string}_{key}__{i}'] = pt2np(v)
      return data_dict
    
    save_dict = {
      'dtype': self.dtype,        # 
      'device': str(self.device), # 
      'n': self.n,                # Number of samples in the dataset
      'n_layers': self.n_layers,	# 
      'col_names_sys': self.col_names_sys, # Column names for system parameters
      'wgn_factor': self.wgn_factor,			# Factor for white Gaussian noise
      'y_return_mode': self.y_return_mode,	# Mode in which the labels are returned (normal, noised, both)
      'return_mode': self.return_mode,		# Mode in which the data is returned (data, model, both)
      'params_msd': self.params_msd,			# Parameters for the multi-scale detector
      'system_dicts': self.system_dicts,	# System parameters
    }

    # Convert dictionaries to saveable format
    data_x_sys_init = convert_dict(self.data_x_sys_init, 'data_x_sys_init')	# 
    data_x_sys_tr = convert_dict(self.data_x_sys_tr, 'data_x_sys_tr')				# 
    data_x_osc_init = convert_dict(self.data_x_osc_init, 'data_x_osc_init')	# 
    data_x_osc_tr = convert_dict(self.data_x_osc_tr, 'data_x_osc_tr')				# 
    
    data_y = convert_dict(self.data_y, 'data_y')    # 
    data_y_noised = convert_dict(self.data_y_noised, 'data_y_noised')  #
    data_tr = convert_dict(self.data_tr, 'data_tr') # 
    
    wn = convert_dict(self.wn, 'wn')
    stepsize = convert_dict(self.stepsize, 'stepsize')
    anchors = convert_dict(self.anchors, 'anchors')
    region_widths = convert_dict(self.region_widths, 'region_widths')
    n_regions_per_scale = convert_dict(self.n_regions_per_scale, 'n_regions_per_scale')
    n_scales = convert_dict(self.n_scales, 'n_scales')
    
    save_dict.update({
      **data_x_sys_init,
      **data_x_osc_init,
      **data_x_osc_tr,
      **data_x_sys_tr,
      **data_y,
      **data_y_noised,
      **data_tr,
      **wn,
      **stepsize,
      **anchors,
      **region_widths,
      **n_regions_per_scale,
      **n_scales,
      
    })

    # Save to file
    np.savez(f"{self.data_directory}/{filename}.npz", **save_dict)
    print(f"Dataset saved to {self.data_directory}/{filename}.npz")
  
  def load_dataset(self, filename:str, data_dir:str|None=None):
    if data_dir is not None:
      data_dir = data_dir
    else:
      data_dir = self.data_directory

    full_path = f"{data_dir}/{filename}.npz"
    # print(f"Trying to load: {full_path}")
    data = _load_npz(full_path)
    
    self.dtype = data['dtype'].item()
    self.n = data['n'].item()
    self.n_layers = data['n_layers'].item()
    self.col_names_sys = data['col_names_sys'].item()
    self.wgn_factor = data['wgn_factor'].item()
    self.y_return_mode = data['y_return_mode'].item()
    self.return_mode = data['return_mode'].item()
    self.params_msd = data['params_msd'].item()
    self.system_dicts = data['system_dicts'].item()
    self.data_x_sys_init = {}
    self.data_x_osc_init = {}
    self.data_x_osc_tr = {}
    self.data_x_sys_tr = {}
    self.data_y = {}
    self.data_y_noised = {}
    self.data_tr = {}
    
    self.wn = {}
    self.stepsize = {}
    self.anchors = {}
    self.region_widths = {}
    self.n_regions_per_scale = {}
    self.n_scales = {}
    
    key:str

    for key in data.files:
      if key.startswith('data_x_sys_init_'):
        system_name = key[len('data_x_sys_init_'):]  # Extract system name from key
        self.data_x_sys_init[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('data_x_osc_init_'):
        system_name = key[len('data_x_osc_init_'):]  # Extract system name from key
        self.data_x_osc_init[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('data_x_osc_tr_'):
        system_name = key[len('data_x_osc_tr_'):]  # Extract system name from key
        self.data_x_osc_tr[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('data_x_sys_tr_'):
        system_name = key[len('data_x_sys_tr_'):]  # Extract system name from key
        self.data_x_sys_tr[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('data_y_') and not key.startswith('data_y_noised_'):
        system_name = key[len('data_y_'):]  # Extract system name from key
        self.data_y[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('data_y_noised_'):
        system_name = key[len('data_y_noised_'):]  # Extract system name from key
        self.data_y_noised[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('data_tr_'):
        system_name = key[len('data_tr_'):]  # Extract system name from key
        self.data_tr[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('wn_'):
        system_name = key[len('wn_'):]  # Extract system name from key
        self.wn[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('stepsize_'):
        system_name = key.split('_')[1]
        self.stepsize[system_name] = data[key].item()
      elif key.startswith('anchors_'):
        system_name = key.split('_')[1]
        self.anchors[system_name] = np2pt(data[key], self.dtype, self.device)
      elif key.startswith('region_widths_'):
        system_name = key[len('region_widths_'):]
        self.region_widths[system_name] = data[key]
      elif key.startswith('n_regions_per_scale_'):
        system_name = key[len('n_regions_per_scale_'):]  # Extract system name from key
        self.n_regions_per_scale[system_name] = data[key]
      elif key.startswith('n_scales_'):
        system_name = key[len('n_scales_'):]  # Extract system name from key
        self.n_scales[system_name] = data[key]
    
    self.tmm = {}
    for system_name, system_params in self.system_dicts.items():
      tmm = TmMethodPyTorch(
        device=self.device,
        dtype=self.dtype,
        dtc=torch.complex64,
        return_mode=system_params.get('return_mode', 'data')
    )
      self.tmm[system_name] = tmm # Initialize TMM method for each system
  
  def purge_ds(self, systems:List[str]):
    list_keys = list(self.wn.keys())
    for key in list_keys:
      if key not in systems:
        if key in self.data_y.keys(): del self.data_y[key]                      # Remove spectra
        if key in self.data_x_osc_init.keys(): del self.data_x_osc_init[key]    # Remove oscillator initialization data
        if key in self.data_x_osc_tr.keys(): del self.data_x_osc_tr[key]        # Remove oscillator true data
        if key in self.data_x_sys_init.keys(): del self.data_x_sys_init[key]    # Remove system initialization data
        if key in self.data_x_sys_tr.keys(): del self.data_x_sys_tr[key]        # Remove system true data
        if key in self.wn.keys(): del self.wn[key]                              # Remove wavenumbers

  # -- PLOT --
  def get_info_for_model(self, print_info:bool=False) -> Dict[str, Any]:
    model_info = {
      'n_samples': self.n,
      'n_layers': self.n_layers,
      'n_scales': self.n_scales,
      'n_regions_per_scale': self.n_regions_per_scale,
      'n_regions_per_scale_max': {k: max(v) for k, v in self.n_regions_per_scale.items()},
      'n_attribs_label': len(self.params_msd['columns_target']),
      'n_system_parameters': n_system_parameters,
      'anchors': self.anchors,
      'region_width': self.region_widths,
      'system_dicts': self.system_dicts,
      'system_function': self.tmm,
      'x_len': {k: len(v) for k, v in self.wn.items()},
      'wn_shape': {k: v.shape for k, v in self.wn.items()},
      'data_y_shape': {k: v.shape for k, v in self.data_y.items()},
      'data_tr_shape': {k: v.shape for k, v in self.data_tr.items()},
    }

    if print_info:
      print("Model Information:")
      for key, value in model_info.items():
        if isinstance(value, dict):
          print(f" - {key}:")
          for sub_key, sub_value in value.items():
            if isinstance(sub_value, dict):
              print(f"   - {sub_key}:")
              for sub_sub_key, sub_sub_value in sub_value.items():
                if isinstance(sub_sub_value, torch.Tensor):
                  print(f"     - {sub_sub_key}: {sub_sub_value.shape}")
                else:
                  print(f"     - {sub_sub_key}: {sub_sub_value}")
            else:
              print(f"   - {sub_key}: {sub_value}")
        else:
          print(f" - {key}: {value}")

    return model_info

  def sample_info(self, idxs:int|List[int], print_info:bool=False, plot_info:bool=False, **kwargs):
    """
    print_info: If True, print information about the sample.
    plot_info:  If True, plot information about the sample.
    multi_fig:  If True, plot each region in a separate figure. If False, plot
                all regions in a single figure.
    """
    show_plots = kwargs.get("show_plots", True)   #
    save_plots = kwargs.get("save_plots", False)  #
    multi_fig = kwargs.get("multi_fig", False)    # Multi-figure mode: each region in separate figure

    plot_anchors = kwargs.get("plot_anchors", True)
    plot_regions = kwargs.get("plot_regions", True)
    bounding_boxes = kwargs.get("bounding_boxes", True)

    if isinstance(idxs, int):
      idxs = [idxs]
    
    # idxs = [i for i in idxs if i < list(self.n.values())[0]]  # TODO: proper handling of different system Ensure idxs are within range
    idxs = [i for i in idxs if i < len(self)]  # TODO: proper handling of different system Ensure idxs are within range

    for idx in idxs:
      # 1. Get sample, signal, label
      # (wn, data_x, data_y_td, data_y_fd, data_yfd_t), (data_tr, label), idx = sample
      # (wn, data_x, data_y_fd_r, data_y_fd_t_real, data_y_fd_t_imag), (data_tr, (label_t, label_r)), idx = sample
      sample = self[idx]          #
      wn, x_sys, x_osc, y, idx = sample  #
      x_osc_init, x_osc_tr = x_osc

      # 2. Get bbox information
      # n_regions = self.anchors.shape[0]
      matches = {key: get_matches(x_osc_tr[key]) for key in x_osc_tr.keys()}

      str_oscs = ""

      if print_info:
        print(f"Information of sample {idx:03d}:")
        for system, data_tr in self.data_tr.items():
          print(f" - System {system}:")
          print(f"  - Model: {self.system_dicts[system].get('model', 'unknown')}")
          p = data_tr[idx]
          print(f"  - Layers {p.shape[0]}:")
          str_oscs += f"  - Layers {p[0]}:\n"
          for i_layer in range(p.shape[0]):
            print(f"   - th0: {p[i_layer,0]:.1f}, d: {p[i_layer,1]:.3f}, ε: {p[i_layer,2]:.3f}")
            str_oscs += f"   - th0: {p[i_layer,0]:.1f}, d: {p[i_layer,1]:.3f}, ε: {p[i_layer,2]:.3f}\n"
            p_layer = p[i_layer, n_system_parameters:]
            for n_oscs in range(len(p_layer)//3):
              print(f"   - Oscillator: ω0: {p_layer[n_oscs*3]:05.1f}, ωp: {p_layer[n_oscs*3+1]:05.2f}, δ: {p_layer[n_oscs*3+2]:05.2f}")
              str_oscs += f"   - Oscillator: ω0: {p_layer[n_oscs*3]:05.1f}, ωp: {p_layer[n_oscs*3+1]:05.2f}, δ: {p_layer[n_oscs*3+2]:05.2f}\n"

            if self.system_dicts[system].get('model') == 'SSD' and bounding_boxes:
              print(f"  - Bounding boxes {system}:")
              for match_ in matches[system]:
                reg_idx = match_[0]
                print(f"  - Region {reg_idx}:")
                bbox_ = pt2np(x_osc_tr[system][match_[0], match_[1], match_[2]])
                self._print_cell_info(bbox_, n_spaces=2)
      
      if plot_info:
        # CREATE FIGURES
        n_systems = len(self.system_dicts)  # Number of systems
        n_y = len(self.data_y)							# Number of output signals
        n_cols_max = int(max(self.n_scales.values())) if self.n_scales else 1  # Max number of scales
        fig: plt.Figure
        ax: plt.Axes|List[plt.Axes]
        plt.ioff()

        fig = plt.figure(layout="constrained", dpi=300, figsize=(12*n_cols_max, 8*n_systems))
        gs = GridSpec(n_y, n_cols_max, figure=fig)
        row_count = 0
        
        # if not multi_fig:
          # fig, ax = plt.subplots(n_y, 1, figsize=(15, 20), dpi=200)
          # if n_systems == 1 and n_y == 1: 
          #   ax = [ax]
        # else:
        #   lst_figs = [
        #     plt.subplots(1, 1, figsize=(20, 20), dpi=200) for _ in range(n_systems)
        #   ]
        # --

        for system_idx, system_name in enumerate(self.system_dicts.keys()):
          wn = self.wn[system_name]  # Get the wavenumbers for the given idx
          n_scales_current_system = self.n_scales.get(system_name, 1)
            
          for y_count, y_name in enumerate([x for x in self.data_y.keys() if system_name in x]):
            # Each row: y of system
            if n_scales_current_system > 1:
              ax = []
              for scale_idx in range(n_scales_current_system):
                ax_scale = fig.add_subplot(gs[row_count, scale_idx])
                ax.append(ax_scale)
            else:
              ax = [fig.add_subplot(gs[row_count, :])]
            
            y_ = y[y_name]
            
            for scale_idx in range(n_scales_current_system):
              # Each column: scale
              if n_scales_current_system > 1:
                ax[scale_idx].set_title(f"System: {system_name} | Scale {scale_idx+1}/{n_scales_current_system} | Sample {idx:03d}")
              else:
                ax[scale_idx].set_title(f"System: {system_name} | Sample {idx:03d}")
                
              if self.y_return_mode == 'both':
                ax[scale_idx].plot(pt2np(wn), pt2np(y_[..., 0]), color='blue', linestyle='-', marker='o')    # Plot the signal
                ax[scale_idx].plot(pt2np(wn), pt2np(y_[..., 1]), color='orange', linestyle='-', marker='o')  # Plot the signal
              else:
                ax[scale_idx].plot(pt2np(wn), pt2np(y_), color='blue', linestyle='-', marker='o')
            
              if self.system_dicts[system_name].get('model') == 'SSD' and plot_regions:
                # Plot anchors
                # if plot_anchors:
                #   for anchor in self.anchors[system_name]:
                #     ax[system_idx+y_count].axvline(x=anchor, color='gray', linestyle='--', alpha=0.5)

                # Plot regions
                region_width = self.region_widths[system_name][scale_idx]
                n_regions = self.n_regions_per_scale[system_name][scale_idx]
                region_centers = [i * region_width + region_width/2 for i in range(n_regions)]
                for center in region_centers:
                  left = center - region_width / 2
                  right = center + region_width / 2
                  if plot_regions:
                    ax[scale_idx].axvspan(left, right, color='gray', alpha=0.00)
                    ax[scale_idx].axvline(x=left, color='gray', linestyle='--', alpha=0.5)
                    ax[scale_idx].axvline(x=right, color='gray', linestyle='--', alpha=0.5)
                    ax[scale_idx].axvline(x=center, color='gray', linestyle='-', alpha=0.5)
                  if plot_anchors:
                    anchor = self.anchors[system_name][scale_idx].item()/2
                    ax[scale_idx].axvspan(center-anchor, center+anchor, ymin=0.25, ymax=0.75, color='gray', linestyle='--', alpha=0.5)

                  if bounding_boxes:
                    # Plot bounding boxes TODO: Implement layer for bounding boxes
                    for match_ in matches[system_name]:
                      if match_[1] == scale_idx: # If scale matches
                        bbox = x_osc_tr[system_name][match_[0], match_[1], match_[2]]
                        has_obj, offset_m, bbox_w, iou_score = pt2np(bbox[:4])
                        opt_params = pt2np(bbox[4:])

                        bbox_left, bbox_right = offset_m - bbox_w / 2, offset_m + bbox_w / 2
                        ax[scale_idx].vlines(x=offset_m, ymin=0, ymax=1, color='red', linewidth=2, linestyles='-')
                        ax[scale_idx].vlines(x=[bbox_left, bbox_right], ymin=0, ymax=1, color='red', linewidth=2, linestyles='--')

                        # Annotate with optical parameters
                        # annotation = f"w0:{opt_params[0]:.1f}\nwp:{opt_params[1]:.2f}\ng:{opt_params[2]:.2f}\nIoU:{iou_score:.2f}"
                        # if plot_regions:
                        #   ax[system_idx+y_count].text((bbox_left + bbox_right) / 2, 1.16, annotation, color='red', fontsize=8, ha='center')

              ax[scale_idx].set(
                xlim=(pt2np(wn[0]), pt2np(wn[-1])),
              )
              ax[scale_idx].grid(True)

            row_count += 1
        
        if save_plots:
          os.makedirs(f"{OUTPUTS_DIR}/sample_info/{self.settype}", exist_ok=True)
          print(f" - Save plot: {OUTPUTS_DIR}/sample_info/{self.settype}/s_{idx:03d}.png")
          plt.savefig(f"{OUTPUTS_DIR}/sample_info/{self.settype}/s_{idx:03d}.png")

        if show_plots:
          plt.ion()
          plt.show()

        plt.close()
      # -------------------

  def _print_cell_info(self, bbox, n_spaces:int=0):
    print(f"{' '*n_spaces}Bbox:")
    print(f"{' '*n_spaces} - has_obj: {bbox[0]:.2f}")
    print(f"{' '*n_spaces} - offset_center: {bbox[1]:.2f}")
    print(f"{' '*n_spaces} - width: {bbox[2]:.2f}")
    print(f"{' '*n_spaces} - iou_score: {bbox[3]:.2f}")
    # print(f"{' '*n_spaces} - class_scores: {bbox[4:4+1]}")
    # print(f"{' '*n_spaces} - optical_params: {bbox[4+1:]}")


class SimulationDatasetBatch(LMIDataset):
  """

  """
  def __init__(self, settype:str, device:str, main_domain:str) -> None:
    super().__init__(settype=settype, device=device)
    self.dielectric_parameters:dict
    self.relu = nn.ReLU()  # ReLU activation function

  # --- NOISE FUNCTIONS ---
  def _noiser(self, size:tuple, noise_type:str='normal', noise_param:float=0.1) -> torch.Tensor:
    """
    Adds noise to data based on the specified noise type and parameters.

    Args:
        size (tuple): Size of the noise tensor.
        noise_type (str): Type of noise ('normal' or 'uniform').
        noise_param (torch.Tensor): Noise parameter.

    Returns:
        torch.Tensor: Noisy data tensor.
    """
    if noise_type == 'normal':
      # Add noise to data by using a normal distribution with mean value_param and std noise_param
      noise = torch.normal(mean=1.0, std=noise_param, size=size, dtype=self.dtype, device=self.device)
    # elif noise_type == 'uniform':
      # Add noise to data by using a uniform distribution with a factor of noise_param
      # noise = (torch.rand_like(self.data) - 0.5) * noise_param
    else:
      raise Exception(f"Noise type {noise_type} not supported. Only (normal/uniform).")

    return noise

  @staticmethod
  def _create_oscs_mask(n_samples, n_oscs, p, dtype:torch.dtype, device:str='cpu') -> torch.Tensor:
    """

    """
    a       = torch.empty(n_samples, n_oscs, dtype=dtype, device=device).uniform_(p, 1)
    b_dist  = torch.bernoulli(a)
    mask    = torch.kron(b_dist, torch.ones((1, 3), dtype=dtype, device=device))
    ones    = torch.ones(n_samples, 2, dtype=dtype, device=device)
    mask    = torch.cat((ones, mask), dim=1)
    return mask

  def _generate_normal(self, min_, max_, mean, std, n):
    min_, max_ = torch.tensor(min_, dtype=self.dtype, device=self.device), torch.tensor(max_, dtype=self.dtype, device=self.device)
    return torch.clip(torch.normal(mean=mean, std=std, size=(n,), dtype=self.dtype, device=self.device), min_, max_)

  def _generate_pink_noise(self, mean, std, size):
    d = self.wn[1] - self.wn[0]  # Sampling interval
    wgn = torch.normal(mean=mean, std=std, size=size, dtype=self.dtype, device=self.device)
    wgn_fft = torch.fft.rfft(wgn)  # Perform FFT to convert to frequency domain
    freqs = torch.fft.rfftfreq(size[1], d=d)  # Frequency bins
    pink_noise = torch.zeros(len(freqs), dtype=self.dtype, device=self.device)
    pink_noise[1:] = 1/torch.sqrt(freqs[1:])  # Apply pink noise filter (1/f)
    pink_noise_fft = wgn_fft * pink_noise  # Apply filter in frequency domain
    pink_noise = torch.fft.irfft(pink_noise_fft)  # Convert back to time domain
    return pink_noise

  def _generate_uniform(self, min_, max_, n):
    return (min_ - max_) * torch.rand(n, dtype=self.dtype, device=self.device) + max_

  # --- LABELING FUNCTIONS ---
  def _iou_shape(
    self, 
    w1:torch.Tensor, 
    w2:torch.Tensor
  ) -> torch.Tensor:
    left = torch.max(-w1/2, -w2/2)
    right = torch.min(w1/2, w2/2)
    intersection = self.relu(right-left)
    union = w1 + w2 - intersection
    iou = intersection / union
    return iou

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
    targets:List[torch.Tensor],
    wn:torch.Tensor
  ):
    """
    Maps a bounding box to the appropriate anchor based on Intersection Over Union (IOU) with predefined anchors.

    Args:
      bbox (np.ndarray): A bounding box defined in terms of wavenumbers, where bbox[0] represents the center position and bbox[1] represents the width.
      params (np.ndarray): Additional parameters associated with the bounding box, such as class labels or other attributes. Given as"
          [class] + [w0, wp, g], where [class] can be a list of classes.
      sample_id (int): An identifier for the sample within the batch, used to index into the targets array.
      targets (List[np.ndarray]): A list of target arrays where each element corresponds to a different scale of anchor boxes. Each target array has dimensions corresponding to (batch_size, num_cells, num_attributes).

    Returns:
      List[np.ndarray]: The updated targets array with the bounding box mapped to the appropriate anchor.

    """
    middle, width = bbox
    iou_anchors = self._iou_shape(width, anchors) # iou_scores

    if osc_id is not None and self.forced_scale[systemname][layer_id][osc_id] >= 0:
      iou_anchors[self.forced_scale[systemname][layer_id][osc_id]] = torch.max(iou_anchors) + 0.1 # Use only the forced scale
    sorted_anchors = torch.flipud(torch.argsort(iou_anchors)) # Sorted in descending order
    pt_one = torch.tensor(1, dtype=self.dtype, device=self.device)

    find_idx_closest = lambda x: torch.argmin(torch.abs(wn - x)) # Function to find the index of the closest wavenumber

    for anchor_idx in sorted_anchors:
      scale = anchor_idx
      n_regs_of_scale = int(torch.ceil((wn[-1]-wn[0])/region_sizes[scale]))
      window_length = len(wn) / n_regs_of_scale # Get the index of the scale of the anchor.
      idx_ = find_idx_closest(middle) # Find the index of the closest wavenumber to the middle of the bounding box
      region = int(idx_ / window_length) # (middle - self.wn[0]) # Get the index of the region from that specific scale. There could be an offset; therefore, we apply the subtraction
      anchor_taken = targets[sample_id, layer_id, scale, region, 0] # Check if anchor is taken.

      if not anchor_taken:
        targets[sample_id, layer_id, scale, region, 0] = pt_one     # Attribute: has_obj
        targets[sample_id, layer_id, scale, region, 1:] = params    # Attribute: classes + (w0,wp,g)
        return targets, True

    print(f"Anchor not found ({sample_id}/{layer_id}/{osc_id}, {sorted_anchors}, {bbox}).")
    return targets, False

  def sample_parameters(self, sys_var:Dict[str, np.ndarray]) -> torch.Tensor:
      """
      Generates parameters for the simulation based on the provided settings.

      Args:
          p: Simulation parameters.

      Returns:
          torch.Tensor: Simulated parameters tensor.
      """
      if 'parameters' in sys_var.keys():
        # Parameters are predefined
        sys_var['n_samples'] = sys_var['parameters'].shape[0]
        sys_var['n_osc_max'] = (sys_var['parameters'].shape[1] - 2) // 3
        return np2pt(sys_var['parameters'], self.dtype, self.device), sys_var

      # Sample d, eps_inf from distribution (uniform/normal)
      n_samples           = sys_var['n_samples']
      n_oscs_max          = sys_var['n_osc_max']
      simulated_params    = torch.ones((n_samples, 2+3*n_oscs_max), dtype=self.dtype, device=self.device)

      simulated_params[:, 0] = self._generate_uniform(min(sys_var['d']), max(sys_var['d']), n_samples)            # [d, _, _, _, _]
      simulated_params[:, 1] = self._generate_uniform(min(sys_var['e_inf']), max(sys_var['e_inf']), n_samples)    # [_, e_inf, _, _, _]

      for i in range(n_oscs_max):
          # Sample w0 from distribution (uniform/normal)
          simulated_params[:, 2 + 3*i] = self._generate_uniform(min(sys_var['w0']), max(sys_var['w0']), n_samples)

          # Sample wp from distribution (uniform/normal)
          S = self._generate_uniform(min(sys_var['S']), max(sys_var['S']), n_samples)
          simulated_params[:, 3 + 3*i] = torch.sqrt(simulated_params[:, 2 + 3*i]**2 * S) # [_, _, _, Wp, _]

          # Sample G from distribution (uniform/normal)
          G_Wo = self._generate_uniform(min(sys_var['G/Wo']), max(sys_var['G/Wo']), n_samples)
          simulated_params[:, 4 + 3*i] = G_Wo * simulated_params[:, 2 + 3*i] # [_, _, _, _, G]

      mask = self._create_oscs_mask(n_samples, n_oscs_max, sys_var['p_oscs'], self.dtype, self.device)
      simulated_params = simulated_params * mask

      return simulated_params, sys_var

  def assign_parameters(
    self,
    parameter_class:Dict[str, ParametersSetGenerator],
    system_dicts:Dict[str, Any],
    **kwargs
  ):
    """

    """

    for systemname, systemclass in parameter_class.items():
      data_tr = systemclass.parameters[self.settype].data 				# True parameters of system
      data_tr_sys = systemclass.parameters[self.settype].data_sys # True parameters of system
      self.col_names_sys.update({systemname: list(systemclass.parameters[self.settype].key_positions.keys())}) # Column names of system parameters
      p_init = system_dicts[systemname]['p_init'] # Initial parameters
      n_samples, n_layers, n_tot = data_tr.shape  #
      self.data_tr[systemname] = data_tr          # 

      self.data_x_sys_init[systemname] = p_init[None,None,:].repeat(n_samples, n_layers, 1)
      self.data_x_sys_tr[systemname] = data_tr_sys

      assert self.data_x_sys_init[systemname].ndim == 3, f"Data should have 3 dimensions [nxlxa]."
      assert self.data_x_sys_init[systemname].shape == self.data_x_sys_tr[systemname].shape, \
        f"Data should be equal for init and training. {self.data_x_sys_init[systemname].shape} != {self.data_x_sys_tr[systemname].shape}"

  def assign_oscs_mapping(self, label_classes:Dict[str, LabelGenerator|str], **kwargs):
    self.oscs_mapping = {}
    forced_scale = kwargs.get("forced_scale", {})

    for systemname, label_class in label_classes.items():
      mapping_system = {}
      if isinstance(label_class, str):
        label_gen = LabelGenerator(self.device, self.settype)
        label_gen.load(label_class)
        mapping_system = label_gen.mapping
      elif isinstance(label_class, LabelGenerator):
        for key, value in label_class.mapping.items():
          mapping_system[key] = value
      else:
        raise Exception(f"Label class {label_class} not supported. Only (LabelGenerator|str).")

      self.oscs_mapping[systemname] = mapping_system
      
      if (forced_scale.get(systemname) is None) and (self.system_dicts[systemname]['model'] == 'SSD'):
        forced_scale[systemname] = -1 * np.ones((self.n_layers, self.n_scales[systemname]), dtype=int)

    self.forced_scale = forced_scale
    # -----------------------------------

  def _add_noise(self, signal:torch.Tensor, model_name:str, **kwargs) -> torch.Tensor:
    time_start = time.time()
    wgn_factor = kwargs.get('wgn_factor', 0.01) 
    wn = kwargs.get('wn', None)
    
    system_name = [key for key in self.system_dicts.keys() if key in model_name][0]
    struct_noise = self.structured_noise.get(system_name, False)

    if wgn_factor == 0 and not struct_noise: return signal
    
    y_noise = signal
    signal_abs = torch.abs(signal)

    if wgn_factor > 0:
      mean_ = torch.mean(signal_abs, axis=1, keepdim=True)
      y_noise = y_noise + torch.normal(0, std=(mean_*wgn_factor).expand(signal.shape[0], signal.shape[-1])) # Add white Gaussian noise -> per-sample, per-wavenumber  

    # Add structured noise
    if struct_noise and wn is not None and usecase == 1:
      pass
    elif struct_noise and wn is not None and usecase == 2:
      if system_name == 'system2':
        noise_points = torch.arange(wn[0], wn[-1], 1, dtype=self.dtype, device=self.device) 
        n_samples = signal.shape[0]
        noise_path = measured_file('STO', 'difference_data2.xlsx')
        noise_profile, wn_mask = generate_noise_profile_uc2(noise_path, wn, noise_points, n_samples, self.dtype, self.device)
        
        noise_profile_power = torch.mean(noise_profile[model_name][:, wn_mask]**2,axis=1)
        signal_power = torch.mean(signal[:, wn_mask]**2,axis=1)
        noise_profile = noise_profile[model_name] * torch.sqrt(signal_power / noise_profile_power)[:,None] * torch.rand(n_samples, dtype=self.dtype, device=self.device)[:, None]
        y_noise = y_noise + noise_profile
      if system_name == 'system1':
        noise_path = measured_file('STO', 'difference_refl.xlsx')
        wn_mask = (wn >= 200) & (wn <= 1000)

        noise_profile = generate_noise_profile_uc2_sys1(noise_path, wn, self.dtype, self.device)
        
        rand_scaling_factor = torch.rand(signal.shape[0], dtype=self.dtype, device=self.device)
        noise_profile_scaled = (noise_profile * rand_scaling_factor[:, None])
        
        y_noise = y_noise + noise_profile_scaled        
    elif struct_noise and wn is not None and usecase == 3:
      noise_points = torch.arange(wn[0], wn[-1], 1, dtype=self.dtype, device=self.device) 
      n_samples = signal.shape[0]
      noise_path = str(EXPERIMENTAL_DIR / 'metastructure' / 'difference_data2.xlsx')
      noise_profile, wn_mask = generate_noise_profile_uc3(noise_path, wn, noise_points, n_samples, self.dtype, self.device)
      
      # Noiseprofile is relative to the signal amplitude
      # TODO: Implement mask
      y_noise = y_noise + noise_profile[model_name]
      
    time_end = time.time()
    print(f"[{model_name}] Noise added ({time_end-time_start:.2f} s).")
    return y_noise
    
  def generate_spectra(self, p_model:torch.Tensor, sys_func, **kwargs) -> torch.Tensor:
    """
    Generates the signal based on the model parameters and simulation settings.

    Args:
        p_model: Model parameters [d, eps_inf, w0_0, wp_0, g_0, ...].
        p: Simulation parameters.

    Returns:
        torch.Tensor: Signal tensor.
    """

    # th0, d, eps_inf, [w0, wp, g]_0, [w0, wp, g]_1, ...
    params_current_signal = p_model.to(self.device)

    # Simulate in batches: 128 (To prevent overflow of memory)
    batch_size = 128  # Batch size
    out = None        # Output tensor

    for i in range(0, self.n_samples, batch_size):
      param_modeling = params_current_signal[i:i+batch_size]

      res = sys_func(param_modeling, input_mode='nxlxa', **kwargs)
      # stack
      if out is None:
        out = res
        continue

      if isinstance(out, tuple):
        out = tuple(torch.cat((o_, r_), dim=0) for o_, r_ in zip(out, res))
      else:
        out = torch.cat((out, res), dim=0)  

    return out

  def generate_labels(
    self,             # 
    systemname:str,   # systemname
    p:torch.Tensor    # [n_samples, n_layers, n_params (typically n_oscs*3)]
  ):
    # 1. Init variables
    n_samples = p.shape[0]  #
    n_layers = p.shape[1]   # Number of layers
    n_scales = self.n_scales[systemname]  # Number of scales for the system
    n_params = p.shape[-1]  # Number of parameters per layer
    domain = self.system_dicts[systemname]['domain']  # Domain of the system (e.g., 't' or 'r')
    prop_dir = self.system_dicts[systemname]['prop_dir']  # Direction of the system (e.g., 'transmission' or 'reflection')
    n_oscs_max = (n_params-n_system_parameters)//3
    df_map_p2mwh = self.oscs_mapping[systemname][f"{domain}_{prop_dir}"]
    col_names = ['sample_id', 'scale_id', 'pc_1', 'w0', 'wp', 'g', 'middle', 'width', 'height']
    class_names = ['pc_1']
    wn = self.wn[systemname]  # Wavenumbers for the system
    n_regions_per_scale = self.n_regions_per_scale[systemname]  # Number of
    targets = torch.zeros((n_samples, n_layers, n_scales, int(max(n_regions_per_scale)), len(self.params_msd['columns_target'])), dtype=self.dtype, device=self.device)

    data_df = np.zeros((n_samples*n_layers*n_oscs_max, len(col_names))) # Create a dataframe with the shape (n_samples*n_layers*n_oscs_max, len(col_names))

    # Assign sample id, osc id, probability class1, and parameters
    data_df[:, 0] = np.repeat(np.arange(n_samples), n_oscs_max*n_layers).astype(int)
    data_df[:, 1] = np.tile(np.arange(n_oscs_max), n_samples*n_layers).astype(int)  # Oscillator ID
    data_df[:, 2] = 1
    data_df[:, 3:6] = pt2np(p[:, :, n_system_parameters:]).reshape(-1, 3)  # [w0, wp, g] for each sample and layer

    # 2. Create interpolation mappings and assign values to df
    middle_mapping  = LinearNDInterpolator(df_map_p2mwh.loc[:, ['w0', 'wp', 'g']].to_numpy(), df_map_p2mwh.loc[:, 'middle'].to_numpy()) #
    width_mapping   = LinearNDInterpolator(df_map_p2mwh.loc[:, ['w0', 'wp', 'g']].to_numpy(), df_map_p2mwh.loc[:, 'width'].to_numpy())  #
    height_mapping  = LinearNDInterpolator(df_map_p2mwh.loc[:, ['w0', 'wp', 'g']].to_numpy(), df_map_p2mwh.loc[:, 'height'].to_numpy()) #

    idx_nonzero = data_df[:, 3:6].sum(axis=1) != 0  #
    df_nonzero = data_df[idx_nonzero, 3:6]          # Non-zero parameters

    data_df[idx_nonzero, 6] = middle_mapping(df_nonzero[:,:3])  # Interpolate middle values
    data_df[idx_nonzero, 7] = width_mapping(df_nonzero[:,:3])   # Interpolate width values
    data_df[idx_nonzero, 8] = height_mapping(df_nonzero[:,:3])  # Interpolate height values

    df = pd.DataFrame(data_df, columns=col_names) # Create a dataframe with the data
    assert not np.any(df.isna()), f"NaN values in mapping dataframe ({systemname}). Please check the ranges of the mappings."

    # 3. Iterate over the rows and create targets
    for i, row in df.iterrows():
      if np.all([row[class_name] for class_name in class_names] == 0): continue

      sample_id = i//(n_oscs_max*n_layers)
      layer_id = (i - sample_id*(n_oscs_max*n_layers))//n_oscs_max # Get the layer id from the index
      scale_id = row['scale_id']  #
      # TODO: Fix scale_id
      assert row['sample_id'] == sample_id, f"Sample ID {row['sample_id']} does not match index {i//(n_oscs_max*n_layers)}."

      bbox_oscs   = torch.tensor([row['middle'], row['width']], dtype=self.dtype, device=self.device)
      params_oscs = torch.tensor([row['w0'], row['wp'], row['g']], dtype=self.dtype, device=self.device)
      if bbox_oscs.sum() == 0: continue

      targets, passed = self._bounding_box_to_label_v2(
        bbox_oscs,
        self.region_widths[systemname],
        self.anchors[systemname],
        params_oscs,
        sample_id,
        layer_id,
        int(scale_id), # used to determine 'forced_scale'
        systemname,
        targets,
        wn
      )
      
      if not passed:
        col_names_ = ['pc_1', 'w0', 'wp', 'g', 'middle', 'width', 'height']
        df.loc[i, col_names_] = 0

    p_osc_true = targets
    p_osc_init = torch.zeros_like(p_osc_true, dtype=self.dtype, device=self.device)  # Initialize the initial parameters
    
    return p_osc_init, p_osc_true
    
  # -- MAIN FUNCTIONS --
  def init_generation(
    self,
    p_gen_dict:dict[str, dict[str, Any]]|dict[str, Any],
    systems:Dict[str, Any],
  ):
    if isinstance(p_gen_dict, dict) and isinstance(next(iter(p_gen_dict.values())), dict):
      p_gen_dict = p_gen_dict.values()[0] # If p_gen_dict is a dictionary of dictionaries, take the first one (assuming all systems have the same generation parameters)
    self.n_samples = p_gen_dict.get("n_samples")  # Number of samples to generate
    self.n_layers = p_gen_dict.get("n_layers")    # Number of layers in the system
    self.wgn_factor = p_gen_dict.get("wgn_factor", 0.0)  # Factor for white Gaussian noise
    self.y_return_mode = p_gen_dict.get("y_return_mode", "normal")  # Return mode for the output signal

    # 1. Initialize tmms
    self.init_tmm(systems['system_dicts'])

    # 2. Assign parameters from ParametersSetGenerator
    self.assign_parameters(
      parameter_class=systems['parameter_classes'],
      system_dicts=systems['system_dicts'],
    )

    # 3. Assign oscs_mapping
    self.assign_oscs_mapping(
      systems['label_classes'],
      forced_scale = p_gen_dict.get("forced_scale", {}),
    )
    
    # Other system parameters
    self.structured_noise = {key: value for key, value in systems.get("structured_noise", {}).items()} # Structured noise flag per system

  def generate_dataset(
    self,
    **kwargs
  ) -> None:
    print_info = kwargs.get("print_info", False)    #  Print information flag
    n_per_system = []

    for systemname, systemvalues in self.system_dicts.items():
      if print_info: print(f"Generating data for system {systemname}")
      wn_ = systemvalues['wn']      # Wavenumbers
      tmm_ = self.tmm[systemname]   # TMM function
      p = self.data_tr[systemname]              # Parameters for the system
      y = self.generate_spectra(p,tmm_,wn=wn_)  # 
      n_per_system.append(p.shape[0])           # Append the number of samples for the sys
      if print_info: print(f"[{systemname}] Generated y for system. Type: {type(y)}, samples: {n_per_system}")

      if isinstance(y, tuple):
        for count, item in enumerate(y):
          if print_info: print(f"[{systemname}] y[{count}] shape: {item.shape}")
          self.data_y[f"{systemname}_{count}"] = item
          self.data_y_noised[f"{systemname}_{count}"] = self._add_noise(
            item, 
            f"{systemname}_{count}", 
            wn=wn_, 
            wgn_factor=self.wgn_factor,
            )
      else:
        if print_info: print(f"[{systemname}] y shape: {y.shape}")
        self.data_y[systemname] = y
        self.data_y_noised[systemname] = self._add_noise(
          y, 
          systemname, 
          wn=wn_, 
          wgn_factor=self.wgn_factor,
        )

      if systemvalues['model'] == 'SSD':
        p_osc_init, p_osc_true = self.generate_labels(systemname, p)    #
      else:
        p_osc_init, p_osc_true = None, None
      
      if p_osc_init is not None:
        self.data_x_osc_init[systemname] = p_osc_init  # Initial parameters for the system
        self.data_x_osc_tr[systemname] = p_osc_true    # True

    if len(n_per_system) > 1:
      # Check if the number of samples per system is the same
      assert sum([n_per_system[i] == n_per_system[i+1] for i in range(len(n_per_system)-1)]) == len(n_per_system)-1, \
      f"Number of samples per system do not match: {n_per_system}. Please check the parameters."

    self.n = n_per_system[0]

  def get_device_dtype(self) -> Tuple[str, torch.dtype]:
    return self.device, self.dtype

  # --- INFORMATION ---
  def plot_distributions(self):
    """
    Plot the distributions of the simulated parameters.
    """
    import matplotlib.pyplot as plt

    # Get the simulated parameters
    params = pt2np(self.data_tr)

    # Plot the distributions
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Simulated Parameter Distributions')

    # Plot d distribution
    axs[0, 0].hist(params[:, 0], bins=20, color='blue', alpha=0.7)
    axs[0, 0].set_title('d Distribution')
    axs[0, 0].set_xlabel('d')
    axs[0, 0].set_ylabel('Frequency')

    # Plot eps_inf distribution
    axs[0, 1].hist(params[:, 1], bins=20, color='green', alpha=0.7)
    axs[0, 1].set_title('eps_inf Distribution')
    axs[0, 1].set_xlabel('eps_inf')
    axs[0, 1].set_ylabel('Frequency')

    # Plot w0 distribution
    axs[0, 2].hist(params[:, 2], bins=20, color='red', alpha=0.7)
    axs[0, 2].set_title('w0 Distribution')
    axs[0, 2].set_xlabel('w0')
    axs[0, 2].set_ylabel('Frequency')

    # Plot wp distribution
    axs[1, 0].hist(params[:, 3], bins=20, color='purple', alpha=0.7)
    axs[1, 0].set_title('wp Distribution')
    axs[1, 0].set_xlabel('wp')
    axs[1, 0].set_ylabel('Frequency')

    # Plot g distribution
    axs[1, 1].hist(params[:, 4], bins=20, color='orange', alpha=0.7)
    axs[1, 1].set_title('g Distribution')
    axs[1, 1].set_xlabel('g')
    axs[1, 1].set_ylabel('Frequency')

    # Remove empty subplot
    fig.delaxes(axs[1, 2])

    # Adjust spacing between subplots
    fig.tight_layout(pad=2.0)

    # Show the plot
    plt.show()
  # -------------------


def load_sim_ds(setname:str, **kwargs) -> SimulationDatasetBatch:
  dataset = SimulationDatasetBatch(setname.split('/')[-1], kwargs.get('device', device), 'fd')
  dataset.load_dataset(setname)

  for key, value in kwargs.items():
    setattr(dataset, key, value)

  return dataset


def init_usecase(usecase:int, device:str='cpu') -> Tuple[Dict[str, Any], Dict[str, Any]]:
  p_gen_params = {
    1: p_gen_usecase1.copy(),
    2: p_gen_usecase2.copy(),
    3: p_gen_usecase3.copy(),
  }[usecase]
  
  sys_pars = {
    1: (system_params_case1.copy(), None),
    2: (system_params_case2_1.copy(), system_params_case2_2.copy()),
    3: (system_params_case3_1.copy(), system_params_case3_2.copy()),
  }
  
  # move p_init to the requested device; clone so the module-level template is not modified in place
  for system in sys_pars[usecase]:
    if system is not None:
      system['p_init'] = system['p_init'].detach().to(device).clone()
  
  sys_pars = init_systems_params(*sys_pars[usecase], device=device)

  sys_pars = {k: v for k, v in sys_pars.items() if v is not None}
  
  return p_gen_params, sys_pars


def create_dataset(
  usecase:int, 
  setname:str, 
  n_samples:int|None=None, 
  system_params:Dict[str, Any]={}, 
  p_gen_params:Dict[str, Any]={}, 
  **kwargs
) -> SimulationDatasetBatch:
  n_systems = len({k: v for k, v in system_params.items() if v is not None})
  system_params_list = list(system_params.values())
  verbose = kwargs.get('verbose', True)
  export = kwargs.get('export', True)
  device = kwargs.get('device', globals()['device'])
  
  print(f"Verbose: {verbose}")
  
  # 1. Create ParameterSet
  p_gen_sys:list[ParametersSetGenerator] = []
  for i in range(n_systems):
    p_gen_sys.append(ParametersSetGenerator('main', 'torch', device, dtype=torch.float32))
    p_gen_current_system = p_gen_params.copy()
    if n_samples is not None:
      p_gen_current_system['n_samples'] = n_samples
    if isinstance(p_gen_current_system['d'], dict):
      p_gen_current_system['d'] = p_gen_params['d'][i]
    p_gen_sys[i](p_gen_current_system, set_name=setname, export=export)
    
    if i > 0 and p_gen_sys[i].parameters[setname].n_layers == 1: # TODO: This is a stupid hotfix, think of a better way to handle this
      p_gen_sys[i].copy(p_gen_sys[0], ) # First copy all values from the system, than change d
      p_gen_sys[i].parameters[setname].set_parameter(
        'd', np.full((n_samples,1), p_gen_current_system['d'][0][0]) #TODO: Fix bug; for multilayer system, this does not work
      )
  
  # 2. Create LabelGenerator
  l_gen = []
  for i in range(n_systems):
    l_gen_ = LabelGenerator(f'mapping_system{i+1}', device=device, dtype=torch.float32)
    mapping_savename = l_gen_(
      p_gen_sys[i].get_ranges(setname),     #
      system_params=system_params_list[i], #
      export=export, #
      resolution=15 #
    )
    l_gen.append(l_gen_)

  # 3. Create systems dict
  structured_noise = kwargs.get('structured_noise', [False for i in range(n_systems)])
  systems = {
    'system_dicts': {
      f'system{i+1}': system_params_list[i] for i in range(n_systems)
    },
    'parameter_classes': {
      f'system{i+1}': p_gen_sys[i] for i in range(n_systems)
    },
    'label_classes': {
      f'system{i+1}': l_gen[i] for i in range(n_systems)
    },
    'structured_noise': {
      f'system{i+1}': structured_noise[i] for i in range(n_systems)
    }
  }

  # -- Create and save --
  ds = SimulationDatasetBatch(setname, device, 'fd')
  ds.init_generation(p_gen_dict=p_gen_current_system, systems=systems)
  ds.generate_dataset(print_info=verbose)
  ds.save_dataset(f"usecase{usecase}/{setname}")

  # -- Load and inspect --
  if verbose:
    ds.info()
    ds.sample_info(
      idxs=range(16),
      print_info=True,  
      plot_info=True,   
      save_plots=True,  
      show_plots=False, 
    )

  _ = ds.get_info_for_model(print_info=verbose)
  
  return ds


p_gen_params, sys_pars = init_usecase(usecase=usecase, device=device) # Initialize usecase


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Generate datasets for the simulation.')
  parser.add_argument('--n_train', type=int, default=12, help='Number of training samples (as power of 2).')
  parser.add_argument('--n_val', type=int, default=10, help='Number of validation samples (as power of 2).')
  parser.add_argument('--n_test', type=int, default=10, help='Number of test samples (as power of 2).')
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
  set_name_train = f"case{usecase}_train_n2_{n_train}_{'nwsn'}" # 
  set_name_val = f"case{usecase}_val_n2_{n_val}_{'nwsn'}" # 
  set_name_test = f"case{usecase}_test_n2_{n_test}_{'nwsn'}" # 
  
  if usecase == 1:
    structured_noise = [False]
  if usecase == 2:
    structured_noise = [False, False]
  
    # -- Create and save: Train --
  
  trainset = create_dataset(
    usecase=usecase,
    setname=set_name_train,
    n_samples=2**n_train,
    system_params=sys_pars,
    p_gen_params=p_gen_params,
    verbose=True,
    export=True,
    structured_noise=structured_noise,
  )

  # -- Create and save: Val --
  valset = create_dataset(
    usecase=usecase,
    setname=set_name_val,
    n_samples=2**n_val,
    system_params=sys_pars,
    p_gen_params=p_gen_params,
    verbose=True,
    export=True,
    structured_noise=structured_noise,
  )
  
  # -- Create and save: Test --
  testset = create_dataset(
    usecase=usecase,
    setname=set_name_test,
    n_samples=2**n_test,
    system_params=sys_pars,
    p_gen_params=p_gen_params,
    verbose=True,
    export=True,
    structured_noise=structured_noise,
  )
  
  if usecase != 2:
    sys.exit(0)
  
  structured_noise = [True, True]
  
  trainset_wsn = create_dataset(
    usecase=usecase,
    setname=set_name_train.replace('nwsn', 'wsn'),
    n_samples=2**n_train,
    system_params=sys_pars,
    p_gen_params=p_gen_params,
    verbose=True,
    export=True,
    structured_noise=structured_noise,
  )
  
  valset_wsn = create_dataset(
    usecase=usecase,
    setname=set_name_val.replace('nwsn', 'wsn'),
    n_samples=2**n_val,
    system_params=sys_pars,
    p_gen_params=p_gen_params,
    verbose=True,
    export=True,
    structured_noise=structured_noise,
  )
  
  testset_wsn = create_dataset(
    usecase=usecase,
    setname=set_name_test.replace('nwsn', 'wsn'),
    n_samples=2**n_test,
    system_params=sys_pars,
    p_gen_params=p_gen_params,
    verbose=True,
    export=True,
    structured_noise=structured_noise,
  )
  