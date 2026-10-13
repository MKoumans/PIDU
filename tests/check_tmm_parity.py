"""Utility to compare the NumPy and PyTorch TMM implementations."""

from __future__ import annotations

import numpy as np
import torch
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pidu.data.physics import TmMethod, TmMethodPyTorch
from pidu.data.utils import pt2np


def _build_demo_params(
    num_samples: int = 2,
    num_layers: int = 2,
    num_oscillators: int = 2,
    num_wn: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    wn = np.linspace(1000, 3000, num_wn, dtype=np.float64)
    param_dim = 5 + 3 * num_oscillators
    params = np.zeros((num_samples, num_layers, param_dim), dtype=np.float64)

    # Angle of incidence and thickness per layer
    params[..., 0] = 0.0
    base_thickness = np.linspace(1e-5, 1e-3, num_layers)
    params[..., 1] = base_thickness

    # Dielectric background and Drude parameters
    params[..., 2] = 4.0    # e_inf
    params[..., 3] = 0.8    # drude_p
    params[..., 4] = 0.05   # drude_g

    # Oscillator triples (w0, wp, gamma)
    for osc_idx in range(num_oscillators):
        offset = 5 + 3 * osc_idx
        params[..., offset] = 50 + 50 * osc_idx  # w0
        params[..., offset + 1] = 20 + 20 * osc_idx  # wp
        params[..., offset + 2] = 5 + 5 * osc_idx  # gamma

    return wn, params


def compare_once(rtol: float = 1e-5, atol: float = 1e-3) -> None:
    wn, params = _build_demo_params()

    torch.manual_seed(0)
    tm_np = TmMethod(return_mode='dict')
    tm_pt = TmMethodPyTorch(device='cpu', dtype=torch.float32, return_mode='dict')

    result_np = tm_np(params, wn)
    result_pt = tm_pt(params, wn)

    keys_to_check = ['R', 'T', 'r_fd', 't_fd', 'r_td', 't_td']

    for key in keys_to_check:
        lhs = result_np[key]
        rhs = pt2np(result_pt[key])
        np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol, err_msg=f"Mismatch for {key}")

    np.testing.assert_allclose(result_np['n'], pt2np(result_pt['n']), rtol=rtol, atol=atol, err_msg="Mismatch for n")


def main() -> None:
    compare_once()
    print('TmMethod (NumPy) matches TmMethodPyTorch for the demo configuration.')


if __name__ == '__main__':
    main()
