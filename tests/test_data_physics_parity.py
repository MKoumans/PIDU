import os
import sys
import unittest

import numpy as np
import torch

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_DIR)

from pidu.data.physics import TmMethod, TmMethodPyTorch
from pidu.data.utils import pt2np


class TestTmMethodParity(unittest.TestCase):
    def setUp(self):
        self.atol = 1e-6
        self.rtol = 1e-5
        self.wn_np = np.linspace(100.0, 1000.0, 64, dtype=np.float64)
        self.params_np = np.array(
            [[0.0, 6.0e-3, 4.5, 0.8, 0.15, 350.0, 25.0, 5.0]],
            dtype=np.float64,
        )

    def test_tmmethod_accepts_numpy_inputs(self):
        model_np = TmMethod(
            dtype=np.float64,
            dtc=np.complex128,
            return_mode='dict',
            return_mode_osc='forward',
        )

        result = model_np(params=self.params_np, wn=self.wn_np, input_mode='lxa')

        self.assertIsInstance(result, dict)
        self.assertIn('R', result)
        self.assertIn('T', result)
        self.assertIsInstance(result['R'], np.ndarray)
        self.assertIsInstance(result['T'], np.ndarray)
        self.assertEqual(result['R'].shape[-1], self.wn_np.shape[0])
        self.assertEqual(result['T'].shape[-1], self.wn_np.shape[0])

    def test_tmmethod_matches_tmmethodpytorch(self):
        model_np = TmMethod(
            dtype=np.float64,
            dtc=np.complex128,
            return_mode='dict',
            return_mode_osc='forward',
        )
        model_pt = TmMethodPyTorch(
            dtype=torch.float64,
            dtc=torch.complex128,
            device='cpu',
            return_mode='dict',
            return_mode_osc='forward',
        )

        result_np = model_np(params=self.params_np, wn=self.wn_np, input_mode='lxa')

        params_pt = torch.tensor(self.params_np, dtype=torch.float64)
        wn_pt = torch.tensor(self.wn_np, dtype=torch.float64)
        result_pt = model_pt(params=params_pt, wn=wn_pt, input_mode='lxa')

        r_np = result_np['R']
        t_np = result_np['T']
        r_pt = pt2np(result_pt['R'])
        t_pt = pt2np(result_pt['T'])

        self.assertTrue(np.allclose(r_np, r_pt, rtol=self.rtol, atol=self.atol), 'R mismatch between TmMethod and TmMethodPyTorch')
        self.assertTrue(np.allclose(t_np, t_pt, rtol=self.rtol, atol=self.atol), 'T mismatch between TmMethod and TmMethodPyTorch')


if __name__ == '__main__':
    unittest.main()
