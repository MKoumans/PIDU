"""
Test all supported initializer modes on MultiModelFramework.
Each test performs real dataset loading and a single forward pass.
"""

import os
import sys
import unittest
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_DIR)

from pidu.data.simulation import init_usecase, load_sim_ds
from pidu.model.initializer import InitMode, NetworkInitializer
from pidu.model.networks.framework import MultiModelFramework
from pidu.model.utils import create_masks_osc, create_masks_sys

USECASE = 1
DATASET_SIZE = 10
BATCH_SIZE = 4


class TestAllInitializersWithForwardPass(unittest.TestCase):
    """Validate every initializer mode with data loading and one forward pass."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cpu')

        try:
            cls.trainset = load_sim_ds(
                f"usecase{USECASE}/case{USECASE}_train_n2_{DATASET_SIZE}",
                y_return_mode='noised',
                device=cls.device,
            )
        except Exception as error:
            raise unittest.SkipTest(f"Could not load dataset for initializer tests: {error}")

        cls.dataset_info = cls.trainset.get_info_for_model(True)
        cls.device, cls.dtype = cls.trainset.get_device_dtype()
        cls.p_gen_params, cls.sys_pars = init_usecase(usecase=USECASE, device=cls.device)

        loader = DataLoader(cls.trainset, batch_size=BATCH_SIZE, shuffle=False)
        cls.batch = next(iter(loader))

    def _create_model(self) -> MultiModelFramework:
        model_info = {
            'physics': False,
            'backbone_act_func': 'tanh',
            'backbone_norm': 'instancenorm',
            'dropout_rate': 0.1,
            'active_models': ['system1'],
            'mask_x_sys': create_masks_sys(
                usecase=USECASE,
                n_layers=self.dataset_info['n_layers'],
                n_sys_pars=self.dataset_info['n_system_parameters'],
                dt=self.dtype,
                dv=self.device,
            ),
            'mask_x_osc': create_masks_osc(
                n_layers=self.dataset_info['n_layers'],
                n_scales=self.dataset_info['n_scales'],
                n_regions_max=self.dataset_info['n_regions_per_scale_max'],
                n_attribs_label=self.dataset_info['n_attribs_label'],
                std_val={'system1': 1},
                dt=self.dtype,
                dv=self.device,
            ),
            'create_wn_mask': False,
            'use_residual': True,
            'conf_threshold': 0.75,
        }

        system_info = {
            'p_gen': self.p_gen_params,
            'dataset_info': self.dataset_info,
            'systems': self.sys_pars,
        }

        return MultiModelFramework(
            n_unfolds=1,
            system_info=system_info,
            model_info=model_info,
            dtype=self.dtype,
            device=self.device,
            verbose=0,
        )

    def _assert_tensor_tree_is_finite(self, value: Any):
        if isinstance(value, torch.Tensor):
            self.assertTrue(torch.isfinite(value).all().item(), "Found non-finite values in output tensor")
            return
        if isinstance(value, dict):
            for item in value.values():
                self._assert_tensor_tree_is_finite(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._assert_tensor_tree_is_finite(item)

    def _run_initializer_forward_case(self, mode: InitMode):
        model = self._create_model()
        initializer = NetworkInitializer(model, verbose=False)

        wn, x_sys, x_osc, y, _ = self.batch
        init_kwargs = {}

        if mode == InitMode.LSUV:
            init_kwargs = {
                'tol': 0.2,
                'max_iter': 2,
                'sample_input': {
                    'wn': wn,
                    'y': y,
                    'x_sys': x_sys,
                    'x_osc': x_osc,
                },
            }

        initializer.initialize(
            mode=mode,
            activation_bb='tanh',
            activation_dh='tanh',
            scale=1.0,
            distribution='normal',
            **init_kwargs,
        )

        with torch.no_grad():
            output = model(wn=wn, y=y, x_sys=x_sys, x_osc=x_osc)

        self.assertIsInstance(output, dict)
        self.assertIn('y', output)
        self.assertIn('p_sys', output)
        self._assert_tensor_tree_is_finite(output)

    def test_all_initializer_modes_with_single_forward_pass(self):
        for mode in InitMode:
            with self.subTest(initializer_mode=mode.value):
                self._run_initializer_forward_case(mode)


if __name__ == '__main__':
    unittest.main()
