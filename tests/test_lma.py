import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_DIR)

import pidu.baselines.lma as lma_mod
from pidu.data.simulation import SimulationDatasetBatch


class DummySimulationDataset(SimulationDatasetBatch):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _make_dummy_dataset() -> DummySimulationDataset:
    dataset = DummySimulationDataset()
    dataset.device = 'cpu'
    dataset.dtype = torch.float32
    dataset.settype = 'test'
    dataset.y_return_mode = 'noised'
    dataset.system_dicts = {'system1': {'return_mode': 'all'}}
    dataset.tmm = {'system1': lambda p, wn: np.zeros((1, 14), dtype=np.float32)}

    dataset.data_x_sys_tr = {
        'system1': torch.tensor(
            [
                [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
                [[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
            ],
            dtype=torch.float32,
        )
    }
    dataset.data_x_osc_tr = {
        'system1': torch.tensor(
            [
                [[[[0.0, 10.0, 20.0, 30.0]]]],
                [[[[0.0, 30.0, 40.0, 50.0]]]],
            ],
            dtype=torch.float32,
        )
    }

    x_sys_init = {'system1': torch.ones((1, 11), dtype=torch.float32)}
    x_sys_tr = {'system1': torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.float32)}

    x_osc_init = {'system1': torch.zeros((1, 1, 1, 1, 4), dtype=torch.float32)}
    x_osc_tr = {'system1': torch.tensor([[[[[0.0, 15.0, 25.0, 35.0]]]]], dtype=torch.float32)}

    y = {'system1': torch.zeros((1, 14), dtype=torch.float32)}
    wn = {'system1': torch.linspace(1.0, 14.0, steps=14, dtype=torch.float32)}

    dataset.samples = [(wn, (x_sys_init, x_sys_tr), (x_osc_init, x_osc_tr), y, 0)]
    return dataset


class _FakeResult:
    def __init__(self, x: np.ndarray, residuals: np.ndarray):
        self.x = np.asarray(x)
        self.success = True
        self.status = 1
        self.message = 'ok'
        self.nfev = 1
        self.njev = 1
        self.optimality = 0.0
        self.cost = 0.5 * float(np.sum(residuals**2))
        self.active_mask = np.zeros_like(self.x, dtype=int)


class TestLevenbergMarquardt(unittest.TestCase):
    def test_numpy_backend_and_records_timing(self):
        class Model:
            system_names = ['system1']

            def __call__(self, wn, params):
                return np.asarray(params).reshape(-1)

        call_info = {}

        def fake_least_squares(fun, x0, **kwargs):
            call_info['max_nfev'] = kwargs.get('max_nfev')
            residuals = fun(np.asarray(x0), kwargs['args'][0])
            return _FakeResult(np.asarray(x0), residuals)

        with mock.patch.object(lma_mod, 'least_squares', side_effect=fake_least_squares):
            optimizer = lma_mod.LevenbergMarquardt(
                wn={'system1': np.arange(4, dtype=np.float64)},
                model=Model(),
                max_nfev=7,
                dtype=np.float64,
                backend='numpy',
            )

            y = {'system1': np.zeros(4, dtype=np.float64)}
            initial = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
            p_true = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
            mask = np.zeros((1, 4), dtype=bool)

            _, history = optimizer.fit(y, initial, p_true, mask)

            self.assertEqual(call_info['max_nfev'], 7)
            self.assertIn('fitting_time', history)
            self.assertGreaterEqual(history['fitting_time'], 0.0)

    def test_all_osc_false_masks_zero_oscillators(self):
        class Model:
            system_names = ['system1']

            def __call__(self, wn, params):
                return np.asarray(params).reshape(-1)

        call_info = {}

        def fake_least_squares(fun, x0, **kwargs):
            call_info['x0_len'] = len(np.asarray(x0).reshape(-1))
            call_info['max_nfev'] = kwargs.get('max_nfev')
            residuals = fun(np.asarray(x0), kwargs['args'][0])
            return _FakeResult(np.asarray(x0), residuals)

        with mock.patch.object(lma_mod, 'least_squares', side_effect=fake_least_squares):
            optimizer = lma_mod.LevenbergMarquardt(
                wn={'system1': np.arange(14, dtype=np.float64)},
                model=Model(),
                max_nfev=9,
                dtype=np.float64,
                all_osc=False,
                backend='numpy',
            )

            y = {'system1': np.zeros(14, dtype=np.float64)}
            initial = np.arange(1, 15, dtype=np.float64).reshape(1, -1)
            p_true = initial.copy()
            p_true[0, 11:] = 0.0  # the one oscillator is absent: its three parameters are fixed
            mask = np.zeros((1, 14), dtype=bool)

            optimizer.fit(y, initial, p_true, mask)

            self.assertEqual(call_info['max_nfev'], 9)
            self.assertEqual(call_info['x0_len'], 11)

    def test_benchmark_initialization_schemes(self):
        dataset = _make_dummy_dataset()
        captured = []

        class FakeWrapper(lma_mod.TmMethodWrapper):
            def __call__(self, wn, params):
                return np.zeros(14, dtype=np.float32)

        class FakeOptimizer:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                self.calls = []
                self._y_mask_flat = None
                captured.append(self)

            def fit(self, y, initial_params, p_true, mask):
                self.calls.append(
                    {
                        'y': y,
                        'initial': np.array(initial_params, copy=True),
                        'p_true': np.array(p_true, copy=True),
                        'mask': np.array(mask, copy=True),
                    }
                )
                history = {
                    'cost': 1.0,
                    'nfev': 3,
                    'success': True,
                    'status': 1,
                    'message': 'ok',
                    'fitting_time': 0.001,
                    'debug_y': {},
                }
                return np.array(initial_params, copy=True), history

        fake_mask = {'system1': torch.zeros((1, 14), dtype=torch.float32)}

        with mock.patch.object(lma_mod, 'TmMethodWrapper', FakeWrapper), \
            mock.patch.object(lma_mod, 'LevenbergMarquardt', FakeOptimizer), \
            mock.patch.object(lma_mod, 'create_masks_p', return_value=fake_mask), \
            mock.patch.object(lma_mod, '_save_timing', return_value=None), \
            mock.patch.object(lma_mod, '_export_outputs', return_value=None), \
            mock.patch.object(lma_mod, '_save_run_summary', return_value=None):

            torch.manual_seed(0)

            baseline_results = lma_mod.benchmark_lma(
                dataset=dataset,
                num_samples=1,
                max_iterations=5,
                initialization_scheme='baseline',
                all_osc=False,
                backend='numpy',
            )

            noisy_results = lma_mod.benchmark_lma(
                dataset=dataset,
                num_samples=1,
                max_iterations=5,
                initialization_scheme='noisy',
                noise_factor=0.0,
                all_osc=False,
                backend='numpy',
            )

            random_results = lma_mod.benchmark_lma(
                dataset=dataset,
                num_samples=1,
                max_iterations=5,
                initialization_scheme='random',
                random_range='dataset',
                all_osc=False,
                backend='numpy',
            )

        baseline_init = captured[0].calls[0]['initial'].reshape(-1)
        noisy_init = captured[1].calls[0]['initial'].reshape(-1)
        noisy_true = captured[1].calls[0]['p_true'].reshape(-1)
        random_init = captured[2].calls[0]['initial'].reshape(-1)

        self.assertTrue(np.allclose(baseline_init, np.ones_like(baseline_init)))
        self.assertTrue(np.allclose(noisy_init, noisy_true))

        self.assertEqual(captured[0].kwargs.get('backend'), 'numpy')
        self.assertEqual(captured[1].kwargs.get('backend'), 'numpy')
        self.assertEqual(captured[2].kwargs.get('backend'), 'numpy')

        self.assertEqual(baseline_results['initialization_scheme'], 'baseline')
        self.assertEqual(noisy_results['initialization_scheme'], 'noisy')
        self.assertEqual(random_results['initialization_scheme'], 'random')

        sys_min = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.float32)
        sys_max = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.float32)
        osc_min = np.array([10, 20, 30], dtype=np.float32)
        osc_max = np.array([30, 40, 50], dtype=np.float32)
        expected_min = np.concatenate([sys_min, osc_min])
        expected_max = np.concatenate([sys_max, osc_max])

        self.assertTrue(np.all(random_init >= expected_min - 1e-6))
        self.assertTrue(np.all(random_init <= expected_max + 1e-6))

    def test_benchmark_uses_backend_specific_tmm(self):
        dataset = _make_dummy_dataset()
        fake_mask = {'system1': torch.zeros((1, 14), dtype=torch.float32)}

        class FakeWrapper(lma_mod.TmMethodWrapper):
            def __call__(self, wn, params):
                return np.zeros(14, dtype=np.float32)

        class FakeOptimizer:
            def __init__(self, *args, **kwargs):
                self._y_mask_flat = None

            def fit(self, y, initial_params, p_true, mask):
                history = {
                    'cost': 1.0,
                    'nfev': 1,
                    'success': True,
                    'status': 1,
                    'message': 'ok',
                    'fitting_time': 0.001,
                    'debug_y': {},
                }
                return np.array(initial_params, copy=True), history

        with mock.patch.object(lma_mod, 'TmMethodWrapper', FakeWrapper), \
            mock.patch.object(lma_mod, 'LevenbergMarquardt', FakeOptimizer), \
            mock.patch.object(lma_mod, 'create_masks_p', return_value=fake_mask), \
            mock.patch.object(lma_mod, '_save_timing', return_value=None), \
            mock.patch.object(lma_mod, '_export_outputs', return_value=None), \
            mock.patch.object(lma_mod, '_save_run_summary', return_value=None), \
            mock.patch.object(lma_mod, 'TmMethod') as mock_tm_numpy, \
            mock.patch.object(lma_mod, 'TmMethodPyTorch') as mock_tm_torch:

            mock_tm_numpy.return_value = lambda p, wn: np.zeros((1, 14), dtype=np.float32)
            mock_tm_torch.return_value = lambda p, wn: torch.zeros((1, 14), dtype=torch.float32)

            lma_mod.benchmark_lma(
                dataset=dataset,
                num_samples=1,
                initialization_scheme='baseline',
                backend='numpy',
            )
            self.assertTrue(mock_tm_numpy.called)

            mock_tm_numpy.reset_mock()
            mock_tm_torch.reset_mock()

            lma_mod.benchmark_lma(
                dataset=dataset,
                num_samples=1,
                initialization_scheme='baseline',
                backend='torch',
            )
            self.assertTrue(mock_tm_torch.called)


if __name__ == '__main__':
    unittest.main()
