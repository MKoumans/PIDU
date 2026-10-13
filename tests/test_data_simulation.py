import copy
import os
import sys
import unittest

import numpy as np
import torch

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_DIR)

from pidu.data.simulation import ParametersSetGenerator, p_gen_usecase1, p_gen_usecase3, random_seed


class TestParametersSetGenerator(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        np.random.seed(random_seed)
        self.default_mode = 'torch'
        self.default_device = 'cpu'
        self.default_dtype = torch.float64
        self.default_set_name = 'testing'
        self.n_samples_testing = 64

        self.generator = ParametersSetGenerator(
            mode=self.default_mode,
            device=self.default_device,
            dtype=self.default_dtype
        )

    def p_gen(self, base=p_gen_usecase1, **updates):
        """A copy of a use case's generation parameters with a small sample count."""
        p_gen = copy.deepcopy(base)
        p_gen['n_samples'] = self.n_samples_testing
        p_gen.update(updates)
        return p_gen

    def create(self, p_gen=None, set_name=None, **kwargs):
        return self.generator.create(
            p_gen if p_gen is not None else self.p_gen(),
            'main', set_name or self.default_set_name, return_data=True, **kwargs
        )

    # -- FUNCTIONS -- #
    def test_class_initialization(self):
        """Test initialization of ParametersSetGenerator."""
        self.assertEqual(self.generator.device, self.default_device)
        self.assertEqual(self.generator.dtype, self.default_dtype)
        self.assertEqual(self.generator.data_mode, self.default_mode)
        self.assertEqual(self.generator.parameters, {})

    def test_class_set_name_all(self):
        """Setting the name to 'all' creates the train, val and test sets."""
        result = self.create(set_name='all')
        for name in ('train', 'val', 'test'):
            self.assertIn(name, result)

    def test_class_set_name_single(self):
        """A single set name creates only that set."""
        result = self.create()
        self.assertEqual(list(result.keys()), [self.default_set_name])

    def test_create_returns_nothing_by_default(self):
        """Without return_data, the sets are only stored on the generator."""
        self.assertIsNone(self.generator.create(self.p_gen(), 'main', self.default_set_name))
        self.assertIn(self.default_set_name, self.generator.parameters)

    def test_class_print_parameters(self):
        """print_parameters works after creating a set."""
        self.create()
        try:
            self.generator.print_parameters()
        except Exception as e:
            self.fail(f"print_parameters raised an exception: {e}")

    # -- CREATE ('main' case) -- #
    def test_get_shape_and_dtype(self):
        """Shape is [samples, layers, system parameters + 3 per oscillator] in the requested dtype/device."""
        p_gen = self.p_gen()
        n_oscs = len(p_gen['w0'][0])
        result = self.create(p_gen)[self.default_set_name]

        self.assertEqual(result.ndim, 3)
        self.assertEqual(tuple(result.shape), (self.n_samples_testing, 1, p_gen['num_sys_params'] + 3 * n_oscs))
        self.assertEqual(result.dtype, self.default_dtype)
        self.assertEqual(torch.device(result.device).type, self.default_device)

    def test_get_values_within_ranges(self):
        """Every sampled parameter lies in the range given for it; zero-width ranges are constant."""
        p_gen = self.p_gen()
        data = self.create(p_gen)[self.default_set_name][None]  # [samples, layers, parameters]
        n_sys = p_gen['num_sys_params']

        for column, key in enumerate(('th0', 'd', 'e_inf', 'drude_p', 'drude_g')):
            low, high = p_gen[key][0]
            self.assertGreaterEqual(data[..., column].min().item(), low - 1e-12, msg=key)
            self.assertLessEqual(data[..., column].max().item(), high + 1e-12, msg=key)

        for i_osc in range(len(p_gen['w0'][0])):
            for offset, key in enumerate(('w0', 'wp', 'g')):
                values = data[..., n_sys + 3 * i_osc + offset]
                values = values[values > 0]  # oscillators are switched off with probability 1 - oscs_chance
                self.assertGreater(len(values), 0, msg=f'{key} {i_osc}: no active oscillator')
                low, high = p_gen[key][0][i_osc]
                self.assertGreaterEqual(values.min().item(), low, msg=f'{key} {i_osc}')
                self.assertLessEqual(values.max().item(), high, msg=f'{key} {i_osc}')

    def test_get_oscs_chance(self):
        """oscs_chance=1 keeps every oscillator; a lower chance switches some off."""
        n_sys = self.p_gen()['num_sys_params']
        on = self.create(self.p_gen(oscs_chance=1.0))[self.default_set_name][None][..., n_sys:]
        some_off = self.create(self.p_gen(oscs_chance=0.5), set_name='half')['half'][None][..., n_sys:]
        self.assertTrue(bool((on > 0).all()))
        self.assertTrue(bool((some_off == 0).any()))
        self.assertTrue(bool((some_off > 0).any()))

    def test_get_seeded_reproducible(self):
        """The same numpy seed gives the same parameters."""
        np.random.seed(0)
        first = self.create()[self.default_set_name][None].clone()
        np.random.seed(0)
        second = self.create(set_name='again')['again'][None]
        torch.testing.assert_close(first, second)

    def test_get_multilayer(self):
        """Use case 3 (two layers): one slice per layer, with the layer-specific ranges."""
        p_gen = self.p_gen(base=p_gen_usecase3)
        result = self.create(p_gen)[self.default_set_name]
        data = result[None]

        self.assertEqual(data.shape[1], 2)
        for layer in range(2):
            low, high = p_gen['d'][layer]
            self.assertGreaterEqual(data[:, layer, 1].min().item(), low - 1e-12)
            self.assertLessEqual(data[:, layer, 1].max().item(), high + 1e-12)
            low, high = p_gen['e_inf'][layer]
            self.assertGreaterEqual(data[:, layer, 2].min().item(), low)
            self.assertLessEqual(data[:, layer, 2].max().item(), high)


if __name__ == '__main__':
    unittest.main()
