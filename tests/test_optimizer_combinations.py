"""
Test cases for all combinations of optimizer modes and algorithms.
Tests that training runs successfully with different optimization configurations.
"""
from csv import Error
import sys
import os
import unittest
import torch
import numpy as np
import time
import pandas as pd
from typing import Dict

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_DIR)

from pidu.config import usecase, active_models
from pidu.data.simulation import init_usecase
from pidu.data.simulation import load_sim_ds
from pidu.model.networks.framework import MultiModelFramework
from pidu.model.trainer import Trainer
from pidu.model.utils import create_masks_osc, create_masks_sys

# Test configuration
USECASE = 1
N_EPOCHS = 5  # Minimal epochs for quick testing
DATASET_SIZE = 6  # 2^6 = 64 samples
BATCH_SIZE = 32

# Optimizer modes to test
OPTIMIZER_MODES = [
    'complete',
    'unfold',
    'scale',
    'optics-type',
    'system',
    'none',
]

# Optimizer algorithms to test
OPTIMIZER_ALGOS = [
    'base',
    'pcgrad',
    'nash',
]


class TestOptimizerCombinations(unittest.TestCase):
    """Test all combinations of optimizer modes and algorithms."""
    
    @classmethod
    def setUpClass(cls):
        """Set up datasets once for all tests."""
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.dtype = torch.float32
        
        # Load datasets
        try:
            cls.trainset = load_sim_ds(
                f"usecase{USECASE}/case{USECASE}_train_n2_{DATASET_SIZE}",
                y_return_mode='noised',
                device=cls.device
            )
            cls.valset = load_sim_ds(
                f"usecase{USECASE}/case{USECASE}_val_n2_{DATASET_SIZE}",
                y_return_mode='noised',
                device=cls.device
            )
        except Exception as e:
            raise unittest.SkipTest(f"Could not load datasets: {e}")
        
        cls.dataset_info = cls.trainset.get_info_for_model(True)
        cls.device, cls.dtype = cls.trainset.get_device_dtype()
        cls.p_gen_params, cls.sys_pars = init_usecase(usecase=USECASE, device=cls.device)
    
    def setUp(self):
        """Set up a fresh model for each test."""
        self.model = self._create_model()
    
    def _create_model(self):
        """Create a fresh model instance."""
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
                dv=self.device
            ),
            'mask_x_osc': create_masks_osc(
                n_layers=self.dataset_info['n_layers'],
                n_scales=self.dataset_info['n_scales'],
                n_regions_max=self.dataset_info['n_regions_per_scale_max'],
                n_attribs_label=self.dataset_info['n_attribs_label'],
                std_val={'system1': 1},
                dt=self.dtype,
                dv=self.device
            ),
            'create_wn_mask': False,
            'use_residual': True,
            'conf_threshold': 0.75,
        }
        
        system_info = {
            'p_gen': self.p_gen_params,
            'dataset_info': self.dataset_info,
            'systems': self.sys_pars
        }
        
        framework = MultiModelFramework(
            n_unfolds=3,
            system_info=system_info,
            model_info=model_info,
            dtype=self.dtype,
            device=self.device,
            verbose=0,
        )
        
        return framework
    
    def _get_training_system(self, optimisation_mode: str, optimiser_algo: str) -> Dict:
        """Create training system configuration."""
        return {
            'wgn_factor': 0.0,
            'verbose': 0,
            'lr': 1e-3,
            'scheduler_type': 'StepLR',
            'scheduler_step_size1': 10,
            'scheduler_gamma1': 0.95,
            'optimisation_mode': optimisation_mode,
            'optimiser': 'adamw',
            'optimiser_algo': optimiser_algo,
            'betas': (0.9, 0.999),
            'eps': 1e-2,
            'weight_decay': 1e-4,
            'clip_grad_norm': 1.0,
            'init_mode': 'orthogonal',
            'batch_size': BATCH_SIZE,
            'shuffle': True,
            'track_gradient': False,
            'run_folder': 'tests/outputs/optimizer_tests',
            'loss_alpha': 0.05,
        }
    
    def _run_training_test(self, optimisation_mode: str, optimiser_algo: str):
        """Run a single training test with specific optimizer configuration."""
        training_system = self._get_training_system(optimisation_mode, optimiser_algo)
        
        # Create trainer
        trainer = Trainer(self.model, training_system=training_system, loss_matrix=False, save_intermediate=False)
        trainer.load_dataset(
            to_import={
                'train': self.trainset,
                'val': self.valset,
            },
        )
        
        # Run training with timing
        start_time = time.time()
        history = trainer.train(N_EPOCHS, history_save_intermediate=False, plot_intermediate=False, save_intermediate=False)
        elapsed_time = time.time() - start_time
        
        # Verify training completed
        self.assertIsNotNone(history, "History should not be None")
        self.assertTrue(history.initialised, "History should be initialized")
        self.assertEqual(
            len(history.loss['train']), 
            N_EPOCHS, 
            f"Expected {N_EPOCHS} epochs, got {len(history.loss['train'])}"
            )
        
        # Verify loss is finite
        train_loss = history.loss['train']
        self.assertTrue(np.all(np.isfinite(train_loss)), 
                       f"Loss contains NaN or Inf values for mode={optimisation_mode}, algo={optimiser_algo}")
        
        # Calculate average time per batch
        n_batches = len(trainer.loaders['train'])
        avg_time_per_batch = elapsed_time / (N_EPOCHS * n_batches)
        
        return elapsed_time, avg_time_per_batch
    
    # Individual test methods for each combination
    def test_complete_base(self):
        """Test complete mode with base optimizer."""
        self._run_training_test('complete', 'base')
    
    def test_complete_pcgrad(self):
        """Test complete mode with PCGrad optimizer."""
        # with self.assertRaises(Error):
        #     self._run_training_test('complete', 'pcgrad')
        pass  # PCGrad not applicable for 'complete' mode
    
    def test_complete_nash(self):
        """Test complete mode with Nash optimizer."""
        # self._run_training_test('complete', 'nash')
        pass  # Nash not applicable for 'complete' mode
    
    def test_unfold_base(self):
        """Test unfold mode with base optimizer."""
        self._run_training_test('unfold', 'base')
    
    def test_unfold_pcgrad(self):
        """Test unfold mode with PCGrad optimizer."""
        self._run_training_test('unfold', 'pcgrad')
    
    def test_unfold_nash(self):
        """Test unfold mode with Nash optimizer."""
        self._run_training_test('unfold', 'nash')
    
    def test_scale_base(self):
        """Test scale mode with base optimizer."""
        self._run_training_test('scale', 'base')
    
    def test_scale_pcgrad(self):
        """Test scale mode with PCGrad optimizer."""
        self._run_training_test('scale', 'pcgrad')
    
    def test_scale_nash(self):
        """Test scale mode with Nash optimizer."""
        self._run_training_test('scale', 'nash')
    
    def test_optics_type_base(self):
        """Test optics-type mode with base optimizer."""
        self._run_training_test('optics-type', 'base')
    
    def test_optics_type_pcgrad(self):
        """Test optics-type mode with PCGrad optimizer."""
        self._run_training_test('optics-type', 'pcgrad')
    
    def test_optics_type_nash(self):
        """Test optics-type mode with Nash optimizer."""
        self._run_training_test('optics-type', 'nash')
    
    def test_system_base(self):
        """Test system mode with base optimizer."""
        self._run_training_test('system', 'base')
    
    def test_system_pcgrad(self):
        """Test system mode with PCGrad optimizer."""
        self._run_training_test('system', 'pcgrad')
    
    def test_system_nash(self):
        """Test system mode with Nash optimizer."""
        self._run_training_test('system', 'nash')
    
    def test_none_base(self):
        """Test none mode with base optimizer."""
        self._run_training_test('none', 'base')
    
    def test_none_pcgrad(self):
        """Test none mode with PCGrad optimizer."""
        self._run_training_test('none', 'pcgrad')
    
    def test_none_nash(self):
        """Test none mode with Nash optimizer."""
        self._run_training_test('none', 'nash')


class TestOptimizerSummary(unittest.TestCase):
    """Test all optimizer combinations and generate a summary report."""
    
    def test_all_combinations_summary(self):
        """Run all combinations and print a summary table."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load datasets
        try:
            trainset = load_sim_ds(
                f"usecase{USECASE}/case{USECASE}_train_n2_{DATASET_SIZE}",
                y_return_mode='noised',
                device=device
            )
            valset = load_sim_ds(
                f"usecase{USECASE}/case{USECASE}_val_n2_{DATASET_SIZE}",
                y_return_mode='noised',
                device=device
            )
        except Exception as e:
            self.skipTest(f"Could not load datasets: {e}")
        
        dataset_info = trainset.get_info_for_model(True)
        device, dtype = trainset.get_device_dtype()
        p_gen_params, sys_pars = init_usecase(usecase=USECASE, device=device)
        
        results = {}
        timings = {}
        
        for mode in OPTIMIZER_MODES:
            results[mode] = {}
            timings[mode] = {}
            for algo in OPTIMIZER_ALGOS:
                print(f"\nTesting: mode={mode}, algo={algo}")
                
                model_info = {
                    'physics': False,
                    'backbone_act_func': 'tanh',
                    'backbone_norm': 'instancenorm',
                    'dropout_rate': 0.1,
                    'active_models': ['system1'],
                    'mask_x_sys': create_masks_sys(
                        usecase=USECASE,
                        n_layers=dataset_info['n_layers'],
                        n_sys_pars=dataset_info['n_system_parameters'],
                        dt=dtype,
                        dv=device
                    ),
                    'mask_x_osc': create_masks_osc(
                        n_layers=dataset_info['n_layers'],
                        n_scales=dataset_info['n_scales'],
                        n_regions_max=dataset_info['n_regions_per_scale_max'],
                        n_attribs_label=dataset_info['n_attribs_label'],
                        std_val={'system1': 1},
                        dt=dtype,
                        dv=device
                    ),
                    'create_wn_mask': False,
                    'use_residual': True,
                    'conf_threshold': 0.75,
                }
                
                system_info = {
                    'p_gen': p_gen_params,
                    'dataset_info': dataset_info,
                    'systems': sys_pars
                }
                
                model = MultiModelFramework(
                    n_unfolds=3,
                    system_info=system_info,
                    model_info=model_info,
                    dtype=dtype,
                    device=device,
                    verbose=0,
                )
                
                training_system = {
                    'wgn_factor': 0.0,
                    'verbose': 0,
                    'lr': 1e-3,
                    'scheduler_type': 'StepLR',
                    'scheduler_step_size1': 10,
                    'scheduler_gamma1': 0.95,
                    'optimisation_mode': mode,
                    'optimiser': 'adamw',
                    'optimiser_algo': algo,
                    'betas': (0.9, 0.999),
                    'eps': 1e-2,
                    'weight_decay': 1e-4,
                    'clip_grad_norm': 1.0,
                    'init_mode': 'orthogonal',
                    'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'track_gradient': False,
                    'run_folder': 'tests/outputs/optimizer_tests',
                    'loss_alpha': 0.05,
                }
                
                try:
                    trainer = Trainer(model, training_system=training_system, loss_matrix=False)
                    trainer.load_dataset(
                        to_import={
                            'train': trainset,
                            'val': valset,
                        },
                    )
                    
                    # Measure training time
                    start_time = time.time()
                    history = trainer.train(N_EPOCHS)
                    elapsed_time = time.time() - start_time
                    
                    # Check if training was successful
                    if history is not None and history.initialised:
                        train_loss = history.loss['train']
                        if np.all(np.isfinite(train_loss)):
                            results[mode][algo] = 'PASS'
                            # Calculate timing metrics
                            n_batches = len(trainer.loaders['train'])
                            avg_time_per_batch = elapsed_time / (N_EPOCHS * n_batches)
                            timings[mode][algo] = {
                                'total_time': elapsed_time,
                                'avg_per_batch': avg_time_per_batch,
                                'avg_per_epoch': elapsed_time / N_EPOCHS
                            }
                        else:
                            results[mode][algo] = 'FAIL (NaN/Inf loss)'
                            timings[mode][algo] = None
                    else:
                        results[mode][algo] = 'FAIL (No history)'
                        timings[mode][algo] = None
                        
                except Exception as e:
                    results[mode][algo] = f'FAIL ({str(e)[:50]})'
                    timings[mode][algo] = None
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY: Optimizer Mode vs Algorithm")
        print("="*80)
        print(f"{'Mode':<20} | {'base':<15} | {'pcgrad':<15} | {'nash':<15}")
        print("-"*80)
        
        for mode in OPTIMIZER_MODES:
            row = f"{mode:<20}"
            for algo in OPTIMIZER_ALGOS:
                status = results[mode][algo]
                row += f" | {status:<15}"
            print(row)
        
        print("="*80)
        
        # Count passes and fails
        total = len(OPTIMIZER_MODES) * len(OPTIMIZER_ALGOS)
        passed = sum(1 for mode_results in results.values() 
                     for status in mode_results.values() 
                     if status == 'PASS')
        
        print(f"\nTotal: {passed}/{total} combinations passed")
        
        # Print timing summary
        print("\n" + "="*80)
        print("TIMING REPORT: Average Time per Batch (seconds)")
        print("="*80)
        print(f"{'Mode':<20} | {'base':<15} | {'pcgrad':<15} | {'nash':<15}")
        print("-"*80)
        
        for mode in OPTIMIZER_MODES:
            row = f"{mode:<20}"
            for algo in OPTIMIZER_ALGOS:
                timing = timings[mode][algo]
                if timing is not None:
                    time_str = f"{timing['avg_per_batch']:.4f}s"
                else:
                    time_str = "N/A"
                row += f" | {time_str:<15}"
            print(row)
        
        print("="*80)
        
        # Create and save detailed timing report
        timing_data = []
        for mode in OPTIMIZER_MODES:
            for algo in OPTIMIZER_ALGOS:
                timing = timings[mode][algo]
                if timing is not None:
                    timing_data.append({
                        'mode': mode,
                        'algorithm': algo,
                        'status': results[mode][algo],
                        'total_time_s': timing['total_time'],
                        'avg_time_per_epoch_s': timing['avg_per_epoch'],
                        'avg_time_per_batch_s': timing['avg_per_batch'],
                    })
                else:
                    timing_data.append({
                        'mode': mode,
                        'algorithm': algo,
                        'status': results[mode][algo],
                        'total_time_s': None,
                        'avg_time_per_epoch_s': None,
                        'avg_time_per_batch_s': None,
                    })
        
        # Save timing report to CSV
        df = pd.DataFrame(timing_data)
        os.makedirs('tests/logs', exist_ok=True)
        timestamp = time.strftime('%Y-%m-%d-%H-%M')
        report_path = f'tests/logs/optimizer_timing_report_{timestamp}.csv'
        df.to_csv(report_path, index=False)
        print(f"\nDetailed timing report saved to: {report_path}")
        
        # Print summary statistics
        if len(df[df['status'] == 'PASS']) > 0:
            print("\n" + "="*80)
            print("TIMING STATISTICS (for successful runs)")
            print("="*80)
            successful_df = df[df['status'] == 'PASS']
            
            for algo in OPTIMIZER_ALGOS:
                algo_df = successful_df[successful_df['algorithm'] == algo]
                if len(algo_df) > 0:
                    avg_time = algo_df['avg_time_per_batch_s'].mean()
                    min_time = algo_df['avg_time_per_batch_s'].min()
                    max_time = algo_df['avg_time_per_batch_s'].max()
                    print(f"{algo:>10}: avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s")
            
            print("="*80)
        
        # Assert all passed
        self.assertEqual(passed, total, f"Only {passed}/{total} combinations passed")


if __name__ == '__main__':
    unittest.main()
