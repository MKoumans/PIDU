import sys
import os
import unittest
import logging

import numpy as np
import torch
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from scipy.optimize import approx_fprime
from typing import List, Tuple, Union

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(PROJECT_DIR)

from pidu.data.physics import TmMethodPyTorch, TmMethod, OscillatorModel
from pidu.data.simulation import pt2np, np2pt

save_log        = False         # 
level_log       = logging.INFO  # 
handler_type    = 'stream'      # 'stream' or 'file': Used for logging
export_image    = True          # 
time_comp_time  = True          # 

reffit_column_name_remapping = {
    "W": "wn",
    "R": "r",
    "T": "t",
    "E1": "e_real",
    "E2": "e_imag",
    "N1": "n_real",
    "N2": "n_imag",
}


class TestTmMethodFunctionality(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)
        self.data_dir   = os.path.join(PROJECT_DIR, 'tests', 'data')
        self.log_dir    = os.path.join(PROJECT_DIR, 'tests', 'logs')
        self.meta_data  = pd.read_excel(os.path.join(self.data_dir, 'meta-data.xlsx'))
        # self.rtol       = 0.01  # 0.5% tolerance (relative)
        self.atol       = 1e-3  # 5e-5 tolerance (absolute)
        self.n_passed   = 0
        self.n_active   = len(self.meta_data[self.meta_data['active'] == 1])
        self.n_passive  = len(self.meta_data[self.meta_data['active'] == 0])

        # tmm
        self.tmm_methods = {
            'TmMethodPyTorch': TmMethodPyTorch,
        }

        # Logging
        self.logger = None
        if save_log:
            self.logger     = logging.getLogger(self.__class__.__name__)
            if handler_type == 'stream':
                handler = logging.StreamHandler()
            else:
                log_filename = self._get_log_file_name()
                handler = logging.FileHandler(log_filename)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            handler.setFormatter(formatter)
            handler.setLevel(level_log)
            self.logger.addHandler(handler)

            self.logger.info(f" - [init] - Import metadata.")
            self.logger.info(f" - [init] - Active test: {self.n_active}. Passive test: {self.n_passive}.")
    
    def __str__(self):
        return "TestTmMethod"

    def _get_log_file_name(self):
        class_name = self.__class__.__name__
        strf_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        name_to_match = f"{class_name}_{strf_time}"

        list_dir = os.listdir(self.log_dir)
        count = sum(name.startswith(name_to_match) for name in list_dir)
        
        log_file_name = os.path.join(self.log_dir, f"{name_to_match}_{count+1:02d}.txt")
        return log_file_name

    def _read_dat_file(self, filename:str, testcase:str, n_layers:int=1):
        """
        testcase: 'drude-lorentz'|'forward'|'complete'
        """
        data = np.loadtxt(f"{self.data_dir}/{filename}", delimiter=" ", skiprows=1)
        with open(f"{self.data_dir}/{filename}", 'r') as f:
            header = f.readline().replace("\n", "")
        
        col_names = [reffit_column_name_remapping[item] for item in header.split(" ")]
        df = pd.DataFrame(data, columns=col_names)

        match testcase:
            case 'drude-lorentz':
                wn = df.loc[:, 'wn'].to_numpy()
                n_real = df.loc[:, 'n_real'].to_numpy()
                n_imag = df.loc[:, 'n_imag'].to_numpy()
                e_real = df.loc[:, 'e_real'].to_numpy()
                e_imag = df.loc[:, 'e_imag'].to_numpy()
                return (wn, n_real, n_imag, e_real, e_imag), df
            case 'drude-lorentz-derivative':
                wn = df.loc[:, 'wn'].to_numpy()
                return wn
            case 'forward':
                wn = df.loc[:, 'wn'].to_numpy()
                R = df.loc[:, 'r'].to_numpy()
                T = df.loc[:, 't'].to_numpy()
                return (wn, R, T), df
            case 'complete':
                wn = df.loc[:, 'wn'].to_numpy()
                n_real = df.loc[:, 'n_real'].to_numpy()
                n_imag = df.loc[:, 'n_imag'].to_numpy()
                e_real = df.loc[:, 'e_real'].to_numpy()
                e_imag = df.loc[:, 'e_imag'].to_numpy()
                R = df.loc[:, 'r'].to_numpy()
                T = df.loc[:, 't'].to_numpy()
                return (wn, n_real, n_imag, e_real, e_imag, R, T), df
            case _:
                raise ValueError(f"Invalid testcase: {testcase}. Must be 'drude-lorentz', 'forward', or 'complete'.")

    def _read_params(self, row:pd.Series, col_names:List[str], n_layers:int=1):
        th0:float = row['deg (˚)']          # 
        layer_data:List[List[str]] = []   # 
        
        for i in range(n_layers): 
            layer_data.append([item for item in col_names if item.startswith(f'l{i}')])
        
        max_p = max([len(item) for item in layer_data])
        params = np.zeros((n_layers, max_p + 1))
        params[:, 0] = th0
        for i, layer in enumerate(layer_data):
            params[i, 1:len(layer)+1] = row[layer]

        # params = params[~np.isnan(params)]
        params = np.nan_to_num(params)
        # [th0, d, eps_inf, w0, wp, g, ...] -> [th0, d, eps_inf, drude_p, drude_g, w0, wp, g, ...] (no Drude term)
        params = np.concatenate([params[:, :3], np.zeros((n_layers, 2)), params[:, 3:]], axis=1)

        return params, 'lxa'  # [layers, attributes]; the models take a batch: add a sample axis and use 'nxlxa'

    @staticmethod
    def _reference_tmm(**kwargs):
        """CPU double precision model without the correcting term, which the Reffit reference does not have."""
        tmm_method = TmMethodPyTorch(device='cpu', dtype=torch.float64, dtc=torch.complex128, **kwargs)
        tmm_method.osc_model.add_correcting_term = False
        return tmm_method

    def test_correcting_term(self):
        """The correcting term adds a Drude term (drude_p = drude_g = 1) to the dielectric function."""
        wn = torch.linspace(10, 1000, 50, dtype=torch.float64)
        params = torch.tensor([[[2.0, 0.0, 0.0, 500.0, 100.0, 10.0]]], dtype=torch.float64)  # eps_inf, drude, 1 oscillator
        with_term = TmMethodPyTorch(device='cpu', dtype=torch.float64, dtc=torch.complex128)
        without_term = self._reference_tmm()
        self.assertTrue(with_term.osc_model.add_correcting_term)

        _, e_with = with_term.osc_model(wn, params, input_mode='nxlxa')
        _, e_without = without_term.osc_model(wn, params, input_mode='nxlxa')
        drude = 1 / (-wn**2 - 1j * wn)
        torch.testing.assert_close(e_with - e_without, drude.reshape(e_with.shape).to(e_with.dtype))

    def test_drude_lorentz(self):
        testcase = 'drude-lorentz'
        meta_data_drude_lorentz = self.meta_data[
            (self.meta_data['testcase'] == testcase) & 
            (self.meta_data['active'] == 1)
        ]

        tmm_method = self._reference_tmm()
        column_names = self.meta_data.columns.to_list()
        
        for i_row, row in meta_data_drude_lorentz.iterrows():
            # 1. Import true data
            filename = row['filename']
            n_layers = row['n_layers']
            (wn, n_real, n_imag, e_real, e_imag), df_true = self._read_dat_file(filename, testcase, n_layers=n_layers)
            # --

            if n_real.ndim == 1:
                n_real = n_real.reshape(1, -1)
                n_imag = n_imag.reshape(1, -1)
                e_real = e_real.reshape(1, -1)
                e_imag = e_imag.reshape(1, -1)
            else:
                n_real = n_real.T
                n_imag = n_imag.T
                e_real = e_real.T
                e_imag = e_imag.T

            # 2. Forward pass
            params, input_mode = self._read_params(row, column_names, n_layers=n_layers)
            params = params[:, 2:] # Remove th0, thickness

            wn_pt   = torch.tensor(wn, dtype=torch.float64)
            params  = torch.tensor(params, dtype=torch.float64)[None]
            n_fit, e_fit = tmm_method.osc_model(wn_pt, params, input_mode='nxlxa')

            n_fit_real, n_fit_imag = pt2np(n_fit[0].real), pt2np(n_fit[0].imag)
            e_fit_real, e_fit_imag = pt2np(e_fit[0].real), pt2np(e_fit[0].imag)
            # --

            if export_image:
                os.makedirs(f"{self.data_dir}/images/{testcase}", exist_ok=True)
                fig:plt.Figure
                ax:plt.Axes|List[plt.Axes]|List[List[plt.Axes]]
                for layer_index in range(n_layers):
                    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
                    # REAL
                    ax[0, 0].plot(wn, n_real[layer_index], label='True')
                    ax[0, 0].plot(wn, n_fit_real[layer_index], label='Fit')
                    ax[1, 0].plot(wn, n_fit_real[layer_index] - n_real[layer_index], label='Diff')
                    ax[2, 0].plot(wn, (n_fit_real[layer_index] - n_real[layer_index])/n_real[layer_index], label='Rel. Diff')
                    # IMAG
                    ax[0, 1].plot(wn, n_imag[layer_index], label='True')
                    ax[0, 1].plot(wn, n_fit_imag[layer_index], label='Fit')
                    ax[1, 1].plot(wn, n_fit_imag[layer_index] - n_imag[layer_index], label='Diff')
                    ax[2, 1].plot(wn, (n_fit_imag[layer_index] - n_imag[layer_index])/n_imag[layer_index], label='Rel. Diff')
                    # Title
                    ax[0, 0].set_title('n1 (real)')
                    ax[0, 1].set_title('n1 (imag)')
                    ax[1, 0].set_title('n1 (real) diff')
                    ax[1, 1].set_title('n1 (imag) diff')
                    ax[2, 0].set_title('n1 (real) rel. diff')
                    ax[2, 1].set_title('n1 (imag) rel. diff')
                    # Legend
                    ax[0, 0].legend()
                    ax[0, 1].legend()
                    ax[1, 0].legend()
                    ax[1, 1].legend()
                    ax[2, 0].legend()
                    ax[2, 1].legend()
                    
                    fig.savefig(f"{self.data_dir}/images/{testcase}/{filename}_l{layer_index}.png")
            
            self.assertTrue(np.allclose(n_fit_real, n_real, atol=self.atol), f"Failed n1 (real) for {filename}")
            self.assertTrue(np.allclose(n_fit_imag, n_imag, atol=self.atol), f"Failed n1 (imag) for {filename}")
            self.assertTrue(np.allclose(e_fit_real, e_real, atol=self.atol), f"Failed e1 (real) for {filename}")
            self.assertTrue(np.allclose(e_fit_imag, e_imag, atol=self.atol), f"Failed e1 (imag) for {filename}")
        
        n = len(meta_data_drude_lorentz)
        self.n_passed += n
        if self.logger is not None: self.logger.info(f" - [test_drude_lorentz] - Passed {n} testcases ({self.n_passed}/{self.n_active}).")

    def test_drude_lorentz_dp(self):
        testcase = 'drude-lorentz-derivative'
        meta_data_drude_lorentz = self.meta_data[
            (self.meta_data['testcase'] == testcase) & 
            (self.meta_data['active'] == 1)
        ]
        dt, dv = torch.float64, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmm_method1 = TmMethodPyTorch(device=dv, return_mode_osc='forward')
        tmm_method2 = TmMethodPyTorch(device=dv, return_mode_osc='backward')

        column_names = self.meta_data.columns.to_list()
        param_names = ['eps_inf', 'w0_0', 'wp_0', 'gamma_0', 'w0_1', 'wp_1', 'gamma_1', 'w0_2', 'wp_2', 'gamma_2', 'w0_3', 'wp_3', 'gamma_3']	
        
        for i_row, row in meta_data_drude_lorentz.iterrows():
            # 1. Import true data
            filename = row['filename']
            n_layers = row['n_layers']
            wn = self._read_dat_file(filename, testcase, n_layers=n_layers)
            # --

            def get_n(params:NDArray) -> NDArray:
                assert tmm_method1.osc_model.return_mode == 'forward', "Return mode must be 'forward'."
                (n_fit, _) = tmm_method1.osc_model(wn_pt, np2pt(params, dt, dv), input_mode='a')
                n_fit = (abs(n_fit))**2
                return pt2np(n_fit)

            # -- FORWARD PASS --
            params, input_mode = self._read_params(row, column_names, n_layers=n_layers)
            params = params[:, 2:] # Remove th0, thickness

            wn_pt   = torch.tensor(wn, dtype=dt, device=dv)
            params  = np.array(params[0])
            # ------------------
            
            # -- GET DERIVATIVE --
            der1 = approx_fprime(params, get_n, epsilon=1e-6)
            der1 = der1[np.newaxis, :, :]
            
            der2 = tmm_method2.osc_model(wn_pt, np2pt(params,dt,dv), input_mode='a')
            der2 = (abs(der2))**2
            
            der1, der2 = der1[0], pt2np(der2[0])
            # --------------------

            if export_image:
                os.makedirs(f"{self.data_dir}/images/{testcase}", exist_ok=True)
                fig:plt.Figure
                ax:plt.Axes|List[plt.Axes]|List[List[plt.Axes]]
                for layer_index in range(n_layers):
                    for i in range(params.shape[-1]):
                        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                        # REAL
                        ax[0].plot(wn, der1[layer_index, :, i], label='Scipy')
                        ax[0].plot(wn, der2[layer_index, :, i], label='Analytical')
                        ax[1].plot(wn, der1[layer_index, :, i] - der2[layer_index, :, i], label='Diff')
                        # Title
                        ax[0].set_title('n1 (real)')
                        ax[1].set_title('n1 (real) diff')
                        # Legend
                        ax[0].legend()
                        ax[1].legend()
                        
                        fig.savefig(f"{self.data_dir}/images/{testcase}/{filename}_l{layer_index}_{param_names[i]}.png")
            
            self.assertTrue(np.allclose(der1, der2, atol=self.atol), f"Failed derivative for {filename}")
        
        n = len(meta_data_drude_lorentz)
        self.n_passed += n
        if self.logger is not None: self.logger.info(f" - [test_drude_lorentz] - Passed {n} testcases ({self.n_passed}/{self.n_active}).")

    def test_forward(self):
        testcase = 'forward'
        meta_data_forward = self.meta_data[
            (self.meta_data['testcase'] == testcase) & 
            (self.meta_data['active'] == 1)
        ]

        tmm_method = self._reference_tmm(time_c=time_comp_time)
        column_names = self.meta_data.columns.to_list()
        
        for i_row, row in meta_data_forward.iterrows():
            # 1. Import true data
            filename = row['filename']
            n_layers = row['n_layers']
            (wn, R_true, T_true), df_true = self._read_dat_file(filename, testcase, n_layers=n_layers)
            # --

            # 2. Forward pass
            params, input_mode = self._read_params(row, column_names, n_layers=n_layers)
            params  = torch.tensor(params, dtype=torch.float64)[None]
            wn_pt   = torch.tensor(wn, dtype=torch.float64)
            res     = tmm_method(params, wn_pt, input_mode='nxlxa')
            # --

            # 3. Compare
            R_sim = res['R'][0].detach().cpu().numpy()
            T_sim = res['T'][0].detach().cpu().numpy()

            max_diff_r = np.max(np.abs(R_sim - R_true))
            max_diff_t = np.max(np.abs(T_sim - T_true))

            wn = wn[1:]
            R_true = R_true[1:]
            T_true = T_true[1:]
            R_sim = R_sim[1:]
            T_sim = T_sim[1:]

            if export_image:
                os.makedirs(f"{self.data_dir}/images/{testcase}", exist_ok=True)
                fig, ax = plt.subplots(3, 2, figsize=(12, 4))
                ax[0,0].plot(wn, R_true, label='R (Reffit)')
                ax[0,0].plot(wn, R_sim, label='R (Python)')
                ax[0,1].plot(wn, T_true, label='T (Reffit)')
                ax[0,1].plot(wn, T_sim, label='T (Python)')
                # 
                ax[1,0].plot(wn, R_sim - R_true, label='Difference R')
                ax[1,1].plot(wn, T_sim - T_true, label='Difference T')
                #
                ax[2,0].plot(wn, (R_sim - R_true)/R_true, label='Rel. Diff R')
                ax[2,1].plot(wn, (T_sim - T_true)/T_true, label='Rel. Diff T')
                # 
                ax[0,0].legend()
                ax[0,1].legend()
                ax[1,0].legend()
                ax[1,1].legend()
                ax[2,0].legend()
                ax[2,1].legend()
                # 
                ax[0,0].set_title('Reflectance R') 
                ax[0,1].set_title('Transmittance T')
                ax[1,0].set_title('Difference R')
                ax[1,1].set_title('Difference T')
                ax[2,0].set_title('R rel. diff')
                ax[2,1].set_title('T rel. diff')
                # 
                fig.legend()
                fig.savefig(f"{self.data_dir}/images/{testcase}/{filename}.png")

            self.assertTrue(np.allclose(R_sim, R_true, atol=self.atol), f"Failed R for {filename}. Max diff: {max_diff_r:.4f}")
            self.assertTrue(np.allclose(T_sim, T_true, atol=self.atol), f"Failed T for {filename}. Max diff: {max_diff_t:.4f}")
        
        n = len(meta_data_forward)
        self.n_passed += n
        if self.logger is not None: self.logger.info(f" - [test_drude_lorentz] - Passed {n} testcases ({self.n_passed}/{self.n_active}).")

    def backward(self):
        pass

class Deprecated_TestTmMethodNumpyFunctionality(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)
        self.data_dir   = os.path.join(PROJECT_DIR, 'tests', 'data')
        self.log_dir    = os.path.join(PROJECT_DIR, 'tests', 'logs')
        self.meta_data  = pd.read_excel(os.path.join(self.data_dir, 'meta-data.xlsx'))
        # self.rtol       = 0.01  # 0.5% tolerance (relative)
        self.atol       = 1e-3  # 5e-5 tolerance (absolute)
        self.n_passed   = 0
        self.n_active   = len(self.meta_data[self.meta_data['active'] == 1])
        self.n_passive  = len(self.meta_data[self.meta_data['active'] == 0])

        # tmm
        self.tmm_methods = {
            'TmMethodNumpy': TmMethod,
        }

        # Logging
        self.logger = None
        if save_log:
            self.logger = logging.getLogger(self.__class__.__name__)
            if handler_type == 'stream':
                handler = logging.StreamHandler()
            else:
                log_filename = self._get_log_file_name()
                handler = logging.FileHandler(log_filename)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            handler.setFormatter(formatter)
            handler.setLevel(level_log)
            self.logger.addHandler(handler)

            self.logger.info(f" - [init] - Import metadata.")
            self.logger.info(f" - [init] - Active test: {self.n_active}. Passive test: {self.n_passive}.")
    
    def __str__(self):
        return "TestTmMethodNumpy"

    def _get_log_file_name(self):
        class_name = self.__class__.__name__
        strf_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        name_to_match = f"{class_name}_{strf_time}"

        list_dir = os.listdir(self.log_dir)
        count = sum(name.startswith(name_to_match) for name in list_dir)
        
        log_file_name = os.path.join(self.log_dir, f"{name_to_match}_{count+1:02d}.txt")
        return log_file_name

    def _read_dat_file(self, filename:str, testcase:str, n_layers:int=1):
        """
        testcase: 'drude-lorentz'|'forward'|'complete'
        """
        data = np.loadtxt(f"{self.data_dir}/{filename}", delimiter=" ", skiprows=1)
        with open(f"{self.data_dir}/{filename}", 'r') as f:
            header = f.readline().replace("\n", "")
        
        col_names = [reffit_column_name_remapping[item] for item in header.split(" ")]
        df = pd.DataFrame(data, columns=col_names)

        match testcase:
            case 'drude-lorentz':
                wn = df.loc[:, 'wn'].to_numpy()
                n_real = df.loc[:, 'n_real'].to_numpy()
                n_imag = df.loc[:, 'n_imag'].to_numpy()
                e_real = df.loc[:, 'e_real'].to_numpy()
                e_imag = df.loc[:, 'e_imag'].to_numpy()
                return (wn, n_real, n_imag, e_real, e_imag), df
            case 'drude-lorentz-derivative':
                wn = df.loc[:, 'wn'].to_numpy()
                return wn
            case 'forward':
                wn = df.loc[:, 'wn'].to_numpy()
                R = df.loc[:, 'r'].to_numpy()
                T = df.loc[:, 't'].to_numpy()
                return (wn, R, T), df
            case 'complete':
                wn = df.loc[:, 'wn'].to_numpy()
                n_real = df.loc[:, 'n_real'].to_numpy()
                n_imag = df.loc[:, 'n_imag'].to_numpy()
                e_real = df.loc[:, 'e_real'].to_numpy()
                e_imag = df.loc[:, 'e_imag'].to_numpy()
                R = df.loc[:, 'r'].to_numpy()
                T = df.loc[:, 't'].to_numpy()
                return (wn, n_real, n_imag, e_real, e_imag, R, T), df
            case _:
                raise ValueError(f"Invalid testcase: {testcase}. Must be 'drude-lorentz', 'forward', or 'complete'.")

    def _read_params(self, row:pd.Series, col_names:List[str], n_layers:int=1):
        th0:float = row['deg (˚)']          # 
        layer_data:List[List[str]] = []   # 
        
        for i in range(n_layers): 
            layer_data.append([item for item in col_names if item.startswith(f'l{i}')])
        
        max_p = max([len(item) for item in layer_data])
        params = np.zeros((n_layers, max_p + 1))
        params[:, 0] = th0
        for i, layer in enumerate(layer_data):
            params[i, 1:len(layer)+1] = row[layer]

        # params = params[~np.isnan(params)]
        params = np.nan_to_num(params)
        # [th0, d, eps_inf, w0, wp, g, ...] -> [th0, d, eps_inf, drude_p, drude_g, w0, wp, g, ...] (no Drude term)
        params = np.concatenate([params[:, :3], np.zeros((n_layers, 2)), params[:, 3:]], axis=1)

        return params, 'lxa'

    def test_forward(self):
        testcase = 'forward'
        meta_data_forward = self.meta_data[
            (self.meta_data['testcase'] == testcase) & 
            (self.meta_data['active'] == 1)
        ]

        tmm_method = TmMethod()
        column_names = self.meta_data.columns.to_list()
        
        for i_row, row in meta_data_forward.iterrows():
            # 1. Import true data
            filename = row['filename']
            n_layers = row['n_layers']
            (wn, n_real, n_imag, e_real, e_imag, R_true, T_true), df = self._read_dat_file(filename, "complete", n_layers=n_layers)
            R_true = pt2np(R_true)
            T_true = pt2np(T_true)
            n = n_real + 1j*n_imag
            # --

            # 2. Forward pass
            params, input_mode = self._read_params(row, column_names, n_layers=n_layers)
            params = torch.tensor(params, dtype=torch.float64)
            wn_pt = torch.tensor(wn, dtype=torch.float64)
            
            params_np   = pt2np(params) # 
            wn_np       = pt2np(wn_pt)  # 
            
            if n.ndim == 2:
                if n.shape[0] > n.shape[1]:
                    n = n.T
            
            ret = tmm_method.forward(params_np, wn_np, input_mode=input_mode, n=n)
            res, _, _ = ret
            # --

            # 3. Compare
            R_sim = res['R'][0]
            T_sim = res['T'][0]

            max_diff_r = np.max(np.abs(R_sim - R_true))
            max_diff_t = np.max(np.abs(T_sim - T_true))

            wn      = wn[1:]
            R_true  = R_true[1:]
            T_true  = T_true[1:]
            R_sim   = R_sim[1:]
            T_sim   = T_sim[1:]

            if export_image:
                os.makedirs(f"{self.data_dir}/images/{testcase}", exist_ok=True)
                fig, ax = plt.subplots(3, 2, figsize=(12, 4))
                ax[0,0].plot(wn, R_true, label='R (Reffit)')
                ax[0,0].plot(wn, R_sim, label='R (Python)')
                ax[0,1].plot(wn, T_true, label='T (Reffit)')
                ax[0,1].plot(wn, T_sim, label='T (Python)')
                # 
                ax[1,0].plot(wn, R_sim - R_true, label='Difference R')
                ax[1,1].plot(wn, T_sim - T_true, label='Difference T')
                #
                ax[2,0].plot(wn, (R_sim - R_true)/R_true, label='Rel. Diff R')
                ax[2,1].plot(wn, (T_sim - T_true)/T_true, label='Rel. Diff T')
                # 
                ax[0,0].legend()
                ax[0,1].legend()
                ax[1,0].legend()
                ax[1,1].legend()
                ax[2,0].legend()
                ax[2,1].legend()
                # 
                ax[0,0].set_title('Reflectance R') 
                ax[0,1].set_title('Transmittance T')
                ax[1,0].set_title('Difference R')
                ax[1,1].set_title('Difference T')
                ax[2,0].set_title('R rel. diff')
                ax[2,1].set_title('T rel. diff')
                # 
                fig.legend()
                fig.savefig(f"{self.data_dir}/images/{testcase}/{filename}.png")

            self.assertTrue(np.allclose(R_sim, R_true, atol=self.atol), f"Failed R for {filename}. Max diff: {max_diff_r:.4f}")
            self.assertTrue(np.allclose(T_sim, T_true, atol=self.atol), f"Failed T for {filename}. Max diff: {max_diff_t:.4f}")
        
        n = len(meta_data_forward)
        self.n_passed += n
        if self.logger is not None: self.logger.info(f" - [test_drude_lorentz] - Passed {n} testcases ({self.n_passed}/{self.n_active}).")

    def backward(self):
        pass

if __name__ == '__main__':
    unittest.main()