import sys, os
import numpy as np
import torch
import torch.fft as fft
import warnings
import matplotlib.pyplot as plt

from time import perf_counter, strftime, gmtime
from subprocess import check_output as sp_check_output
from numpy import ndarray
from numpy.typing import NDArray
from typing import Any, Tuple, List, Dict

from pidu.data.utils import *
from pidu.data.optical_parameters import *
from pidu.paths import OUTPUTS_DIR

class TmMethod:
	"""NumPy implementation mirroring the TmMethodPyTorch interface."""

	def __init__(self, **kwargs) -> None:
		self.dtype = self._resolve_dtype(kwargs.pop('dtype', np.float64), np.float64)
		self.dtc = self._resolve_dtype(kwargs.pop('dtc', np.complex128), np.complex128)
		self.device = 'cpu'
		self.time_c = kwargs.pop('time_c', False)
		self.return_mode = kwargs.pop('return_mode', 'dict')
		osc_model_name = kwargs.pop('osc_model', 'drude_lorentz')
		return_mode_osc = kwargs.pop('return_mode_osc', 'forward')
		self.osc_model = OscillatorModelNumpy(osc_model_name, return_mode_osc, dtype=self.dtc)
		self.forward_func = forward_func_numpy

	def __call__(self, params: Any, wn: Any, **kwargs):
		t_call_s = perf_counter()
		wn_np = self._to_numpy(wn, self.dtype)
		params_np = self._to_numpy(params, self.dtype)
		res, dims, timings = self.forward(params_np, wn_np, **kwargs)
		t_call_e = perf_counter()
		if self.time_c:
			hash_ = sp_check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
			date = strftime("%Y%m%d-%H_%M", gmtime())
			os.makedirs(OUTPUTS_DIR, exist_ok=True)
			with open(OUTPUTS_DIR / 'physics_timings.csv', 'a') as f:
				my_arr = [date, hash_, 'cpu'] + dims + [t_call_e - t_call_s] + timings
				my_str = '\t'.join([str(x) for x in my_arr])
				f.write(f"{my_str}\n")
		return res

	def forward(self, params: np.ndarray, wn: np.ndarray, **kwargs):
		input_mode = kwargs.pop('input_mode', 'nxlxa')
		assert len(input_mode) == 1 + 2 * (params.ndim - 1), (
			f"Input mode is not compatible with the input dimensions {len(input_mode)}-{1 + 2 * (params.ndim - 1)}."
		)
		wn = np.where(wn == 0, wn + NUMPY_EPSILON, wn)
		freq = 1 / wn
		params, (num_samples, num_layers, _) = remap_params(input_mode, params)
		th_0_layers = params[..., 0]
		thickness = params[..., 1]
		opt_params = params[..., 2:]
		n = kwargs.pop('n', None)
		t_n_s = perf_counter()
		if n is None:
			resp_osc = self.osc_model(wn, opt_params, input_mode)
			if self.osc_model.return_mode == 'forward':
				n, _ = resp_osc
			else:
				(n, _), _ = resp_osc
		else:
			n = self._to_numpy(n, self.dtc)
		t_n_e = perf_counter()
		num_wn = wn.shape[-1]
		n_basic = np.ones((num_samples, num_layers + 2, num_wn), dtype=self.dtc)
		d_basic = np.ones((num_samples, num_layers + 2), dtype=self.dtype) * 1e12
		d_basic[:, 1:-1] = thickness
		n_basic[:, 1:-1] = n
		th_zeros = np.zeros((num_samples, 1), dtype=self.dtype)
		th_0 = np.concatenate((th_zeros, th_0_layers, th_zeros), axis=1)
		t_rt_s = perf_counter()
		res = self.coh_tmm('s', n_basic, d_basic, th_0, freq)
		t_rt_e = perf_counter()
		return res, [num_samples, num_layers, num_wn], [t_n_e - t_n_s, t_rt_e - t_rt_s]

	def coh_tmm(self, pol: str, n_list: np.ndarray, d_list: np.ndarray, th_0: np.ndarray, wn: np.ndarray):
		num_samples, num_layers, num_wn = n_list.shape
		num_calc_layers = num_layers - 2
		M_list = np.zeros((num_samples, num_layers, num_wn, 2, 2), dtype=self.dtc)
		A_tensor = np.zeros((num_samples, num_calc_layers, num_wn, 2, 2), dtype=self.dtc)
		B_tensor = np.ones((num_samples, num_calc_layers, num_wn, 2, 2), dtype=self.dtc)
		th_list = self._list_snell(n_list, th_0)
		kz_list = 2 * np.pi * n_list * np.cos(th_list) / wn
		delta = kz_list * d_list[..., None]
		delta_to_check = delta[:, 1:-1].copy()
		mask = delta_to_check.imag > 35
		delta_to_check[mask] = delta_to_check[mask].real + 35j
		delta[:, 1:-1] = delta_to_check
		t_list = self._interface_t(pol, n_list[:, :-1], n_list[:, 1:], th_list[:, :-1], th_list[:, 1:])
		r_list = self._interface_r(pol, n_list[:, :-1], n_list[:, 1:], th_list[:, :-1], th_list[:, 1:])
		exp_neg_delta = np.exp(-1j * delta[:, 1:-1])
		exp_pos_delta = np.exp(1j * delta[:, 1:-1])
		A_tensor[..., 0, 0] = exp_neg_delta
		A_tensor[..., 1, 1] = exp_pos_delta
		B_tensor[..., 0, 1] = r_list[:, 1:]
		B_tensor[..., 1, 0] = r_list[:, 1:]
		t_inv_0 = 1 / (t_list[:, 0] + NUMPY_EPSILON)
		t_inv_1 = 1 / (t_list[:, 1:] + NUMPY_EPSILON)
		M_list[:, 1:-1] = t_inv_1[..., None, None] * np.matmul(A_tensor, B_tensor)
		Mtilde = np.tile(np.eye(2, dtype=self.dtc), (num_samples, num_wn, 1, 1))
		for i in range(1, num_layers - 1):
			Mtilde = np.matmul(Mtilde, M_list[:, i])
		arr = np.ones_like(Mtilde)
		arr[..., 0, 1] = r_list[:, 0]
		arr[..., 1, 0] = r_list[:, 0]
		M = np.matmul(arr * t_inv_0[..., None, None], Mtilde)
		n_i, n_f = n_list[:, 0], n_list[:, -1]
		th_i = th_0[:, 0]
		th_f = th_list[:, -1]
		cos_tf = np.cos(th_f)
		cos_ti = np.cos(th_i)[:, None]
		if pol == 's':
			t_pow = (np.real(n_f * cos_tf) / (np.real(n_i * cos_ti) + NUMPY_EPSILON))
		elif pol == 'p':
			t_pow = (np.real(n_f * np.conj(cos_tf)) / (np.real(n_i * np.conj(cos_ti)) + NUMPY_EPSILON))
		else:
			raise ValueError("Polarization must be 's' or 'p'")
		res = self.forward_func[self.return_mode](M, t_pow)
		if self.return_mode == 'dict':
			res['n'] = n_list
		return res

	@staticmethod
	def _resolve_dtype(value, default):
		if isinstance(value, torch.dtype):
			mapping = {
				torch.float32: np.float32,
				torch.float64: np.float64,
				torch.complex64: np.complex64,
				torch.complex128: np.complex128,
			}
			return mapping.get(value, default)
		return value

	def _to_numpy(self, value: Any, dtype):
		if isinstance(value, torch.Tensor):
			return value.detach().cpu().numpy().astype(dtype)
		return np.asarray(value, dtype=dtype)

	def _list_snell(self, n_list: np.ndarray, th_0: np.ndarray):
		sin_th_0 = np.sin(th_0)[..., None]
		n0 = n_list[:, 0, :][:, None, :]
		return np.arcsin(n0 * sin_th_0 / (n_list + NUMPY_EPSILON))

	@staticmethod
	def _interface_r(polarization, n_i, n_f, th_i, th_f):
		cos_th_i = np.cos(th_i)
		cos_th_f = np.cos(th_f)
		if polarization == 's':
			return (n_i * cos_th_i - n_f * cos_th_f) / (n_i * cos_th_i + n_f * cos_th_f + NUMPY_EPSILON)
		if polarization == 'p':
			return (n_f * cos_th_i - n_i * cos_th_f) / (n_f * cos_th_i + n_i * cos_th_f + NUMPY_EPSILON)
		raise ValueError("Polarization must be 's' or 'p'")

	@staticmethod
	def _interface_t(polarization, n_i, n_f, th_i, th_f):
		cos_th_i = np.cos(th_i)
		cos_th_f = np.cos(th_f)
		if polarization == 's':
			return 2 * n_i * cos_th_i / (n_i * cos_th_i + n_f * cos_th_f + NUMPY_EPSILON)
		if polarization == 'p':
			return 2 * n_i * cos_th_i / (n_f * cos_th_i + n_i * cos_th_f + NUMPY_EPSILON)
		raise ValueError("Polarization must be 's' or 'p'")

EPSILON = torch.finfo(torch.float64).eps

dict_add_dims = {
	'nxlxa': lambda x: x,
	'lxa': lambda x: x[None, ...],
	'nxa': lambda x: x[:, None, ...],
	'a': lambda x: x[None, None, ...],
}
dict_dim_sizes = {
	'nxlxa': lambda x: [x[0], x[1], x[2]],
	'lxa': lambda x: [1, x[0], x[1]],
	'nxa': lambda x: [x[0], 1, x[1]],
	'a': lambda x: [1, 1, x[0]],
}

def remap_params(input_mode:str, params:torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
		"""
		Initialize the input mode
		"""
		return dict_add_dims[input_mode](params), dict_dim_sizes[input_mode](params.shape)

class OscillatorModel():
	"""
	A model for simulating optical properties using various oscillator models.

	Attributes:
			osc_model (Callable): The oscillator model function to be used.
			epsilon (float): A small value added to avoid division by zero or other numerical issues.
			wn1 (torch.Tensor): The input wavenumber as a tensor, used in calculations.
			wn2 (torch.Tensor): The squared input wavenumber as a tensor, used in calculations.
			mapping_dict (Dict[str, Callable]): A dictionary that maps model names to their corresponding functions.
	"""
	def __init__(self, model:str='drude_lorentz', return_mode:str='forward', **kwargs) -> None:
			self.return_mode = return_mode # forward|backward|both
			self.osc_model:function
			self.epsilon:torch.Tensor = kwargs.pop('epsilon', EPSILON)
			# Adds a fixed Drude term (wp = g = 1 cm^-1) to every layer; part of the forward model the
			# datasets and released models were generated with (and mirrored in OscillatorModelNumpy).
			self.add_correcting_term = True

			self.wn1:torch.Tensor
			self.wn2:torch.Tensor

			self.mapping_dict = {
					'drude_lorentz': self.drude_lorentz,
					'drude_lorentz_dp': self.drude_lorentz_dp,
			}

			self.assign_model(model, **kwargs)

	def __call__(self, wn:torch.Tensor|NDArray, params:torch.Tensor|NDArray, input_mode:str) -> torch.Tensor:
			"""
			Computes the dielectric function based on the input parameters and the selected oscillator model.

			Args:
					wn (torch.Tensor): The input wavenumber tensor.
					params (torch.Tensor): The parameters tensor with shape [n, l, a] or \
					[l, a], where n is the number of samples, l is the number of layers, \
					and a is the number of parameters. a = [e_inf, wp_0, g_0, oscs] \
					where e_inf is the static dielectric constant, [wp_0, g_0] the drude \
					component, and oscs are the oscillator parameters.
					input_mode (str): A string indicating the dimension configuration of the input parameters.

			Returns:
					torch.Tensor: The computed dielectric function tensor.
			"""
			params, dim_sizes = remap_params(input_mode, params)

			wn = torch.where(wn==0, wn+EPSILON, wn)
			assert torch.where(wn==0, wn+EPSILON, wn).sum() > 0, "Wavenumber cannot be zero."

			if isinstance(wn, ndarray): wn = torch.tensor(wn, dtype=torch.float32, device='cpu')
			if isinstance(params, ndarray): params = torch.tensor(params, dtype=torch.float32, device='cpu')

			wn = wn
			params = params

			self.wn = wn # Wavenumber
			if self.wn.ndim == 1:
				self.wn1    = wn[None, None, None, ...]       # Wavenumber
				self.wn2    = (wn**2)[None, None, None, ...]  # Wavenumber squared
			else:
				self.wn1 = wn[:,None,None,:] # Wavenumber
				self.wn2 = (wn**2)[:,None,None,:] # Wavenumber squared

			(n, e), oscs_p = self.osc_model(params)

			if self.return_mode == 'forward':
					return (n, e)
			elif self.return_mode == 'backward':
					dn_dp = self.osc_model_dp(params, n, oscs_p)
					return dn_dp
			elif self.return_mode == 'both':
					dn_dp = self.osc_model_dp(params, n, oscs_p)
					return (n, e), dn_dp
			else:
					raise Exception(f"Return mode {self.return_mode} not supported.")

	def assign_model(self, model:str) -> None:
			"""
			Assigns the oscillator model based on the specified model name.

			Args:
					model (str): The name of the oscillator model to use.

			Raises:
					Exception: If the specified model is not supported.
			"""
			if model not in self.mapping_dict.keys():
					raise Exception(f"Model {model} not supported.")

			self.osc_model      = self.mapping_dict[model]
			self.osc_model_dp   = self.mapping_dict[f"{model}_dp"]

	def drude_lorentz(self, params:torch.Tensor):
		"""
		Implements the Drude-Lorentz model to compute the dielectric function. The Drude-Lorentz model is a
		classical theory used to describe the optical and electronic properties of materials, particularly
		their dielectric function, which governs how they interact with electromagnetic waves (like light).

		Args:
				params (torch.Tensor): A tensor containing the parameters with shape [sample, layer, opt_params].
																opt_params consists of [e_inf, oscs] where oscs are oscillator parameters.

		Returns:
				Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
						- sqrt_e: The square root of the dielectric function tensor.
						- e: The dielectric function tensor.
		"""
		e_inf = params[..., 0][..., None] # Static dielectric constant
		drude_p, drude_g = params[..., 1], params[..., 2] # Drude parameters
		oscs = params[..., 3:]        # Oscillator parameters
		n_oscs = oscs.shape[-1]//3    # Number of oscillators
		n_layers = oscs.shape[-2]     # Number of layers
		n_samples = oscs.shape[-3]    # Number of samples
		correcting_term = torch.zeros_like(e_inf)

		oscs = oscs.view(n_samples, n_layers, n_oscs, 3) # Reshape oscs to have shape (n_batches * n_params/3, 3)

		# given oscs, if one of three values (last dim) is zero, set all values of that oscillator to zero
		mask_0 = (oscs == 0).any(dim=-1)[..., None]
		oscs = oscs * (~mask_0)

		oscs_1 = (oscs[..., 0]**2)[..., None] # oscs[0+i_osc*3] w0
		oscs_2 = (oscs[..., 1]**2)[..., None] # oscs[1+i_osc*3] wp
		oscs_3 = (oscs[..., 2])[..., None]    # oscs[2+i_osc*3] g

		denom = (oscs_1 - self.wn2 - 1j *oscs_3*self.wn1 + EPSILON)   # Denominator
		oscs = torch.sum(oscs_2/denom, 2)                             # Oscillator term
		drude_term = ((drude_p**2)[..., None])/(-self.wn2[0,...] - 1j*drude_g[..., None]*self.wn1[0,...] + EPSILON) # Drude term
		if self.add_correcting_term:
			drude_p_corr = torch.ones_like(drude_p)
			drude_g_corr = torch.ones_like(drude_g)
			correcting_term = ((drude_p_corr**2)[..., None])/(-self.wn2[0,...] - 1j*drude_g_corr[..., None]*self.wn1[0,...] + EPSILON)
		
		e = e_inf + oscs + drude_term + correcting_term                               # Dielectric function

		sqrt_e  = torch.sqrt(e)
		oscs_p  = (oscs_1, oscs_2, oscs_3)

		return (sqrt_e, e), oscs_p

	def drude_lorentz_dp(self, params:torch.Tensor, n:torch.Tensor, osc_p:Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
			"""

			"""
			n_samples, n_layers, n_oscs, n_wn = n.shape[0], n.shape[1], (params.shape[-1]//3), n.shape[-1]

			de_dp = torch.ones((n_samples, n_layers, 1+n_oscs*3, n_wn), dtype=n.dtype, device=n.device)
			dn_de = 1/(2*n)

			oscs_1 = osc_p[0]  # [batch, oscs, param (0)]
			oscs_2 = osc_p[1]  # [batch, oscs, param (1)]
			oscs_3 = osc_p[2]  # [batch, oscs, param (2)]

			NUM = oscs_2
			DEN = oscs_1 - self.wn2 - 1j *oscs_3*self.wn1 # Denominator

			de_dp0 = (-2*torch.sqrt(oscs_1) * NUM/(DEN**2))
			de_dp1 = (2*torch.sqrt(oscs_2) * 1/DEN)
			de_dp2 = (1j*self.wn1*NUM/(DEN**2))

			#
			de_dp[:, :, 1:, :] = torch.stack((de_dp0, de_dp1, de_dp2), dim=2).view(n_samples, n_layers, n_oscs*3, n_wn)
			de_dp = torch.transpose(de_dp, 2, 3)

			dp_dn = dn_de[..., None] * de_dp

			return dp_dn

class OscillatorModelNumpy:
	"""NumPy version of the oscillator model used by TmMethod."""

	def __init__(self, model: str = 'drude_lorentz', return_mode: str = 'forward', **kwargs) -> None:
		if return_mode != 'forward':
			raise NotImplementedError("OscillatorModelNumpy currently supports return_mode_osc='forward' only.")
		self.return_mode = return_mode
		self.dtype = kwargs.pop('dtype', np.complex128)
		self.epsilon = NUMPY_EPSILON
		self.add_correcting_term = True
		self.mapping_dict = {
			'drude_lorentz': self.drude_lorentz,
		}
		self.assign_model(model)

	def assign_model(self, model: str) -> None:
		if model not in self.mapping_dict:
			raise Exception(f"Model {model} not supported.")
		self.osc_model = self.mapping_dict[model]

	def __call__(self, wn: np.ndarray, params: np.ndarray, input_mode: str):
		params, _ = remap_params(input_mode, params)
		params = np.asarray(params, dtype=self.dtype)
		wn = np.asarray(wn, dtype=self.dtype)
		wn = np.where(wn == 0, wn + self.epsilon, wn)
		if wn.ndim == 1:
			self.wn1 = wn[None, None, None, :]
			self.wn2 = (wn ** 2)[None, None, None, :]
		else:
			self.wn1 = wn[:, None, None, :]
			self.wn2 = (wn ** 2)[:, None, None, :]
		return self.osc_model(params)

	def drude_lorentz(self, params: np.ndarray):
		e_inf = params[..., 0][..., None]
		drude_p = params[..., 1]
		drude_g = params[..., 2]
		oscs = params[..., 3:]
		n_wn = self.wn1.shape[-1]
		if oscs.shape[-1] == 0:
			osc_term = np.zeros(drude_p[..., None].shape[:-1] + (n_wn,), dtype=self.dtype)
		else:
			n_oscs = oscs.shape[-1] // 3
			oscs = oscs.reshape(params.shape[0], params.shape[1], n_oscs, 3)
			mask = np.any(oscs == 0, axis=-1, keepdims=True)
			oscs = np.where(mask, 0.0, oscs)
			oscs_1 = (oscs[..., 0] ** 2)[..., None]
			oscs_2 = (oscs[..., 1] ** 2)[..., None]
			oscs_3 = (oscs[..., 2])[..., None]
			denom = oscs_1 - self.wn2 - 1j * oscs_3 * self.wn1 + self.epsilon
			osc_term = np.sum(oscs_2 / denom, axis=2)
		denominator = -self.wn2[0, ...] - 1j * drude_g[..., None] * self.wn1[0, ...] + self.epsilon
		drude_term = ((drude_p ** 2)[..., None]) / denominator
		if self.add_correcting_term:
			drude_p_corr = np.ones_like(drude_p)
			drude_g_corr = np.ones_like(drude_g)
			correction = ((drude_p_corr ** 2)[..., None]) / (-self.wn2[0, ...] - 1j * drude_g_corr[..., None] * self.wn1[0, ...] + self.epsilon)
		else:
			correction = np.zeros_like(drude_term)
		e = e_inf + osc_term + drude_term + correction
		sqrt_e = np.sqrt(e)
		return sqrt_e.astype(self.dtype), e.astype(self.dtype)

class TmMethodPyTorch():
	"""
	A PyTorch-based implementation of the Transfer Matrix Method (TMM) for simulating
	the propagation of electromagnetic waves through layered media. This class supports
	both the Drude-Lorentz oscillator model and other custom models for calculating
	the refractive index and other optical parameters.

	Attributes:
		dtype (torch.dtype): The data type for real-valued tensors (default: torch.float32).
		dtc (torch.dtype): The data type for complex-valued tensors (default: torch.complex64).
		device (str): The device to perform computations on ('cuda' if available, otherwise 'cpu').
		time_c (bool): Whether to time the computations and log the results (default: False).
		osc_model (OscillatorModel): The oscillator model used for calculating the refractive index.
	"""
	def __init__(self, **kwargs) -> None:
			"""
			Initializes the class with the given configuration options. Sets up the
			data types, computation device, timing options, and oscillator model used
			for simulating optical properties.

			Keyword Args (**kwargs):
					dtype (torch.dtype): Data type for real-valued tensors, default is torch.float32.
					dtc (torch.dtype): Data type for complex-valued tensors, default is torch.complex64.
					device (str): Device for computation, either 'cpu' or 'cuda'. Default is 'cuda'
							if available, otherwise 'cpu'.
					time_c (bool): Flag to enable or disable timing of computations, default is False.
					osc_model (str): Name of the oscillator model to use, default is 'drude_lorentz'.

			Notes:
					- The `dtype` and `dtc` settings determine the precision of the computations
					and should be chosen based on the needs of the application.
					- The `device` setting allows the model to leverage GPU acceleration if a
					CUDA-enabled GPU is available.
					- The `osc_model` parameter specifies the physical model for calculating
					the refractive index and permittivity of the layers.
			"""
			# META DATA
			self.dtype  = kwargs.pop('dtype', torch.float32)    #
			self.dtc    = kwargs.pop('dtc', torch.complex64)   #
			self.device = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
			self.time_c = kwargs.pop('time_c', False)           # Time the computation

			#
			osc_model_name  = kwargs.pop('osc_model', 'drude_lorentz')  #
			return_mode_osc = kwargs.pop('return_mode_osc', 'forward') #
			self.osc_model  = OscillatorModel(osc_model_name, return_mode_osc) #

			self.return_mode = kwargs.pop('return_mode', 'dict')        # dict|all|fd_r|fd_t|td_r|td_t|r|t
			self.forward_func = forward_func

	def __call__(
		self, 
    params:NDArray|torch.Tensor|OpticalParameters, 
		wn:Any, 
		**kwargs
  ) -> Tuple[torch.Tensor, torch.Tensor]|Dict[str, torch.Tensor]:
			"""
			Executes the forward pass for the model.

			Args:
					wn (Any): The input wavenumber(s), either as a tensor or convertible to a tensor.
					params (Any): The input parameters, either as a tensor or convertible to a tensor.
												The complete shape of params is [n,l,a], where n is the number of
												samples, l is the number of layers, and a is the number of parameters.
					**kwargs: Additional arguments for the forward pass.
										An important optional argument is 'input_mode', which specifies the
										dimension configuration of the input parameters. If one uses params with
										shape other than [n,l,a], one must specify the input_mode.

			Returns:
					Tuple[torch.Tensor, torch.Tensor]: The results of the forward pass, including the output
					tensor and timing information if enabled.
			"""
			t_call_s = perf_counter()

			if not isinstance(wn, torch.Tensor):
					wn = torch.tensor(wn, dtype=self.dtype, device=self.device)

			# if isinstance(params, SystemParameters):
			#     pass
			# elif not isinstance(params, torch.Tensor):
			#     params = torch.tensor(params, dtype=self.dtype, device=self.device)

			if isinstance(params, np.ndarray):
				params = torch.tensor(params, dtype=self.dtype, device=self.device)

			res, n, t = self.forward(params, wn, **kwargs)
			t_call_e = perf_counter()

			if self.time_c:
				hash_   = sp_check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
				gpu_    = torch.cuda.get_device_name() if self.device == 'cuda' else 'cpu'
				date    = strftime("%Y%m%d-%H_%M", gmtime())
				os.makedirs(OUTPUTS_DIR, exist_ok=True)
				with open(OUTPUTS_DIR / 'physics_timings.csv', 'a') as f:
					my_arr = [date, hash_, gpu_] + n + [t_call_e-t_call_s] + t
					my_str = '\t'.join([str(x) for x in my_arr])
					f.write(f"{my_str}\n")
					print(f"Save to timings.csv: {my_str}")

			return res
     
	# -- DIRECT STATIC METHODS --
	@staticmethod
	def _is_forward_angle(n:torch.Tensor, theta):
			"""
			Determines if a wave is traveling forward based on the refractive index and propagation angle.

			Args:
					n (torch.Tensor): The refractive index of the medium.
					theta: The propagation angle in the medium.

			Returns:
					torch.Tensor: A boolean tensor indicating whether the wave is forward-traveling.
			"""
			# -- Calculate the answer --
			ncostheta = n * torch.cos(theta)
			imaginary_mask = torch.abs(ncostheta.imag) > 100 * EPSILON

			answer = torch.where(imaginary_mask, ncostheta.imag > 0, ncostheta.real > 0)
			# ---------------------------

			return answer

	@staticmethod
	def _R_from_r(r):
			"""
			Calculates the reflected power (R) given the reflection amplitude (r).

			Args:
					r: Reflection amplitude.

			Returns:
					torch.Tensor: Reflected power as a fraction of the incident power.
			"""
			return torch.abs(r)**2

	@staticmethod
	def _T_from_t(pol, t, n_i, n_f, th_i, th_f):
			"""
			Calculates the transmitted power (T) given the transmission amplitude (t).

			Args:
					pol (str): Polarization ('s' or 'p').
					t: Transmission amplitude.
					n_i: Refractive index of the incident medium.
					n_f: Refractive index of the final medium.
					th_i: Propagation angle in the incident medium.
					th_f: Propagation angle in the final medium.

			Returns:
					torch.Tensor: Transmitted power as a fraction of the incident power.
			"""
			cos_tf = torch.cos(th_f)
			cos_ti = torch.cos(th_i)

			if pol == 's':
					return abs(t**2) * (((n_f*cos_tf).real) / ((n_i*cos_ti).real + EPSILON))
			elif pol == 'p':
					return abs(t**2) * (((n_f*torch.conj(cos_tf)).real) /
																	((n_i*torch.conj(cos_ti)).real + EPSILON))
			else:
					raise ValueError("Polarization must be 's' or 'p'")

	@staticmethod
	def _power_entering_from_r(pol, r, n_i, th_i):
			"""
			Calculates the power entering the first interface of the stack, starting with
			the reflection amplitude r.

			Args:
					pol (str): Polarization ('s' or 'p').
					r: Reflection amplitude.
					n_i: Refractive index of the incident medium.
					th_i: Propagation angle in the incident medium.

			Returns:
					torch.Tensor: Power entering the first interface, accounting for complex refractive indices.
			"""
			if pol == 's':
					return ((n_i*torch.cos(th_i)*(1+torch.conj(r))*(1-r)).real / (n_i*torch.cos(th_i)).real)
			elif pol == 'p':
					return ((n_i*torch.conj(torch.cos(th_i))*(1+r)*(1-torch.conj(r))).real / (n_i*torch.conj(torch.cos(th_i))).real)
			else:
					raise ValueError("Polarization must be 's' or 'p'")
	# ---------------------------

	def _list_snell(self, n_list:torch.Tensor, th_0:torch.Tensor):
			"""
			Computes the list of angles in each layer using Snell's law.

			Args:
					n_list (torch.Tensor): The refractive index of each layer.
					th_0 (torch.Tensor): The angle of incidence in the first layer.

			Returns:
					torch.Tensor: A tensor of angles for each layer, potentially complex.
			"""
			sin_th_0 = torch.sin(th_0).unsqueeze(2)
			angles  = torch.arcsin(n_list[:,0,:].unsqueeze(1)*sin_th_0 / (n_list + EPSILON))    #

			# aanname: incident n is altijd lucht: ofwel n.imag == 0 ofwel het blijft hetzelfde

		#   mask0   = self._is_forward_angle(n_list[:,0,:], angles[:,0,:])    #
		#   mask1   = self._is_forward_angle(n_list[:,-1,:], angles[:,-1,:])  #

		#   angles[:,0,:]   = torch.where(mask0, angles[:,0,:], torch.pi - angles[:,0,:])
		#   angles[:,-1,:]  = torch.where(mask1, angles[:,-1,:], torch.pi - angles[:,-1,:])

			return angles

	def _interface_r(self, polarization, n_i, n_f, th_i, th_f):
			"""
			Calculates the reflection amplitude using the Fresnel equations.

			Args:
					polarization (str): Polarization ('s' or 'p').
					n_i (torch.Tensor): Refractive index of the incident medium.
					n_f (torch.Tensor): Refractive index of the final medium.
					th_i (torch.Tensor): Propagation angle in the incident medium.
					th_f (torch.Tensor): Propagation angle in the final medium.

			Returns:
					torch.Tensor: Reflection amplitude.
			"""
			th_i_cos = torch.cos(th_i)
			th_f_cos = torch.cos(th_f)
			if polarization == 's':
					return ((n_i * th_i_cos - n_f * th_f_cos) /
									(n_i * th_i_cos + n_f * th_f_cos + EPSILON))
			elif polarization == 'p':
					return ((n_f * th_i_cos - n_i * th_f_cos) /
									(n_f * th_i_cos + n_i * th_f_cos + EPSILON))
			else:
					raise ValueError("Polarization must be 's' or 'p'")

	def _interface_t(self, polarization, n_i, n_f, th_i, th_f):
			"""
			Calculates the transmission amplitude using the Fresnel equations.

			Args:
					polarization (str): Polarization ('s' or 'p').
					n_i (torch.Tensor): Refractive index of the incident medium.
					n_f (torch.Tensor): Refractive index of the final medium.
					th_i (torch.Tensor): Propagation angle in the incident medium.
					th_f (torch.Tensor): Propagation angle in the final medium.

			Returns:
					torch.Tensor: Transmission amplitude.
			"""
			th_i_cos = torch.cos(th_i)
			th_f_cos = torch.cos(th_f)

			if polarization == 's':
					return 2 * n_i * th_i_cos / (n_i * th_i_cos + n_f * th_f_cos + EPSILON)
			elif polarization == 'p':
					return 2 * n_i * th_i_cos / (n_f * th_i_cos + n_i * th_f_cos + EPSILON)
			else:
					raise ValueError("Polarization must be 's' or 'p'")
				
	# FORWARD METHOD
	def coh_tmm(
		self, 
		pol:str, 
		n_list:torch.Tensor, 
		d_list:torch.Tensor, 
		th_0:torch.Tensor, 
		wn:torch.Tensor, 
		**kwargs
	)  -> Tuple[torch.Tensor, torch.Tensor]|Dict[str, torch.Tensor]:
		"""
		Main "coherent transfer matrix method" calculation. This function simulates how
		light propagates through a stack of layers with different refractive indices.

		Args:
			pol (str): Polarization of the light, either "s" or "p".
			n_list (torch.Tensor): Tensor of refractive indices for each layer in the
				stack, with dimensions [num_samples, num_layers, num_wn].
			d_list (torch.Tensor): Tensor of layer thicknesses corresponding to n_list,
				with dimensions [num_samples, num_layers].
			th_0 (torch.Tensor): Angle of incidence in radians, with dimensions
				[num_samples, 1]. Should be complex if n_list[0] is complex.
			wn (torch.Tensor): Wavenumber of the incident light, with dimensions [num_wn].

		Returns:
			dict: Dictionary containing the following keys:
				- 'r': Reflection amplitude.
				- 't': Transmission amplitude.
				- 'R': Reflected power (fraction of incident power).
				- 'T': Transmitted power (fraction of incident power).
				- 'power_entering': Power entering the first layer.
				- 'vw_list': List of forward- and backward-traveling wave amplitudes
				in each layer (currently set to None).
				- 'kz_list': Normal component of the complex angular wavenumber in each layer.
				- 'th_list': Propagation angle (complex) in each layer.
				- 'pol': Input polarization.
				- 'n_list': Input refractive indices.
				- 'd_list': Input layer thicknesses.
				- 'th_0': Input angle of incidence.
				- 'lam_vac': Input vacuum wavelength (wn).

		Notes:
			The function handles cases where the imaginary part of the phase thickness
			(delta) exceeds a certain threshold (35). In such cases, the value is
			clamped to avoid numerical issues.
		"""
		num_samples     = n_list.shape[-3]
		num_layers      = n_list.shape[-2]
		num_wn          = n_list.shape[-1]
		num_calc_layers = num_layers - 2

		# Predefine tensors
		M_list = torch.zeros((num_samples, num_layers, num_wn, 2, 2), dtype=self.dtc, device=self.device)
		A_tensor = torch.zeros((num_samples, num_calc_layers, num_wn, 2, 2), dtype=self.dtc, device=self.device)
		B_tensor = torch.ones((num_samples, num_calc_layers, num_wn, 2, 2), dtype=self.dtc, device=self.device)

		# BLOCK 1
		# th_list = self._list_snell(n_list, th_0.clone())
		th_list = self._list_snell(n_list, th_0)
		kz_list = 2 * torch.pi * n_list * torch.cos(th_list) / wn
		delta   = kz_list * d_list.unsqueeze(2)

		# BLOCK 2
		delta_to_check = delta[:, 1:num_layers-1]
		mask = delta_to_check.imag > 35
		delta_to_check[mask] = delta_to_check[mask].real + 35j
		delta[:, 1:num_layers-1] = delta_to_check

		t_list = self._interface_t(pol, n_list[:,:-1], n_list[:,1:], th_list[:,:-1], th_list[:,1:]) #
		r_list = self._interface_r(pol, n_list[:,:-1], n_list[:,1:], th_list[:,:-1], th_list[:,1:]) #

		exp_neg_delta = torch.exp(-1j * delta[:,1:num_layers-1])
		exp_pos_delta = torch.exp(1j * delta[:,1:num_layers-1])

		#
		A_tensor[..., 0,0] = exp_neg_delta
		A_tensor[..., 1,1] = exp_pos_delta

		B_tensor[..., 0,1] = r_list[:,1:]
		B_tensor[..., 1,0] = r_list[:,1:]

		t_values_inv_0 = 1 / (t_list[:,0] + EPSILON) # Inverse of t values for division
		t_values_inv_1 = 1 / (t_list[:,1:] + EPSILON) # Inverse of t values for division

		M_list[:,1:num_layers-1] = t_values_inv_1.view(num_samples,num_calc_layers,num_wn,1,1) * torch.matmul(A_tensor,B_tensor)

		Mtilde = torch.eye(2, dtype=self.dtc, device=self.device).repeat(num_samples, num_wn, 1, 1)
		for i in range(1, num_layers-1):
				Mtilde = torch.matmul(Mtilde, M_list[:,i,...])

		arr = torch.ones((num_samples, num_wn, 2, 2), dtype=self.dtc, device=self.device)
		arr[..., 0, 1] = r_list[:,0]
		arr[..., 1, 0] = r_list[:,0]
		M = torch.matmul((arr * t_values_inv_0[..., None, None]), Mtilde)

		# n_i, n_f, th_i, th_f = n_list[:,0].clone(), n_list[:,-1].clone(), th_0.clone(), th_list[:,-1].clone()
		n_i, n_f, th_i, th_f = n_list[:,0], n_list[:,-1], th_0[:,0], th_list[:,-1]

		cos_tf = torch.cos(th_f)
		cos_ti = torch.cos(th_i)[:, None]

		if pol == 's': t_pow = (n_f*cos_tf).real / ((n_i*cos_ti).real + EPSILON)
		elif pol == 'p': t_pow = (n_f*torch.conj(cos_tf)).real / ((n_i*torch.conj(cos_ti)).real + EPSILON)
		else: raise ValueError("Polarization must be 's' or 'p'")

		res = self.forward_func[self.return_mode](M, t_pow)

		if self.return_mode == 'dict':
			res['n'] = n_list

		return res

	def forward(self, params:torch.Tensor|OpticalParameters, wn:torch.Tensor, **kwargs):
		"""
		Forward pass for calculating the reflection and transmission spectra of a
		layered structure.

		Args:
			wn (torch.Tensor): Tensor of wavenumbers with shape [num_wn].
			params (torch.Tensor): Tensor of optical parameters with shape
					[nxlxa] where:
					- n is the number of samples.
					- l is the number of layers.
					- a is the number of parameters (angle, thickness, oscillator parameters).

		Keyword Args (**kwargs):
			input_mode (str): Specifies the input mode for `params`, default is 'nxlxa'.
			n (torch.Tensor, optional): Precomputed refractive index tensor to bypass
					the oscillator model computation.

		Returns:
			Tuple[Dict[str, torch.Tensor], List[int], List[float]]:
					- Dictionary containing the computed refractive indices, reflection,
					transmission, reflected power, and transmitted power.
					- List of dimensions [num_samples, num_layers, num_wn].
					- List of elapsed times for the refractive index computation and TMM
					computation.

		Notes:
			- The method computes the refractive indices using the oscillator model
			unless provided through the keyword argument `n`.
			- Ensures that the wavenumber tensor `wn` does not contain zero values,
			replacing them with a small epsilon if necessary.
		"""
		input_mode = kwargs.pop('input_mode', 'nxlxa')

		assert_str_1 = f"Input mode is not compatible with the input dimensions {len(input_mode)}-{1 + 2*(params.ndim-1)}."
		assert_str_2 = "Wavenumber cannot be zero."

		assert len(input_mode) == 1 + 2*(params.ndim-1), assert_str_1
  
		if isinstance(params, torch.Tensor):
			params, (num_samples, num_layers, num_params) = remap_params(input_mode, params)
		else:
			params = params[None]
			(num_samples, num_layers, num_params) = params.shape

		num_wn = wn.shape[-1]           # Number of wavenumbers
		th_0 = params[..., 0]           # There is only one angle of incidence; it's the same for all layers
		th_0_ = torch.zeros((num_samples, 1), dtype=self.dtype, device=self.device)
		th_0 = torch.concat((th_0_, th_0, th_0_), axis=1)
		thickness = params[..., 1]      #
		opt_params = params[..., 2:]    #

		#
		wn = torch.where(wn==0, wn+EPSILON, wn)
		freq = 1/wn
		assert torch.where(wn==0, wn+EPSILON, wn).sum() > 0, assert_str_2

		# 1. Calculate n, e
		n = kwargs.pop('n', None)
		t_n_s = perf_counter()
		if n is None:
				# resp_osc = self.osc_model(wn, opt_params.clone(), input_mode)
				resp_osc = self.osc_model(wn, opt_params, input_mode)
				if self.osc_model.return_mode == 'both':
						(n, e), dn_dp = resp_osc
				else:
						n, e = resp_osc
		elif not isinstance(n, torch.Tensor): n = torch.tensor(n, dtype=self.dtc, device=self.device)
		t_n_e = perf_counter()

		# 2. Calculate TMM
		t_rt_s = perf_counter()
		n_basic = torch.ones(num_samples, num_layers+2, num_wn, dtype=self.dtc, device=self.device)
		d_basic = torch.ones(num_samples, num_layers+2, dtype=self.dtype, device=self.device) * 1e12
		d_basic[:,1:-1] = thickness
		n_basic[:, 1:-1] = n

		res = self.coh_tmm('s', n_basic, d_basic, th_0, freq)
		t_rt_e = perf_counter()

		# 3. Result
		return res, [num_samples, num_layers, num_wn], [t_n_e - t_n_s, t_rt_e - t_rt_s]

	# PLOT
	def plot_result(self, res:Dict[str, torch.Tensor], wn:torch.Tensor|NDArray|None=None, plot:bool=True):
		fig:plt.Figure
		ax:plt.Axes
		fig, ax = plt.subplots(1, 2, figsize=(12, 6))

		if wn is None:
				x = np.arange(res['R'].shape[-1])
				x_label = 'Index'
		elif isinstance(wn, torch.Tensor):
				x = wn.detach().cpu().numpy()
				x_label = 'Wavenumber'
		else:
				x = wn
				x_label = 'Wavenumber'

		ax[0].plot(x, pt2np(res['T'][0]), label='T')
		ax[1].plot(x, pt2np(res['R'][0]), label='R')

		ax[0].set_title('Transmitted Power')
		ax[1].set_title('Reflected Power')
		ax[0].set_xlabel(x_label)
		ax[1].set_xlabel(x_label)
		ax[0].set_ylabel('Power')
		ax[1].set_ylabel('Power')
		ax[0].legend()
		ax[1].legend()

		if plot:
				plt.show()

		return fig, ax

class NoTmm():
  def __init__(self, dt, dv, export_mode:str) -> None:
    translate_export_mode = {
      'fd_R': 0,
      'fd_r': 1,
      'fd_T': 0,
      'fd_t': 1,
      'fd_r_real': 0,
      'fd_r_imag': 0,
      'fd_t_real': 0,
      'fd_t_imag': 0,
      'td_r': 0,
      'td_t': 0,
      'R': 1,
      'T': 1,
    }
    self.return_tuple = translate_export_mode.get(export_mode, 1)
    self.dtype = dt
    self.device = dv
  
  def __call__(self, p:torch.Tensor, wn:torch.Tensor, *args) -> torch.Tensor|Tuple[torch.Tensor, torch.Tensor]:
    if self.return_tuple:
      return torch.zeros(p.shape[0], wn.shape[-1], dtype=self.dtype, device=self.device), torch.zeros(p.shape[0], wn.shape[-1], dtype=self.dtype, device=self.device)
    return torch.zeros(p.shape[0], wn.shape[-1], dtype=self.dtype, device=self.device)
