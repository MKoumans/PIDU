from typing import List, Tuple, Callable, Dict

from numpy import ndarray, abs as np_abs, flip as np_flip
from numpy.fft import fftshift as np_fftshift, irfft as np_irfft, \
    rfft as np_rfft
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from pandas import read_excel
import numpy as np

from torch import tensor, Tensor, flip as t_flip, device as pt_device, \
    abs as t_abs, real as t_real, imag as t_imag, nonzero, cuda, \
    stack as t_stack, min as t_min, max as t_max, cat as t_cat, argsort, \
    where as t_where, zeros as pt_zeros, float64 as pt_float64, float32 as \
    pt_float32, linspace, logspace as pt_logspace, log10, bucketize, normal, \
    dtype as pt_dtype, zeros_like as pt_zeros_like
from torch.nn import ReLU
from torch.fft import fftshift, irfft, rfft
import matplotlib.pyplot as plt
import time

numerical_stability = 1e-6

NUMPY_EPSILON = np.finfo(np.float64).eps

# --- FUNCTIONS ---
def create_wn(wn_spacing:str, wn_ran:Tensor, device:pt_device, dtype:pt_dtype) -> Tensor:
  if wn_spacing == 'linear':
    return linspace(*wn_ran, dtype=dtype, device=device)  # Wavenumber
  
  return pt_logspace(
    start=log10(wn_ran[0]), 
    end=log10(wn_ran[1]),
    steps=wn_ran[2], 
    base=10,
    dtype=dtype, device=device
  )  # Wavenumber

def set_device():
  if cuda.is_available():
    cuda.empty_cache()
    cuda.reset_max_memory_allocated()
    cuda.reset_max_memory_cached()
    device = pt_device("cuda") # Device
  else:
    device = pt_device("cpu") # Device
  return device

def pt2np(pt:Tensor) -> NDArray:
  if isinstance(pt, Tensor):
    return pt.cpu().detach().numpy()
  return pt

def np2pt(np_arr:NDArray, dtype:pt_dtype, device:pt_device) -> Tensor:
  if not isinstance(np_arr, Tensor):
    return tensor(np_arr, dtype=dtype, device=device)
  return np_arr

def get_matches(labels:Tensor, **kwargs):
  """

  """
  sample  = kwargs.get('sample', None)    #
  dt_to   = kwargs.get('dt_to', 'torch')  # Data type to return (numpy/torch)
  # dt      = kwargs.get('dt', torch.int64) #
  # dv      = kwargs.get('dv', torch.device("cuda" if torch.cuda.is_available() else "cpu"))

  if sample is not None: labels = labels[sample, ...]

  has_bbox = labels[..., 0]     # Mask for object where object is present
  idx_true = nonzero(has_bbox)  # Returns indices of the elements that are non-zero

  if dt_to == 'numpy':
    return pt2np(idx_true)

  return idx_true

def translate_anchor_to_bbox(scale:int, position:int, anchor:Tensor, region_sizes:List[int], stepsize:float):
  """
  Translates an anchor representation into a bounding box format.

  Args:
    scale (int): The scale of the anchor.
    position (int): The position of the anchor.
    anchor (Tensor): The anchor sizes (given in wavenumbers)
    region_sizes (List[int]): The region sizes (given in datapoints)

  Returns:
    List[float]: A list representing the bounding box with [start, middle, end] coordinates in wavenumbers.
  """
  region_size = region_sizes[scale]
  middle      = position*region_size + region_size/2 #
  side_width  = anchor[scale]/2 #
  return [middle-side_width, middle, middle+side_width]

def gamma_scale(x:Tensor, gamma:Tensor, a:float=1e-4, b:float=1.0) -> Tensor:
    y = x**gamma - a**gamma
    y = y / (b**gamma - a**gamma)  # Scale to [0, 1]
    return y

# def translate_anchor_to_bbox2(scale:int, position:int, anchor:Tensor, region_sizes:List[int], wn:Tensor) -> List[float]:
#   n_regs = int((wn[-1] - wn[0]) / region_sizes[scale])
#   factor = 
#   middle_points = linspace(
#     start=wn[0],
#     end=wn[-1],
#     steps=n_regs    
#   )
#   middle_points += (middle_points[1]-middle_points[0])/2  # Adjust middle points to start at wn[0]
#   side_positions = 

def translate_target_to_bbox(target:Tensor, middle:callable, width:callable) -> List[float]:
  """
  Translates a target into a bounding box format.

  Args:
    target (Tensor): The target tensor.

  Returns:
    List[float]: A list representing the bounding box with [start, middle, end] coordinates in wavenumbers.
  """
  middle_value = middle(*pt2np(target[1:]))
  width_value  = width(*pt2np(target[1:]))
  side_width  = width_value/2
  vlines = [middle_value - side_width, middle_value, middle_value + side_width]
  return vlines

def iou(bbox1:List[Tensor], sorted_bb:Tensor):
  relu    = ReLU()
  bbox2   = t_stack((sorted_bb[:, -3], sorted_bb[:, 2]), dim=1)

  left    = t_max(bbox1[0] - bbox1[1] / 2, bbox2[:, 0] - bbox2[:, 1] / 2)
  right   = t_min(bbox1[0] + bbox1[1] / 2, bbox2[:, 0] + bbox2[:, 1] / 2)
  intersection = relu(right-left)
  union = bbox1[1] + bbox2[:, 1] - intersection

  return intersection / union

def nms(target:List[Tensor], iou_threshold:float=0.5, conf_threshold:float=0.5) -> Tensor:
  is_conf = [target[i][:,:,:,0] >= conf_threshold for i in range(len(target))] # Get the confidence values for all bounding boxes.

  # filter out bounding boxes with confidence below the threshold.
  bboxes      = [target[i][is_conf[i]] for i in range(len(target))] # Get the bounding boxes with confidence above the threshold.
  stacked_bb  = t_cat(bboxes) # Stack the bounding boxes.
  sorted_bb   = stacked_bb[stacked_bb[:, 0].sort(descending=True)[1]]
  bboxes_nms  = []

  while sorted_bb.size(0) > 0:
    max_conf_box = sorted_bb[0]     # Select the bounding box with the highest confidence.
    bboxes_nms.append(max_conf_box) #

    if sorted_bb.size(0) == 1:
        break

    # Compute IoU of the highest confidence box with the rest.
    max_conf_dims = [max_conf_box[-3], max_conf_box[2]]

    ious = iou(max_conf_dims, sorted_bb[1:])

    # Keep only boxes with IoU below the threshold.
    sorted_bb = sorted_bb[1:][ious < iou_threshold]

  return t_stack(bboxes_nms, dim=0)

def get_slope(arr: Tensor) -> Tensor:
    # TODO: Implement this function correctly
    max_val, _ = arr.max(dim=1)
    # Filter for values within 5% of max
    filtered_indices = t_where(arr >= 0.90 * max_val[:, None], arr, -10)
    _, lowest_index = filtered_indices.sort(dim=1, descending=True)
    return lowest_index

def ret_zeros(x:Tensor, p:Tensor):
  return pt_zeros((p.shape[0], len(x)), dtype=x.dtype, device=x.device, requires_grad=False)

def savefig_debug(x:Tensor, name='debug.png'):
  plt.plot(pt2np(x))
  plt.savefig(name)
  plt.close()

def digitize_data(x:Tensor, y:Tensor, bins:Tensor) -> Tuple[Tensor, Tensor]:
  if y.ndim == 1:
    y = y[None,:]
  
  if x.ndim > 1:
    x = x[0]
  
  # get x-
  
  bin_indices = bucketize(x, bins, right=False)
  n_bins = len(bins)
  
  # Vectorized approach: compute means and stds for each bin
  bin_means = []
  bin_stds = []
  
  for bin_idx in range(n_bins):
    mask = bin_indices == bin_idx
    if mask.any():
      # Extract values for this bin across all samples
      bin_values = y[:, mask]  # Shape: (n_samples, n_values_in_bin)
      bin_means.append(bin_values.mean(dim=1).mean())  # Average across samples and values
      bin_stds.append(bin_values.std())
    else:
      bin_means.append(tensor(0.0, dtype=y.dtype, device=y.device))
      bin_stds.append(tensor(0.0, dtype=y.dtype, device=y.device))
  
  return t_stack(bin_means), t_stack(bin_stds)

def compute_spline(x, x_p, means, stds, n:int=1):
  from scipy.interpolate import CubicSpline
  
  f_p = normal(means[None,:].repeat(n, 1), stds[None,:].repeat(n, 1))
  
  y = CubicSpline(pt2np(x_p), pt2np(f_p), bc_type='natural', axis=1)
  random_points = y(pt2np(x))
  
  rp = np2pt(random_points, dtype=x.dtype, device=x.device)

  return rp, f_p

def filter_data(x, x_min:float, x_max:float):
  mask = (x >= x_min) & (x <= x_max)
  return mask

def generate_noise_profile_uc2(ds_path:str, wn, bins, n_samples:int=1, dtype=pt_float32, device=pt_device("cpu")) -> Dict[str, Tensor]:
  data = read_excel(ds_path, sheet_name=None, index_col=0)
  
  data_x_real = np2pt(data['real_x'].to_numpy(), dtype=dtype, device=device)
  data_y_real = np2pt(data['real_y'].to_numpy(), dtype=dtype, device=device)
  data_x_imag = np2pt(data['imag_x'].to_numpy(), dtype=dtype, device=device)
  data_y_imag = np2pt(data['imag_y'].to_numpy(), dtype=dtype, device=device)
  data_real_diff = np2pt(data['real_diff'].to_numpy(), dtype=dtype, device=device)
  data_imag_diff = np2pt(data['imag_diff'].to_numpy(), dtype=dtype, device=device)
  
  mask = filter_data(data_x_real[0], bins[0], bins[-1]) # create mask x-meas 
  wn_mask = filter_data(wn, bins[0], bins[-1])

  means_r, stds_r = digitize_data(data_x_real[:,mask], data_real_diff[:,mask], bins)
  means_i, stds_i = digitize_data(data_x_imag[:,mask], data_imag_diff[:,mask], bins)

  random_points_r, _ = compute_spline(wn, bins, means_r, stds_r, n_samples)
  random_points_i, _ = compute_spline(wn, bins, means_i, stds_i, n_samples)
  
  return {'system2_0': random_points_r, 'system2_1': random_points_i}, wn_mask

def generate_noise_profile_uc2_sys1(ds_path:str, wn, bins, n_samples:int=1, dtype=pt_float32, device=pt_device("cpu")) -> Dict[str, Tensor]:
  data = read_excel(ds_path, sheet_name=None, index_col=0)
  data_x_real = np2pt(data['refl_x'].to_numpy(), dtype=dtype, device=device)  # 
  data_y_real = np2pt(data['refl_y'].to_numpy(), dtype=dtype, device=device)  # 
  data_diff = np2pt(data['refl_diff'].to_numpy(), dtype=dtype, device=device) #
  
  diff_mean = data_diff.mean(0)
  
  # wn_mask, should be between both wn and data_x_real
  wn_mask = filter_data(wn, data_x_real[0].min(), data_x_real[0].max())
  
  # stupid non-monotonic x-value; should be fixed before interpolation
  for i in range(1, len(data_x_real[0])):
    if i >= len(data_x_real[0]):
      break
    if data_x_real[0, i] <= data_x_real[0, i-1]:
      # pop the non-monotonic point
      data_x_real = t_cat((data_x_real[:, :i], data_x_real[:, i+1:]), dim=1)  # 
      data_y_real = t_cat((data_y_real[:, :i], data_y_real[:, i+1:]), dim=1)  # 
      diff_mean = t_cat((diff_mean[:i], diff_mean[i+1:]), dim=0)              # 
  
  diff_mean_interpolated = CubicSpline(pt2np(data_x_real[0]), pt2np(diff_mean), bc_type='natural')(pt2np(wn))
  
  return t_where(wn_mask, np2pt(diff_mean_interpolated, dtype=wn.dtype, device=wn.device), pt_zeros_like(wn))

def generate_noise_profile_uc3(ds_path:str, wn, bins, n_samples:int=1, dtype=pt_float32, device=pt_device("cpu")) -> Dict[str, Tensor]:
  data = read_excel(ds_path, index_col=0)
  
  if isinstance(data, dict):
    data = data['Sheet1']
  
  data_diff = np2pt(data.to_numpy(), dtype=dtype, device=device) # First row is x, rest are y-differences
  
  mask = filter_data(data_diff[0], bins[0], bins[-1]) # mask for the x values in the dataset
  wn_mask = filter_data(wn, bins[0], bins[-1])        # mask for the wavenumbers from simulator

  means_r, stds_r = digitize_data(data_diff[0,mask], data_diff[1:,mask], bins) # digitize the data to get means and stds for each bin
  random_points_r, _ = compute_spline(wn, bins, means_r, stds_r, n_samples) # compute the spline to get random points for each wavenumber in the simulator
  
  return {'system1': random_points_r}, wn_mask

# ------------------

# Forwardpropagation [Pytorch]
def signal_forward_simulate(r, t, eta_sub, eta_ss, t_pow):
  """

  """
  t_fd = t_pow*abs(t)**2
  r_fd = abs(r)**2
  t_td = t_flip(fftshift(irfft(r)), [1])
  r_td = t_flip(fftshift(irfft(t)), [1])
  return t_td, r_td, t_fd, r_fd

# Backpropagation for d
def signal_backwards_d_fd_t(dr_dd, dt_dd, **kwargs):
  # dra_dd, drb_dd  = torch.real(dr_dd), torch.imag(dr_dd)
  dta_dd, dtb_dd  = t_real(dt_dd), t_imag(dt_dd)
  der_fd_t        = t_real(kwargs['eta_sub']/kwargs['eta_ss'])*2*(kwargs['ta'] * dta_dd + kwargs['tb'] * dtb_dd)
  return der_fd_t.squeeze()

def signal_backwards_d_fd_r(dr_dd, dt_dd, **kwargs):
  dra_dd, drb_dd  = t_real(dr_dd), t_imag(dr_dd)
  # dta_dd, dtb_dd  = torch.real(dt_dd), torch.imag(dt_dd)
  der_fd_t = 2 * (kwargs['ra'] * dra_dd + kwargs['rb'] * drb_dd)
  return der_fd_t.squeeze()

def signal_backwards_d_td_t(dr_dd, dt_dd, **kwargs):
  der_td_t        = t_flip(fftshift(irfft(dr_dd, dim=1), dim=1), [1]).squeeze()
  return der_td_t

def signal_backwards_d_td_r(dr_dd, dt_dd, **kwargs):
  der_td_r        = t_flip(fftshift(irfft(dt_dd, dim=1), dim=1), [1]).squeeze()
  return der_td_r

# Backpropagation for n
def signal_backwards_n_fd_t(dr_ddl, dt_ddl, **kwargs):
  return kwargs['t_fac'][None, :, None]*2*(kwargs['ta'][:, :, None]*t_real(dt_ddl) + kwargs['tb'][:, :, None]*t_imag(dt_ddl))

def signal_backwards_n_fd_r(dr_ddl, dt_ddl, **kwargs):
  return 2*(kwargs['ra'][:, :, None]*t_real(dr_ddl) + kwargs['rb'][:, :, None]*t_imag(dr_ddl))

def signal_backwards_n_td_t(dr_ddl, dt_ddl, **kwargs):
  return t_flip(fftshift(irfft(dt_ddl, dim=1), dim=1), [1])

def signal_backwards_n_td_r(dr_ddl, dt_ddl, **kwargs):
  return t_flip(fftshift(irfft(dr_ddl, dim=1), dim=1), [1])

# NEW FUNCTIONS
def pt2np(pt:Tensor) -> ndarray:
  if isinstance(pt, Tensor):
    return pt.cpu().detach().numpy()
  if pt is None:
    return None
  return pt

def np2pt(np_arr:ndarray, dtype:pt_dtype, device:pt_device) -> Tensor:
  if not isinstance(np_arr, Tensor):
    return tensor(np_arr, dtype=dtype, device=device)
  return np_arr

# -- PYTORCH FORWARD --
def signal_forward_dict(M, t_pow):
  """

  """
  out = {}
  r = M[..., 1, 0]/M[..., 0, 0]
  t = 1/M[..., 0, 0]

  out['r_fd'] = r
  out['t_fd'] = t
  out['T'] = t_abs(t)**2 * t_pow
  out['R'] = t_abs(r)**2
  out['t_td'] = t_flip(fftshift(irfft(t)), [1])
  out['r_td'] = t_flip(fftshift(irfft(r)), [1])
  return out

def signal_forward_all(M, t_pow):
  """

  """
  r = M[..., 1, 0]/M[..., 0, 0]
  t = 1/M[..., 0, 0]

  t_fd = t_abs(t)**2 * t_pow
  r_fd = t_abs(r)**2

  t_td = t_flip(fftshift(irfft(t)), [1])
  r_td = t_flip(fftshift(irfft(r)), [1])

  t_fd_real = (t * t_pow).real
  t_fd_imag = (t * t_pow).imag

  return ((t_td, r_td), (t_fd, r_fd), (t_fd_real, t_fd_imag))

def signal_forward_r(M):
  """

  """
  r = M[..., 1, 0]/M[..., 0, 0]
  r_fd = t_abs(r)**2
  r_td = t_flip(fftshift(irfft(r)), [1])
  return r_td, r_fd

def signal_forward_t(M, t_pow):
  """

  """
  t = 1/M[..., 0, 0]
  t_fd = t_abs(t)**2 * t_pow
  t_td = t_flip(fftshift(irfft(t)), [1])
  return t_td, t_fd

def forward_fd_R(*args):
  return t_abs(args[0][..., 1, 0]/args[0][..., 0, 0])**2

def forward_fd_T(*args):
  return t_abs(1/args[0][..., 0, 0])**2*args[1]

def forward_fd_r_real(*args):
  return (args[0][..., 1, 0]/args[0][..., 0, 0]).real

def forward_fd_r_imag(*args):
  return (args[0][..., 1, 0]/args[0][..., 0, 0]).imag

def forward_fd_r(*args):
  return (args[0][..., 1, 0]/args[0][..., 0, 0]).real, (args[0][..., 1, 0]/args[0][..., 0, 0]).imag

def forward_fd_t_real(*args):
  return (1/args[0][..., 0, 0]*args[1]).real

def forward_fd_t_imag(*args):
  return (1/args[0][..., 0, 0]*args[1]).imag

def forward_fd_t(*args):
  return (1/args[0][..., 0, 0]*args[1]).real, (1/args[0][..., 0, 0]*args[1]).imag
  # return gamma_scale(1/args[0][..., 0, 0]*args[1].real, 0.15), gamma_scale(1/args[0][..., 0, 0]*args[1].imag, 0.15)

def forward_td_r(*args):
  return t_flip(fftshift(irfft(args[0][..., 1, 0]/args[0][..., 0, 0])), [1])

def forward_td_t(*args):
  return t_flip(fftshift(irfft(1/args[0][..., 0, 0])), [1]),

forward_func = {
  'dict': signal_forward_dict,
  'all': signal_forward_all,
  'fd_R': forward_fd_R,
  'fd_r': forward_fd_r,
  'fd_T': forward_fd_T,
  'fd_t': forward_fd_t,
  'fd_r_real': forward_fd_r_real,
  'fd_r_imag': forward_fd_r_imag,
  'fd_t_real': forward_fd_t_real,
  'fd_t_imag': forward_fd_t_imag,
  'td_r': forward_td_r,
  'td_t': forward_td_t,
  'R': signal_forward_r,
  'T': signal_forward_t,
}
# ----------------------

# -- 

def _np_td_transform(signal: np.ndarray) -> np.ndarray:
	"""Return time-domain signal using numpy FFT utilities."""
	irfft_signal = np.fft.irfft(signal, axis=-1)
	shifted = np.fft.fftshift(irfft_signal, axes=-1)
	return np.flip(shifted, axis=1)


def _np_signal_forward_dict(M: np.ndarray, t_pow: np.ndarray) -> Dict[str, np.ndarray]:
	r = M[..., 1, 0] / M[..., 0, 0]
	t = 1 / M[..., 0, 0]
	return {
		'r_fd': r,
		't_fd': t,
		'T': np.abs(t) ** 2 * t_pow,
		'R': np.abs(r) ** 2,
		't_td': _np_td_transform(t),
		'r_td': _np_td_transform(r),
	}


def _np_signal_forward_all(M: np.ndarray, t_pow: np.ndarray):
	r = M[..., 1, 0] / M[..., 0, 0]
	t = 1 / M[..., 0, 0]
	t_fd = np.abs(t) ** 2 * t_pow
	r_fd = np.abs(r) ** 2
	t_td = _np_td_transform(t)
	r_td = _np_td_transform(r)
	t_fd_real = (t * t_pow).real
	t_fd_imag = (t * t_pow).imag
	return (t_td, r_td), (t_fd, r_fd), (t_fd_real, t_fd_imag)


def _np_signal_forward_r(M: np.ndarray, _: np.ndarray):
	r = M[..., 1, 0] / M[..., 0, 0]
	r_fd = np.abs(r) ** 2
	r_td = _np_td_transform(r)
	return r_td, r_fd


def _np_signal_forward_t(M: np.ndarray, t_pow: np.ndarray):
	t = 1 / M[..., 0, 0]
	t_fd = np.abs(t) ** 2 * t_pow
	t_td = _np_td_transform(t)
	return t_td, t_fd


def _np_forward_fd_R(M: np.ndarray, _: np.ndarray):
	return np.abs(M[..., 1, 0] / M[..., 0, 0]) ** 2


def _np_forward_fd_T(M: np.ndarray, t_pow: np.ndarray):
	return np.abs(1 / M[..., 0, 0]) ** 2 * t_pow


def _np_forward_fd_r(M: np.ndarray, _: np.ndarray):
	r = M[..., 1, 0] / M[..., 0, 0]
	return r.real, r.imag


def _np_forward_fd_t(M: np.ndarray, t_pow: np.ndarray):
	t = (1 / M[..., 0, 0]) * t_pow
	return t.real, t.imag


def _np_forward_fd_component(M: np.ndarray, t_pow: np.ndarray, component: str, part: str):
	if component == 'r':
		value = M[..., 1, 0] / M[..., 0, 0]
	else:
		value = (1 / M[..., 0, 0]) * (t_pow if component == 't' else 1)
	return getattr(value, part)


def _np_forward_td_r(M: np.ndarray, _: np.ndarray):
	return _np_td_transform(M[..., 1, 0] / M[..., 0, 0])


def _np_forward_td_t(M: np.ndarray, _: np.ndarray):
	return (_np_td_transform(1 / M[..., 0, 0])),


forward_func_numpy: Dict[str, callable] = {
	'dict': _np_signal_forward_dict,
	'all': _np_signal_forward_all,
	'fd_R': _np_forward_fd_R,
	'fd_r': _np_forward_fd_r,
	'fd_T': _np_forward_fd_T,
	'fd_t': _np_forward_fd_t,
	'fd_r_real': lambda M, t_pow: _np_forward_fd_component(M, t_pow, 'r', 'real'),
	'fd_r_imag': lambda M, t_pow: _np_forward_fd_component(M, t_pow, 'r', 'imag'),
	'fd_t_real': lambda M, t_pow: _np_forward_fd_component(M, t_pow, 't', 'real'),
	'fd_t_imag': lambda M, t_pow: _np_forward_fd_component(M, t_pow, 't', 'imag'),
	'td_r': _np_forward_td_r,
	'td_t': _np_forward_td_t,
	'R': _np_signal_forward_r,
	'T': _np_signal_forward_t,
}
