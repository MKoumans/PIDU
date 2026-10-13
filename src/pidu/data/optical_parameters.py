import torch
from numpy.typing import NDArray

class OpticalParameters():
  def __init__(
    self, 
    th0:float|torch.Tensor, 
    d:float|torch.Tensor, 
    eps_inf:float|torch.Tensor, 
    drude_p:torch.Tensor, 
    drude_g:torch.Tensor, 
    w0_before:float|torch.Tensor,
    wp_before:float|torch.Tensor,
    g_before:float|torch.Tensor,
    w0_after:float|torch.Tensor,
    wp_after:float|torch.Tensor,
    g_after:float|torch.Tensor,
    opt_params:torch.Tensor|None=None, 
    **kwargs
  ) -> None:
    self.device = kwargs.get('device', 'cpu')       # 
    self.dtype = kwargs.get('dtype', torch.float64) # 
    
    self.th0 = self._make_tensor(th0, self.device, self.dtype)               # [num_samples] | [num_samples, num_layers] 
    self.d = self._make_tensor(d, self.device, self.dtype)                   # [num_samples] | [num_samples, num_layers]
    self.eps_inf = self._make_tensor(eps_inf, self.device, self.dtype)       # [num_samples] | [num_samples, num_layers]
    self.drude_p = self._make_tensor(drude_p, self.device, self.dtype)       # [num_samples, num_layers, 2]
    self.drude_g = self._make_tensor(drude_g, self.device, self.dtype)       # [num_samples, num_layers, 2]
    self.w0_before = self._make_tensor(w0_before, self.device, self.dtype)   # [num_samples] | [num_samples, num_layers]
    self.wp_before = self._make_tensor(wp_before, self.device, self.dtype)   # [num_samples] | [num_samples, num_layers]
    self.g_before = self._make_tensor(g_before, self.device, self.dtype)     # [num_samples] | [num_samples, num_layers]
    self.w0_after = self._make_tensor(w0_after, self.device, self.dtype)     # [num_samples] | [num_samples, num_layers]
    self.wp_after = self._make_tensor(wp_after, self.device, self.dtype)     # [num_samples] | [num_samples, num_layers]
    self.g_after = self._make_tensor(g_after, self.device, self.dtype)       # [num_samples] | [num_samples, num_layers]
    self.opt_params = self._make_tensor(opt_params, self.device, self.dtype) # [num_samples] | [num_samples, num_opt_params] | [num_samples, num_layers, num_opt_params]
    
    assert len(th0) == len(d) == len(eps_inf) == len(drude_p) == len(drude_g) == len(opt_params), "All input parameters must have the same number of samples."
    assert self.d.shape[1] == self.eps_inf.shape[1] == self.opt_params.shape[1], "All input parameters must have the same number of layers."
    
    self.system_params_per_layer = 5 + 2*3 # 
    self.n = self.d.shape[0]            # 
    self.n_layers = self.d.shape[1]     #
    self.data = self._make_dataset()    # 
    self.data_sys = self._make_dataset_sys()  #
    self.ndim = self.data.ndim          # 
    self.shape = self.data.shape        # 
    self.input_mode = 'nxlxa'           # [num_samples, num_layers, num_attributes]  
      
  def __len__(self) -> int:
    return self.n
  
  def __getitem__(self, idx:int|slice|torch.Tensor|None) -> torch.Tensor:
    if idx is None:
      return self.data
    elif isinstance(idx, int):
      return self.data[idx][None, ...]
    else:
      return self.data[idx]
  
  def __repr__(self) -> str:
    return f"SystemParameters(n={self.n}, n_layers={self.n_layers}, attributes={self.data.shape[2]}, device={self.device}, dtype={self.dtype})"
  
  # -- PRIVATE FUNCTIONS -- #
  @staticmethod
  def _make_tensor(item, device, dtype) -> torch.Tensor:
    if isinstance(item, torch.Tensor):
      return item.to(device=device, dtype=dtype)
    else:
      return torch.tensor(item, device=device, dtype=dtype)

  def _make_dataset(self) -> torch.Tensor:
    if self.th0.ndim == 1:
      self.th0 = self.th0.unsqueeze(1)
    
    data = torch.zeros((self.n, self.n_layers, self.system_params_per_layer + self.opt_params.shape[2]), device=self.device, dtype=self.dtype)
    data[:,:,0] = self.th0            # 
    data[:,:,1] = self.d              # 
    data[:,:,2] = self.eps_inf        # 
    data[:,:,3] = self.drude_p        #
    data[:,:,4] = self.drude_g        #
    data[:,:,5] = self.w0_before
    data[:,:,6] = self.wp_before
    data[:,:,7] = self.g_before
    data[:,:,8] = self.w0_after
    data[:,:,9] = self.wp_after
    data[:,:,10] = self.g_after
    data[:,:,self.system_params_per_layer:] = self.opt_params    # 
    
    self.key_positions = {
      'th0': 0,
      'd': 1,
      'eps_inf': 2,
      'drude_p': 3,
      'drude_g': 4,
      'w0_before': 5,
      'wp_before': 6,
      'g_before': 7,
      'w0_after': 8,
      'wp_after': 9,
      'g_after': 10
    }
        
    return data

  def _make_dataset_sys(self) -> torch.Tensor:
    if self.th0.ndim == 1:
      self.th0 = self.th0.unsqueeze(1)
    
    data = torch.zeros((self.n, self.n_layers, self.system_params_per_layer), device=self.device, dtype=self.dtype)
    data[:,:,0] = self.th0            # 
    data[:,:,1] = self.d              #
    data[:,:,2] = self.eps_inf        #
    data[:,:,3] = self.drude_p        #
    data[:,:,4] = self.drude_g        #
    data[:,:,5] = self.w0_before
    data[:,:,6] = self.wp_before
    data[:,:,7] = self.g_before
    data[:,:,8] = self.w0_after
    data[:,:,9] = self.wp_after
    data[:,:,10] = self.g_after
    
    return self._make_tensor(data, self.device, self.dtype)

  # -- FUNCTIONS -- #
  def get_np(self, idx:int|slice|torch.Tensor|None=None) -> NDArray:
    return self[idx].detach().cpu().numpy()

  def set_parameter(
    self, 
    key:str, 
    value:float|torch.Tensor, 
    idx_sample:int|slice|torch.Tensor|None=None, 
    idx_layer:int|None=None
  ) -> None:
    if idx_sample is None:
      idx_sample = slice(None)
    if idx_layer is None:
      idx_layer = slice(None)
    
    if key not in self.key_positions:
      raise KeyError(f"Key '{key}' not found in system parameters.")
    
    pos = self.key_positions[key]
    new_value = self._make_tensor(value, self.device, self.dtype)

    self.data[idx_sample, idx_layer, pos] = new_value     # Update data tensor
    self.data_sys[idx_sample, idx_layer, pos] = new_value # Update system data tensor
  
  def copy(self) -> 'OpticalParameters':
    new_instance = OpticalParameters(
      th0=self.th0.clone(),
      d=self.d.clone(),
      eps_inf=self.eps_inf.clone(),
      drude_p=self.drude_p.clone(),
      drude_g=self.drude_g.clone(),
      w0_before=self.w0_before.clone(),
      wp_before=self.wp_before.clone(),
      g_before=self.g_before.clone(),
      w0_after=self.w0_after.clone(),
      wp_after=self.wp_after.clone(),
      g_after=self.g_after.clone(),
      opt_params=self.opt_params.clone(),
      device=self.device,
      dtype=self.dtype
    )

    return new_instance