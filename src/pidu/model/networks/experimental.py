"""
Archive of superseded / abandoned architecture experiments.

Nothing in this module is imported by any live code path (verified by a
repo-wide import search when this package was split out of the old
model/Layers.py and model/Functions.py). It is kept only as a historical
reference for past detection-head and physics-layer designs. Contents are
NOT maintained, NOT covered by tests, and may not even run (e.g. the
original DetectionHead references a variable, `strides_sys_obj`, that is
never assigned within its scope).

Live equivalents:
  blocks.py  - act_str_to_fn, ScalableTanh/ScalableTanhOutput, Backbone
  heads.py   - HeadCnnModel, HeadSSDModel, HeadSSDScaleSmall(+Cell/SysNetwork)
"""
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.functional import pad
import matplotlib.pyplot as plt

from pidu.data.physics import TmMethodPyTorch
from pidu.data.utils import pt2np
from pidu.model.networks.blocks import act_str_to_fn, ScalableTanh


def fig_debug(data:torch.Tensor, filepath:str):
  if data.ndim == 1:
    data = [data]
  
  if len(data) > 16:
    data = data[:16] # Limit to first 16 plots

  for id, d in enumerate(data):
    fig, ax = plt.subplots()
    ax.plot(pt2np(d))
    fig.savefig(f"{filepath}{id}.png")
    plt.close(fig)


def plot_dist(data:torch.Tensor, filepath:str):
  fig, ax = plt.subplots()
  ax.hist(pt2np(data).flatten(), bins=100)
  fig.savefig(filepath)
  plt.close(fig)


class PhysicsLayerBatch(nn.Module):
  """
  PhysicsLayer: Simulates the complex refractive index of a material (frequency-dependant).
  1: layer_number: Number of the layer
  2: wn: Wavenumbers
  3: direction: Direction of propagation (t, r)
  4: domain_id: Domain in which the input is (Index dependant on domains)
  5: domains: Domains in which the input can be. Default: [td, fd]
  device: Device to run on (cpu/cuda)
  """
  def __init__(self, wn: torch.Tensor, device: str, mode: str):
    super(PhysicsLayerBatch, self).__init__()
    self.wn   = wn    # x-axis: wavenumbers (torch.Tensor)
    self.dir  = mode[3:]  # Direction of propagation: t, r
    self.domain = mode[:2]  # Domains: td, fd
    self.mode   = mode    # Mode: (td_t, td_r, fd_t, fd_r)
    self.device = device  # Device: cpu, cuda
    self.name   = f'PhysicsLayer_{mode}' # Name
    self.dtype  = torch.float64

    self.tmm_system = TmMethodPyTorch(device=self.device, dtype=self.dtype, return_mode='all')

  def forward(self, p) -> torch.tensor:
    simulated_signal = SimulateFunction.apply(p, self.tmm_system)

    return simulated_signal


class CNNOut(nn.Module):
  def __init__(self, shape_info:Dict[str, int], sys_info:Dict[str, int], *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.n_features_in = shape_info['in_features']      # Number of input features
    self.n_features_out = shape_info['out_features']    # 
    self.n_total_regions = sys_info['n_total_regions']  # 
    self.n_sys_pars = sys_info['n_sys_pars']            #
    self.dt = sys_info['dtype']   # Data type
    self.dv = sys_info['device']  # Device to run on
    
    self.fc1 = nn.Linear(self.n_features_in + self.n_sys_pars, self.n_features_in, dtype=self.dt, device=self.dv)
    self.fc2 = nn.Linear(self.n_features_in, self.n_features_in//2, dtype=self.dt, device=self.dv)
    
    self.cnnout_fc_sys_system = nn.Linear(self.n_features_in//2, 3, dtype=self.dt, device=self.dv)  # System parameters output
    self.cnnout_fc_sys_drude = nn.Linear(self.n_features_in//2, 2, dtype=self.dt, device=self.dv)   # Drude parameters output
    self.cnnout_fc_sys_w0b = nn.Linear(self.n_features_in//2, 1, dtype=self.dt, device=self.dv)     # W0b output
    self.cnnout_fc_sys_wpb = nn.Linear(self.n_features_in//2, 1, dtype=self.dt, device=self.dv)     # Wpb output
    self.cnnout_fc_sys_gb = nn.Linear(self.n_features_in//2, 1, dtype=self.dt, device=self.dv)      # Gb output
    self.cnnout_fc_sys_w0a = nn.Linear(self.n_features_in//2, 1, dtype=self.dt, device=self.dv)     # W0a output
    self.cnnout_fc_sys_wpa = nn.Linear(self.n_features_in//2, 1, dtype=self.dt, device=self.dv)     # Wpa output
    self.cnnout_fc_sys_ga = nn.Linear(self.n_features_in//2, 1, dtype=self.dt, device=self.dv)      # Ga output
    
    self.cnnout_fc_osc_obj = nn.Linear(self.n_features_in//2, self.n_total_regions, dtype=self.dt, device=self.dv)  
    self.cnnout_fc_osc_w0 = nn.Linear(self.n_features_in//2, self.n_total_regions, dtype=self.dt, device=self.dv)
    self.cnnout_fc_osc_wb = nn.Linear(self.n_features_in//2, self.n_total_regions, dtype=self.dt, device=self.dv)
    self.cnnout_fc_osc_g = nn.Linear(self.n_features_in//2, self.n_total_regions, dtype=self.dt, device=self.dv)

    self.norms = nn.ModuleDict({
      'instance1': nn.InstanceNorm1d(self.n_features_in, dtype=self.dt, device=self.dv),
      'instance2': nn.InstanceNorm1d(self.n_features_in, dtype=self.dt, device=self.dv),
    })
      
    self.act1 = act_str_to_fn('tanh')  # Activation function for the first layer
    self.act2 = act_str_to_fn('tanh')  # Activation function for the first layer
    
    self.fc1 = self._init_weights(self.fc1)
    self.fc2 = self._init_weights(self.fc2)
    
    self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
  
  def _init_weights(self, net:nn.Linear):
    """
    Initialize weights of the network according to diagonal init for matrix, with small noise for other stuff.
    :param net: Linear layer to initialize
    :return: Initialized linear layer
    """
    net.weight.data = torch.eye(net.weight.shape[0], net.weight.shape[1], dtype=self.dt, device=self.dv) + \
      torch.normal(0, 0.05, size=net.weight.shape, dtype=self.dt, device=self.dv)
    
    return net
  
  def forward(self, x_in:torch.Tensor, z_sys:torch.Tensor=None) -> torch.Tensor:
    """
    Forward pass through the CNN output layer.
    :param x: Input tensor of shape [batch_size, n_features_in, signal_length]
    :return: Output tensor of shape [batch_size, n_features_out, signal_length]
    """
    x_input_network = torch.cat([x_in, z_sys], dim=-1) if z_sys is not None else x_in  # Concatenate system parameters if provided
    x = self.act1(self.fc1(x_input_network))
    # x = self.norms['instance1'](x) # Apply first instance normalization
    # x = x_in + self.act1(x) # Apply activation function
    
    x = self.act2(self.fc2(x))      # Apply second fully connected layer
    # x = self.norms['instance2'](x)  # Apply second instance normalization
    # x = x + self.act2(x)    # Apply activation function
    
    x_sys_system = self.cnnout_fc_sys_system(x, z_sys) # System parameters output
    x_sys_drude = self.cnnout_fc_sys_drude(x) # Drude parameters output
    x_sys_w0b = self.cnnout_fc_sys_w0b(x) # W0b output
    x_sys_wpb = self.cnnout_fc_sys_wpb(x) # Wpb output
    x_sys_gb = self.cnnout_fc_sys_gb(x) # Gb output
    x_sys_w0a = self.cnnout_fc_sys_w0a(x) # W0a output
    x_sys_wpa = self.cnnout_fc_sys_wpa(x) # Wpa output
    x_sys_ga = self.cnnout_fc_sys_ga(x) # Ga output
    
    x_osc_obj = self.cnnout_fc_osc_obj(x)        # Oscillator object output
    x_osc_w0 = self.cnnout_fc_osc_w0(x)          # Oscillator W0 output
    x_osc_wb = self.cnnout_fc_osc_wb(x)          # Oscillator Wb output
    x_osc_g = self.cnnout_fc_osc_g(x)            # Oscillator G output
    
    x_system_out = torch.cat([
      x_sys_system, 
      x_sys_drude, 
      x_sys_w0b, 
      x_sys_wpb, 
      x_sys_gb, 
      x_sys_w0a, 
      x_sys_wpa, 
      x_sys_ga
    ], dim=-1)  # Concatenate system outputs
    
    x_osc_out = torch.stack([x_osc_obj, x_osc_w0, x_osc_wb, x_osc_g], dim=-1) # Return N x 
    x_osc_out = x_osc_out.view(x_osc_obj.shape[0], self.n_total_regions*4)  # Reshape to [batch_size, n_attributes, n_regions]
    
    x_out = torch.cat([x_system_out, x_osc_out], dim=-1)  # Concatenate system and oscillator outputs
    
    return x_out  # Return the output tensor  


class DetectionHead(nn.Module):
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    system_info:dict,
    use_residual: bool = False,
  ):
    super(DetectionHead, self).__init__()
    self.unfold_idx = unfold_idx    # 
    self.scale_idx = scale_idx      # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.name = 'DetectionHead'     # 
    self.n_regions = n_regions = system_info['n_regions'] #
    self.n_layers = n_layers = system_info['n_layers']    #
    self.n_sys_pars = system_info['n_sys_pars']           #
    
    out_channels = 4

    self.flatten = nn.Flatten()         #
    self.act_fn1_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_3 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn2_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn2_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    
    self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    
    self.bn_1 = nn.BatchNorm1d(16, dtype=self.dt, device=self.dv)
    self.bn_2 = nn.BatchNorm1d(16, dtype=self.dt, device=self.dv)
    self.bn_3 = nn.BatchNorm1d(out_channels, dtype=self.dt, device=self.dv)
    
    stride = stride
    kernel_size = kernel_size
    
    # Define downsample layers for residual connections
    self.cell_downsample1 = nn.Conv1d(
        in_channels=in_features, 
        out_channels=16, 
        kernel_size=1, 
        stride=2, 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.cell_downsample2 = nn.Conv1d(
        in_channels=16, 
        out_channels=16, 
        kernel_size=1, 
        stride=2, 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    # System residual connections
    self.sys_downsample1 = nn.Conv1d(
        in_channels=in_features, 
        out_channels=4*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[0], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.sys_downsample2 = nn.Conv1d(
        in_channels=4*n_layers, 
        out_channels=1*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[1], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.sys_downsample3 = nn.Conv1d(
        in_channels=1*n_layers, 
        out_channels=1*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[2], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    # FC layer residual connections
    self.fc_downsample = nn.Linear(
        in_features=(n_regions*out_channels + self.n_sys_pars+n_regions)*n_layers,
        out_features=(n_regions*out_channels)*self.n_layers,
        dtype=self.dt,
        device=self.dv
    ) if use_residual else None
    
    self.conv1 = nn.Conv1d(
      in_channels=in_features, 
      out_channels=16, 
      kernel_size=3, 
      stride=1, 
      padding=(3-1)//2,
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    self.conv2 = nn.Conv1d(
      in_channels=16, 
      out_channels=16, 
      kernel_size=3, 
      stride=1, 
      padding=1,
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    self.conv3 = nn.Conv1d(
      in_channels=16, 
      out_channels=out_channels, 
      kernel_size=kernel_size, 
      stride=stride, 
      padding=(kernel_size - stride),
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    self.fc2 = nn.Linear(
      in_features=n_regions*out_channels,
      out_features=(n_regions*4 + self.n_sys_pars)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
    self.fc3 = nn.Linear(
      in_features=(n_regions*4 + self.n_sys_pars)*self.n_layers,
      out_features=(n_regions*4 + self.n_sys_pars)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
    self.fc4 = nn.Linear(
      in_features=(n_regions*4 + self.n_sys_pars)*self.n_layers,
      out_features=(n_regions*4 + self.n_sys_pars)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
    # store intermediate tensors
    self.activation_outputs = {}
    
    # register hook for the conv layer
    # self.conv.register_forward_hook(self._get_activation('conv')) 

  def _get_activation(self, name):
    def hook_fn(module, input, output):
      # store the output of the module
      self.activation_outputs[name] = input[0].detach()
    return hook_fn

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    for name, param in self.named_parameters():
      if param.grad is not None:
        gradients.append(param.grad)
        gradient_norms.append(param.grad.norm())
    
    gradients = [pt2np(g) for g in gradients if g is not None]
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _clear_activation(self):
    self.activation_outputs = {}

  def forward(self, x_in:torch.Tensor):
    n_samples = x_in.shape[0]                     # 
    x_cnn1 = self.act_fn1_1(self.conv1(x_in))     # 
    # x_cnn1 = self.bn_1(x_cnn1)                    #
    x_cnn2 = self.act_fn1_2(self.conv2(x_cnn1))   # 
    # x_cnn2 = self.bn_2(x_cnn2)                    #
    x_cnn3 = self.act_fn1_3(self.conv3(x_cnn2))   # [n_samples, 4, n_regions]
    # x_cnn3 = self.bn_3(x_cnn3)                    #
    x_f = self.flatten(x_cnn3)              # 
    x_fc2 = self.act_fn2_1(self.fc2(x_f))   # 
    x_fc3 = self.act_fn2_2(self.fc3(x_fc2)) # 
    x_fc4 = self.fc4(x_fc3)                 #   
    
    # reshape [64 -> (self.n_regions, 4)]
    x_oscs = x_fc4[:, :(self.n_regions*4)*self.n_layers].view(n_samples, self.n_layers, self.n_regions, -1) # 
    x_sys = x_fc4[:, (self.n_regions*4)*self.n_layers:].view(n_samples, self.n_layers, -1)   # 
    # out = x_oscs.view(n_samples, 4, self.n_regions) # will be permuted in the model
    
    return x_oscs, x_sys

  @property
  def fc_weights(self):
    """Return the weights of fc2, fc3 and fc4 for regularization purposes"""
    return [self.fc2.weight, self.fc3.weight, self.fc4.weight]


class DetectionHead2(nn.Module):
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    system_info:dict,
    communication:bool=True,
  ):
    super(DetectionHead2, self).__init__()
    self.unfold_idx = unfold_idx      # 
    self.scale_idx = scale_idx        # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.name = 'DetectionHead2'      # 
    self.n_regions = system_info['n_regions']        #
    self.n_layers = system_info['n_layers']          #
    self.output_size = 4 # output cell
    self.act_fn_str = act_fun_str #'stanh' # 
    self.signal_size = system_info['in_size'] #
    self.communication:bool = communication
    
    out_channels = 8

    self.flatten = nn.Flatten() #
    self.act_fn_cnn = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn_sys = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn_dec = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    
    stride = stride
    kernel_size = kernel_size
    
    self.conv1 = nn.Conv1d(
      in_channels=in_features, 
      out_channels=16, 
      kernel_size=3, 
      stride=1, 
      padding=(3-1)//2,
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    self.conv2 = nn.Conv1d(
      in_channels=16, 
      out_channels=out_channels, 
      kernel_size=kernel_size, 
      stride=stride, 
      padding=(kernel_size - stride),
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    self.conv_sys = nn.Sequential(
      nn.Conv1d(in_features, 4, 3, 2, 1, padding_mode='replicate', dtype=self.dt, device=self.dv),
      self.act_fn_sys,
      nn.Conv1d(4, 1, 5, 2, 2, padding_mode='replicate', dtype=self.dt, device=self.dv),
      nn.Flatten(),
    )
      
    self._init_cells(system_info['n_regions'], 5, 8, input_size_sys=self.signal_size//4)

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    gradients.append(self.conv1.weight.grad)
    gradients.append(self.conv2.weight.grad)
    
    for cell in self.cells:
      gradients.append(cell.conn_1.weight.grad)
      gradients.append(cell.conn_2.weight.grad)
      gradients.append(cell.conn_3.weight.grad)
    
    gradients.append(self.cell_system.conn_1.weight.grad)
    gradients.append(self.cell_system.conn_2.weight.grad)
    gradients.append(self.cell_system.conn_3.weight.grad)
      
    try:
      gradient_norms.append(self.conv1.weight.grad.norm())
      gradient_norms.append(self.conv2.weight.grad.norm())
      
      for cell in self.cells:
        gradient_norms.append(cell.conn_1.weight.grad.norm())
        gradient_norms.append(cell.conn_2.weight.grad.norm())
        gradient_norms.append(cell.conn_3.weight.grad.norm())
    
      gradient_norms.append(self.cell_system.conn_1.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_2.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_3.weight.grad.norm())
    except:
      # No gradient
      gradient_norms = [0]*len(gradients)
    
    gradients = [pt2np(g) for g in gradients if g is not None] 
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _init_weights(self, net:nn.Linear):
    """Initialize weights of the network according to diagonal init for matrix, with small noise for other stuff"""
    net.weight.data = torch.eye(net.weight.shape[0], net.weight.shape[1], dtype=self.dt, device=self.dv) + \
      torch.normal(0, 0.01, size=net.weight.shape, dtype=self.dt, device=self.dv)
    
    return net

  def _init_cells(self, number_of_cells:int, n_sys_pars:int, input_size:int, input_size_sys:int, output_size:int=4):
    list_of_cells = []
    
    for i in range(number_of_cells):
      cell = DetectionHeadCell(
        input_size=input_size,
        output_size=output_size,
        act_fun_str=self.act_fn_str, # 
        system_info={'dtype': self.dt, 'device': self.dv},
      )
      
      list_of_cells.append(cell)
      
      # Define communication layers
      if i == 0:
        comm_1 = cell.comm_out_1 # [2]
        comm_2 = cell.comm_out_2 # [2]
      else:
        comm_1 = torch.concat([comm_1, cell.comm_out_1], axis=1) # [2,...,2] + [2]
        comm_2 = torch.concat([comm_2, cell.comm_out_2], axis=1) # [2,...,2] + [2]
    
    # -- Add system head --
    self.cell_system = DetectionHeadSystem(
      input_size=input_size_sys,
      output_size=n_sys_pars,
      act_fun_str=self.act_fn_str, # 
      system_info={'dtype': self.dt, 'device': self.dv},
    )
    
    list_of_cells.append(self.cell_system)
    comm_1 = torch.concat([comm_1, self.cell_system.comm_out_1], axis=1)
    comm_2 = torch.concat([comm_2, self.cell_system.comm_out_2], axis=1)
    # ---------------------
    
    comm_1_size = comm_1.shape[0]*comm_1.shape[1]
    comm_2_size = comm_2.shape[0]*comm_2.shape[1]
    
    self.decoder_1 = nn.Linear(comm_1_size, comm_1_size, dtype=self.dt, device=self.dv, bias=False)
    self.decoder_1 = self._init_weights(self.decoder_1)
    
    self.decoder_2 = nn.Linear(comm_2_size, comm_1_size, dtype=self.dt, device=self.dv, bias=False)
    self.decoder_2 = self._init_weights(self.decoder_2)
    # self.encoder_2 = nn.Linear(8, comm_2_size, dtype=self.dt, device=self.dv)
    
    self.cells:List[DetectionHeadCell] = nn.ModuleList(list_of_cells)
  
  def forward(self, x_in:torch.Tensor):
    """
    1. Receive signal from backbone [n_samples, in_features, signal_length]
    2. Reduce size to [n_samples, 8, cells]
    3. (1) Run through cells
    4. (1) Decode output
    5. (1) Encode output
    6. (2) Run through cells
    7. (2) Decode output
    8. (2) Encode output
    9. Return output
    """
    comm_1 = torch.zeros((x_in.shape[0], 2*len(self.cells)), dtype=self.dt, device=self.dv)
    comm_2 = torch.zeros((x_in.shape[0], 2*len(self.cells)), dtype=self.dt, device=self.dv)
    
    # 2. Reduce size to [n_samples, 8, cells]
    x = self.act_fn_cnn(self.conv1(x_in))   # 
    x = self.conv2(x)           # [n_samples, 8, cells]
    x_sys = self.conv_sys(x_in)     # [n_samples, 32]
    
    # ---
    # 3. (1) Run through cells
    for idx, cell in enumerate(self.cells[:-1]):
      # For each cell in the list, run forward
      cell.forward_1(x[...,idx])
    
    self.cell_system.forward_1(x_sys)
    
    # 4. (1) Decode output
    if self.communication:
      comm_1 = torch.concat([cell.comm_out_1 for cell in self.cells], axis=1)
      comm_1 = self.act_fn_dec(self.decoder_1(comm_1)) # n_cells*2 -> 8
      # comm_1 = self.act_fn(self.encoder_1(comm_1)) # 8 -> n_cells*2
    
    # 5. (1) Encode output
    for idx, cell in enumerate(self.cells):
      cell.comm_in_1 = comm_1[...,(idx)*2:(idx+1)*2]
    
    # ---
    # 6. (2) Run through cells
    for idx, cell in enumerate(self.cells):
      cell.forward_2()
    
    # 7. (2) Decode output
    if self.communication:
      comm_2 = torch.concat([cell.comm_out_2 for cell in self.cells], axis=1)
      comm_2 = self.act_fn_dec(self.decoder_2(comm_2))
      # comm_2 = self.act_fn(self.encoder_2(comm_2))
    
    # 8. (2) Encode output
    for idx, cell in enumerate(self.cells):
      cell.comm_in_2 = comm_2[...,(idx)*2:(idx+1)*2]
    
    cells_out = []
    for idx, cell in enumerate(self.cells):
      cell_out = cell.forward_3()
      cells_out.append(cell_out)
    
    # 9. Return output
    out_cells = torch.concat(cells_out, axis=1)
    out_oscs = out_cells[:, :self.n_regions*self.output_size].view(-1, self.n_regions, 4)
    out_oscs = out_oscs.permute(0, 2, 1) # [n_samples, 4, n_regions]
    out_sys = out_cells[:, self.n_regions*self.output_size:]
    
    # -- single layer --
    out_oscs = out_oscs[:, None, :, :]
    out_sys = out_sys[:, None, :]
    
    out_oscs = out_oscs.permute(0,1,3,2)
    
    return out_oscs, out_sys


class DetectionHead3(nn.Module):
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    n_regions:int,
    n_layers:int,
    system_info:dict,
  ):
    super(DetectionHead3, self).__init__()
    self.unfold_idx = unfold_idx    # 
    self.scale_idx = scale_idx      # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.name = 'DetectionHead'     # 
    self.n_regions = n_regions      #
    self.n_layers = n_layers      #
    
    out_channels = 3

    self.flatten = nn.Flatten() #
    self.act_fn1_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_3 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    
    self.act_fn2_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn2_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    
    self.act_fn3_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn3_2 = ScalableTanh(self.dt, self.dv)
    
    self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    
    stride = stride
    kernel_size = kernel_size
    
    # convsys, convobj info
    strides_sys_obj = [2,2,2]
    in_length = stride * self.n_regions
    sys_obj_size = in_length // 2**(sum(strides_sys_obj)-3)
    
    self.conv1 = nn.Conv1d(
      in_channels=in_features, 
      out_channels=16, 
      kernel_size=3, 
      stride=1, 
      padding=(3-1)//2,
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    self.conv2 = nn.Conv1d(
      in_channels=16, 
      out_channels=16, 
      kernel_size=3, 
      stride=1, 
      padding=(3-1)//2,
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    self.conv3 = nn.Conv1d(
      in_channels=16, 
      out_channels=out_channels*n_layers, 
      kernel_size=kernel_size, 
      stride=stride, 
      padding=(kernel_size - stride),
      padding_mode='replicate',
      dtype=self.dt, 
      device=self.dv
    )
    
    self.convsys_cnn1 = nn.Conv1d(in_features, 4*n_layers, 3, strides_sys_obj[0], (3-strides_sys_obj[0])//2 + 1, padding_mode='replicate', dtype=self.dt, device=self.dv)
    self.convsys_cnn2 = nn.Conv1d(4*n_layers, 1*n_layers, 3, strides_sys_obj[1], (3-strides_sys_obj[0])//2 + 1, padding_mode='replicate', dtype=self.dt, device=self.dv)
    self.convsys_cnn3 = nn.Conv1d(1*n_layers, 1*n_layers, 5, strides_sys_obj[2], (5-strides_sys_obj[0])//2 + 1, padding_mode='replicate', dtype=self.dt, device=self.dv)
    self.convsys_fc = nn.Linear(sys_obj_size*n_layers, (5+n_regions)*n_layers , dtype=self.dt, device=self.dv)
    
    self.convobj_cnn1 = nn.Conv1d(in_features, n_layers*2, 3, 1, 1, padding_mode='replicate', dtype=self.dt, device=self.dv)
    self.convobj_cnn2 = nn.Conv1d(n_layers*2, n_layers, kernel_size, stride, (kernel_size - stride), padding_mode='replicate', dtype=self.dt, device=self.dv)
    self.convobj_cnn3 = nn.Conv1d(n_layers, n_layers, 3, 1, 1, groups=n_layers, padding_mode='zeros', dtype=self.dt, device=self.dv)
    
    self.fc1 = nn.Linear(
      in_features=(n_regions*(out_channels+1+1)+5)*n_layers,
      out_features=(n_regions*out_channels)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
    self.fc2 = nn.Linear(
      in_features=(n_regions*out_channels)*self.n_layers,
      out_features=(n_regions*out_channels)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
    
    # store intermediate tensors
    self.activation_outputs = {}

  def _get_activation(self, name):
    def hook_fn(module, input, output):
      # store the output of the module
      self.activation_outputs[name] = input[0].detach()
    return hook_fn

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    for name, param in self.named_parameters():
      if param.grad is not None:
        gradients.append(param.grad)
        gradient_norms.append(param.grad.norm())
    
    gradients = [pt2np(g) for g in gradients if g is not None]
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _clear_activation(self):
    self.activation_outputs = {}

  def forward_convobj(self, x_in:torch.Tensor):
    """
    Forward pass for the convobj part of the detection head.
    This part processes the input signal to extract object-related features.
    """
    x_obj1 = self.act_fn1_1(self.convobj_cnn1(x_in))  # [n_samples, n_layers*2, n_regions]
    x_obj2 = self.act_fn1_2(self.convobj_cnn2(x_obj1))
    x_obj_out = self.convobj_cnn3(x_obj2) # [n_samples, n_layers, n_regions]
    
    return x_obj_out

  def forward_convsys(self, x_in:torch.Tensor):
    """
    Forward pass for the convsys part of the detection head.
    This part processes the input signal to extract system-related features.
    """
    x_sys1 = self.act_fn1_1(self.convsys_cnn1(x_in))  # [n_samples, 4*n_layers, n_regions]
    x_sys2 = self.act_fn1_2(self.convsys_cnn2(x_sys1))
    x_sys3 = self.act_fn1_3(self.convsys_cnn3(x_sys2))
    x_sys_out = self.convsys_fc(self.flatten(x_sys3)) # [n_samples, 5*n_layers + n_regions]
    
    return x_sys_out

  def forward(self, x_in):
    n_samples = x_in.shape[0]                   # 
    x_cnn1 = self.act_fn1_1(self.conv1(x_in))   # 
    x_cnn2 = self.act_fn1_2(self.conv2(x_cnn1)) # 
    x_oscs = self.act_fn1_3(self.conv3(x_cnn2)) # [n_samples, 4, n_regions]
    
    x_sys = self.forward_convsys(x_in) # [n_samples, 4, n_regions]
    x_obj = self.forward_convobj(x_in) # [n_samples, 1, n_regions]
    
    x_f = self.flatten(x_oscs)
    x_concat = torch.concat([x_f, self.flatten(x_obj), x_sys], axis=1)
    
    x_fc1 = self.act_fn2_1(self.fc1(x_concat)) #
    x_fc2 = self.fc2(x_fc1) #
    
    x_oscs = x_fc2.view(n_samples, self.n_layers, self.n_regions, -1) #
    x_sys = x_sys[:,:5].view(n_samples, self.n_layers, -1) # 
    
    x_out = torch.concat([x_obj.unsqueeze(-1), x_oscs], axis=-1) # 
    
    # x_oscs = x_oscs.view(n_samples, self.n_layers, self.n_regions, -1) # [n_samples, n_layers, n_regions, 4]
    # x_sys = x_sys.unsqueeze(1)  # [n_samples, 1, 5]
    
    return x_out, x_sys

  @property
  def fc_weights(self):
    """Return the weights of fc2, fc3 and fc4 for regularization purposes"""
    return [self.fc2.weight, self.fc3.weight, self.fc4.weight]


class DetectionHead4(nn.Module):
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    system_info:dict,
    use_residual:bool = False,  # New parameter to toggle residual connections
  ):
    super(DetectionHead4,self).__init__()
    self.name = 'DetectionHead'       # 
    self.unfold_idx = unfold_idx      # 
    self.scale_idx = scale_idx        # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.n_regions = n_regions = system_info['n_regions'] #
    self.n_layers = n_layers = system_info['n_layers']    #
    self.n_sys_pars = system_info['n_sys_pars']           #
    self.use_residual = use_residual  # Store the residual flag
    
    self.kernel_size = kernel_size  #
    self.stride = stride            # 
    padding_mode = 'replicate'      # 
    out_channels = 4                # 
    self.flatten = nn.Flatten()     #
    
    self.act_fn1_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_3 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    
    self.act_fn2_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn2_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    
    self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    
    # convsys, convobj info
    strides_sys_obj = [2,2,2]
    in_length = stride * self.n_regions
    sys_obj_size = in_length // 2**(sum(strides_sys_obj)-3)
    new_tot_size = kernel_size
    
    # Define downsample layers for residual connections
    self.cell_downsample1 = nn.Conv1d(
        in_channels=in_features, 
        out_channels=16, 
        kernel_size=1, 
        stride=2, 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.cell_downsample2 = nn.Conv1d(
        in_channels=16, 
        out_channels=32, 
        kernel_size=1, 
        stride=2, 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    # System residual connections
    self.sys_downsample1 = nn.Conv1d(
        in_channels=in_features, 
        out_channels=8*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[0], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.sys_downsample2 = nn.Conv1d(
        in_channels=8*n_layers, 
        out_channels=4*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[1], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.sys_downsample3 = nn.Conv1d(
        in_channels=4*n_layers, 
        out_channels=1*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[2], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    # FC layer residual connections
    self.fc_downsample = nn.Linear(
        in_features=(n_regions*out_channels + self.n_sys_pars+n_regions)*n_layers,
        out_features=(n_regions*out_channels)*self.n_layers,
        dtype=self.dt,
        device=self.dv
    ) if use_residual else None
    
    self.cell_net_cnn1 = nn.Conv1d(
      in_channels=in_features,
      out_channels=16,
      kernel_size=3,
      stride=2,
      padding=1,
      padding_mode=padding_mode,
      dtype=self.dt,
      device=self.dv
    )
    self.cell_net_cnn2 = nn.Conv1d(
      in_channels=16,
      out_channels=32,
      kernel_size=3,
      stride=2,
      padding=1,
      groups=4*n_layers,
      padding_mode=padding_mode,
      dtype=self.dt,
      device=self.dv
    )
    self.cell_net_cnn3 = nn.Conv1d(
      in_channels=32,
      out_channels=4*4*n_layers,
      kernel_size=int(kernel_size/(4*5))+4,
      stride=int(kernel_size/(4*5)),
      padding=2,
      groups=4*n_layers,
      padding_mode=padding_mode,
      dtype=self.dt,
      device=self.dv
    )
    
    self.cell_net_fc1 = nn.Linear(
      in_features=5*(4*4*n_layers) + 4 + 1, # 5*4*n_layers for convsys, n_regions for objnet, 1 for region index
      out_features=16+n_layers*4,
      bias=False,
      dtype=self.dt,
      device=self.dv
    )
    self.cell_net_fc2 = nn.Linear(
      in_features=16+n_layers*4,
      out_features=out_channels*n_layers,
      bias=False,
      dtype=self.dt,
      device=self.dv
    )
    
    self.convsys_cnn1 = nn.Conv1d(in_features, 8*n_layers, 3, strides_sys_obj[0], (3-strides_sys_obj[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn2 = nn.Conv1d(8*n_layers, 4*n_layers, 3, strides_sys_obj[1], (3-strides_sys_obj[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn3 = nn.Conv1d(4*n_layers, 1*n_layers, 5, strides_sys_obj[2], (5-strides_sys_obj[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_fc = nn.Linear(sys_obj_size*n_layers, (self.n_sys_pars+n_regions)*n_layers , dtype=self.dt, device=self.dv)
    
    self.fc1 = nn.Linear(
      in_features=(n_regions*out_channels + self.n_sys_pars+n_regions)*n_layers, #  + n_regions # n_regions for objnet
      out_features=(n_regions*out_channels)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
    self.fc2 = nn.Linear(
      in_features=(n_regions*out_channels)*self.n_layers,
      out_features=(n_regions*out_channels)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
        
    # store intermediate tensors
    self.activation_outputs = {}

    # Add instance norm layers for cell and system
    self.instancenorm_cell = nn.InstanceNorm1d(16, affine=True, dtype=self.dt, device=self.dv)
    self.instancenorm_sys = nn.InstanceNorm1d(8*system_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    for name, param in self.named_parameters():
      if param.grad is not None:
        gradients.append(param.grad)
        gradient_norms.append(param.grad.norm())
    
    gradients = [pt2np(g) for g in gradients if g is not None]
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _clear_activation(self):
    self.activation_outputs = {}

  def forward_cell(self, x_in:torch.Tensor, out:torch.Tensor, idx_region:int):
    region_idx = torch.full((x_in.shape[0], 1), fill_value=idx_region, dtype=self.dt, device=self.dv) # [n_samples, 1]
    
    # Apply CNN1 with residual connection
    x_cnn1_pre = self.cell_net_cnn1(x_in)
    # Apply instance norm after first conv
    x_cnn1_pre = self.instancenorm_cell(x_cnn1_pre)
    if self.use_residual and self.cell_downsample1 is not None:
      x_cnn1_pre = x_cnn1_pre + self.cell_downsample1(x_in)
    x_cnn1 = self.act_fn1_1(x_cnn1_pre)
    
    # Apply CNN2 with residual connection
    x_cnn2_pre = self.cell_net_cnn2(x_cnn1)
    if self.use_residual and self.cell_downsample2 is not None:
      x_cnn2_pre = x_cnn2_pre + self.cell_downsample2(x_cnn1)
    x_cnn2 = self.act_fn1_2(x_cnn2_pre)
    
    # Apply CNN3 (no residual due to significant size reduction and kernel size)
    x_cnn3 = self.act_fn1_3(self.cell_net_cnn3(x_cnn2))
    
    x_fc = self.flatten(x_cnn3) #
    out_f = self.flatten(out)   # [n_samples, 4*n_layers, n_regions]
    x_fc = torch.concat([x_fc, out_f, region_idx], dim=-1) # [n_samples, 4*n_layers + n_regions]
    
    # Apply FC1 with residual connection not appropriate here due to dimensional changes
    x_fc1 = self.act_fn2_1(self.cell_net_fc1(x_fc))
    
    # Apply FC2 with residual not applicable (different sizes)
    x_fc2 = self.act_fn2_1(self.cell_net_fc2(x_fc1))
    
    return x_fc2

  def forward_convsys(self, x_in:torch.Tensor):
    """
    Forward pass for the convsys part of the detection head.
    This part processes the input signal to extract system-related features.
    """
    # Apply CNN1 with residual connection
    x_sys1_pre = self.convsys_cnn1(x_in)
    # Apply instance norm after first conv
    x_sys1_pre = self.instancenorm_sys(x_sys1_pre)
    if self.use_residual and self.sys_downsample1 is not None:
        x_sys1_pre = x_sys1_pre + self.sys_downsample1(x_in)
    x_sys1 = self.act_fn1_1(x_sys1_pre)
    
    # Apply CNN2 with residual connection
    x_sys2_pre = self.convsys_cnn2(x_sys1)
    if self.use_residual and self.sys_downsample2 is not None:
        x_sys2_pre = x_sys2_pre + self.sys_downsample2(x_sys1)
    x_sys2 = self.act_fn1_2(x_sys2_pre)
    
    # Apply CNN3 with residual connection
    x_sys3_pre = self.convsys_cnn3(x_sys2)
    if self.use_residual and self.sys_downsample3 is not None:
        x_sys3_pre = x_sys3_pre + self.sys_downsample3(x_sys2)
    x_sys3 = self.act_fn1_3(x_sys3_pre)
    
    x_sys_out = self.convsys_fc(self.flatten(x_sys3))
    
    return x_sys_out

  def forward(self, x_in, out):
    n_samples = x_in.shape[0] # 
    ks, ks_2, s = self.kernel_size, (self.kernel_size - self.stride) // 2, self.stride
    x_f = []
    
    x_in_padded = pad(x_in, (ks_2, ks_2), mode='constant')  # Pad input to match kernel size
    for idx_region in range(self.n_regions):
      # For each region, apply the forward cell with the padded input
      x_in_padded_region = x_in_padded[...,int((idx_region+1/2)*s-(ks//2))+ks_2:int((idx_region+1/2)*s+(ks//2))+ks_2]
      x_out_region = out[:,:,self.scale_idx,idx_region,:]
      x_region_forward = self.forward_cell(x_in_padded_region, x_out_region, idx_region) # [n_samples, 4*n_layers]
      x_f.append(x_region_forward)  # Collect outputs for each region
    
    x_fc = torch.concat(x_f, dim=-1) # [n_samples, 8*4*n_layers, n_regions]
    
    x_sys = self.forward_convsys(x_in) # [n_samples, 4, n_regions]    
    x_concat = torch.concat([x_fc, x_sys], axis=1) # , self.flatten(x_obj)
    
    # Apply FC1 with residual connection
    x_fc1_pre = self.fc1(x_concat)
    if self.use_residual and self.fc_downsample is not None:
        x_fc1_pre = x_fc1_pre + self.fc_downsample(x_concat)
    x_fc1 = self.act_fn2_1(x_fc1_pre)
    
    # Apply FC2 with residual connection - direct as input and output dimensions match
    x_fc2_pre = self.fc2(x_fc1)
    if self.use_residual:
        x_fc2_pre = x_fc2_pre + x_fc1  # Direct residual connection
    x_fc2 = x_fc2_pre  # No activation after final layer
    
    x_oscs = x_fc2.view(n_samples, self.n_layers, self.n_regions, -1) #
    x_sys = x_sys[:,:self.n_sys_pars].view(n_samples, self.n_layers, -1) # 
    
    x_out = torch.concat([x_oscs], axis=-1) # x_obj.unsqueeze(-1), 
    
    return x_out, x_sys

  @property
  def fc_weights(self):
    """Return the weights of fc2, fc3 and fc4 for regularization purposes"""
    return [self.fc2.weight, self.fc3.weight, self.fc4.weight]



class DetectionHead42(nn.Module):
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    system_info:dict,
    use_residual:bool = False,  # New parameter to toggle residual connections
  ):
    super(DetectionHead42,self).__init__()
    self.name = 'DetectionHead'       # 
    self.unfold_idx = unfold_idx      # 
    self.scale_idx = scale_idx        # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.n_regions = n_regions = system_info['n_regions'] #
    self.n_layers = n_layers = system_info['n_layers']    #
    self.n_sys_pars = system_info['n_sys_pars']           #
    self.use_residual = use_residual  # Store the residual flag
    
    self.kernel_size = kernel_size  #
    self.stride = stride            # 
    padding_mode = 'replicate'      # 
    out_channels = 4                # 

    self.flatten = nn.Flatten()     #
    
    self.act_fn1_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn1_3 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    
    self.act_fn2_1 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    self.act_fn2_2 = ScalableTanh(self.dt, self.dv) # act_str_to_fn('tanh') #
    
    self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    
    # convsys, convobj info
    strides_sys_obj = [2,2,2]
    in_length = stride * self.n_regions
    sys_obj_size = in_length // 2**(sum(strides_sys_obj)-3)
    new_tot_size = kernel_size
    
    # Define downsample layers for residual connections
    self.cell_downsample1 = nn.Conv1d(
        in_channels=in_features, 
        out_channels=16, 
        kernel_size=1, 
        stride=2, 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.cell_downsample2 = nn.Conv1d(
        in_channels=16, 
        out_channels=32, 
        kernel_size=1, 
        stride=2, 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    # System residual connections
    self.sys_downsample1 = nn.Conv1d(
        in_channels=in_features, 
        out_channels=8*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[0], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.sys_downsample2 = nn.Conv1d(
        in_channels=8*n_layers, 
        out_channels=4*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[1], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    self.sys_downsample3 = nn.Conv1d(
        in_channels=4*n_layers, 
        out_channels=1*n_layers, 
        kernel_size=1, 
        stride=strides_sys_obj[2], 
        dtype=self.dt, 
        device=self.dv
    ) if use_residual else None
    
    # FC layer residual connections
    self.fc_downsample = nn.Linear(
        in_features=(n_regions*out_channels + self.n_sys_pars+n_regions)*n_layers,
        out_features=(n_regions*out_channels)*self.n_layers,
        dtype=self.dt,
        device=self.dv
    ) if use_residual else None
    
    self.cell_net_cnn1 = nn.Conv1d(
      in_channels=in_features,
      out_channels=16,
      kernel_size=3,
      stride=2,
      padding=1,
      padding_mode=padding_mode,
      dtype=self.dt,
      device=self.dv
    )
    self.cell_net_cnn2 = nn.Conv1d(
      in_channels=16,
      out_channels=32,
      kernel_size=3,
      stride=2,
      padding=1,
      groups=4*n_layers,
      padding_mode=padding_mode,
      dtype=self.dt,
      device=self.dv
    )
    self.cell_net_cnn3 = nn.Conv1d(
      in_channels=32,
      out_channels=4*4*n_layers,
      kernel_size=int(kernel_size/(4*5))+4,
      stride=int(kernel_size/(4*5)),
      padding=2,
      groups=4*n_layers,
      padding_mode=padding_mode,
      dtype=self.dt,
      device=self.dv
    )
    
    self.cell_net_fc1 = nn.Linear(
      in_features=5*(4*4*n_layers) + 4 + 1, # 5*4*n_layers for convsys, n_regions for objnet, 1 for region index
      out_features=16+n_layers*4,
      bias=False,
      dtype=self.dt,
      device=self.dv
    )
    self.cell_net_fc2 = nn.Linear(
      in_features=16+n_layers*4,
      out_features=out_channels*n_layers,
      bias=False,
      dtype=self.dt,
      device=self.dv
    )
    
    self.convsys_cnn1 = nn.Conv1d(in_features, 8*n_layers, 3, strides_sys_obj[0], (3-strides_sys_obj[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn2 = nn.Conv1d(8*n_layers, 4*n_layers, 3, strides_sys_obj[1], (3-strides_sys_obj[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn3 = nn.Conv1d(4*n_layers, 1*n_layers, 5, strides_sys_obj[2], (5-strides_sys_obj[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_fc = nn.Linear(sys_obj_size*n_layers, (self.n_sys_pars+n_regions)*n_layers , dtype=self.dt, device=self.dv)
    
    self.fc1 = nn.Linear(
      in_features=(n_regions*out_channels + self.n_sys_pars+n_regions)*n_layers, #  + n_regions # n_regions for objnet
      out_features=(n_regions*out_channels)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
    self.fc2 = nn.Linear(
      in_features=(n_regions*out_channels)*self.n_layers,
      out_features=(n_regions*out_channels)*self.n_layers,
      dtype=self.dt,
      device=self.dv
    )
        
    # store intermediate tensors
    self.activation_outputs = {}

    # Add instance norm layers for cell and system
    self.instancenorm_cell1 = nn.InstanceNorm1d(16, affine=True, dtype=self.dt, device=self.dv)
    self.instancenorm_cell2 = nn.InstanceNorm1d(32, affine=True, dtype=self.dt, device=self.dv)
    
    self.instancenorm_sys = nn.InstanceNorm1d(8*system_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)

  def _get_activation(self, name):
    def hook_fn(module, input, output):
      # store the output of the module
      self.activation_outputs[name] = input[0].detach()
    return hook_fn

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    for name, param in self.named_parameters():
      if param.grad is not None:
        gradients.append(param.grad)
        gradient_norms.append(param.grad.norm())
    
    gradients = [pt2np(g) for g in gradients if g is not None]
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _clear_activation(self):
    self.activation_outputs = {}

  def forward_cell(self, x_in:torch.Tensor, out:torch.Tensor, idx_region:int):
    region_idx = torch.full((x_in.shape[0], 1), fill_value=idx_region, dtype=self.dt, device=self.dv) # [n_samples, 1]
    
    # Apply CNN1 with residual connection
    x_cnn1_pre = self.cell_net_cnn1(x_in)
    x_cnn1_norm = self.instancenorm_cell1(x_cnn1_pre)
    
    if self.use_residual and self.cell_downsample1 is not None:
      x_cnn1_norm = x_cnn1_norm + self.cell_downsample1(x_in)
    x_cnn1 = self.act_fn1_1(x_cnn1_norm)
    
    # Apply CNN2 with residual connection
    x_cnn2_pre = self.cell_net_cnn2(x_cnn1)           # 
    x_cnn2_norm = self.instancenorm_cell2(x_cnn2_pre) # 
    
    if self.use_residual and self.cell_downsample2 is not None:
      x_cnn2_norm = x_cnn2_norm + self.cell_downsample2(x_cnn1)
    x_cnn2 = self.act_fn1_2(x_cnn2_pre)
    
    # Apply CNN3 (no residual due to significant size reduction and kernel size)
    x_cnn3 = self.act_fn1_3(self.cell_net_cnn3(x_cnn2))
    
    x_fc = self.flatten(x_cnn3) #
    out_f = self.flatten(out)   # [n_samples, 4*n_layers, n_regions]
    x_fc = torch.concat([x_fc, out_f, region_idx], dim=-1) # [n_samples, 4*n_layers + n_regions]
    
    # Apply FC1 with residual connection not appropriate here due to dimensional changes
    x_fc1 = self.act_fn2_1(self.cell_net_fc1(x_fc))
    
    # Apply FC2 with residual not applicable (different sizes)
    x_fc2 = self.act_fn2_1(self.cell_net_fc2(x_fc1))
    
    return x_fc2

  def forward_convsys(self, x_in:torch.Tensor):
    """
    Forward pass for the convsys part of the detection head.
    This part processes the input signal to extract system-related features.
    """
    # Apply CNN1 with residual connection
    x_sys1_pre = self.convsys_cnn1(x_in)
    # Apply instance norm after first conv
    x_sys1_pre = self.instancenorm_sys(x_sys1_pre)
    if self.use_residual and self.sys_downsample1 is not None:
        x_sys1_pre = x_sys1_pre + self.sys_downsample1(x_in)
    x_sys1 = self.act_fn1_1(x_sys1_pre)
    
    # Apply CNN2 with residual connection
    x_sys2_pre = self.convsys_cnn2(x_sys1)
    if self.use_residual and self.sys_downsample2 is not None:
        x_sys2_pre = x_sys2_pre + self.sys_downsample2(x_sys1)
    x_sys2 = self.act_fn1_2(x_sys2_pre)
    
    # Apply CNN3 with residual connection
    x_sys3_pre = self.convsys_cnn3(x_sys2)
    if self.use_residual and self.sys_downsample3 is not None:
        x_sys3_pre = x_sys3_pre + self.sys_downsample3(x_sys2)
    x_sys3 = self.act_fn1_3(x_sys3_pre)
    
    x_sys_out = self.convsys_fc(self.flatten(x_sys3))
    
    return x_sys_out

  def forward(self, x_in, out):
    n_samples = x_in.shape[0] # 
    ks, ks_2, s = self.kernel_size, (self.kernel_size - self.stride) // 2, self.stride
    x_f = []
    
    x_in_padded = pad(x_in, (ks_2, ks_2), mode='replicate') # Pad input to match kernel size
    for idx_region in range(self.n_regions):
      # For each region, apply the forward cell with the padded input
      x_in_padded_region = x_in_padded[...,int((idx_region+1/2)*s-(ks//2))+ks_2:int((idx_region+1/2)*s+(ks//2))+ks_2]
      x_out_region = out[:,:,self.scale_idx,idx_region,:]
      x_region_forward = self.forward_cell(x_in_padded_region, x_out_region, idx_region) # [n_samples, 4*n_layers]
      x_f.append(x_region_forward)  # Collect outputs for each region
    
    x_fc = torch.concat(x_f, dim=-1) # [n_samples, 8*4*n_layers, n_regions]
    
    x_sys = self.forward_convsys(x_in) # [n_samples, 4, n_regions]    
    x_concat = torch.concat([x_fc, x_sys], axis=1) # , self.flatten(x_obj)
    
    # Apply FC1 with residual connection
    x_fc1_pre = self.fc1(x_concat)
    if self.use_residual and self.fc_downsample is not None:
        x_fc1_pre = x_fc1_pre + self.fc_downsample(x_concat)
    x_fc1 = self.act_fn2_1(x_fc1_pre)
    
    # Apply FC2 with residual connection - direct as input and output dimensions match
    x_fc2_pre = self.fc2(x_fc1)
    if self.use_residual:
        x_fc2_pre = x_fc2_pre + x_fc1  # Direct residual connection
    x_fc2 = x_fc2_pre  # No activation after final layer
    
    x_oscs = x_fc2.view(n_samples, self.n_layers, self.n_regions, -1) #
    x_sys = x_sys[:,:self.n_sys_pars].view(n_samples, self.n_layers, -1) # 
    
    x_out = torch.concat([x_oscs], axis=-1) # x_obj.unsqueeze(-1), 
    
    return x_out, x_sys

  @property
  def fc_weights(self):
    """Return the weights of fc2, fc3 and fc4 for regularization purposes"""
    return [self.fc2.weight, self.fc3.weight, self.fc4.weight]


class DetectionHead5(nn.Module):
  """
    For each cell; different network, no communication between cells.
  """
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    system_info:dict,
  ):
    super(DetectionHead5, self).__init__()
    self.name = 'DetectionHead'       # 
    self.unfold_idx = unfold_idx      # 
    self.scale_idx = scale_idx        # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.n_regions = n_regions = system_info['n_regions'] #
    self.n_layers = n_layers = system_info['n_layers']    #
    self.n_sys_pars = system_info['n_sys_pars']           #

    self.flatten = nn.Flatten() #
    self.act_fn_cnn = act_str_to_fn(act_fun_str) #
    self.act_fn_sys = act_str_to_fn(act_fun_str) #
    self.act_fn_dec = act_str_to_fn(act_fun_str) #
    
    self.stride = stride
    self.kernel_size = kernel_size
    sys_obj_size = int((stride*n_regions) / (2**(sum([2, 2, 2]) - 3)))  # Calculate size of system object
    
    self._init_system(
      shapes_info={
        'in_features': in_features,
        'n_layers': self.n_layers,
        'n_regions': self.n_regions,
        'n_sys_pars': system_info['n_sys_pars'],
        'sys_obj_size': sys_obj_size
      },
      strides=[2, 2, 2],
      padding_mode='replicate',
      act_fun_str=act_fun_str,
    )
    
    self.cells:List[DetectionHeadCell5] = self._init_cells(
      system_info['n_regions'],
      input_channels=in_features,
      output_size=out_features,
      act_str_fn=act_fun_str
    )

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    gradients.append(self.convsys_cnn1.weight.grad)
    gradients.append(self.convsys_cnn2.weight.grad)
    gradients.append(self.convsys_cnn3.weight.grad)
    gradients.append(self.convsys_fc.weight.grad)	
    
    for cell in self.cells:
      gradients.append(cell.cnn1.weight.grad)
      gradients.append(cell.cnn2.weight.grad)
      gradients.append(cell.cnn3.weight.grad)
      gradients.append(cell.fc1.weight.grad)
      gradients.append(cell.fc2.weight.grad)
      
    try:
      gradient_norms.append(self.conv1.weight.grad.norm())
      gradient_norms.append(self.conv2.weight.grad.norm())
      
      for cell in self.cells:
        gradient_norms.append(cell.conn_1.weight.grad.norm())
        gradient_norms.append(cell.conn_2.weight.grad.norm())
        gradient_norms.append(cell.conn_3.weight.grad.norm())
    
      gradient_norms.append(self.cell_system.conn_1.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_2.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_3.weight.grad.norm())
    except:
      # No gradient
      gradient_norms = [0]*len(gradients)
    
    gradients = [pt2np(g) for g in gradients if g is not None] 
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _init_weights(self, net:nn.Linear):
    """Initialize weights of the network according to diagonal init for matrix, with small noise for other stuff"""
    net.weight.data = torch.eye(net.weight.shape[0], net.weight.shape[1], dtype=self.dt, device=self.dv) + \
      torch.normal(0, 0.01, size=net.weight.shape, dtype=self.dt, device=self.dv)
    
    return net

  def _init_cells(self, number_of_cells:int, input_channels:int, output_size:int=4, act_str_fn:str='tanh'):
    list_of_cells = []
    
    for i in range(number_of_cells):
      cell = DetectionHeadCell5(
        input_channels=input_channels,
        output_channels=output_size,
        act_fun_str=act_str_fn,
        system_info={'dtype': self.dt, 'device': self.dv},
      )
      
      list_of_cells.append(cell)
    
    return nn.ModuleList(list_of_cells)
  
  def _init_system(self, shapes_info:Dict[str, int], strides:List[int], padding_mode:str='replicate', act_fun_str:str='tanh'):
    in_features:int = shapes_info['in_features']
    n_layers:int = shapes_info['n_layers']
    n_regions:int = shapes_info['n_regions']
    n_sys_pars:int = shapes_info['n_sys_pars']
    sys_obj_size:int = shapes_info['sys_obj_size']
    self.convsys_cnn1 = nn.Conv1d(in_features, 4*n_layers, 3, strides[0], (3-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn2 = nn.Conv1d(4*n_layers, 1*n_layers, 3, strides[1], (3-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn3 = nn.Conv1d(1*n_layers, 1*n_layers, 5, strides[2], (5-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_fc = nn.Linear(sys_obj_size*n_layers, (n_sys_pars+n_regions)*n_layers , dtype=self.dt, device=self.dv)

  def forward_convsys(self, x_in:torch.Tensor):
    """
    Forward pass for the convsys part of the detection head.
    This part processes the input signal to extract system-related features.
    """
    x_sys1 = self.act_fn_sys(self.convsys_cnn1(x_in))  # [n_samples, 4*n_layers, n_regions]
    x_sys2 = self.act_fn_sys(self.convsys_cnn2(x_sys1))
    x_sys3 = self.act_fn_sys(self.convsys_cnn3(x_sys2))
    x_sys_out = self.convsys_fc(self.flatten(x_sys3))
    
    return x_sys_out  # [n_samples, 5*n_layers + n_regions]

  def forward_oscs(self, x_in:torch.Tensor, out:torch.Tensor):
    ks, ks_2, s = self.kernel_size, (self.kernel_size - self.stride) // 2, self.stride
    res = []
    
    x_in_padded = pad(x_in, (ks_2, ks_2), mode='constant')  # Pad input to match kernel size
    for idx_unfold in range(self.n_regions):
      x_region = x_in_padded[...,int((idx_unfold+1/2)*s-(ks//2))+ks_2:int((idx_unfold+1/2)*s+(ks//2))+ks_2]
      out_region = out[:,:,:,idx_unfold,:]  # [n_samples, n_layers, n_regions, 4]
      x_region_out = self.cells[idx_unfold].forward(x_region, out_region)
      res.append(x_region_out)
    
    return torch.concat(res, dim=-2)  # [n_samples, n_regions, 4*n_layers]

  def forward(self, x_in:torch.Tensor, out:torch.Tensor):
    """
    1. Receive signal from backbone [n_samples, in_features, signal_length]
    2. Reduce size to [n_samples, 8, cells]
    3. (1) Run through cells
    4. (1) Decode output
    5. (1) Encode output
    6. (2) Run through cells
    7. (2) Decode output
    8. (2) Encode output
    9. Return output
    """
    
    x_sys = self.forward_convsys(x_in)    # [n_samples, 4, n_regions]    
    x_oscs = self.forward_oscs(x_in, out) # [n_samples, n_regions, 4*n_layers]
    
    x_oscs = x_oscs.view(x_oscs.shape[0], self.n_layers, self.n_regions, -1)  # [n_samples, n_layers, n_regions, 4]4
    x_sys = x_sys[:,:self.n_sys_pars].view(x_sys.shape[0], self.n_layers, -1)  # [n_samples, n_layers, 5]
    
    return x_oscs, x_sys  # [n_samples, n_layers, n_regions, 4], [n_samples, n_layers, 5]


class DetectionHead6(nn.Module):
  """
    For each cell; different network, no communication between cells.
  """
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    system_info:dict,
  ):
    super(DetectionHead6, self).__init__()
    self.name = 'DetectionHead'       # 
    self.unfold_idx = unfold_idx      # 
    self.scale_idx = scale_idx        # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.n_regions = n_regions = system_info['n_regions'] #
    self.n_layers = n_layers = system_info['n_layers']    #
    self.n_sys_pars = system_info['n_sys_pars']           #

    self.flatten = nn.Flatten() #
    self.act_fn_cnn = act_str_to_fn(act_fun_str) #
    self.act_fn_sys = act_str_to_fn(act_fun_str) #
    self.act_fn_dec = act_str_to_fn(act_fun_str) #
    
    self.shape_norm = nn.Upsample(
      size=(64),
      mode='linear',
      align_corners=True,
    )
    
    self.stride = stride
    self.kernel_size = kernel_size
    sys_obj_size = int((2048) / (2**(sum([3, 3, 2, 2, 2]) - 5))) * 4  # Calculate size of system object
    
    self._init_sys(
      shapes_info={
        'in_features': in_features,
        'n_layers': self.n_layers,
        'n_regions': self.n_regions,
        'n_sys_pars': system_info['n_sys_pars'],
        'sys_obj_size': sys_obj_size
      },
      strides=[4, 4, 2, 2, 2],
      padding_mode='replicate',
      act_fun_str=act_fun_str,
    )
    
    self._init_osc(
      shapes_info={
        'in_features': in_features,
        'n_layers': self.n_layers,
        'n_regions': self.n_regions,
        'n_sys_pars': system_info['n_sys_pars'],
        'sys_obj_size': sys_obj_size
      },
      strides=[2, 2, 2],
      padding_mode='replicate',
      act_fun_str=act_fun_str,
    )

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    gradients.append(self.convsys_cnn1.weight.grad)
    gradients.append(self.convsys_cnn2.weight.grad)
    gradients.append(self.convsys_cnn3.weight.grad)
    gradients.append(self.convsys_fc.weight.grad)	
    
    for cell in self.cells:
      gradients.append(cell.cnn1.weight.grad)
      gradients.append(cell.cnn2.weight.grad)
      gradients.append(cell.cnn3.weight.grad)
      gradients.append(cell.fc1.weight.grad)
      gradients.append(cell.fc2.weight.grad)
      
    try:
      gradient_norms.append(self.conv1.weight.grad.norm())
      gradient_norms.append(self.conv2.weight.grad.norm())
      
      for cell in self.cells:
        gradient_norms.append(cell.conn_1.weight.grad.norm())
        gradient_norms.append(cell.conn_2.weight.grad.norm())
        gradient_norms.append(cell.conn_3.weight.grad.norm())
    
      gradient_norms.append(self.cell_system.conn_1.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_2.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_3.weight.grad.norm())
    except:
      # No gradient
      gradient_norms = [0]*len(gradients)
    
    gradients = [pt2np(g) for g in gradients if g is not None] 
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _init_weights(self, net:nn.Linear):
    """Initialize weights of the network according to diagonal init for matrix, with small noise for other stuff"""
    net.weight.data = torch.eye(net.weight.shape[0], net.weight.shape[1], dtype=self.dt, device=self.dv) + \
      torch.normal(0, 0.01, size=net.weight.shape, dtype=self.dt, device=self.dv)
    
    return net

  def _init_sys(self, shapes_info:Dict[str, int], strides:List[int], padding_mode:str='replicate', act_fun_str:str='tanh'):
    in_features:int = shapes_info['in_features']
    n_layers:int = shapes_info['n_layers']
    n_regions:int = shapes_info['n_regions']
    n_sys_pars:int = shapes_info['n_sys_pars']
    sys_obj_size:int = shapes_info['sys_obj_size']
    
    self.convsys_cnn1 = nn.Conv1d(in_features, 4*n_layers, 5, strides[0], (5-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn2 = nn.Conv1d(4*n_layers, 8*n_layers, 5, strides[1], (5-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn3 = nn.Conv1d(8*n_layers, 16*n_layers, 5, strides[2], (5-strides[0])//2 + 2, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn4 = nn.Conv1d(16*n_layers, 32*n_layers, 5, strides[3], (5-strides[0])//2 + 2, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn5 = nn.Conv1d(32*n_layers, 4*n_layers, 7, strides[4], (7-strides[0])//2 + 2, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    
    # self.sys_norm1 = nn.InstanceNorm1d(2*n_layers, affine=True, dtype=self.dt, device=self.dv)
    self.sys_norm2 = nn.InstanceNorm1d(8*n_layers, affine=True, dtype=self.dt, device=self.dv)
    # self.sys_norm3 = nn.InstanceNorm1d(8*n_layers, affine=True, dtype=self.dt, device=self.dv)
    self.sys_norm4 = nn.InstanceNorm1d(32*n_layers, affine=True, dtype=self.dt, device=self.dv)
    # self.sys_norm5 = nn.InstanceNorm1d(4*n_layers, affine=True, dtype=self.dt, device=self.dv)
    
    
    self.sys_act1 = act_str_to_fn(act_fun_str)  # Activation function for first conv
    self.sys_act2 = act_str_to_fn(act_fun_str)  # Activation function for
    self.sys_act3 = act_str_to_fn(act_fun_str)  # Activation function for third conv
    self.sys_act4 = act_str_to_fn(act_fun_str)  # Activation function for
    
    self.sys_norm5 = nn.InstanceNorm1d(4*n_layers, affine=True, dtype=self.dt, device=self.dv)
    self.convsys_fc = nn.Linear(sys_obj_size*n_layers, (n_sys_pars)*n_layers , dtype=self.dt, device=self.dv)

  def _init_osc(
    self, 
    shapes_info:Dict[str, int], 
    strides:List[int], 
    padding_mode:str='replicate', 
    act_fun_str:str='tanh'
  ):
    self.convosc_cnn1 = nn.Conv1d(shapes_info['in_features'], 8*shapes_info['n_layers'], 3, 2, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convosc_cnn2 = nn.Conv1d(8*shapes_info['n_layers'], 32*shapes_info['n_layers'], 3, 1, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convosc_cnn3 = nn.Conv1d(32*shapes_info['n_layers'], 4*shapes_info['n_layers'], 1, 1, 0, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    
    self.instance_norm1 = nn.InstanceNorm1d(8*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)
    self.instance_norm2 = nn.InstanceNorm1d(32*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)
    
    self.convosc_act1 = act_str_to_fn(act_fun_str)
    self.convosc_act2 = act_str_to_fn(act_fun_str)
    
    self.convosc_fc1 = nn.Linear(
      in_features=32*4 + 1 + 4, 
      out_features=32, 
      dtype=self.dt,
      device=self.dv
    )
    self.convosc_fc2 = nn.Linear(
      in_features=32, 
      out_features=4, 
      dtype=self.dt,
      device=self.dv
    )
    self.convosc_act3 = act_str_to_fn(act_fun_str)          #
    self.convosc_fc1 = self._init_weights(self.convosc_fc1) # 
    self.convosc_fc2 = self._init_weights(self.convosc_fc2) # 

  def forward_convsys(self, x_in:torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the convsys part of the detection head.
    This part processes the input signal to extract system-related features.
    """
    x_sys1 = self.convsys_cnn1(x_in)                # [n_samples, 4*n_layers, n_regions]
    x_sys1 = self.sys_act1(x_sys1)  #
    
    x_sys2 = self.convsys_cnn2(x_sys1)  # 
    x_sys2 = self.sys_act2(self.sys_norm2(x_sys2))      #  
    
    x_sys3 = self.convsys_cnn3(x_sys2)  # 
    x_sys3 = self.sys_act3(x_sys3)  # [n_samples, 8*n_layers, n_regions]
    
    x_sys4 = self.convsys_cnn4(x_sys3)  #
    x_sys4 = self.sys_act4(self.sys_norm4(x_sys4))  #
    
    x_sys5 = self.convsys_cnn5(x_sys4)  # [n_samples, 4*n_layers, n_regions]
    
    x_sys_out = self.convsys_fc(self.flatten(x_sys5))   # 
    
    return x_sys_out  # [n_samples, 11*n_layers]

  def forward_convosc(self, x_in:torch.Tensor, x_extra:torch.Tensor) -> torch.Tensor:
    x_64 = self.shape_norm(x_in)  # [n_samples, in_features, 64]
    
    x_cnn1 = self.convosc_cnn1(x_64)  # [n_samples, 4*n_layers, 64]
    x_cnn1_norm = self.instance_norm1(x_cnn1) 
    x_cnn1_act = self.convosc_act1(x_cnn1_norm)  # [n_samples, 4*n_layers, 64]
    
    x_cnn2 = self.convosc_cnn2(x_cnn1_act)  # [n_samples, 8*n_layers, 64]
    x_cnn2_norm = self.instance_norm2(x_cnn2)
    x_cnn2_act = self.convosc_act2(x_cnn2_norm)
    
    x_cnn3 = self.convosc_cnn3(x_cnn2_act)  # [n_samples, 16*n_layers, 64]
    x_cnn3_flat = self.flatten(x_cnn3)  # [n_samples, 16*n_layers*64]
    x_extra_flat = self.flatten(x_extra)  # [n_samples, 4*n_layers]
    
    x = torch.concat([x_cnn3_flat, x_extra_flat], dim=-1)  # [n_samples, 16*n_layers*64 + 4*n_layers]
    
    x_fc1 = self.convosc_act3(self.convosc_fc1(x))  # [n_samples, 4]
    x_fc2 = self.convosc_fc2(x_fc1)  # [n_samples, 4]
    
    return x_fc2[:,None,:]  # [n_samples, region, 4]

  def forward_oscs(self, x_in:torch.Tensor, out:torch.Tensor):
    ks, ks_2, s = self.kernel_size, (self.kernel_size - self.stride) // 2, self.stride
    res = []
    
    x_in_padded = pad(x_in, (ks_2, ks_2), mode='constant')  # Pad input to match kernel size
    for idx_reg in range(self.n_regions):
      x_region = x_in_padded[...,int((idx_reg+1/2)*s-(ks//2))+ks_2:int((idx_reg+1/2)*s+(ks//2))+ks_2]
      out_region = out[:,:,self.scale_idx,idx_reg,:].flatten(1)  # [n_samples, n_layers, n_regions, 4]
      idx_region = torch.full((x_region.shape[0], 1), fill_value=idx_reg, dtype=self.dt, device=self.dv)  # [n_samples, 1]
      x_extra = torch.concat([out_region, idx_region], dim=-1)  # [n_samples, n_layers, n_regions, 4 + in_features]
      
      x_region_out = self.forward_convosc(x_region, x_extra)  # [n_samples, 4]
      res.append(x_region_out)
    
    return torch.concat(res, dim=1)  # [n_samples, n_layers, n_regions]

  def forward(self, x_in:torch.Tensor, out:torch.Tensor):
    """
    1. Receive signal from backbone [n_samples, in_features, signal_length]
    2. Reduce size to [n_samples, 8, cells]
    3. (1) Run through cells
    4. (1) Decode output
    5. (1) Encode output
    6. (2) Run through cells
    7. (2) Decode output
    8. (2) Encode output
    9. Return output
    """
    
    x_sys = self.forward_convsys(x_in)    # [n_samples, 4, n_regions]    
    x_oscs = self.forward_oscs(x_in, out) # [n_samples, n_regions, 4*n_layers]
    
    x_oscs = x_oscs.view(x_oscs.shape[0], self.n_layers, self.n_regions, -1)  # [n_samples, n_layers, n_regions, 4]4
    x_sys = x_sys.view(x_sys.shape[0], self.n_layers, -1)  # [n_samples, n_layers, 11]
    
    return x_oscs, x_sys  # [n_samples, n_layers, n_regions, 4], [n_samples, n_layers, 11]


class DetectionHead62(nn.Module):
  """
    For each cell; different network, no communication between cells.
  """
  def __init__(
    self, 
    unfold_idx:int,
    scale_idx:int,
    in_features:int, 
    out_features:int, 
    act_fun_str:str,
    stride:int,
    kernel_size:int,
    system_info:dict,
  ):
    super(DetectionHead6, self).__init__()
    self.name = 'DetectionHead'       # 
    self.unfold_idx = unfold_idx      # 
    self.scale_idx = scale_idx        # 
    self.in_features = in_features    # 
    self.out_features = out_features  # 
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   # 
    self.n_regions = n_regions = system_info['n_regions'] #
    self.n_layers = n_layers = system_info['n_layers']    #
    self.n_sys_pars = system_info['n_sys_pars']           #

    self.flatten = nn.Flatten() #
    self.act_fn_cnn = act_str_to_fn(act_fun_str) #
    self.act_fn_sys = act_str_to_fn(act_fun_str) #
    self.act_fn_dec = act_str_to_fn(act_fun_str) #
    
    self.shape_norm = nn.Upsample(
      size=(64),
      mode='linear',
      align_corners=True,
    )
    
    self.stride = stride
    self.kernel_size = kernel_size
    sys_obj_size = int((stride*n_regions) / (2**(sum([2, 2, 2]) - 3)))  # Calculate size of system object
    
    self._init_sys(
      shapes_info={
        'in_features': in_features,
        'n_layers': self.n_layers,
        'n_regions': self.n_regions,
        'n_sys_pars': system_info['n_sys_pars'],
        'sys_obj_size': sys_obj_size
      },
      strides=[2, 2, 2, 2, 2],
      padding_mode='replicate',
      act_fun_str=act_fun_str,
    )
    
    self._init_osc(
      shapes_info={
        'in_features': in_features,
        'n_layers': self.n_layers,
        'n_regions': self.n_regions,
        'n_sys_pars': system_info['n_sys_pars'],
        'sys_obj_size': sys_obj_size
      },
      strides=[2, 2, 2],
      padding_mode='replicate',
      act_fun_str=act_fun_str,
    )

  def get_gradients(self):
    gradients = []
    gradient_norms = []
    
    gradients.append(self.convsys_cnn1.weight.grad)
    gradients.append(self.convsys_cnn2.weight.grad)
    gradients.append(self.convsys_cnn3.weight.grad)
    gradients.append(self.convsys_fc.weight.grad)	
    
    for cell in self.cells:
      gradients.append(cell.cnn1.weight.grad)
      gradients.append(cell.cnn2.weight.grad)
      gradients.append(cell.cnn3.weight.grad)
      gradients.append(cell.fc1.weight.grad)
      gradients.append(cell.fc2.weight.grad)
      
    try:
      gradient_norms.append(self.conv1.weight.grad.norm())
      gradient_norms.append(self.conv2.weight.grad.norm())
      
      for cell in self.cells:
        gradient_norms.append(cell.conn_1.weight.grad.norm())
        gradient_norms.append(cell.conn_2.weight.grad.norm())
        gradient_norms.append(cell.conn_3.weight.grad.norm())
    
      gradient_norms.append(self.cell_system.conn_1.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_2.weight.grad.norm())
      gradient_norms.append(self.cell_system.conn_3.weight.grad.norm())
    except:
      # No gradient
      gradient_norms = [0]*len(gradients)
    
    gradients = [pt2np(g) for g in gradients if g is not None] 
    gradient_norms = [pt2np(g) for g in gradient_norms]
    
    return gradients, gradient_norms

  def _init_weights(self, net:nn.Linear):
    """Initialize weights of the network according to diagonal init for matrix, with small noise for other stuff"""
    net.weight.data = torch.eye(net.weight.shape[0], net.weight.shape[1], dtype=self.dt, device=self.dv) + \
      torch.normal(0, 0.01, size=net.weight.shape, dtype=self.dt, device=self.dv)
    
    return net

  def _init_sys(self, shapes_info:Dict[str, int], strides:List[int], padding_mode:str='replicate', act_fun_str:str='tanh'):
    in_features:int = shapes_info['in_features']
    n_layers:int = shapes_info['n_layers']
    n_regions:int = shapes_info['n_regions']
    n_sys_pars:int = shapes_info['n_sys_pars']
    sys_obj_size:int = shapes_info['sys_obj_size']
    
    self.convsys_cnn1 = nn.Conv1d(in_features, 2*n_layers, 3, strides[0], (3-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn2 = nn.Conv1d(2*n_layers, 4*n_layers, 3, strides[1], (3-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn3 = nn.Conv1d(4*n_layers, 8*n_layers, 5, strides[2], (5-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn4 = nn.Conv1d(8*n_layers, 16*n_layers, 5, strides[3], (5-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.convsys_cnn5 = nn.Conv1d(16*n_layers, 4*n_layers, 7, strides[4], (7-strides[0])//2 + 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    
    # self.sys_norm1 = nn.InstanceNorm1d(2*n_layers, affine=True, dtype=self.dt, device=self.dv)
    self.sys_norm2 = nn.InstanceNorm1d(4*n_layers, affine=True, dtype=self.dt, device=self.dv)
    # self.sys_norm3 = nn.InstanceNorm1d(8*n_layers, affine=True, dtype=self.dt, device=self.dv)
    self.sys_norm4 = nn.InstanceNorm1d(16*n_layers, affine=True, dtype=self.dt, device=self.dv)
    # self.sys_norm5 = nn.InstanceNorm1d(4*n_layers, affine=True, dtype=self.dt, device=self.dv)
    
    
    self.sys_act1 = act_str_to_fn(act_fun_str)  # Activation function for first conv
    self.sys_act2 = act_str_to_fn(act_fun_str)  # Activation function for
    self.sys_act3 = act_str_to_fn(act_fun_str)  # Activation function for third conv
    self.sys_act4 = act_str_to_fn(act_fun_str)  # Activation function for
    
    self.sys_norm5 = nn.InstanceNorm1d(4*n_layers, affine=True, dtype=self.dt, device=self.dv)
    self.convsys_fc = nn.Linear(sys_obj_size*n_layers, (n_sys_pars)*n_layers , dtype=self.dt, device=self.dv)

  def _init_osc(self, shapes_info:Dict[str, int], padding_mode:str='replicate', act_fun_str:str='tanh'):
    self.cnn1_w0 = nn.Conv1d(shapes_info['in_features'], 8*shapes_info['n_layers'], 3, 2, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.cnn2_w0 = nn.Conv1d(8*shapes_info['n_layers'], 16*shapes_info['n_layers'], 3, 2, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.cnn3_w0 = nn.Conv1d(16*shapes_info['n_layers'], 2*shapes_info['n_layers'], 1, 1, 0, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.norm1_w0 = nn.InstanceNorm1d(8*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)
    self.norm2_w0 = nn.InstanceNorm1d(16*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)
    
    self.cnn1_wp = nn.Conv1d(shapes_info['in_features'], 8*shapes_info['n_layers'], 3, 2, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.cnn2_wp = nn.Conv1d(8*shapes_info['n_layers'], 16*shapes_info['n_layers'], 3, 2, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.cnn3_wp = nn.Conv1d(16*shapes_info['n_layers'], 2*shapes_info['n_layers'], 1, 1, 0, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.norm1_wp = nn.InstanceNorm1d(8*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)
    self.norm2_wp = nn.InstanceNorm1d(16*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)
    
    self.cnn1_g = nn.Conv1d(shapes_info['in_features'], 8*shapes_info['n_layers'], 3, 2, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.cnn2_g = nn.Conv1d(8*shapes_info['n_layers'], 16*shapes_info['n_layers'], 3, 2, 1, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.cnn3_g = nn.Conv1d(16*shapes_info['n_layers'], 2*shapes_info['n_layers'], 1, 1, 0, padding_mode=padding_mode, dtype=self.dt, device=self.dv)
    self.norm1_g = nn.InstanceNorm1d(8*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)
    self.norm2_g = nn.InstanceNorm1d(16*shapes_info['n_layers'], affine=True, dtype=self.dt, device=self.dv)  
    
    self.convosc_act1 = act_str_to_fn(act_fun_str)
    self.convosc_act2 = act_str_to_fn(act_fun_str)
    
    self.convosc_fc1 = nn.Linear(
      in_features=16*2*3 + 5, 
      out_features=32, 
      dtype=self.dt,
      device=self.dv
    )
    self.convosc_fc2 = nn.Linear(
      in_features=32, 
      out_features=4, 
      dtype=self.dt,
      device=self.dv
    )
    
    self.convosc_act3 = act_str_to_fn(act_fun_str)          #
    self.convosc_fc1 = self._init_weights(self.convosc_fc1) # 
    self.convosc_fc2 = self._init_weights(self.convosc_fc2) # 

  def forward_convsys(self, x_in:torch.Tensor) -> torch.Tensor:
    """
    Forward pass for the convsys part of the detection head.
    This part processes the input signal to extract system-related features.
    """
    x_sys1 = self.convsys_cnn1(x_in)                # [n_samples, 4*n_layers, n_regions]
    x_sys1 = self.sys_act1(x_sys1)  #
    
    x_sys2 = self.convsys_cnn2(x_sys1)  # 
    x_sys2 = self.sys_act2(self.sys_norm2(x_sys2))      #  
    
    x_sys3 = self.convsys_cnn3(x_sys2)  # 
    x_sys3 = self.sys_act3(x_sys3)  # [n_samples, 8*n_layers, n_regions]
    
    x_sys4 = self.convsys_cnn4(x_sys3)  #
    x_sys4 = self.sys_act4(self.sys_norm4(x_sys4))  #
    
    x_sys5 = self.convsys_cnn5(x_sys4)  # [n_samples, 4*n_layers, n_regions]
    
    x_sys_out = self.convsys_fc(self.flatten(x_sys5))   # 
    
    return x_sys_out  # [n_samples, 11*n_layers]

  def forward_convosc(self, x_in:torch.Tensor, x_extra:torch.Tensor) -> torch.Tensor:
    x_64 = self.shape_norm(x_in)  # [n_samples, in_features, 64]
    
    # - W0 -
    x_cnn1 = self.cnn1_w0(x_64)  # [n_samples, 8*n_layers, 64]
    x_cnn1_norm = self.norm1_w0(x_cnn1)
    x_cnn1_act = self.convosc_act1(x_cnn1_norm)
    x_cnn2 = self.cnn2_w0(x_cnn1_act)  # [n_samples, 16*n_layers, 64]
    x_cnn2_norm = self.norm2_w0(x_cnn2)
    x_cnn2_act = self.convosc_act2(x_cnn2_norm)
    x_cnn3 = self.cnn3_w0(x_cnn2_act)  # [n_samples, 2*n_layers, 64]
    x_cnn3_w0_flat = self.flatten(x_cnn3)  # [n_samples, 2*n_layers*64]
    
    # - WP -
    x_cnn1_wp = self.cnn1_wp(x_64)  # [n_samples, 8*n_layers, 64]
    x_cnn1_wp_norm = self.norm1_wp(x_cnn1_wp)
    x_cnn1_wp_act = self.convosc_act1(x_cnn1_wp_norm)
    x_cnn2_wp = self.cnn2_wp(x_cnn1_wp_act)  # [n_samples, 16*n_layers, 64]
    x_cnn2_wp_norm = self.norm2_wp(x_cnn2_wp)
    x_cnn2_wp_act = self.convosc_act2(x_cnn2_wp_norm)
    x_cnn3_wp = self.cnn3_wp(x_cnn2_wp_act)  # [n_samples, 2*n_layers, 64]
    x_cnn3_wp_flat = self.flatten(x_cnn3_wp)  # [n_samples, 2*n_layers*64]
    
    # - G -
    x_cnn1_g = self.cnn1_g(x_64)  # [n_samples, 8*n_layers, 64]
    x_cnn1_g_norm = self.norm1_g(x_cnn1_g)
    x_cnn1_g_act = self.convosc_act1(x_cnn1_g_norm)
    x_cnn2_g = self.cnn2_g(x_cnn1_g_act)  # [n_samples, 16*n_layers, 64]
    x_cnn2_g_norm = self.norm2_g(x_cnn2_g)
    x_cnn2_g_act = self.convosc_act2(x_cnn2_g_norm)
    x_cnn3_g = self.cnn3_g(x_cnn2_g_act)  # [n_samples, 2*n_layers, 64]
    x_cnn3_g_flat = self.flatten(x_cnn3_g)  # [n_samples, 2*n_layers*64]
    
    # Concatenate all outputs
    x_extra_flat = self.flatten(x_extra)  # [n_samples, 4*n_layers]
    x = torch.concat([
      x_cnn3_w0_flat, 
      x_cnn3_wp_flat,
      x_cnn3_g_flat,
      x_extra_flat
    ], dim=-1)  # [n_samples, 2*n_layers*64 + 4*n_layers]
    
    x_fc1 = self.convosc_act3(self.convosc_fc1(x))  # [n_samples, 4]
    x_fc2 = self.convosc_fc2(x_fc1)  # [n_samples, 4]
    
    return x_fc2[:,None,:]  # [n_samples, region, 4]

  def forward_oscs(self, x_in:torch.Tensor, out:torch.Tensor):
    ks, ks_2, s = self.kernel_size, (self.kernel_size - self.stride) // 2, self.stride
    res = []
    
    x_in_padded = pad(x_in, (ks_2, ks_2), mode='constant')  # Pad input to match kernel size
    for idx_reg in range(self.n_regions):
      x_region = x_in_padded[...,int((idx_reg+1/2)*s-(ks//2))+ks_2:int((idx_reg+1/2)*s+(ks//2))+ks_2]
      out_region = out[:,:,:,idx_reg,:].flatten(1)  # [n_samples, n_layers, n_regions, 4]
      idx_region = torch.full((x_region.shape[0], 1), fill_value=idx_reg, dtype=self.dt, device=self.dv)  # [n_samples, 1]
      x_extra = torch.concat([out_region, idx_region], dim=-1)  # [n_samples, n_layers, n_regions, 4 + in_features]
      
      x_region_out = self.forward_convosc(x_region, x_extra)  # [n_samples, 4]
      res.append(x_region_out)
    
    return torch.concat(res, dim=1)  # [n_samples, n_layers, n_regions]

  def forward(self, x_in:torch.Tensor, out:torch.Tensor):
    """
    1. Receive signal from backbone [n_samples, in_features, signal_length]
    2. Reduce size to [n_samples, 8, cells]
    3. (1) Run through cells
    4. (1) Decode output
    5. (1) Encode output
    6. (2) Run through cells
    7. (2) Decode output
    8. (2) Encode output
    9. Return output
    """
    
    x_sys = self.forward_convsys(x_in)    # [n_samples, 4, n_regions]    
    x_oscs = self.forward_oscs(x_in, out) # [n_samples, n_regions, 4*n_layers]
    
    x_oscs = x_oscs.view(x_oscs.shape[0], self.n_layers, self.n_regions, -1)  # [n_samples, n_layers, n_regions, 4]4
    x_sys = x_sys.view(x_sys.shape[0], self.n_layers, -1)  # [n_samples, n_layers, 11]
    
    return x_oscs, x_sys  # [n_samples, n_layers, n_regions, 4], [n_samples, n_layers, 11]


class DetectionHeadCell(nn.Module):
  def __init__(
    self, 
    input_size: int,
    output_size: int,
    act_fun_str: str,
    system_info:dict,
    communication:bool=False,
    *args, 
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   #
    
    comm_size = 2
    self.comm_out_1 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv) #
    self.comm_out_2 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv) #
    self.comm_in_1 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv) #
    self.comm_in_2 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv)
    
    self.out_1 = torch.zeros((1,4), dtype=self.dt, device=self.dv)  # 
    self.out_2 = torch.zeros((1,2), dtype=self.dt, device=self.dv)  #
    self.output = torch.zeros((output_size), dtype=self.dt, device=self.dv) # [obj_score, w0, wp, g]
    
    self.act_fn = act_str_to_fn(act_fun_str)  # 
    self.conn_1 = nn.Linear(input_size, self.out_1.shape[1]+comm_size, dtype=self.dt, device=self.dv)             # 
    self.conn_2 = nn.Linear(self.out_1.shape[1]+comm_size, self.out_2.shape[1]+comm_size, dtype=self.dt, device=self.dv)  # 
    self.conn_3 = nn.Linear(self.out_2.shape[1]+comm_size, output_size, dtype=self.dt, device=self.dv)            # 
    # self.conn_3 = nn.Linear(input_size, output_size, dtype=self.dt, device=self.dv)  # 
    # self.conn_out = nn.Linear(self.out_2.shape[0]+comm_size, self.output.shape[0], dtype=self.dt, device=self.dv) #
    
    # self.scaling_factor = torch.tensor([5.0, 1.0, 1.0, 1.0], dtype=self.dt, device=self.dv) # 
  
  def forward_1(self, x_in):
    out = self.act_fn(self.conn_1(x_in))
    self.out_1 = out[:, :self.out_1.shape[1]]
    self.comm_out_1 = out[:, self.out_1.shape[1]:]

  def forward_2(self):
    x_in = torch.concat([self.out_1, self.comm_in_1], axis=1)
    out = self.act_fn(self.conn_2(x_in))
    self.out_2 = out[:, :self.out_2.shape[1]]
    self.comm_out_2 = out[:, self.out_2.shape[1]:]
    
  def forward_3(self) -> torch.Tensor:
    x_in = torch.concat([self.out_2, self.comm_in_2], axis=1)   # 4
    output = self.conn_3(x_in) # 4
    
    self.output = output_scaled = output # * self.scaling_factor  # Scale objectness score
    
    return output_scaled


class DetectionHeadSystem(nn.Module):
  def __init__(
    self, 
    input_size: int,
    output_size: int,
    act_fun_str: str,
    system_info:dict,
    *args, 
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   #
    
    comm_size = 2
    
    self.comm_out_1 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv) #
    self.comm_out_2 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv) #
    self.comm_in_1 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv)  #
    self.comm_in_2 = torch.zeros((comm_size,1), dtype=self.dt, device=self.dv)  # 
    
    self.out_1 = torch.zeros((1,output_size), dtype=self.dt, device=self.dv)  # 
    self.out_2 = torch.zeros((1,output_size), dtype=self.dt, device=self.dv)  #
    self.output = torch.zeros((output_size), dtype=self.dt, device=self.dv) # [obj_score, w0, wp, g]
    
    self.act_fn = act_str_to_fn(act_fun_str)  # 
    self.conn_1 = nn.Linear(input_size, self.out_1.shape[1]+comm_size, dtype=self.dt, device=self.dv)   # 
    self.conn_2 = nn.Linear(self.out_1.shape[1]+comm_size, self.out_2.shape[1]+comm_size, dtype=self.dt, device=self.dv)  # 
    self.conn_3 = nn.Linear(self.out_2.shape[1]+comm_size, output_size, dtype=self.dt, device=self.dv)  # 
    # self.conn_3 = nn.Linear(input_size, output_size, dtype=self.dt, device=self.dv)  # 
    # self.conn_out = nn.Linear(self.out_2.shape[0]+comm_size, self.output.shape[0], dtype=self.dt, device=self.dv) #
  
  def forward_1(self, x_in):
    out = self.act_fn(self.conn_1(x_in))
    self.out_1 = out[:, :self.out_1.shape[1]]
    self.comm_out_1 = out[:, self.out_1.shape[1]:]

  def forward_2(self):
    x_in = torch.concat([self.out_1, self.comm_in_1], axis=1)
    out = self.act_fn(self.conn_2(x_in))
    self.out_2 = out[:, :self.out_2.shape[1]]
    self.comm_out_2 = out[:, self.out_2.shape[1]:]
    
  def forward_3(self) -> torch.Tensor:
    x_in = torch.concat([self.out_2, self.comm_in_2], axis=1)   # 4
    self.output = output = self.conn_3(x_in)          # 4
    
    return output


class DetectionHeadCell5(nn.Module):
  def __init__(
    self, 
    input_channels: int,
    output_channels: int,
    act_fun_str: str,
    system_info:dict,
    *args, 
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    
    self.dt = system_info['dtype']    #
    self.dv = system_info['device']   #

    self.cnn1 = nn.Conv1d(
      in_channels=input_channels,
      out_channels=16,
      kernel_size=3,
      stride=2,
      padding=1,
      padding_mode='replicate',
      dtype=self.dt,
      device=self.dv
    )
    self.cnn2 = nn.Conv1d(
      in_channels=16,
      out_channels=32,
      kernel_size=5,
      stride=1,
      padding=2,
      groups=4,  # n_layers
      padding_mode='replicate',
      dtype=self.dt,
      device=self.dv
    )
    self.cnn3 = nn.Conv1d(
      in_channels=32,
      out_channels=8,  # n_layers*4
      kernel_size=7,
      stride=3,
      padding=3,
      groups=4,  # n_layers
      padding_mode='replicate',
      dtype=self.dt,
      device=self.dv
    )
    
    self.fc1 = nn.Linear(
      in_features=108,
      out_features=32,  # n_layers*4
      bias=False,
      dtype=self.dt,
      device=self.dv
    )
    self.fc2 = nn.Linear(
      in_features=32,
      out_features=output_channels,  # n_layers*4
      bias=False,
      dtype=self.dt,
      device=self.dv
    )
  
    self.act_cnn1 = act_str_to_fn(act_fun_str)  # Activation function for CNN layers
    self.act_cnn2 = act_str_to_fn(act_fun_str)  # Activation function for CNN layers
    self.act_cnn3 = act_str_to_fn(act_fun_str)  # Activation function for CNN layers
    self.act_fc1 = act_str_to_fn(act_fun_str)   # Activation function for FC layers
    self.act_fc2 = act_str_to_fn(act_fun_str)   # Activation function for FC layers
    self.flatten = nn.Flatten()  # Flatten layer to reshape tensors

    # self.batchnorm1 = nn.BatchNorm1d(16, dtype=self.dt, device=self.dv)  # Batch normalization for CNN layer 1
    # self.batchnorm2 = nn.BatchNorm1d(16, dtype=self.dt, device=self.dv)  # Batch normalization for CNN layer 2
    # self.batchnorm_fc1 = nn.BatchNorm1d(8, dtype=self.dt, device=self.dv)  # Batch normalization for FC layer 1
  
  def forward(self, x_in:torch.Tensor, out:torch.Tensor):
    """
    Forward pass for the cell detection head.
    This part processes the input signal to extract object-related features.
    """
    x_cnn1 = self.act_cnn1(self.cnn1(x_in))     # [n_samples, 16, n_regions]
    # x_cnn1 = self.batchnorm1(x_cnn1)            # Apply batch normalization
    x_cnn2 = self.act_cnn2(self.cnn2(x_cnn1))   # [n_samples, 16, n_regions]
    # x_cnn2 = self.batchnorm2(x_cnn2)            # Apply batch normalization
    x_cnn3 = self.act_cnn3(self.cnn3(x_cnn2))   # [n_samples, 4*n_layers, n_regions]
    
    x_f   = self.flatten(x_cnn3)                # Flatten the output of the CNN layers
    x_f   = torch.concat([x_f, self.flatten(out)], axis=1)  # Concatenate with the flattened output from the previous layer
    x_fc1 = self.act_fc1(self.fc1(x_f))         # [n_samples, 4*n_layers]
    # x_fc1 = self.batchnorm_fc1(x_fc1)           # Apply batch normalization to the first FC layer output
    x_out = self.act_fc2(self.fc2(x_fc1))       # [n_samples, 4*n_layers]
    x_out = x_out.view(x_in.shape[0], -1, 4)    # Reshape to [n_samples, n_layers, 4]
    # x_out = x_out.permute(0, 2, 1)  # Permute to [n_samples, 4, n_layers]
    
    return x_out  # Return the output tensor


class HeadSSDScaleModel(nn.Module):
  def __init__(self, i_scale:int, shape_info:Dict[str, int], sys_info:Dict[str, int], *args, **kwargs):
    super(HeadSSDScaleModel, self).__init__()
    self.i_scale = i_scale
    self.n_features_in = shape_info['in_features']      # Number of input features
    self.n_sys_pars = shape_info['n_sys_pars']  # Number of system parameters
    self.dt = sys_info['dtype']   # Data type
    self.dv = sys_info['device']  # Device to run on
    
    self.n_layers = n_layers = shape_info['n_layers'] # Number of layers
    n_scales = shape_info['osc_shape'][1] # Number of scales
    self.n_regions = n_regions = shape_info['n_regions'] # Number of regions
    n_attribs = shape_info['osc_shape'][3]  # Number of attributes

    out_channels = 4
    
    # -- convolution info --
    stride = shape_info['stride'] // 16          # Stride for convolution
    kernel_size = shape_info['kernel_size'] // 16 # Kernel size for convolution
    
    # -- fc info --
    in_features = n_regions*out_channels + n_regions*n_attribs + self.n_sys_pars

    self.flatten = nn.Flatten()

    self.cnn = nn.ModuleDict({
      'cnn1': nn.Conv1d(
        in_channels=self.n_features_in, 
        out_channels=8, 
        kernel_size=5, 
        stride=2, 
        padding=2,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      ),
      'cnn2': nn.Conv1d(
        in_channels=8, 
        out_channels=16, 
        kernel_size=7, 
        stride=2, 
        padding=3,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      ),
      'cnn3': nn.Conv1d(
        in_channels=16, 
        out_channels=16, 
        kernel_size=7, 
        stride=2, 
        padding=3,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      ),
      'cnn4': nn.Conv1d(
        in_channels=16, 
        out_channels=32, 
        kernel_size=9, 
        stride=2, 
        padding=4,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      ),
      'cnn5': nn.Conv1d(
        in_channels=32, 
        out_channels=out_channels*n_layers, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=(kernel_size - stride),
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      )
    })

    self.norm = nn.ModuleDict({
      'norm1': nn.InstanceNorm1d(8, dtype=self.dt, device=self.dv),
      'norm2': nn.InstanceNorm1d(16, dtype=self.dt, device=self.dv),
      'norm3': nn.InstanceNorm1d(16, dtype=self.dt, device=self.dv),
      'norm4': nn.InstanceNorm1d(32, dtype=self.dt, device=self.dv),
      'norm5': nn.InstanceNorm1d(out_channels, dtype=self.dt, device=self.dv)
    })
    
    self.act = nn.ModuleDict({
      'act1': ScalableTanh(self.dt, self.dv),
      'act2': ScalableTanh(self.dt, self.dv),
      'act3': ScalableTanh(self.dt, self.dv),
      'act4': ScalableTanh(self.dt, self.dv),
      'act5': ScalableTanh(self.dt, self.dv),
      'fc1': ScalableTanhOutput(((n_regions*n_attribs + self.n_sys_pars)*n_layers), self.dt, self.dv),
      'fc2': ScalableTanhOutput(((n_regions*n_attribs + self.n_sys_pars)*n_layers), self.dt, self.dv),
    })
    
    self.fc = nn.ModuleDict({
      'fc1': nn.Linear(
        in_features=in_features*n_layers,
        out_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        dtype=self.dt,
        device=self.dv
      ),
      'fc2': nn.Linear(
        in_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        out_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        dtype=self.dt,
        device=self.dv
      ),
      'fc3': nn.Linear(
        in_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        out_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        dtype=self.dt,
        device=self.dv
      )
    })

    self.n_params_cnn = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
    self.n_params_fc = sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
    self.n_params = self.n_params_cnn + self.n_params_fc

  def forward(
    self, 
    x_in:torch.Tensor, 
    z_sys:torch.Tensor|None, 
    z_osc_scale:torch.Tensor|None, 
    **kwargs
  ):
    n_samples = x_in.shape[0] # 
    x_cnn1 = self.act['act1'](self.norm['norm1'](self.cnn['cnn1'](x_in)))     # 
    x_cnn2 = self.act['act2'](self.norm['norm2'](self.cnn['cnn2'](x_cnn1)))   # 
    x_cnn3 = self.act['act3'](self.norm['norm3'](self.cnn['cnn3'](x_cnn2)))   # 
    x_cnn4 = self.act['act4'](self.norm['norm4'](self.cnn['cnn4'](x_cnn3)))   # 
    x_cnn5 = self.act['act5'](self.norm['norm5'](self.cnn['cnn5'](x_cnn4)))   # [n_samples, 4, n_regions]

    x_flat = self.flatten(x_cnn5)
    x_fc_in = torch.concat((x_flat, z_sys.flatten(1), z_osc_scale.flatten(1)), axis=1) # [n_samples, in_features*n_layers]
    x_fc2 = self.act['fc1'](self.fc['fc1'](x_fc_in))
    x_fc3 = self.act['fc2'](self.fc['fc2'](x_fc2)) 
    x_fc4 = self.fc['fc3'](x_fc3) 

    # reshape [64 -> (self.n_regions, 4)]
    x_oscs = x_fc4[:, :(self.n_regions*4)*self.n_layers].view(n_samples, self.n_layers, self.n_regions, -1) # 
    x_sys = x_fc4[:, (self.n_regions*4)*self.n_layers:].view(n_samples, self.n_layers, -1) # 
    
    return x_oscs, x_sys

  def get_weights(self):
    return {name: param for name, param in self.named_parameters() if param.requires_grad}


class HeadSSDScaleModel_complex(nn.Module):
  def __init__(self, i_scale:int, shape_info:Dict[str, int], sys_info:Dict[str, int], *args, **kwargs):
    super(HeadSSDScaleModel_complex, self).__init__()
    self.i_scale = i_scale
    self.n_features_in = shape_info['in_features']  # Number of input features
    self.n_sys_pars = shape_info['n_sys_pars']      # Number of system parameters
    self.dt = sys_info['dtype']   # Data type
    self.dv = sys_info['device']  # Device to run on
    
    self.n_layers = n_layers = shape_info['osc_shape'][0] # Number of layers
    n_scales = shape_info['osc_shape'][1] # Number of scales
    self.n_regions = n_regions = shape_info['n_regions'] # Number of regions
    n_attribs = shape_info['osc_shape'][3]  # Number of attributes

    out_channels = 1
    
    # -- convolution info --
    self.stride = shape_info['stride']           # Stride for convolution
    self.kernel_size = shape_info['kernel_size'] # Kernel size for convolution
    self.zero_padding = (self.kernel_size - self.stride)
    
    standard_size = 64
    
    # -- fc info --
    # in_features = n_regions*out_channels + n_regions*n_attribs + self.n_sys_pars
    in_features = min(64, (self.n_regions * 8)) * out_channels + n_regions*n_attribs + self.n_sys_pars

    self.flatten = nn.Flatten()
    self.downsample = nn.Upsample(size=standard_size, mode='linear', align_corners=True) # Downsample to standard size

    self.cnn = nn.ModuleDict({
      'cnn1': nn.Conv1d(
        in_channels=self.n_features_in, 
        out_channels=16, 
        kernel_size=5, 
        stride=2, 
        padding=2,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      ),
      'cnn2': nn.Conv1d(
        in_channels=16, 
        out_channels=32, 
        kernel_size=5, 
        stride=2, 
        padding=2,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      ),
      'cnn3': nn.Conv1d(
        in_channels=32, 
        out_channels=32, 
        kernel_size=5, 
        stride=2, 
        padding=2,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      ),
      'cnn4': nn.Conv1d(
        in_channels=32, 
        out_channels=out_channels, 
        kernel_size=3, 
        stride=1, 
        padding=1,
        padding_mode='replicate',
        dtype=self.dt, 
        device=self.dv
      )
    })

    if self.n_regions > 4:
      self.cnn_downsample_to_64 = nn.Conv1d(out_channels, out_channels, ((self.n_regions*8)//64)*2, stride=((self.n_regions*8)//64), padding=((self.n_regions*8)//64)//2, dtype=self.dt, device=self.dv)

    self.norm = nn.ModuleDict({
      'norm1': nn.InstanceNorm1d(16, dtype=self.dt, device=self.dv),
      'norm2': nn.InstanceNorm1d(32, dtype=self.dt, device=self.dv),
      'norm3': nn.InstanceNorm1d(32, dtype=self.dt, device=self.dv),
      'norm4': nn.InstanceNorm1d(out_channels, dtype=self.dt, device=self.dv)
    })
    
    self.act = nn.ModuleDict({
      'act1': ScalableTanh(self.dt, self.dv),
      'act2': ScalableTanh(self.dt, self.dv),
      'act3': ScalableTanh(self.dt, self.dv),
      'act4': ScalableTanh(self.dt, self.dv),
      'fc1': ScalableTanh(self.dt, self.dv),
      'fc2': ScalableTanh(self.dt, self.dv),
    })
    
    # first item, 32
    self.fc = nn.ModuleDict({
      'fc1': nn.Linear(
        in_features=in_features*n_layers,
        out_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        dtype=self.dt,
        device=self.dv
      ),
      'fc2': nn.Linear(
        in_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        out_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        dtype=self.dt,
        device=self.dv
      ),
      'fc3': nn.Linear(
        in_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        out_features=(n_regions*n_attribs + self.n_sys_pars)*n_layers,
        dtype=self.dt,
        device=self.dv
      )
    })

    self.n_params_cnn = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
    self.n_params_fc = sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
    self.n_params = self.n_params_cnn + self.n_params_fc

  def forward(
    self, 
    x_in:torch.Tensor, 
    z_sys:torch.Tensor|None, 
    z_osc_scale:torch.Tensor|None, 
    **kwargs
  ):
    n_samples = x_in.shape[0] # 
    x_in_padded = F.pad(x_in, (self.zero_padding, 0), mode='constant') # Pad input to ensure correct size after conv
    x_f = []
    for step in range(self.n_regions):
      x_region = x_in_padded[:, :, self.stride*step : self.stride*(step+1)] # Extract region
      x_in_downsampled = self.downsample(x_region) # Downsample to standard size
      x_cnn1 = self.act['act1'](self.norm['norm1'](self.cnn['cnn1'](x_in_downsampled))) # 
      x_cnn2 = self.act['act2'](self.norm['norm2'](self.cnn['cnn2'](x_cnn1))) # 
      x_cnn3 = self.act['act3'](self.norm['norm3'](self.cnn['cnn3'](x_cnn2))) # [n_samples, 4, n_regions]
      x_cnn4 = self.act['act4'](self.norm['norm4'](self.cnn['cnn4'](x_cnn3))) # [n_samples, 4, n_regions]
      x_f.append(x_cnn4)

    x_fc = torch.concat(x_f, axis=2) # [n_samples, 4, n_regions]
    x_flat = self.flatten(x_fc)
    
    if x_flat.shape[1] > 64:
      x_flat = self.cnn_downsample_to_64(x_flat[:,None,:]).flatten(1)
    
    x_fc_in = torch.concat((x_flat, z_sys.flatten(1), z_osc_scale.flatten(1)), axis=1) # [n_samples, in_features*n_layers]
    x_fc2 = self.act['act1'](self.fc['fc1'](x_fc_in))
    x_fc3 = self.act['act2'](self.fc['fc2'](x_fc2)) 
    x_fc4 = self.act['act3'](self.fc['fc3'](x_fc3))

    # reshape [64 -> (self.n_regions, 4)]
    x_oscs = x_fc4[:, :(self.n_regions*4)*self.n_layers].view(n_samples, self.n_layers, self.n_regions, -1) # 
    x_sys = x_fc4[:, (self.n_regions*4)*self.n_layers:].view(n_samples, self.n_layers, -1) # 
    
    return x_oscs, x_sys

  def get_weights(self):
    return {name: param for name, param in self.named_parameters() if param.requires_grad}

  def get_n_params(self):
    return self.n_params



def plot_pt_histogram(
  pred:torch.Tensor,
  target:torch.Tensor,
  name:str='',
) -> None:
  """
  Plot a histogram of the predicted and target values.
  Can only work for 1D or 2D tensors
  """
  import matplotlib.pyplot as plt
  
  n_dims = len(pred.shape)
  fig, ax = plt.subplots(n_dims, 1)
  
  if n_dims == 1:
    pred = pred.unsqueeze(0)
    target = target.unsqueeze(0)
    ax = [ax]
  elif n_dims > 2:
    raise ValueError("Input tensors must be 1D or 2D")
  
  for i in range(n_dims):
    ax[i].hist(target.cpu().detach().numpy(), bins=100, alpha=0.5, color='tab:blue', label='target')
    ax[i].hist(pred.cpu().detach().numpy(), bins=100, alpha=0.5, color='tab:orange', label='pred')
    ax[i].set(
      title=f'Histogram of {name}'
    )
    ax[i].legend()
  
  fig.savefig('outputs/histogram_' + name + '.png', dpi=300, bbox_inches='tight')
  plt.close(fig) # Close the figure to free memor


class ActivationG(torch.nn.Module):
  def __init__(self, slope=5):
    super().__init__()
    self.slope = slope  # Slope for x ≥ 0 (default: 5 for smoothness)
    self.relu = nn.ReLU()

  def __repr__(self):
    return f"ActivationG(slope={self.slope})"

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    # For x < 0: 5 * exp(x)
    # For x ≥ 0: 5 + slope * x
    # return torch.where(x < 0, self.slope * torch.exp(x), self.slope * (1+x))
    # return torch.where(x < 0, self.slope * torch.exp(2*x), self.slope + (2*self.slope) * x)
    # return self.relu(x*self.slope)
    return torch.exp(x)
    # return self.relu(x*self.slope)  # ReLU activation function for positive values
  
  def get_slope(self):
    return self.slope


class ActivationWp(torch.nn.Module):
  def __init__(self, slope=250):
    super().__init__()    # 
    self.slope = slope    # Slope for x ≥ 0 (default: 5 for smoothness)
    self.relu = nn.ReLU() # 
  
  def __repr__(self):
    return f"ActivationWp(slope={self.slope})"
  
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    # return torch.where(x < 0, self.slope * torch.exp(x), self.slope + (self.slope) * x)
    # return torch.where(x < 0, self.slope * torch.exp(2*x), self.slope + (2*self.slope) * x)
    return self.relu(x*self.slope)
  
  def get_slope(self):
    return self.slope



class SimulateFunction(Function):
    """
    Custom autograd function to simulate a batch of inputs through a given physical model.
    Version 1: Calculates the gradient during the backward pass.

    Attributes:
        None
    """
    @staticmethod
    def forward(ctx, p:torch.Tensor, system):
        """
        Forward pass of the custom autograd function. 

        Args:
            ctx (Context): A context object to save tensors for the backward pass.
            p (torch.Tensor): Input parameters.
            system: The physical model to simulate. (TmMethodPyTorch / TmMethodPyTorchBatch)

        Returns:
            torch.Tensor: Output of the forward pass.
        """
        ctx.save_for_backward(p)

        ctx.system = system    # system (physical model)
        out = system(p)

        ctx.out = out # Save output for backward pass

        return out

    @staticmethod
    def backward(ctx, grad_out:torch.Tensor):
        """
        Backward pass of the custom autograd function.

        Args:
            ctx (Context): A context object containing saved tensors from the forward pass.
            grad_out (torch.Tensor): Gradient of the loss with respect to the output of this function.

        Returns:
            tuple: Gradients of the loss with respect to the input parameters and the system (None in this case).
        """
        p_pt = ctx.saved_tensors[0]
        grad = ctx.system.Jacobian(p_pt)

        # grad_norm = grad/torch.max(torch.abs(grad), dim=1, keepdim=True).values
        out = torch.einsum('ij,ijl->il', grad_out, grad)

        return out, None