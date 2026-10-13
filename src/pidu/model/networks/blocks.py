"""
Generic, reusable building blocks shared by the detection heads
(heads.py) and by Backbone itself: an activation-function factory, a 1D
residual conv block, two learnable-slope tanh activations, and the CNN
backbone that turns a raw signal into a feature map.
"""
import torch
import torch.nn as nn
from numpy import concatenate as np_concatenate, array as np_array, mean as np_mean

from pidu.data.utils import pt2np


def act_str_to_fn(act_str: str, **kwargs) -> nn.Module:
  if act_str == 'relu':
    return nn.ReLU()
  elif act_str == 'leakyrelu':
    return nn.LeakyReLU()
  elif act_str == 'tanh':
    return nn.Tanh()
  elif act_str == 'tanhshrink':
    return nn.Tanhshrink()
  elif act_str == 'sigmoid':
    return nn.Sigmoid()
  elif act_str == 'swish':
    return nn.SiLU()
  elif act_str == 'mish':
    return nn.Mish()
  elif act_str == 'stanh':
    return ScalableTanh(**kwargs)
  else:
    raise ValueError(f'Activation function {act_str} not supported.')


class ScalableTanh(nn.Module):
  """tanh(x) scaled by a single learnable slope: (1 + beta*x) * tanh(x)."""

  def __init__(self, dtype: torch.dtype = torch.float32, device: torch.device = 'cpu'):
    super().__init__()
    self.beta = nn.Parameter(torch.tensor(0.5, dtype=dtype, device=device))
    self.tanh = nn.Tanh()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (1 + self.beta * x) * self.tanh(x)

  def get_beta(self) -> nn.Parameter:
    return self.beta


class ScalableTanhOutput(nn.Module):
  """ScalableTanh with one learnable slope per output feature."""

  def __init__(self, out_size: int, dtype: torch.dtype = torch.float32, device: torch.device = 'cpu'):
    super().__init__()
    self.beta = nn.Parameter(torch.ones(out_size, dtype=dtype, device=device))
    self.tanh = nn.Tanh()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (1 + self.beta * x) * self.tanh(x)

  def get_beta(self) -> nn.Parameter:
    return self.beta


class ResidualBlock1D(nn.Module):
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    act_fun_str: str = 'relu',
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
    dropout_rate: float = 0.1,
    norm: str = 'none',  # none|instancenorm
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu',
  ):
    super().__init__()
    padding = (kernel_size - 1) // 2 * dilation
    self.conv1 = nn.Conv1d(
      in_channels, out_channels, kernel_size, stride, padding,
      dilation=dilation, padding_mode='replicate', dtype=dtype, device=device,
    )
    self.conv2 = nn.Conv1d(
      out_channels, out_channels, kernel_size, 1, padding,
      dilation=dilation, padding_mode='replicate', dtype=dtype, device=device,
    )
    self.act_fun = act_str_to_fn(act_fun_str)
    self.dropout = nn.Dropout(dropout_rate)

    self.downsample = nn.Identity()
    if stride != 1 or in_channels != out_channels:
      if norm == 'instancenorm':
        self.downsample = nn.Sequential(
          nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, dtype=dtype, device=device),
          nn.InstanceNorm1d(out_channels, dtype=dtype, device=device),
        )
      else:
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, dtype=dtype, device=device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = self.downsample(x)     # Apply downsample to input for skip connection

    out = self.act_fun(self.conv1(x))  # First convolution + activation
    out = self.dropout(out)
    out = self.conv2(out)              # Second convolution (no activation yet)

    out += identity                    # Add skip connection
    out = self.act_fun(out)            # Final activation

    return out

  def get_weights(self, mode='list', type='data'):
    if mode == 'list':
      if type == 'data':
        return np_concatenate([
          pt2np(self.conv1.weight.data).flatten(),
          pt2np(self.conv2.weight.data).flatten(),
        ])
      elif type == 'grad':
        return np_concatenate([
          pt2np(self.conv1.weight.grad).flatten(),
          pt2np(self.conv2.weight.grad).flatten(),
        ])
    elif mode == 'norm':
      if type == 'data':
        return np_array([pt2np(self.conv1.weight.data.norm()), pt2np(self.conv2.weight.data.norm())])
      elif type == 'grad':
        return np_mean([pt2np(self.conv1.weight.grad.norm()), pt2np(self.conv2.weight.grad.norm())])


class Backbone(nn.Module):
  """CNN feature extractor shared by CNNModel/SSDModel: either a stack of plain
  conv+activation layers, or (when use_residual=True) a stack of ResidualBlock1D."""

  def __init__(
    self,
    in_channels: int = 2,
    out_channels: int = 4,
    act_fun_str: str = 'relu',
    dropout_rate: float = 0.1,
    norm: str = 'none',  # none|instancenorm
    dtype: torch.dtype = torch.float32,
    device: str = 'cpu',
    use_residual: bool = False,
    *args,
    **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.dt = dtype
    self.dv = device
    self.use_residual = use_residual

    if use_residual:
      # First layer - simple conv to get to first feature size
      self.conv1 = nn.Sequential(
        nn.Conv1d(in_channels, 4, 3, 1, 1, padding_mode='replicate', dtype=self.dt, device=self.dv),
        act_str_to_fn(act_fun_str),
      )

      # Residual blocks with increasing channels
      self.res_block1 = ResidualBlock1D(4, 8, act_fun_str, 3, 1, 1, dropout_rate, norm, dtype, device)
      self.res_block2 = ResidualBlock1D(8, 16, act_fun_str, 5, 1, 3, dropout_rate, norm, dtype, device)
      self.res_block3 = ResidualBlock1D(16, 16, act_fun_str, 5, 1, 5, dropout_rate, norm, dtype, device)
      self.res_block4 = ResidualBlock1D(16, out_channels, act_fun_str, 5, 1, 7, dropout_rate, norm, dtype, device)

    else:
      # Plain sequential layers without residual connections
      act_layer = act_str_to_fn(act_fun_str)
      strides = [1, 1, 1, 1, 1]
      kernel_sizes = [3, 3, 5, 5, 5]

      self.network = nn.ModuleDict({
        'backbone_conv1': nn.Sequential(
          nn.Conv1d(in_channels, 4, kernel_sizes[0], strides[0], (kernel_sizes[0] - strides[0]) // 2, padding_mode='replicate', dtype=self.dt, device=self.dv),
          act_layer,
        ),
        'backbone_conv2': nn.Sequential(
          nn.Conv1d(4, 8, kernel_sizes[1], strides[1], (kernel_sizes[1] - strides[1]) // 2, padding_mode='replicate', dtype=self.dt, device=self.dv),
          act_layer,
        ),
        'backbone_conv3': nn.Sequential(
          nn.Conv1d(8, 16, kernel_sizes[2], strides[2], (kernel_sizes[2] - strides[2]) // 2, padding_mode='replicate', dtype=self.dt, device=self.dv),
          act_layer,
        ),
        'backbone_conv4': nn.Sequential(
          nn.Conv1d(16, 16, kernel_sizes[3], strides[3], (kernel_sizes[3] - strides[3]) // 2, padding_mode='replicate', dtype=self.dt, device=self.dv),
          act_layer,
        ),
        'backbone_conv5': nn.Sequential(
          nn.Conv1d(
            16, out_channels, kernel_sizes[4], strides[4],
            padding=(kernel_sizes[4] - strides[4]) // 2,
            padding_mode='replicate',
            dtype=self.dt,
            device=self.dv,
          ),
          act_layer,
        ),
      })

    self.div_factor = 0  # No stride reduction with residual blocks
    self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.use_residual:
      x = self.conv1(x)
      x = self.res_block1(x)
      x = self.res_block2(x)
      x = self.res_block3(x)
      x = self.res_block4(x)
      return x
    else:
      x1 = self.network['backbone_conv1'](x)
      x2 = self.network['backbone_conv2'](x1)
      x3 = self.network['backbone_conv3'](x2)
      x4 = self.network['backbone_conv4'](x3)
      x5 = self.network['backbone_conv5'](x4)
      return x5

  def get_weights(self, mode='list', type='data'):
    weight_layers = []

    if self.use_residual:
      for block in [self.res_block1, self.res_block2, self.res_block3, self.res_block4]:
        weight_layers.extend(block.get_weights(mode, type))
      for sub_layer in self.conv1:
        if isinstance(sub_layer, nn.Conv1d):
          if type == 'data':
            weight_layers.append(pt2np(sub_layer.weight.data))
          elif type == 'grad':
            grad = sub_layer.weight.grad
            if grad is None:
              grad = torch.zeros_like(sub_layer.weight.data)
            weight_layers.append(pt2np(grad))
    else:
      for name, layer in self.network.items():
        if isinstance(layer, nn.Sequential):
          for sub_layer in layer:
            if isinstance(sub_layer, nn.Conv1d):
              if type == 'data':
                weight_layers.append(pt2np(sub_layer.weight.data))
              elif type == 'grad':
                grad = sub_layer.weight.grad
                if grad is None:
                  grad = torch.zeros_like(sub_layer.weight.data)
                weight_layers.append(pt2np(grad))

    if mode in ('list', 'norm'):
      return np_concatenate([x.flatten() for x in weight_layers])
