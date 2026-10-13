"""
Primitive activation modules that map one raw network output to one
physically-meaningful quantity (thickness, eps_inf, resonance frequency,
...). These are the building blocks assembled by ParameterActivations
(activations.py) into the full system/oscillator parameterization.
"""
from typing import List

import torch
import torch.nn as nn


class Activation(nn.Module):
  """ReLU-clamped affine map, used to constrain a raw output to [x_min, x_max]-ish range."""

  def __init__(self, x_min: torch.Tensor = -1, x_max: torch.Tensor = 1, factor=8):
    super().__init__()
    self.offset = 2
    self.mean = ((x_min + x_max) / 2) - self.offset
    self.slope = (x_max - x_min) / factor
    self.relu = nn.ReLU()

  def __repr__(self):
    return f"Activation(mean={self.mean},slope={self.slope})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.relu(self.mean + self.slope * x + self.offset)

  def get_slope(self):
    return self.slope

  def get_mean(self):
    return self.mean


class ActivationOScore(nn.Module):
  """Objectness-score activation: identity scaled by a slope."""

  def __init__(self, slope: float = 1.0):
    super().__init__()
    self.slope = slope

  def __repr__(self):
    return f"ActivationOScore(slope={self.slope})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.slope

  def get_slope(self):
    return self.slope


class ActivationW0(nn.Module):
  """
  Maps a raw oscillator output to a resonance frequency w0, confined to one
  of several fixed-width regions tiling the wavenumber range (one region per
  output channel), via offset + slope * tanh(x).
  """

  def __init__(
    self,
    wn: torch.Tensor,
    regions_width: List[int],
    idx_scale: int = 0,
    dt: torch.dtype = torch.float32,
    dv: torch.device = torch.device('cpu'),
  ):
    super().__init__()
    self.tanh = nn.Tanh()

    wn_range = wn.max() - wn.min()
    n_regions = [int(torch.ceil(wn_range / rw)) for rw in regions_width]

    offsets = torch.linspace(
      wn.min(),
      wn.max() - regions_width[idx_scale],
      n_regions[idx_scale],
      dtype=dt, device=dv,
    ) + 0.5 * regions_width[idx_scale]
    slopes = torch.tensor([regions_width[idx_scale] / 2], dtype=dt, device=dv)

    self.offsets = offsets[None, None, :]  # [samples, n_layers] | [n_regions]
    self.slopes = slopes[None, None, :]    # [samples, n_layers] | [n_regions]

  def __repr__(self):
    return f"ActivationW0(slope={float(self.slopes)})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.offsets + self.slopes * self.tanh(x)

  def get_range(self):
    x = torch.linspace(-5, 5, 1000).repeat_interleave(self.offsets.shape[-1]).view(-1, 1, self.offsets.shape[-1])
    y = self.forward(x)

    return x, y


class ActivationEinf(nn.Module):
  """Maps a raw output to eps_inf (>= 0) via softplus."""

  def __init__(self, slope: float = 5):
    super().__init__()
    self.slope = slope
    self.softplus = nn.Softplus(beta=1)

  def __repr__(self):
    return f"ActivationEinf(slope={self.slope})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.softplus(x) * self.slope

  def get_factor(self):
    return self.slope


class ActivationD(nn.Module):
  """Maps a raw output to a layer thickness (>= 0) via softplus."""

  def __init__(self, slope: float = 1e-4):
    super().__init__()
    self.slope = slope
    self.exp = nn.Softplus(beta=1)

  def __repr__(self):
    return f"ActivationD(slope={self.slope})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.slope * self.exp(x)

  def get_slope(self):
    return self.slope


class ActivationdG(nn.Module):
  """Maps a raw output to a Drude plasma frequency (>= 0) via ReLU."""

  def __init__(self, slope: float = 50):
    super().__init__()
    self.slope = slope
    self.relu = nn.ReLU()

  def __repr__(self):
    return f"ActivationdG(slope={self.slope})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.relu(x * self.slope)

  def get_slope(self):
    return self.slope

  def get_inverse(self, y: torch.Tensor) -> torch.Tensor:
    return y / self.slope


class ActivationdWp(nn.Module):
  """Maps a raw output to a Drude damping factor (>= 0) via ReLU."""

  def __init__(self, slope: float = 75):
    super().__init__()
    self.slope = slope
    self.relu = nn.ReLU()

  def __repr__(self):
    return f"ActivationdWp(slope={self.slope})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.relu(x * self.slope)

  def get_slope(self):
    return self.slope

  def get_inverse(self, y: torch.Tensor) -> torch.Tensor:
    return y / self.slope


class ActivationW0B(nn.Module):
  """Maps a raw output to the boundary-oscillator w0 "before" the stack (>= 0)."""

  def __init__(self, w0_max: torch.Tensor, dt: torch.dtype = torch.float32, dv: torch.device = torch.device('cpu'), *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.relu = nn.ReLU()
    self.w0_max = w0_max.to(dtype=dt, device=dv)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.relu(x) * 25

  def __repr__(self):
    return f"ActivationW0B(w0_max={self.w0_max})"


class ActivationW0A(nn.Module):
  """Maps a raw output to the boundary-oscillator w0 "after" the stack (>= 0)."""

  def __init__(self, w0_min: torch.Tensor, dt: torch.dtype = torch.float32, dv: torch.device = torch.device('cpu'), *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.relu = nn.ReLU()
    self.softplus = nn.Softplus(beta=1)
    self.w0_min = w0_min.to(dtype=dt, device=dv)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w0_min / 4 * self.relu(x)

  def __repr__(self):
    return f"ActivationW0A(w0_min={self.w0_min})"
