from typing import List, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from abc import abstractmethod
import cvxpy as cp

from pidu.data.utils import pt2np

# -- Optimisers --
class BaseOptimiser():
  def __init__(self, optimiser:torch.optim.Optimizer, dt, device, reduction='mean', optimisation_mode:str='complete'):
    self._optim, self._reduction = optimiser, reduction
    self.dt = dt
    self.dv = device
    self.optimisation_mode = optimisation_mode

  @property
  def optimiser(self):
    return self._optim

  def zero_grad(self, **kwargs):
    return self._optim.zero_grad(**kwargs)

  def step(self):
    return self._optim.step()
  
  def backward(self, objectives: torch.Tensor, model: nn.Module = None, scaler=None, **kwargs):
    if self._reduction == 'mean':
      total = objectives.mean()
    elif self._reduction == 'sum':
      total = objectives.sum()
    if scaler is not None:
      return scaler.scale(total).backward()
    return total.backward()
    
    # return total.item()

class PCGrad(BaseOptimiser):
  """Gradient surgery for multi-task learning (Yu et al., NeurIPS 2020)."""
  def __init__(self, optimiser:torch.optim.Optimizer, dt, device, reduction='mean', optimisation_mode:str='complete'):
    super().__init__(optimiser, dt, device, reduction, optimisation_mode)
    
    # --
    params = []
    for g in self._optim.param_groups:
      params.extend(g['params'])
    self._params = params
    self._shapes = [p.shape for p in params]
    self._sizes = [p.numel() for p in params]
    self._offsets = torch.as_tensor([0] + list(torch.cumsum(torch.tensor(self._sizes), dim=0)[:-1]))
    self._total_numel = sum(self._sizes)

  def backward(self, objectives: Dict[str, torch.Tensor]|torch.Tensor, model: nn.Module = None, scaler=None, **kwargs):
    # Collect per-task grads without touching .grad
    if isinstance(objectives, dict):
      task_names = list(objectives.keys())
      losses = [objectives[k] for k in task_names]
    else:
      losses = objectives
    
    grads_flat = []
    has_mask = []
    
    loss_scale = float(scaler.get_scale()) if scaler is not None else 1.0

    for i_loss in range(len(losses)):
      objective = losses[i_loss] * loss_scale if scaler is not None else losses[i_loss]
      g = torch.autograd.grad(
        objective,
        self._params,
        retain_graph=True,
        allow_unused=True
      )
      buf = torch.zeros(self._total_numel, dtype=self.dt, device=self.dv)
      mask = torch.zeros_like(buf, dtype=self.dt, device=self.dv)
      for tensor_grad, offset, size in zip(g, self._offsets, self._sizes):
        if tensor_grad is not None:
          flat = tensor_grad.reshape(-1)
          buf[offset:offset + size] = flat
          mask[offset:offset + size] = 1
      
      grads_flat.append(buf)
      has_mask.append(mask)
    
    grads_flat = torch.stack(grads_flat) 
    has_mask = torch.stack(has_mask) 
    shared = has_mask.bool().all(dim=0) 

    proj_grads = self._project_conflicting(grads_flat)

    # Reduce across tasks for shared / non-shared dims
    if self._reduction == 'mean':
      merged = proj_grads[:, shared].mean(0)
    elif self._reduction == 'sum':
      merged = proj_grads[:, shared].sum(0)
    elif self._reduction == 'debug':
      merged = proj_grads  # special mode
    else:
      raise ValueError(f"Unknown reduction {self._reduction}")

    if self._reduction != 'debug':
      # For non-shared dims just sum
      merged_full = torch.zeros(self._total_numel, dtype=self.dt, device=self.dv)
      merged_full[shared] = merged
      merged_full[~shared] = proj_grads[:, ~shared].sum(0)
      if scaler is not None:
        merged_full = merged_full / loss_scale
      self._assign_back(merged_full)
    else:
      # debug: skip assigning
      pass
  
    # return sum([l.item() for l in losses])

  def _project_conflicting(self, grads_flat: torch.Tensor) -> torch.Tensor:
    # grads_flat: (T, N)
    T = grads_flat.size(0)
    out = grads_flat.clone()
    for i in range(T):
      order = torch.randperm(T, device=grads_flat.device)
      for j in order:
        if i == j:
          continue
        gi = out[i]
        gj = grads_flat[j]
        dot = torch.dot(gi, gj)
        if dot < 0:
          denom = gj.norm().pow(2).clamp_min(1e-12)
          out[i] = gi - dot * gj / denom
    return out

  def _assign_back(self, flat: torch.Tensor):
    # Write flattened grad back into .grad
    idx = 0
    for p, size in zip(self._params, self._sizes):
        p.grad = flat[idx:idx + size].view(p.shape).to(p.dtype)
        idx += size

# WeightMethod and NashMTL are adapted from the Nash-MTL reference implementation
# (Navon et al., ICML 2022; https://github.com/AvivNavon/nash-mtl, MIT License).
class WeightMethod:
    def __init__(self, optimiser:torch.optim.Optimizer, n_tasks: int, device: torch.device):
        super().__init__()
        self._optim = optimiser
        self.n_tasks = n_tasks
        self.device = device

    @property
    def optimiser(self):
      return self._optim
    
    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ],
        last_shared_parameters: Union[List[torch.nn.parameter.Parameter], torch.Tensor],
        representation: Union[torch.nn.parameter.Parameter, torch.Tensor],
        **kwargs,
    ):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
            representation=representation,
            **kwargs,
        )
        loss.backward()
        return loss, extra_outputs

    def __call__(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []

class NashMTL(WeightMethod):
    def __init__(
        self,
        optimiser:torch.optim.Optimizer,
        n_tasks: int,
        device: torch.device,
        max_norm: float = 1.0,
        update_weights_every: int = 1,
        optim_niter=20,
    ):
        super(NashMTL, self).__init__(
            optimiser=optimiser,
            n_tasks=n_tasks,
            device=device,
        )

        self.optim_niter = optim_niter
        self.update_weights_every = update_weights_every
        self.max_norm = max_norm

        self.prvs_alpha_param = None
        self.normalization_factor = np.ones((1,))
        self.init_gtg = self.init_gtg = np.eye(self.n_tasks)
        self.i_step = 0.0
        self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)

    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )

    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha

    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.n_tasks,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.n_tasks,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.n_tasks, self.n_tasks), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.n_tasks):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)

    def get_weighted_loss(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        """

        Parameters
        ----------
        losses :
        shared_parameters : shared parameters
        kwargs :

        Returns
        -------

        """
        shared_parameters = list(shared_parameters)
        
        extra_outputs = dict()
        
        # Update n_tasks if the number of losses changed
        actual_n_tasks = len(losses)
        if actual_n_tasks != self.n_tasks:
            self.n_tasks = actual_n_tasks
            self.prvs_alpha = np.ones(self.n_tasks, dtype=np.float32)
            self.init_gtg = np.eye(self.n_tasks)
            self.i_step = 0  # Reset step counter to reinitialize
        
        if self.i_step == 0:
            self._init_optim_problem()

        if (self.i_step % self.update_weights_every) == 0:
            self.i_step += 1

            grads = {}
            for i, loss in enumerate(losses):
                g = list(
                    torch.autograd.grad(
                        loss,
                        shared_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                )
                
                if isinstance(shared_parameters, (list, tuple)):
                    filled = []
                    for gg, p in zip(g, shared_parameters):
                        if gg is None:
                            filled.append(torch.zeros_like(p, dtype=p.dtype, device=p.device).view(-1))
                        else:
                               filled.append(gg.view(-1))
                    grad = torch.cat(filled)
                else:
                    grad = g[0].view(-1) if g[0] is not None else torch.zeros_like(shared_parameters, dtype=shared_parameters.dtype, device=shared_parameters.device).view(-1)
                grads[i] = grad.detach()

            G = torch.stack(tuple(v for v in grads.values()))
            GTG = torch.mm(G, G.t())

            self.normalization_factor = (
                torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            )
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
            alpha = torch.from_numpy(alpha)

        else:
            self.i_step += 1
            alpha = self.prvs_alpha

        weighted_loss = sum([losses[i] * alpha[i] for i in range(len(alpha))])
        extra_outputs["weights"] = alpha
        return weighted_loss, extra_outputs

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        task_specific_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        last_shared_parameters: Union[
            List[torch.nn.parameter.Parameter], torch.Tensor
        ] = None,
        representation: Union[List[torch.nn.parameter.Parameter], torch.Tensor] = None,
        scaler=None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        loss, extra_outputs = self.get_weighted_loss(
            losses=losses,
            shared_parameters=shared_parameters,
            **kwargs,
        )
        if scaler is not None:
          scaler.scale(loss).backward()
        else:
          loss.backward()

        # make sure the solution for shared params has norm <= self.eps
        if self.max_norm > 0 and scaler is None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)

        return loss, extra_outputs

    def zero_grad(self, **kwargs):
        return self._optim.zero_grad(**kwargs)
    
    def step(self):
        return self._optim.step()

# -- Loss functions --
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        # alpha weights positive class (target=1), (1-alpha) weights negative class (target=0)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class LossFunction(nn.Module):
  def __init__(
    self, 
    active_systems:str, 
    dtype=torch.float32, 
    device=torch.device('cpu'), 
    **kwargs
  ):
    super().__init__()
    self.active_systems = active_systems  # 
    self.dtype = dtype                    # pytorch datatype
    self.device = device                  # pytorch device

    self.bce_logits = nn.BCEWithLogitsLoss()
    self.focal_loss = FocalLoss(gamma=2, alpha=0.9, reduction='sum', task_type='binary')
    
    self.system_loss_scaling = torch.tensor([0,0,1e1,0,0], dtype=dtype, device=device)  # Scale for the system loss
    self.osc_loss_scaling = torch.tensor([1e0,1e0,1e0], dtype=dtype, device=device)  # Scale for the oscillator loss
    self.scaling_sys = torch.concat([self.system_loss_scaling, self.osc_loss_scaling, self.osc_loss_scaling])
    # self.scaling = torch.tensor([[[1e3], [1e0]]], dtype=dtype, device=device)  # Scale for the losses: [system, osc]
    self.scaling = torch.ones((1,1,1), dtype=dtype, device=device)  # Scale for the losses: [system, osc]
    self.n_regions_per_scale_per_system = kwargs.get('n_regions_per_scale_per_system', {})
    
    self.return_residuals = kwargs.get('residuals', False)      # 
    self.return_loss_matrix = kwargs.get('loss_matrix', False)  #
    
    # Hybrid loss parameters
    self.loss_alpha = kwargs.get('loss_alpha', 0.1)  # 0.0 = pure log error, 1.0 = pure MSE 

  def __call__(self, *args, **kwds):
    return self.forward(*args, **kwds)

  def _init_loss_tensor(self, sys_pred:torch.Tensor, n_scales_max:Dict[str, int]) -> Tuple[torch.Tensor, int]:    
    n_scales_max = max(n_scales_max.values())
    n_unfolds = list(sys_pred.values())[0].shape[0]  # Number of unfolds
    
    # loss tensor: [unfolds, scales, [w0, wp, g, system/obj]]
    loss_system = {key: torch.zeros((value.shape[0], n_scales_max, 4), dtype=self.dtype, device=self.device) for key, value in sys_pred.items()}  # [unfolds, scales, [system, osc, has, no]]

    return loss_system, n_unfolds

  def init_epochs(self, epochs:int, n_unfolds:int):
    fns = []
    u = []
    
    x_axis = torch.arange(epochs, dtype=self.dtype, device=self.device)
    
    for i in range(n_unfolds-1):
      fns.append(lambda a, i=i: 1/n_unfolds * (1/2)**(a/(15*(i+1))))  # Decay functions for each unfold; determines the weight of each unfolding
      u.append(fns[i](x_axis))
    
    fns_final = lambda a: torch.ones_like(a, dtype=self.dtype, device=self.device) - sum([fn(a) for fn in fns])  # Function for the last unfold; ensures the weights sum to 1
    u_final = fns_final(x_axis)
    
    self.unfold_scaling = torch.stack(u + [u_final], axis=1) 
    
    self.tracking = False
    self.tracking_information = None
    self.weight_history = None
    
    if self.tracking:
      number_of_parameters_to_track = 36 # TODO: Write function to track this
      self.thl = tracking_history_length = 10 # 
      
      self.tracking_information = np.zeros((tracking_history_length, number_of_parameters_to_track), dtype=np.float32)  # [time, parameters]
      self.weight_history = np.zeros((epochs - tracking_history_length, number_of_parameters_to_track), dtype=np.float32)  # [time, parameters]
      
  def _loss_fn(self, x, y, sum_):
    # x = pred, y = true
    # Hybrid loss: alpha * MSE + (1 - alpha) * log_error
    mse_loss = ((x - y)**2).mean(sum_)  # MSE loss
    log_loss = 1e3 * ((torch.log(1 + x) - torch.log(1 + y))**2).mean(sum_)  # Log loss
    
    hybrid_loss = self.loss_alpha * mse_loss + (1 - self.loss_alpha) * log_loss
    
    return hybrid_loss
    
    # Alternative loss functions (commented out):
    # return torch.sqrt(((torch.log(1 + x) - torch.log(1 + y))**2).sum(sum_)) # Root mean squared log error
    # return torch.log(torch.cosh((x - y) + 1e-12)).sum(sum_) # Logcosh
  
  def _calculate_tracking_weight(self, tracking_matrix:np.ndarray):
    # tracking_matrix: [time, parameters]
    percentage_decrease = np.abs(tracking_matrix[0] - tracking_matrix.mean(0))  # [time-1, parameters]
    weights = percentage_decrease / (np.sum(percentage_decrease) + 1e-12)   # Normalize to sum to 1
    return weights
  
  def loss_system(
    self, 
    p_sys_pred:torch.Tensor, 
    p_sys_true:torch.Tensor, 
    scale:torch.Tensor|None=None
  ):
    res = (p_sys_pred - p_sys_true).abs() * self.scaling
    # l2 = (1/2)*res**2 # MSE loss
    # logloss = (torch.log1p(p_sys_pred) - torch.log1p(p_sys_true))**2 # Log loss
    
    # logloss = torch.arcsinh((1 + p_sys_pred)/(1 + p_sys_true + 1e-8))**2
    # loss = logloss.mean((0,1))
    loss = self._loss_fn(p_sys_pred, p_sys_true, (0,1)) #
    
    if scale is not None:
      loss = loss * scale

    if self.return_residuals:
      return loss, pt2np(res)

    return loss, None

  def loss_params_osc(
    self, 
    p_osc_pred:torch.Tensor, 
    p_osc_true:torch.Tensor, 
    scale=None
  ):
    if p_osc_pred.shape[0] == 0:
      if self.return_residuals:
        return torch.tensor(0.0, dtype=self.dtype, device=self.device), np.array([])
      return torch.tensor(0.0, dtype=self.dtype, device=self.device), None
    
    res = (p_osc_pred - p_osc_true)
    
    loss = self._loss_fn(p_osc_pred, p_osc_true, (0)) #

    if scale is not None:
      loss = loss * scale
    
    if self.return_residuals:
      return loss, pt2np(res)

    return loss, None

  def loss_has_osc(
    self,
    p_osc_pred:torch.Tensor,
    p_osc_true:torch.Tensor,
    scale=None
  ):
    
    pred_obj_inv = torch.log((p_osc_pred + 1e-8) / (1 - p_osc_pred + 1e-8))
    # bce = self.bce_logits(pred_obj_inv, p_osc_true)
    bce = self.focal_loss(pred_obj_inv, p_osc_true)
    
    if scale is not None:
      return bce * scale

    return bce

  def reduce(self, loss:Dict[str, torch.Tensor], reduce:str='complete'):
    # loss shape: [system: [unfolds, scales, [w0, wp, g, system/obj]]]
    # complete: reduce all
    # type: reduce different loss types
    if reduce == 'complete':
      sum_ = torch.stack([v.sum() for v in loss.values()]).sum()
      return sum_
    elif reduce == 'unfold':
      reduced_loss = {}
      for key, value in loss.items():
        reduced_loss[key] = torch.sum(value, axis=(1,2))
      sum_ = torch.stack(list(reduced_loss.values())).sum(0) # n_unfolds
      return sum_
    elif reduce == 'scale':
      # Different systems may have different number of scales
      reduced_loss = {}
      n_scales_max = 0
      for key, value in loss.items():
        n_scales_max = max(n_scales_max, len(value.sum((0,2))))
      reduced_loss = {key: torch.zeros((n_scales_max,), dtype=self.dtype, device=self.device) for key in loss.keys()}
      for key, value in loss.items():
        reduced_loss[key][:value.shape[1]] = value.sum((0,2)) # [scales]

      sum_ = torch.stack(list(reduced_loss.values())).sum(0) # Sum over systems
      return sum_
    elif reduce == 'system':
      reduced_loss = torch.stack([x.sum() for x in loss.values()])
      return reduced_loss
    elif reduce == 'none':
      flattened = torch.stack(list(loss.values())).flatten() # 
      return flattened[flattened != 0] # Remove zeros; check if possible with e.g. Nash
    elif reduce == 'optics-type':
      reduced_loss = {}
      for key, value in loss.items():
        reduced_loss[key] = value.sum((0,1)) # [unfolds, scales]
      sum_ = torch.stack(list(reduced_loss.values())).sum(0) # Sum over systems
      return sum_
    elif reduce == 'scale-optics-type':
      reduced_loss = {}
      n_scales_max = 0
      for key, value in loss.items():
        n_scales_max = max(n_scales_max, value.shape[1])
      reduced_loss = {key: torch.zeros((n_scales_max,value.shape[2]), dtype=self.dtype, device=self.device) for key in loss.keys()}
      for key, value in loss.items():
        reduced_loss[key][:value.shape[1]] = value.sum(0) # [scales]

      sum_ = torch.stack(list(reduced_loss.values())).sum(0).flatten() # Sum over systems
      return sum_
    else:
      raise ValueError(f"Unknown average mode: {reduce}.")

  def forward_unfold(
    self, 
    systemname:str,
    shape_unfold, 
    sys_pred, 
    sys_true, 
    osc_pred, 
    osc_true
  ):
    losses = torch.zeros(shape_unfold, device=self.device, dtype=self.dtype) # [scales, [system, osc, has, no]]
    # 1. system loss
    losses_sys, res_system = self.loss_system(
      sys_pred,
      sys_true,
      self.scaling_sys
    )
    losses[0, 3] = losses_sys[:5].sum()
    losses[0, 0:3] = losses_sys[5:].reshape(2,-1).sum(0)
    
    if osc_pred is None:
      return losses, {'sys': res_system}

    res_oscillators_scales = []
    for i_scale in range(losses.shape[0]-1):
      cutoff = self.n_regions_per_scale_per_system[systemname][i_scale]
      scale_pred = osc_pred[:, :, i_scale, :cutoff, :] # 
      scale_true = osc_true[:, :, i_scale, :cutoff, :] # 
      has_obj = scale_true[..., 0] == 1   # Get the indices of the targets which have an object
      no_obj = scale_true[..., 0] == 0    # Get the indices of the targets which do not have an object
      
      losses[i_scale+1, 0:3], res_oscillators = self.loss_params_osc(
        scale_pred[..., -3:][has_obj],
        scale_true[..., -3:][has_obj],
        # self.osc_loss_scaling
      )
      losses[i_scale+1, 3] = self.loss_has_osc(
        scale_pred[..., 0], # [has_obj]
        scale_true[..., 0], # [has_obj]
        1e2
      )
      # losses[i_scale+1, 5] = self.loss_has_osc(
      #   scale_pred[..., 0][no_obj],
      #   scale_true[..., 0][no_obj],
      #   1e1
      # )

      res_oscillators_scales.append(res_oscillators)

    return losses, {'sys': res_system, 'osc': res_oscillators_scales}

  def forward(
    self,
    epoch:int,
    p_sys_pred:Dict[str, torch.Tensor],
    p_osc_pred:Dict[str, torch.Tensor],
    p_sys_true:Dict[str, torch.Tensor],
    p_osc_true:Dict[str, torch.Tensor],
    reduce:str='complete',
  ):
    assert p_sys_pred.keys() == p_sys_true.keys(), "'system names' for system parameters do not match"
    # assert p_osc_pred.keys() == p_osc_true.keys(), "'system names' for oscillator parameters do not match" # TODO: Fix logic too handle the case of NO oscs

    self.i_unfold = -1  # Current unfold index
    system_losses = {}  # Dictionary to store system losses
    system_residuals = {}
    
    if not hasattr(self, 'unfold_scaling'):
      self.unfold_scaling = torch.zeros((epoch+1, 3), dtype=self.dtype, device=self.device)
      self.unfold_scaling[:, -1] = 1.0
    
    # -- Define maximum number of scales --
    n_scales_max = {k: 1 for k in p_sys_pred.keys()} # Get the maximum number of scales across all systems
    for system_key, system_value in p_osc_pred.items():
      if isinstance(system_value, list): continue
      n_scales_max[system_key] = 1 + system_value.shape[-3]
    
    loss_system, n_unfolds = self._init_loss_tensor(p_sys_pred, n_scales_max) # 

    for key in p_sys_pred.keys():
      sys_pred = p_sys_pred[key] if key in p_sys_true else None # 
      osc_pred = p_osc_pred[key] if key in p_osc_true else None # 
      sys_true = p_sys_true[key] if key in p_sys_true else None # 
      osc_true = p_osc_true[key] if key in p_osc_true else None # 
      
      res_system = []

      for i_unfold in range(n_unfolds):
        self.i_unfold = i_unfold
        loss, res = self.forward_unfold(
          key,
          loss_system[key].shape[1:], 
          sys_pred[i_unfold], 
          sys_true, 
          osc_pred[i_unfold] if osc_pred is not None else None, 
          osc_true,
        )
        
        loss_system[key][i_unfold] = loss
        res_system.append(res)

      system_losses[key] = loss_system[key] * self.unfold_scaling[epoch][:,None,None] # 
      system_residuals[key] = res_system  # 

    reduced_loss = self.reduce(system_losses, reduce=reduce) # Reduce the losses based on the specified mode 
    complete_loss = {k: pt2np(v) for k,v in system_losses.items()} if self.return_loss_matrix else None # 
    system_residuals = system_residuals if self.return_residuals else None # 
    
    if self.tracking:
      self.tracking_information = np.roll(self.tracking_information, 1, axis=0)
      self.tracking_information[0] = reduced_loss.detach().cpu().numpy().flatten()
      
      # Calculate weights
      if epoch > self.tracking_information.shape[0]:
        weights = self._calculate_tracking_weight(self.tracking_information)
        # print(f"Calculated tracking weights:\n{weights}")
        reduced_loss = reduced_loss * torch.from_numpy(weights).to(self.device, self.dtype)
        self.weight_history[epoch - self.thl] = weights
      
    return reduced_loss, complete_loss, system_residuals