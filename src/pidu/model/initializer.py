import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple, List, Dict, Callable, Any
from enum import Enum
import math
import logging
from collections import defaultdict

from pidu.model.networks.framework import MultiModelFramework

class InitMode(Enum):
    KAIMING = "kaiming"
    XAVIER = "xavier"
    ORTHOGONAL = "orthogonal"
    LSUV = "lsuv"
    NORMAL = "normal"
    TRUNCATED_NORMAL = "truncated_normal"
    DELTA_ORTHOGONAL = "delta_orthogonal"
    SPARSE = "sparse"
    FIXUP = "fixup"
    SPECTRAL = "spectral"
    VARIANCE_SCALING = "variance_scaling"
    MIDAS = "midas" 

class NetworkInitializer:
    """
    A comprehensive network initializer implementing various state-of-the-art initialization methods.
    Supports different activation functions and layer types with specific initialization strategies.
    """
    def __init__(self, model: MultiModelFramework, verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.logger = self._setup_logger()
        self.statistics = defaultdict(dict)
        
        self.activation_gains = {
            'relu': 'relu',
            'lrelu': 'leaky_relu',
            'gelu': 'relu',         # Approximation for GELU
            'swish': 'relu',        # Approximation for Swish
            'tanh': 'tanh',
            'sigmoid': 'sigmoid',
            'selu': 'selu',
            'elu': 'elu',
            'linear': 'linear',
            # 'mish': 'relu',     # Approximation for Mish
        }
        
        # Custom gain values for activations not covered by PyTorch
        self.custom_gains = {
            'gelu': 1.7,
            'swish': 1.8,
            'mish': 1.75,
        }
        
        # Track initialization operations
        self.init_history = []

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for initialization information"""
        logger = logging.getLogger('NetworkInitializer')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def initialize(
        self, 
        mode: Union[str, InitMode],
        activation_bb: str = None,
        activation_dh: str = None,
        scale: float = 1.0,
        distribution: str = 'normal',
        layer_specific: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Initialize the network using the specified method.
        
        Args:
            mode: Initialization mode
            activation: Activation function used in the network
            scale: Scaling factor for the initialization
            distribution: 'normal' or 'uniform' distribution
            layer_specific: Whether to use layer-specific initialization strategies
            **kwargs: Additional arguments for specific initialization methods
        """
        if isinstance(mode, str):
            try:
                mode = InitMode(mode.lower())
            except ValueError:
                self.logger.error(f"Unsupported initialization mode: {mode}")
                raise ValueError(f"Unsupported initialization mode: {mode}")
        
        self.logger.info(f"Initializing network using {mode.value} initialization")
        
        # Record initial parameter statistics
        self._record_statistics("before")
        
        # Apply the selected initialization method
        activations = [activation_bb, activation_dh]
        if mode == InitMode.KAIMING:
            self._init_kaiming(activations[0], distribution, **kwargs)
        elif mode == InitMode.XAVIER:
            self._init_xavier(activations[0], distribution, **kwargs)
        elif mode == InitMode.ORTHOGONAL:
            self._init_orthogonal(scale, **kwargs)
        elif mode == InitMode.LSUV:
            self._init_lsuv(**kwargs)
        elif mode == InitMode.NORMAL:
            self._init_normal(scale, **kwargs)
        elif mode == InitMode.TRUNCATED_NORMAL:
            self._init_truncated_normal(scale, **kwargs)
        elif mode == InitMode.DELTA_ORTHOGONAL:
            self._init_delta_orthogonal(**kwargs)
        elif mode == InitMode.SPARSE:
            self._init_sparse(**kwargs)
        elif mode == InitMode.FIXUP:
            self._init_fixup(**kwargs)
        elif mode == InitMode.SPECTRAL:
            self._init_spectral(**kwargs)
        elif mode == InitMode.VARIANCE_SCALING:
            self._init_variance_scaling(**kwargs)
        elif mode == InitMode.MIDAS:
            self._init_midas(activations[0], **kwargs)
        else:
            raise ValueError(f"Unsupported initialization mode: {mode}")

        # Apply special initialization for specific layers if requested
        if layer_specific:
            self._apply_layer_specific_init(activations)
        
        # Initialize biases with best practices
        self._init_biases(**kwargs)
        
        # Record final parameter statistics
        self._record_statistics("after")
        
        # Log summary
        if self.verbose:
            self._log_initialization_summary()
            
        return self.model

    def _record_statistics(self, stage: str):
        """Record statistics about model parameters"""
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        self.statistics[stage] = stats

    def _log_initialization_summary(self):
        """Log a summary of the initialization effects"""
        before = self.statistics.get("before", {})
        after = self.statistics.get("after", {})
        
        if not before or not after:
            return
            
        self.logger.info("=== Initialization Summary ===")
        for name in after:
            if name in before:
                self.logger.info(f"Layer: {name}")
                self.logger.info(f"  Before: mean={before[name]['mean']:.4f}, std={before[name]['std']:.4f}")
                self.logger.info(f"  After:  mean={after[name]['mean']:.4f}, std={after[name]['std']:.4f}")

    def _init_kaiming(self, activation: str, distribution: str, mode: str = 'fan_out', **kwargs):
        """
        Kaiming (He) initialization with activation function consideration.
        
        Args:
            activation: The activation function used
            distribution: 'normal' or 'uniform'
            mode: 'fan_in' or 'fan_out'
        """
        nonlinearity = self.activation_gains.get(activation, 'relu')
        self.logger.info(f"Kaiming init with {nonlinearity} nonlinearity, {distribution} distribution, {mode} mode")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                if distribution == 'normal':
                    nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
                    self.init_history.append(f"kaiming_normal: {name}")
                else:
                    nn.init.kaiming_uniform_(m.weight, mode=mode, nonlinearity=nonlinearity)
                    self.init_history.append(f"kaiming_uniform: {name}")
            elif isinstance(m, nn.LSTM):
                for param_name, param in m.named_parameters():
                    if 'weight' in param_name:
                        nn.init.orthogonal_(param)
                        self.init_history.append(f"orthogonal (LSTM): {name}.{param_name}")

    def _init_xavier(self, activation: str, distribution: str, **kwargs):
        """
        Xavier (Glorot) initialization with gain calculation based on activation.
        
        Args:
            activation: The activation function used
            distribution: 'normal' or 'uniform'
        """
        gain = self.custom_gains.get(activation, nn.init.calculate_gain(
            self.activation_gains.get(activation, 'linear')))
            
        self.logger.info(f"Xavier init with gain={gain}, {distribution} distribution")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                if distribution == 'normal':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                    self.init_history.append(f"xavier_normal: {name}")
                else:
                    nn.init.xavier_uniform_(m.weight, gain=gain)
                    self.init_history.append(f"xavier_uniform: {name}")
            elif isinstance(m, (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                # Special handling for transformer layers
                self._init_transformer_layer(m, gain, distribution)

    def _init_transformer_layer(self, layer, gain: float, distribution: str):
        """Special initialization for transformer layers"""
        # Apply Xavier init to all weight parameters
        for name, param in layer.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Only apply to weight matrices
                if distribution == 'normal':
                    nn.init.xavier_normal_(param, gain=gain)
                else:
                    nn.init.xavier_uniform_(param, gain=gain)
                self.init_history.append(f"xavier_{distribution} (transformer): {name}")

    def _init_orthogonal(self, scale: float, **kwargs):
        """
        Orthogonal initialization with scaling factor.
        
        Args:
            scale: Scaling factor for the orthogonal initialization
        """
        self.logger.info(f"Orthogonal init with scale={scale}")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTMCell, nn.GRUCell)):
                nn.init.orthogonal_(m.weight, gain=scale)
                self.init_history.append(f"orthogonal: {name}")
            elif isinstance(m, (nn.LSTM, nn.GRU)):
                for param_name, param in m.named_parameters():
                    if 'weight' in param_name:
                        nn.init.orthogonal_(param, gain=scale)
                        self.init_history.append(f"orthogonal (RNN): {name}.{param_name}")
    
    def _init_lsuv(self, tol: float = 0.1, max_iter: int = 20, batch_size: int = 8, **kwargs):
        """
        Enhanced Layer-Sequential Unit-Variance (LSUV) initialization.
        Iteratively normalizes each layer to have unit variance.
        
        Args:
            tol: Tolerance for variance convergence
            max_iter: Maximum number of iterations for each layer
            batch_size: Batch size for sample input
        """
        # Start with orthogonal initialization as base
        self._init_orthogonal(1.0)
        
        def _forward_with_sample(sample):
            if isinstance(sample, dict):
                return self.model(**sample)
            if isinstance(sample, (tuple, list)):
                return self.model(*sample)
            return self.model(sample)

        sample_input = kwargs.get('sample_input', None)
        if sample_input is None:
            # Determine input shape
            input_shape = kwargs.get('input_shape')
            if input_shape is None:
                # Try to infer from model
                if hasattr(self.model, 'input_shape'):
                    input_shape = self.model.input_shape
                else:
                    raise ValueError("Input shape must be provided for LSUV initialization")

            # Generate sample input
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            sample_input = torch.randn(batch_size, *input_shape, dtype=dtype, device=device)
        
        self.logger.info(f"LSUV init with tol={tol}, max_iter={max_iter}")
        
        # Get all layers in forward order
        layers = []
        hooks = []
        layer_outputs: Dict[nn.Module, torch.Tensor] = {}
        
        def hook_fn(module, input, output):
            # Store output for variance calculation
            if isinstance(output, torch.Tensor):
                layer_outputs[module] = output
        
        # Register forward hooks
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
                layers.append((name, module))
        
        # Forward pass to register all outputs
        with torch.no_grad():
            _forward_with_sample(sample_input)
        
        # Process layers sequentially
        for name, layer in layers:
            self.logger.info(f"LSUV processing layer: {name}")
            var = float('inf')
            
            for i in range(max_iter):
                # Forward pass through the model
                with torch.no_grad():
                    _ = _forward_with_sample(sample_input)
                    output = layer_outputs.get(layer)
                    if output is None:
                        raise RuntimeError(f"LSUV hook did not capture output for layer: {name}")
                    
                    # Calculate variance of the output
                    var = torch.var(output.view(output.size(0), -1), dim=1).mean().item()
                    
                    if abs(var - 1.0) < tol:
                        self.logger.info(f"  Converged at iter {i}, variance: {var:.4f}")
                        break
                        
                    # Update weights
                    layer.weight.data /= math.sqrt(var)
                
                if i == max_iter - 1:
                    self.logger.warning(f"  Failed to converge, final variance: {var:.4f}")
            
            self.init_history.append(f"lsuv (var={var:.4f}): {name}")
        
        # Remove hooks
        for hook in hooks:
            hook.remove()

    def _init_normal(self, scale: float, **kwargs):
        """
        Normal initialization with specified scale.
        
        Args:
            scale: Standard deviation for normal distribution
        """
        self.logger.info(f"Normal init with scale={scale}")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=scale)
                self.init_history.append(f"normal: {name}")

    def _init_truncated_normal(self, scale: float, mean: float = 0.0, std: float = 1.0, 
                               a: float = -2.0, b: float = 2.0, **kwargs):
        """
        Truncated normal initialization, similar to TensorFlow's truncated normal.
        
        Args:
            scale: Overall scaling factor
            mean: Mean of the normal distribution
            std: Standard deviation of the normal distribution
            a: Lower bound in standard deviations
            b: Upper bound in standard deviations
        """
        self.logger.info(f"Truncated normal init with scale={scale}, mean={mean}, std={std}, bounds=[{a}, {b}]")
        
        def truncated_normal_(tensor, mean=0, std=1, a=-2, b=2):
            # Generate a truncated normal distribution
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < b) & (tmp > a)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                truncated_normal_(m.weight, mean=mean, std=std*scale, a=a, b=b)
                self.init_history.append(f"truncated_normal: {name}")

    def _init_delta_orthogonal(self, gain: float = 1.0, **kwargs):
        """
        Delta Orthogonal initialization for convolutional layers.
        Places orthogonal matrix in the center of the convolutional kernel.
        
        Args:
            gain: Scaling factor for the orthogonal matrix
        """
        self.logger.info(f"Delta Orthogonal init with gain={gain}")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # First zero out the whole tensor
                nn.init.zeros_(m.weight)
                
                # Get the middle indices for each dimension
                if isinstance(m, nn.Conv1d):
                    k_size = m.kernel_size[0]
                    mid = k_size // 2
                    orthogonal_tensor = torch.randn(m.out_channels, m.in_channels)
                    nn.init.orthogonal_(orthogonal_tensor, gain=gain)
                    m.weight.data[:, :, mid] = orthogonal_tensor
                
                elif isinstance(m, nn.Conv2d):
                    k_h, k_w = m.kernel_size
                    mid_h, mid_w = k_h // 2, k_w // 2
                    orthogonal_tensor = torch.randn(m.out_channels, m.in_channels)
                    nn.init.orthogonal_(orthogonal_tensor, gain=gain)
                    m.weight.data[:, :, mid_h, mid_w] = orthogonal_tensor
                
                elif isinstance(m, nn.Conv3d):
                    k_d, k_h, k_w = m.kernel_size
                    mid_d, mid_h, mid_w = k_d // 2, k_h // 2, k_w // 2
                    orthogonal_tensor = torch.randn(m.out_channels, m.in_channels)
                    nn.init.orthogonal_(orthogonal_tensor, gain=gain)
                    m.weight.data[:, :, mid_d, mid_h, mid_w] = orthogonal_tensor
                
                self.init_history.append(f"delta_orthogonal: {name}")
            elif isinstance(m, nn.Linear):
                # Use regular orthogonal for linear layers
                nn.init.orthogonal_(m.weight, gain=gain)
                self.init_history.append(f"orthogonal (from delta): {name}")

    def _init_sparse(self, sparsity: float = 0.1, std: float = 0.01, **kwargs):
        """
        Sparse initialization that sets most of the weights to zero.
        
        Args:
            sparsity: Fraction of weights to initialize as non-zero
            std: Standard deviation for the non-zero weights
        """
        self.logger.info(f"Sparse init with sparsity={sparsity}, std={std}")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                # First zero out everything
                nn.init.zeros_(m.weight)
                
                # Create a mask for non-zero elements
                mask = torch.zeros_like(m.weight)
                mask.bernoulli_(p=sparsity)
                
                # Initialize non-zero values
                m.weight.data = mask * torch.randn_like(m.weight) * std
                self.init_history.append(f"sparse: {name}")

    def _init_fixup(self, depth: int = None, **kwargs):
        """
        Fixup initialization for residual networks.
        
        Args:
            depth: Network depth (number of residual blocks)
        """
        if depth is None:
            # Try to estimate depth from the model
            depth = self._estimate_network_depth()
            if depth is None:
                self.logger.warning("Could not determine network depth for Fixup, defaulting to 12")
                depth = 12
        
        # L is the number of layers
        L = depth
        layer_scale = L ** (-0.5)
        
        self.logger.info(f"Fixup init with estimated depth={depth}, layer_scale={layer_scale:.4f}")
        
        # Count residual branches
        residual_branches = 0
        
        # For each module that might be a residual block
        for name, m in self.model.named_modules():
            # Check if this looks like a residual branch (has .conv layers and possibly a shortcut)
            conv_layers = [mod for mod_name, mod in m.named_modules() 
                           if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) 
                           and 'conv' in mod_name.lower()]
            
            if len(conv_layers) > 1:  # Likely a residual block
                residual_branches += 1
                
                # Get all Conv layers and apply special initialization
                scale_idx = 0
                for i, layer in enumerate(conv_layers):
                    if i == 0:  # First conv in residual branch
                        # Regular initialization
                        nn.init.normal_(layer.weight, mean=0, std=1.0)
                    elif i == len(conv_layers) - 1:  # Last conv
                        # Initialize to zero
                        nn.init.zeros_(layer.weight)
                        # If it has a bias, initialize with zeros
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                    else:  # Middle convs
                        # Scale based on depth
                        nn.init.normal_(layer.weight, mean=0, std=layer_scale)
                    
                    scale_idx += 1
                
                self.init_history.append(f"fixup (residual block): {name}")
                
        self.logger.info(f"Fixup detected {residual_branches} residual branches")

    def _estimate_network_depth(self):
        """Estimate the depth of the network by counting Conv/Linear layers"""
        depth = 0
        for m in self.model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                depth += 1
        return depth if depth > 0 else None

    def _init_spectral(self, n_power_iterations: int = 1, eps: float = 1e-12, **kwargs):
        """
        Spectral Normalization that normalizes weight matrices by their spectral norm.
        
        Args:
            n_power_iterations: Number of power iterations for estimating spectral norm
            eps: Small constant for numerical stability
        """
        self.logger.info(f"Spectral normalization with n_power_iterations={n_power_iterations}")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                # Use PyTorch's built-in spectral_norm
                m = nn.utils.spectral_norm(
                    m, 
                    name='weight',
                    n_power_iterations=n_power_iterations,
                    eps=eps,
                    dim=0
                )
                self.init_history.append(f"spectral_norm: {name}")
    
    def _init_variance_scaling(self, scale: float = 1.0, mode: str = 'fan_in', 
                              distribution: str = 'truncated_normal', **kwargs):
        """
        Variance Scaling initialization similar to TensorFlow's variance_scaling_initializer.
        
        Args:
            scale: Scaling factor
            mode: 'fan_in', 'fan_out', or 'fan_avg'
            distribution: 'truncated_normal', 'normal', or 'uniform'
        """
        self.logger.info(f"Variance scaling with scale={scale}, mode={mode}, distribution={distribution}")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                fan_in, fan_out = self.compute_fans(m)
                
                if mode == 'fan_in':
                    scale_factor = scale / max(1., fan_in)
                elif mode == 'fan_out':
                    scale_factor = scale / max(1., fan_out)
                else:  # fan_avg
                    scale_factor = scale / max(1., (fan_in + fan_out) / 2.)
                
                if distribution == 'truncated_normal':
                    # Truncated normal with 2 standard deviations
                    std = math.sqrt(scale_factor) / 0.87962566103423978
                    self._init_truncated_normal(std, mean=0.0, std=1.0, a=-2.0, b=2.0)
                elif distribution == 'normal':
                    std = math.sqrt(scale_factor)
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                else:  # uniform
                    limit = math.sqrt(3.0 * scale_factor)
                    nn.init.uniform_(m.weight, -limit, limit)
                
                self.init_history.append(f"variance_scaling: {name}")

    def _init_midas(self, activation: str, scale: float = 1.0, **kwargs):
        """
        Model-specific Initialization for Deep Architecture Stability.
        Analyzes layer connectivity and applies tailored initialization.
        
        Args:
            activation: Activation function
            scale: Global scaling factor
        """
        self.logger.info(f"MIDAS initialization with activation={activation}, scale={scale}")
        
        # 1. First apply a base initialization (Xavier)
        self._init_xavier(activation, 'normal', **kwargs)
        
        # 2. Analyze model architecture
        depth = self._estimate_network_depth()
        if depth is None:
            depth = 10  # Default estimate
            
        # Scale factor based on network depth
        depth_scale = 1.0 / math.sqrt(depth)
        
        # Connection count for each layer (estimate of fan-in and fan-out)
        connections = {}
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                fan_in, fan_out = self.compute_fans(m)
                connections[name] = {'fan_in': fan_in, 'fan_out': fan_out}
        
        # 3. Apply tailored initialization
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                fan_in, fan_out = connections[name]['fan_in'], connections[name]['fan_out']
                
                # Calculate initialization scale based on connectivity
                connectivity_scale = math.sqrt(2.0 / (fan_in + fan_out))
                
                # Final scale combines depth and connectivity information
                final_scale = scale * depth_scale * connectivity_scale
                
                # Apply scaled initialization
                nn.init.normal_(m.weight, mean=0.0, std=final_scale)
                self.init_history.append(f"midas: {name}")
                
                # Special handling for attention and transformers
                if 'attention' in name.lower() or 'transformer' in name.lower():
                    # Apply scaled attention initialization
                    m.weight.data *= 0.67  # Empirical scale for attention

    def _apply_layer_specific_init(self, activation: str):
        """
        Apply layer-specific initialization strategies.
        
        Args:
            activation: The activation function used
        """
        self.logger.info("Applying layer-specific initialization strategies")
        
        for name, m in self.model.named_modules():
            # Special handling for different layer types
            if isinstance(m, (nn.LSTM, nn.GRU)):
                # Special handling for recurrent layers
                for param_name, param in m.named_parameters():
                    if 'weight_ih' in param_name:
                        # Input weights
                        nn.init.xavier_uniform_(param)
                        self.init_history.append(f"xavier_uniform (RNN-ih): {name}.{param_name}")
                    elif 'weight_hh' in param_name:
                        # Hidden weights - use orthogonal
                        nn.init.orthogonal_(param)
                        self.init_history.append(f"orthogonal (RNN-hh): {name}.{param_name}")
            
            elif isinstance(m, nn.MultiheadAttention):
                # Initialize attention heads with special scales
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight, gain=1/math.sqrt(2))
                    self.init_history.append(f"xavier_uniform (attention): {name}.in_proj_weight")
                
                if hasattr(m, 'out_proj') and hasattr(m.out_proj, 'weight'):
                    nn.init.xavier_uniform_(m.out_proj.weight, gain=1/math.sqrt(2))
                    self.init_history.append(f"xavier_uniform (attention): {name}.out_proj.weight")
            
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm layers
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                self.init_history.append(f"batchnorm: {name}")
            
            elif isinstance(m, nn.LayerNorm):
                # Initialize LayerNorm layers
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                self.init_history.append(f"layernorm: {name}")
            
            elif isinstance(m, nn.Embedding):
                # Initialize embedding layers
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                self.init_history.append(f"embedding_normal: {name}")

    def _init_biases(self, bias_scale: float = 0.01, **kwargs):
        """
        Initialize biases using best practices for different layer types.
        
        Args:
            bias_scale: Scale for non-zero bias initializations
        """
        self.logger.info(f"Initializing biases with bias_scale={bias_scale}")
        
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def compute_fans(layer: nn.Module) -> Tuple[int, int]:
        """
        Compute the fan-in and fan-out of a layer.
        """
        if isinstance(layer, nn.Linear):
            fan_in = layer.in_features
            fan_out = layer.out_features
        elif isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            receptive_field_size = np.prod(layer.kernel_size)
            fan_in = layer.in_channels * receptive_field_size
            fan_out = layer.out_channels * receptive_field_size
        else:
            fan_in = fan_out = 0
        return fan_in, fan_out