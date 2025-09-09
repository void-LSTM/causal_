"""
Neural network utilities for CSP framework.
Provides common NN components, initialization, and analysis tools.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import math
import warnings


class MLP(nn.Module):
    """
    Multi-layer perceptron with flexible configuration.
    Enhanced version with additional features for CSP framework.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: List[int] = [128, 128],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 final_activation: bool = False,
                 bias: bool = True,
                 init_method: str = 'xavier_uniform'):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            layer_norm: Whether to use layer normalization
            final_activation: Whether to apply activation to final layer
            bias: Whether to use bias in linear layers
            init_method: Weight initialization method
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_p = dropout
        self.use_batch_norm = batch_norm
        self.use_layer_norm = layer_norm
        self.final_activation = final_activation
        
        # Build layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Create layer dimensions
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # Build each layer
        for i in range(len(dims) - 1):
            # Linear layer
            layer = nn.Linear(dims[i], dims[i + 1], bias=bias)
            self.layers.append(layer)
            
            # Skip normalization and activation for final layer unless specified
            is_final = (i == len(dims) - 2)
            
            # Normalization
            if (batch_norm or layer_norm) and (not is_final or final_activation):
                if batch_norm:
                    norm = nn.BatchNorm1d(dims[i + 1])
                else:  # layer_norm
                    norm = nn.LayerNorm(dims[i + 1])
                self.norms.append(norm)
            else:
                self.norms.append(nn.Identity())
            
            # Activation
            if not is_final or final_activation:
                act = self._get_activation(activation)
                self.activations.append(act)
            else:
                self.activations.append(nn.Identity())
            
            # Dropout
            if dropout > 0 and (not is_final or final_activation):
                self.dropouts.append(nn.Dropout(dropout))
            else:
                self.dropouts.append(nn.Identity())
        
        # Initialize weights
        self.apply(lambda m: self._init_weights(m, init_method))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        for layer, norm, activation, dropout in zip(
            self.layers, self.norms, self.activations, self.dropouts
        ):
            x = layer(x)
            x = norm(x)
            x = activation(x)
            x = dropout(x)
        
        return x
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'identity': nn.Identity()
        }
        
        if name.lower() not in activations:
            warnings.warn(f"Unknown activation {name}, using ReLU")
            return activations['relu']
        
        return activations[name.lower()]
    
    def _init_weights(self, module: nn.Module, method: str):
        """Initialize weights using specified method."""
        if isinstance(module, nn.Linear):
            if method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif method == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            else:
                # Default to Xavier uniform
                nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about layer structure."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'num_layers': len(self.layers),
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'activation': self.activation_name,
            'dropout': self.dropout_p,
            'batch_norm': self.use_batch_norm,
            'layer_norm': self.use_layer_norm
        }


class ResidualBlock(nn.Module):
    """
    Residual block for building deeper networks.
    """
    
    def __init__(self,
                 dim: int,
                 hidden_dim: Optional[int] = None,
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 norm_type: str = 'layer'):
        """
        Initialize residual block.
        
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (default: same as dim)
            activation: Activation function
            dropout: Dropout probability
            norm_type: Normalization type ('layer', 'batch', 'none')
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = dim
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Build block
        layers = []
        
        # First layer
        layers.append(nn.Linear(dim, hidden_dim))
        
        # Normalization
        if norm_type == 'layer':
            layers.append(nn.LayerNorm(hidden_dim))
        elif norm_type == 'batch':
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'swish':
            layers.append(nn.SiLU())
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Second layer
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.block = nn.Sequential(*layers)
        
        # Skip connection projection if needed
        self.skip_proj = nn.Identity() if hidden_dim == dim else nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.block(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for sequence modeling.
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 bias: bool = True):
        """
        Initialize attention block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention.
        
        Args:
            x: Input tensor (batch_size, seq_len, dim)
            
        Returns:
            Output tensor (batch_size, seq_len, dim)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    """
    Count parameters in a model.
    
    Args:
        model: PyTorch model
        only_trainable: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive parameter statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter statistics
    """
    stats = {
        'total_params': 0,
        'trainable_params': 0,
        'param_by_layer': {},
        'param_by_type': {},
        'weight_stats': {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        },
        'grad_stats': {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'norm': 0.0
        }
    }
    
    all_weights = []
    all_grads = []
    
    for name, param in model.named_parameters():
        # Count parameters
        param_count = param.numel()
        stats['total_params'] += param_count
        
        if param.requires_grad:
            stats['trainable_params'] += param_count
        
        # Store by layer
        layer_name = name.split('.')[0]
        if layer_name not in stats['param_by_layer']:
            stats['param_by_layer'][layer_name] = 0
        stats['param_by_layer'][layer_name] += param_count
        
        # Store by parameter type
        param_type = 'weight' if 'weight' in name else 'bias'
        if param_type not in stats['param_by_type']:
            stats['param_by_type'][param_type] = 0
        stats['param_by_type'][param_type] += param_count
        
        # Collect weights for statistics
        all_weights.extend(param.data.view(-1).cpu().numpy())
        
        # Collect gradients if available
        if param.grad is not None:
            all_grads.extend(param.grad.data.view(-1).cpu().numpy())
    
    # Compute weight statistics
    if all_weights:
        all_weights = np.array(all_weights)
        stats['weight_stats'] = {
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights))
        }
    
    # Compute gradient statistics
    if all_grads:
        all_grads = np.array(all_grads)
        stats['grad_stats'] = {
            'mean': float(np.mean(all_grads)),
            'std': float(np.std(all_grads)),
            'min': float(np.min(all_grads)),
            'max': float(np.max(all_grads)),
            'norm': float(np.linalg.norm(all_grads))
        }
    
    return stats


def init_weights(model: nn.Module, method: str = 'xavier_uniform'):
    """
    Initialize model weights using specified method.
    
    Args:
        model: PyTorch model
        method: Initialization method
    """
    def _init_fn(module):
        if isinstance(module, nn.Linear):
            if method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif method == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            elif method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif method == 'uniform':
                nn.init.uniform_(module.weight, -0.1, 0.1)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            if method in ['kaiming_uniform', 'kaiming_normal']:
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            else:
                nn.init.xavier_uniform_(module.weight)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    model.apply(_init_fn)


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Freeze specified layers by setting requires_grad=False.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                break


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specified layers by setting requires_grad=True.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                break


def get_activation_stats(model: nn.Module, 
                        input_tensor: torch.Tensor) -> Dict[str, Dict[str, float]]:
    """
    Get activation statistics by running forward pass with hooks.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor for forward pass
        
    Returns:
        Dictionary with activation statistics per layer
    """
    activation_stats = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                output_np = output.detach().cpu().numpy()
                activation_stats[name] = {
                    'mean': float(np.mean(output_np)),
                    'std': float(np.std(output_np)),
                    'min': float(np.min(output_np)),
                    'max': float(np.max(output_np)),
                    'zero_fraction': float(np.mean(output_np == 0))
                }
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_stats


def compute_model_flops(model: nn.Module, 
                       input_shape: Tuple[int, ...]) -> int:
    """
    Estimate FLOPs for model (simplified calculation).
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape (without batch dimension)
        
    Returns:
        Estimated FLOPs count
    """
    total_flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Linear):
            # Linear layer: input_size * output_size * 2 (multiply-add)
            flops = module.in_features * module.out_features * 2
            if module.bias is not None:
                flops += module.out_features  # Bias addition
            total_flops += flops
        
        elif isinstance(module, nn.Conv2d):
            # Conv2d: kernel_size * in_channels * out_channels * output_h * output_w * 2
            if isinstance(output, torch.Tensor):
                output_h, output_w = output.shape[-2:]
                kernel_flops = module.kernel_size[0] * module.kernel_size[1]
                flops = kernel_flops * module.in_channels * module.out_channels * output_h * output_w * 2
                if module.bias is not None:
                    flops += module.out_channels * output_h * output_w
                total_flops += flops
    
    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hook = module.register_forward_hook(flop_count_hook)
            hooks.append(hook)
    
    # Forward pass
    dummy_input = torch.randn(1, *input_shape)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


# Test functions
def test_mlp():
    """Test MLP functionality."""
    print("Testing MLP...")
    
    # Create MLP
    mlp = MLP(
        input_dim=64,
        output_dim=10,
        hidden_dims=[128, 64, 32],
        activation='relu',
        dropout=0.1,
        batch_norm=True,
        final_activation=False
    )
    
    # Test forward pass
    x = torch.randn(16, 64)
    output = mlp(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {count_parameters(mlp)}")
    
    # Get layer info
    info = mlp.get_layer_info()
    print(f"Layer info keys: {list(info.keys())}")
    print(f"Total layers: {info['num_layers']}")
    
    return mlp


def test_residual_block():
    """Test ResidualBlock functionality."""
    print("\nTesting ResidualBlock...")
    
    # Create residual block
    block = ResidualBlock(
        dim=128,
        hidden_dim=256,
        activation='gelu',
        dropout=0.1,
        norm_type='layer'
    )
    
    # Test forward pass
    x = torch.randn(8, 128)
    output = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Residual connection works: {output.shape == x.shape}")
    
    return block


def test_attention_block():
    """Test AttentionBlock functionality."""
    print("\nTesting AttentionBlock...")
    
    # Create attention block
    attn = AttentionBlock(
        dim=256,
        num_heads=8,
        dropout=0.1
    )
    
    # Test forward pass
    x = torch.randn(4, 32, 256)  # (batch, seq_len, dim)
    output = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shape preserved: {output.shape == x.shape}")
    
    return attn


def test_parameter_analysis():
    """Test parameter analysis functions."""
    print("\nTesting parameter analysis...")
    
    # Create test model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Test parameter counting
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters trainable: {total_params == trainable_params}")
    
    # Test parameter statistics
    stats = get_parameter_stats(model)
    print(f"Parameter stats keys: {list(stats.keys())}")
    print(f"Layers found: {list(stats['param_by_layer'].keys())}")
    
    # Test weight initialization
    init_weights(model, method='xavier_uniform')
    print("Weight initialization completed")
    
    return model, stats


def test_activation_analysis():
    """Test activation analysis."""
    print("\nTesting activation analysis...")
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Sigmoid()
    )
    
    # Test activation statistics
    x = torch.randn(5, 10)
    activation_stats = get_activation_stats(model, x)
    
    print(f"Activation stats for {len(activation_stats)} layers")
    for layer_name, stats in activation_stats.items():
        if stats:  # Only show non-empty stats
            print(f"  {layer_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    return activation_stats


def test_flops_estimation():
    """Test FLOPs estimation."""
    print("\nTesting FLOPs estimation...")
    
    # Create test model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Estimate FLOPs
    flops = compute_model_flops(model, input_shape=(784,))
    
    print(f"Estimated FLOPs: {flops:,}")
    
    # Manual calculation for verification
    manual_flops = (784 * 256 + 256) * 2 + (256 * 128 + 128) * 2 + (128 * 10 + 10) * 2
    print(f"Manual calculation: {manual_flops:,}")
    print(f"Estimates match: {abs(flops - manual_flops) < 100}")
    
    return flops


if __name__ == "__main__":
    print("="*50)
    print("CSP Neural Network Utilities Test")
    print("="*50)
    
    # Run all tests
    mlp = test_mlp()
    residual_block = test_residual_block()
    attention_block = test_attention_block()
    model, param_stats = test_parameter_analysis()
    activation_stats = test_activation_analysis()
    flops = test_flops_estimation()
    
    print("\n" + "="*50)
    print("All neural network utilities tests completed!")
    print("="*50)