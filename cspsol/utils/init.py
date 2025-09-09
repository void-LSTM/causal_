"""
Utility functions and classes for CSP framework.
Provides common functionality across modules.
"""

from .grl import (
    create_grl,
    StaticGRL,
    AdaptiveGRL,
    GradientReversalFunction
)

__all__ = [
    'create_grl',
    'StaticGRL', 
    'AdaptiveGRL',
    'GradientReversalFunction'
]

# Utility functions that will be implemented
def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(device=None):
    """
    Get appropriate device for computation.
    
    Args:
        device: Preferred device ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        torch.device instance
    """
    import torch
    
    if device is None or device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device)

def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds):
    """
    Format time duration for display.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def get_memory_usage():
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory information
    """
    import torch
    import psutil
    
    info = {
        'cpu_percent': psutil.virtual_memory().percent,
        'cpu_available_gb': psutil.virtual_memory().available / (1024**3)
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'gpu_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        })
    
    return info