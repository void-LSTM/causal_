"""
Gradient Reversal Layer implementation for adversarial training.
Used in style decoupling to reverse gradients during backpropagation.
Based on "Unsupervised Domain Adaptation by Backpropagation" (Ganin & Lempitsky, 2015).
"""

import torch
import torch.nn as nn
from typing import Optional, Union


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Function for adversarial training.
    
    Forward pass: identity function (output = input)
    Backward pass: multiply gradients by -alpha
    """
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Forward pass - identity function.
        
        Args:
            input_tensor: Input tensor
            alpha: Gradient reversal strength
            
        Returns:
            Same as input tensor
        """
        ctx.alpha = alpha
        return input_tensor.view_as(input_tensor)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass - reverse gradients.
        
        Args:
            grad_output: Gradients from upstream
            
        Returns:
            Tuple of (reversed_gradients, None)
        """
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer with configurable reversal strength.
    
    Can be used with fixed alpha or with adaptive scheduling.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize GRL.
        
        Args:
            alpha: Initial gradient reversal strength
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRL.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor (same as input in forward, gradients reversed in backward)
        """
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha: float):
        """Set gradient reversal strength."""
        self.alpha = alpha


class AdaptiveGRL(nn.Module):
    """
    Adaptive Gradient Reversal Layer with automatic alpha scheduling.
    
    Alpha increases during training following different scheduling strategies.
    """
    
    def __init__(self, 
                 max_alpha: float = 1.0,
                 schedule: str = 'linear',
                 warmup_steps: int = 1000):
        """
        Initialize adaptive GRL.
        
        Args:
            max_alpha: Maximum alpha value
            schedule: Scheduling strategy ('linear', 'exponential', 'cosine')
            warmup_steps: Number of steps to reach max_alpha
        """
        super().__init__()
        self.max_alpha = max_alpha
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.current_alpha = 0.0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive alpha."""
        return GradientReversalFunction.apply(x, self.current_alpha)
    
    def update_alpha(self, step: int):
        """
        Update alpha based on current training step.
        
        Args:
            step: Current training step
        """
        self.current_step = step
        
        if step >= self.warmup_steps:
            self.current_alpha = self.max_alpha
        else:
            progress = step / self.warmup_steps
            
            if self.schedule == 'linear':
                self.current_alpha = progress * self.max_alpha
            elif self.schedule == 'exponential':
                self.current_alpha = self.max_alpha * (progress ** 2)
            elif self.schedule == 'cosine':
                self.current_alpha = self.max_alpha * 0.5 * (1 - torch.cos(torch.tensor(progress * 3.14159)).item())
            else:
                raise ValueError(f"Unknown schedule: {self.schedule}")
    
    def get_alpha(self) -> float:
        """Get current alpha value."""
        return self.current_alpha


class DomainAdversarialNetwork(nn.Module):
    """
    Complete domain adversarial network with GRL and classifier.
    
    Combines feature extractor -> GRL -> domain classifier.
    """
    
    def __init__(self,
                 input_dim: int,
                 num_domains: int,
                 hidden_dims: list = [256, 256],
                 grl_alpha: float = 1.0,
                 adaptive_grl: bool = False,
                 grl_schedule: str = 'linear',
                 grl_warmup_steps: int = 1000):
        """
        Initialize domain adversarial network.
        
        Args:
            input_dim: Input feature dimension
            num_domains: Number of domains to classify
            hidden_dims: Hidden layer dimensions for classifier
            grl_alpha: GRL alpha (if not adaptive)
            adaptive_grl: Whether to use adaptive GRL
            grl_schedule: GRL schedule type (if adaptive)
            grl_warmup_steps: Warmup steps for adaptive GRL
        """
        super().__init__()
        
        # Gradient reversal layer
        if adaptive_grl:
            self.grl = AdaptiveGRL(
                max_alpha=grl_alpha,
                schedule=grl_schedule,
                warmup_steps=grl_warmup_steps
            )
        else:
            self.grl = GradientReversalLayer(alpha=grl_alpha)
        
        # Domain classifier
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_domains))
        self.classifier = nn.Sequential(*layers)
        
        self.adaptive_grl = adaptive_grl
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRL and classifier.
        
        Args:
            features: Input features
            
        Returns:
            Domain classification logits
        """
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)
    
    def update_grl(self, step: int):
        """Update GRL alpha if using adaptive scheduling."""
        if self.adaptive_grl and hasattr(self.grl, 'update_alpha'):
            self.grl.update_alpha(step)
    
    def get_grl_alpha(self) -> float:
        """Get current GRL alpha value."""
        if hasattr(self.grl, 'get_alpha'):
            return self.grl.get_alpha()
        else:
            return self.grl.alpha


# Convenience function for creating GRL
def create_grl(alpha: float = 1.0, 
               adaptive: bool = False,
               max_alpha: float = 1.0,
               schedule: str = 'linear',
               warmup_steps: int = 1000) -> Union[GradientReversalLayer, AdaptiveGRL]:
    """
    Factory function for creating GRL layers.
    
    Args:
        alpha: Fixed alpha value (if not adaptive)
        adaptive: Whether to use adaptive scheduling
        max_alpha: Maximum alpha for adaptive GRL
        schedule: Schedule type for adaptive GRL
        warmup_steps: Warmup steps for adaptive GRL
        
    Returns:
        GRL layer instance
    """
    if adaptive:
        return AdaptiveGRL(
            max_alpha=max_alpha,
            schedule=schedule,
            warmup_steps=warmup_steps
        )
    else:
        return GradientReversalLayer(alpha=alpha)


# Test implementation
if __name__ == "__main__":
    print("=== Testing Gradient Reversal Layer ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 16
    feature_dim = 64
    num_domains = 3
    
    print(f"Test parameters:")
    print(f"  batch_size: {batch_size}")
    print(f"  feature_dim: {feature_dim}")
    print(f"  num_domains: {num_domains}")
    
    print("\n--- Testing GradientReversalFunction ---")
    try:
        # Create test input that requires gradients
        x = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
        alpha = 0.5
        
        # Forward pass
        y = GradientReversalFunction.apply(x, alpha)
        
        # Check forward pass (should be identity)
        assert torch.allclose(x, y), "Forward pass should be identity"
        print(f"Forward pass: input.shape={x.shape}, output.shape={y.shape}")
        print(f"Forward identity check: {'✓' if torch.allclose(x, y) else '✗'}")
        
        # Backward pass test
        loss = y.sum()
        loss.backward()
        
        # Check if gradients exist and have correct shape
        assert x.grad is not None, "Gradients should be computed"
        assert x.grad.shape == x.shape, "Gradient shape should match input shape"
        print(f"Backward pass: grad.shape={x.grad.shape}")
        print(f"Gradient reversal: alpha={alpha}, mean_grad={x.grad.mean().item():.4f}")
        print("✓ GradientReversalFunction test passed")
        
    except Exception as e:
        print(f"GradientReversalFunction test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing GradientReversalLayer ---")
    try:
        grl = GradientReversalLayer(alpha=1.0).to(device)
        x = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
        
        # Forward pass
        y = grl(x)
        
        # Test alpha modification
        grl.set_alpha(0.8)
        y2 = grl(x)
        
        print(f"GRL forward: input.shape={x.shape}, output.shape={y.shape}")
        print(f"Identity check: {'✓' if torch.allclose(x, y) else '✗'}")
        print(f"Alpha change test: new_alpha={grl.alpha}")
        print("✓ GradientReversalLayer test passed")
        
    except Exception as e:
        print(f"GradientReversalLayer test failed: {e}")
    
    print("\n--- Testing AdaptiveGRL ---")
    try:
        adaptive_grl = AdaptiveGRL(
            max_alpha=1.0,
            schedule='linear',
            warmup_steps=100
        ).to(device)
        
        x = torch.randn(batch_size, feature_dim, device=device)
        
        # Test alpha progression
        alphas = []
        steps = [0, 25, 50, 75, 100, 150]
        
        for step in steps:
            adaptive_grl.update_alpha(step)
            alpha = adaptive_grl.get_alpha()
            alphas.append(alpha)
            y = adaptive_grl(x)
            
            print(f"Step {step:3d}: alpha={alpha:.3f}, output.shape={y.shape}")
        
        # Check alpha progression
        assert alphas[0] == 0.0, "Alpha should start at 0"
        assert alphas[-2] == 1.0, "Alpha should reach max at warmup_steps"
        assert alphas[-1] == 1.0, "Alpha should stay at max after warmup"
        assert all(alphas[i] <= alphas[i+1] for i in range(len(alphas)-1)), "Alpha should be non-decreasing"
        
        print("✓ AdaptiveGRL test passed")
        
    except Exception as e:
        print(f"AdaptiveGRL test failed: {e}")
    
    print("\n--- Testing DomainAdversarialNetwork ---")
    try:
        # Test with fixed GRL
        dan_fixed = DomainAdversarialNetwork(
            input_dim=feature_dim,
            num_domains=num_domains,
            hidden_dims=[128, 64],
            grl_alpha=0.5,
            adaptive_grl=False
        ).to(device)
        
        # Test with adaptive GRL
        dan_adaptive = DomainAdversarialNetwork(
            input_dim=feature_dim,
            num_domains=num_domains,
            hidden_dims=[128, 64],
            grl_alpha=1.0,
            adaptive_grl=True,
            grl_schedule='exponential',
            grl_warmup_steps=50
        ).to(device)
        
        x = torch.randn(batch_size, feature_dim, device=device)
        
        # Test fixed GRL
        logits_fixed = dan_fixed(x)
        alpha_fixed = dan_fixed.get_grl_alpha()
        
        print(f"Fixed DAN: input.shape={x.shape}, output.shape={logits_fixed.shape}")
        print(f"Fixed GRL alpha: {alpha_fixed}")
        assert logits_fixed.shape == (batch_size, num_domains), f"Expected shape ({batch_size}, {num_domains}), got {logits_fixed.shape}"
        
        # Test adaptive GRL
        for step in [0, 25, 50]:
            dan_adaptive.update_grl(step)
            logits_adaptive = dan_adaptive(x)
            alpha_adaptive = dan_adaptive.get_grl_alpha()
            
            print(f"Adaptive DAN step {step}: alpha={alpha_adaptive:.3f}, output.shape={logits_adaptive.shape}")
            assert logits_adaptive.shape == (batch_size, num_domains), f"Expected shape ({batch_size}, {num_domains}), got {logits_adaptive.shape}"
        
        print("✓ DomainAdversarialNetwork test passed")
        
    except Exception as e:
        print(f"DomainAdversarialNetwork test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing create_grl factory function ---")
    try:
        # Test fixed GRL creation
        grl_fixed = create_grl(alpha=0.7, adaptive=False)
        assert isinstance(grl_fixed, GradientReversalLayer), "Should create GradientReversalLayer"
        assert grl_fixed.alpha == 0.7, "Alpha should be set correctly"
        
        # Test adaptive GRL creation
        grl_adaptive = create_grl(
            adaptive=True,
            max_alpha=2.0,
            schedule='cosine',
            warmup_steps=200
        )
        assert isinstance(grl_adaptive, AdaptiveGRL), "Should create AdaptiveGRL"
        assert grl_adaptive.max_alpha == 2.0, "Max alpha should be set correctly"
        assert grl_adaptive.schedule == 'cosine', "Schedule should be set correctly"
        
        print(f"Fixed GRL: type={type(grl_fixed).__name__}, alpha={grl_fixed.alpha}")
        print(f"Adaptive GRL: type={type(grl_adaptive).__name__}, max_alpha={grl_adaptive.max_alpha}")
        print("✓ create_grl factory test passed")
        
    except Exception as e:
        print(f"create_grl factory test failed: {e}")
    
    print("\n--- Testing gradient flow ---")
    try:
        # Create a simple network with GRL
        class TestNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = nn.Linear(feature_dim, 32)
                self.grl = GradientReversalLayer(alpha=1.0)
                self.classifier = nn.Linear(32, 2)
                
            def forward(self, x):
                features = self.feature_extractor(x)
                reversed_features = self.grl(features)
                return self.classifier(reversed_features)
        
        net = TestNetwork().to(device)
        x = torch.randn(batch_size, feature_dim, device=device)
        target = torch.randint(0, 2, (batch_size,), device=device)
        
        # Forward pass
        output = net(x)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        feature_grad = net.feature_extractor.weight.grad
        classifier_grad = net.classifier.weight.grad
        
        assert feature_grad is not None, "Feature extractor should have gradients"
        assert classifier_grad is not None, "Classifier should have gradients"
        
        print(f"Gradient flow test:")
        print(f"  Feature extractor grad norm: {feature_grad.norm().item():.4f}")
        print(f"  Classifier grad norm: {classifier_grad.norm().item():.4f}")
        print(f"  Output shape: {output.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print("✓ Gradient flow test passed")
        
    except Exception as e:
        print(f"Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== GRL Test Complete ===")