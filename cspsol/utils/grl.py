"""
Gradient Reversal Layer (GRL) implementation for domain adaptation and adversarial training.
Supports both static and adaptive alpha scheduling for gradient reversal strength.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Union, Callable
import math


class GradientReversalFunction(Function):
    """
    Gradient Reversal Function that reverses gradients during backpropagation.
    Forward pass: y = x
    Backward pass: grad_x = -alpha * grad_y
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass - identity function.
        
        Args:
            ctx: Context for saving information for backward pass
            input: Input tensor
            alpha: Gradient reversal strength
            
        Returns:
            Output tensor (same as input)
        """
        ctx.alpha = alpha
        return input.view_as(input)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass - reverse gradients.
        
        Args:
            ctx: Context with saved information
            grad_output: Gradients from subsequent layers
            
        Returns:
            Tuple of (reversed_gradients, None)
        """
        return -ctx.alpha * grad_output, None


class StaticGRL(nn.Module):
    """
    Static Gradient Reversal Layer with fixed alpha.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize static GRL.
        
        Args:
            alpha: Fixed gradient reversal strength
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal with fixed alpha."""
        return GradientReversalFunction.apply(x, self.alpha)
    
    def get_alpha(self) -> float:
        """Get current alpha value."""
        return self.alpha
    
    def set_alpha(self, alpha: float):
        """Set new alpha value."""
        self.alpha = alpha


class AdaptiveGRL(nn.Module):
    """
    Adaptive Gradient Reversal Layer with scheduling.
    Supports linear, exponential, and custom scheduling of alpha.
    """
    
    def __init__(self,
                 max_alpha: float = 1.0,
                 schedule: str = 'linear',
                 warmup_steps: int = 1000,
                 total_steps: Optional[int] = None,
                 gamma: float = 10.0):
        """
        Initialize adaptive GRL.
        
        Args:
            max_alpha: Maximum alpha value
            schedule: Scheduling type ('linear', 'exponential', 'sigmoid')
            warmup_steps: Number of steps to reach max_alpha
            total_steps: Total training steps (for some schedules)
            gamma: Parameter for exponential/sigmoid schedules
        """
        super().__init__()
        self.max_alpha = max_alpha
        self.schedule = schedule.lower()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gamma = gamma
        
        # Current state
        self.current_step = 0
        self.alpha = 0.0
        
        # Validate schedule
        valid_schedules = ['linear', 'exponential', 'sigmoid', 'cosine']
        if self.schedule not in valid_schedules:
            raise ValueError(f"Unknown schedule: {schedule}. Valid: {valid_schedules}")
    
    def _compute_alpha(self, step: int) -> float:
        """
        Compute alpha value based on current step and schedule.
        
        Args:
            step: Current training step
            
        Returns:
            Alpha value for gradient reversal
        """
        if step <= 0:
            return 0.0
        
        if self.schedule == 'linear':
            # Linear increase to max_alpha
            progress = min(step / self.warmup_steps, 1.0)
            return self.max_alpha * progress
        
        elif self.schedule == 'exponential':
            # Exponential growth: alpha = max_alpha * (1 - exp(-gamma * progress))
            progress = min(step / self.warmup_steps, 1.0)
            return self.max_alpha * (1 - math.exp(-self.gamma * progress))
        
        elif self.schedule == 'sigmoid':
            # Sigmoid schedule: alpha = max_alpha * sigmoid(gamma * (progress - 0.5))
            progress = min(step / self.warmup_steps, 1.0)
            sigmoid_input = self.gamma * (progress - 0.5)
            sigmoid_val = 1 / (1 + math.exp(-sigmoid_input))
            return self.max_alpha * sigmoid_val
        
        elif self.schedule == 'cosine':
            # Cosine annealing: smooth increase then decrease
            if self.total_steps is None:
                # If no total steps, just increase to max
                progress = min(step / self.warmup_steps, 1.0)
                return self.max_alpha * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            else:
                # Full cosine schedule
                progress = min(step / self.total_steps, 1.0)
                return self.max_alpha * 0.5 * (1 + math.cos(math.pi * progress))
        
        else:
            return self.max_alpha
    
    def update_alpha(self, step: int):
        """
        Update alpha based on current training step.
        
        Args:
            step: Current training step
        """
        self.current_step = step
        self.alpha = self._compute_alpha(step)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal with current alpha."""
        return GradientReversalFunction.apply(x, self.alpha)
    
    def get_alpha(self) -> float:
        """Get current alpha value."""
        return self.alpha
    
    def get_statistics(self) -> dict:
        """Get GRL statistics."""
        return {
            'current_step': self.current_step,
            'alpha': self.alpha,
            'max_alpha': self.max_alpha,
            'schedule': self.schedule,
            'warmup_steps': self.warmup_steps
        }


def create_grl(alpha: Optional[float] = None,
               adaptive: bool = False,
               max_alpha: float = 1.0,
               schedule: str = 'linear',
               warmup_steps: int = 1000,
               **kwargs) -> Union[StaticGRL, AdaptiveGRL]:
    """
    Factory function to create gradient reversal layers.
    
    Args:
        alpha: Fixed alpha for static GRL (ignored if adaptive=True)
        adaptive: Whether to use adaptive scheduling
        max_alpha: Maximum alpha value for adaptive GRL
        schedule: Scheduling type for adaptive GRL
        warmup_steps: Warmup steps for adaptive GRL
        **kwargs: Additional arguments for adaptive GRL
        
    Returns:
        GRL layer (static or adaptive)
    """
    if adaptive:
        return AdaptiveGRL(
            max_alpha=max_alpha,
            schedule=schedule,
            warmup_steps=warmup_steps,
            **kwargs
        )
    else:
        if alpha is None:
            alpha = max_alpha
        return StaticGRL(alpha=alpha)


# Test implementation
if __name__ == "__main__":
    print("=== Testing Gradient Reversal Layer ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test data
    batch_size = 32
    feature_dim = 64
    x = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Input requires_grad: {x.requires_grad}")
    
    print("\n--- Testing StaticGRL ---")
    try:
        # Test static GRL
        static_grl = StaticGRL(alpha=0.5).to(device)
        
        # Forward pass
        y = static_grl(x)
        print(f"Output shape: {y.shape}")
        print(f"Output requires_grad: {y.requires_grad}")
        
        # Backward pass
        loss = y.sum()
        loss.backward()
        
        print(f"Input gradients sum: {x.grad.sum().item():.4f}")
        print(f"Expected gradient sum: {(-0.5 * batch_size * feature_dim):.4f}")
        
        # Check if gradients are reversed
        expected_grad_sum = -0.5 * batch_size * feature_dim
        actual_grad_sum = x.grad.sum().item()
        
        if abs(actual_grad_sum - expected_grad_sum) < 1e-3:
            print("✓ StaticGRL gradient reversal working correctly")
        else:
            print(f"✗ StaticGRL gradient reversal failed: {actual_grad_sum} != {expected_grad_sum}")
        
    except Exception as e:
        print(f"✗ StaticGRL test failed: {e}")
    
    print("\n--- Testing AdaptiveGRL ---")
    try:
        # Test adaptive GRL
        adaptive_grl = AdaptiveGRL(
            max_alpha=1.0,
            schedule='linear',
            warmup_steps=100
        ).to(device)
        
        # Test alpha progression
        test_steps = [0, 25, 50, 75, 100, 150]
        print("Alpha progression:")
        
        for step in test_steps:
            adaptive_grl.update_alpha(step)
            alpha = adaptive_grl.get_alpha()
            print(f"  Step {step:3d}: alpha = {alpha:.4f}")
        
        # Test forward/backward with adaptive alpha
        x_new = torch.randn(batch_size, feature_dim, device=device, requires_grad=True)
        adaptive_grl.update_alpha(50)  # Set to middle of warmup
        
        y_adaptive = adaptive_grl(x_new)
        loss_adaptive = y_adaptive.sum()
        loss_adaptive.backward()
        
        expected_alpha = 0.5  # At step 50 of 100 warmup steps
        actual_alpha = adaptive_grl.get_alpha()
        print(f"Alpha at step 50: {actual_alpha:.4f} (expected ~0.5)")
        
        print("✓ AdaptiveGRL test passed")
        
    except Exception as e:
        print(f"✗ AdaptiveGRL test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing different schedules ---")
    try:
        schedules = ['linear', 'exponential', 'sigmoid', 'cosine']
        
        for schedule in schedules:
            grl = AdaptiveGRL(
                max_alpha=1.0,
                schedule=schedule,
                warmup_steps=100,
                gamma=5.0
            )
            
            # Test a few key points
            alphas = []
            for step in [0, 25, 50, 75, 100]:
                grl.update_alpha(step)
                alphas.append(grl.get_alpha())
            
            print(f"{schedule:12s}: {[f'{a:.3f}' for a in alphas]}")
        
        print("✓ Schedule tests passed")
        
    except Exception as e:
        print(f"✗ Schedule tests failed: {e}")
    
    print("\n--- Testing create_grl factory ---")
    try:
        # Test static creation
        static_grl = create_grl(alpha=0.8, adaptive=False)
        assert isinstance(static_grl, StaticGRL)
        assert static_grl.get_alpha() == 0.8
        
        # Test adaptive creation
        adaptive_grl = create_grl(
            adaptive=True,
            max_alpha=2.0,
            schedule='exponential',
            warmup_steps=200
        )
        assert isinstance(adaptive_grl, AdaptiveGRL)
        assert adaptive_grl.max_alpha == 2.0
        assert adaptive_grl.schedule == 'exponential'
        
        print("✓ Factory function tests passed")
        
    except Exception as e:
        print(f"✗ Factory function tests failed: {e}")
    
    print("\n--- Testing integration with model training ---")
    try:
        # Simulate training scenario
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(feature_dim, 32)
                self.grl = create_grl(adaptive=True, max_alpha=1.0, warmup_steps=10)
                self.discriminator = nn.Linear(32, 2)
            
            def forward(self, x, step=None):
                if step is not None:
                    self.grl.update_alpha(step)
                
                features = self.encoder(x)
                reversed_features = self.grl(features)
                output = self.discriminator(reversed_features)
                return output, features
        
        model = SimpleModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Simulate a few training steps
        for step in range(15):
            x_batch = torch.randn(8, feature_dim, device=device)
            labels = torch.randint(0, 2, (8,), device=device)
            
            optimizer.zero_grad()
            output, features = model(x_batch, step=step)
            loss = nn.CrossEntropyLoss()(output, labels)
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                alpha = model.grl.get_alpha()
                print(f"Training step {step}: GRL alpha = {alpha:.4f}, Loss = {loss.item():.4f}")
        
        print("✓ Integration test passed")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== GRL Test Complete ===")