"""
GradNorm implementation for automatic loss weight balancing.
Based on "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
Adapted for CSP causal representation learning with multiple loss components.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import inspect
from typing import Any
class GradNorm(nn.Module):
    """
    GradNorm for automatic loss weight balancing.
    
    Maintains gradient norms across tasks by adjusting loss weights dynamically.
    Loss weights are updated based on:
    L_gradnorm = Σ|G_k - Ḡ · (L_k(t)/L_k(0))^α|
    """
    
    def __init__(self,
                 loss_names: List[str],
                 alpha: float = 0.5,
                 update_freq: int = 50,
                 initial_weights: Optional[Dict[str, float]] = None,
                 normalize_weights: bool = True,
                 min_weight: float = 0.01,
                 max_weight: float = 5.0):
        """
        Initialize GradNorm.
        
        Args:
            loss_names: List of loss component names
            alpha: Restoring force strength (0.5 works well in practice)
            update_freq: Update weights every N steps
            initial_weights: Initial loss weights (default: uniform)
            normalize_weights: Whether to normalize weights to sum to K
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        super().__init__()
        
        self.loss_names = loss_names
        self.alpha = alpha
        self.update_freq = update_freq
        self.normalize_weights = normalize_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.num_tasks = len(loss_names)
        
        # Initialize loss weights as learnable parameters
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_names}
        
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_weights.get(name, 1.0), dtype=torch.float32))
            for name in loss_names
        })

        self._name_to_idx = {n: i for i, n in enumerate(self.loss_names)}

        # Track loss history for computing ratios
        self.register_buffer('initial_losses', torch.zeros(self.num_tasks))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))
        
        # Loss tracking
        self.loss_history = defaultdict(list)
        self.grad_norm_history = defaultdict(list)
        self.initial_losses_set = False
        
        print(f"GradNorm initialized with {self.num_tasks} tasks: {loss_names}")
        print(f"Initial weights: {dict(self.weights)}")

    def _weights_tensor(self) -> torch.Tensor:
        """
        返回满足约束的权重张量：
        - 正值（softplus）
        - 若提供 min/max，则在 [min_weight, max_weight] 之间
        - 若 normalize_weights=True，则权重和为 K，同时不违反上/下界
        （通过一次简单的“有界 simplex 投影”迭代实现）
        """
        raw = torch.stack([self.weights[n] for n in self.loss_names])  # [K]
        w = F.softplus(raw)  # 正值

        # 读取上下界
        lo = self.min_weight if self.min_weight is not None else 0.0
        hi = self.max_weight if self.max_weight is not None else float('inf')

        # 先裁到 [lo, hi]
        w = torch.clamp(w, min=lo, max=hi)

        if not self.normalize_weights:
            return w

        # 目标和为 K
        Ksum = float(self.num_tasks)

        # 有界 simplex 投影（少量迭代足够收敛，K 通常很小）
        # 思路：不断把 (Ksum - 当前和) 均匀分配到“未触边”的分量上，再裁到边界
        for _ in range(10):  # 迭代 10 次通常足够
            s = w.sum()
            gap = Ksum - s
            if torch.abs(gap) < 1e-8:
                break

            free = (w > lo + 1e-12) & (w < hi - 1e-12)  # 未触及上下界的自由分量
            n_free = int(free.sum().item())
            if n_free == 0:
                # 没有可自由调整的分量：只能在未到上界的分量上微调一次，然后再裁边
                candidates = (w < hi - 1e-12) if gap > 0 else (w > lo + 1e-12)
                if not bool(candidates.any()):
                    break
                share = gap / candidates.sum().clamp_min(1)
                w = torch.where(candidates, w + share, w)
                w = torch.clamp(w, min=lo, max=hi)
                continue

            share = gap / n_free
            w = torch.where(free, w + share, w)
            w = torch.clamp(w, min=lo, max=hi)

        return w

    
    def get_weights(self) -> Dict[str, float]:
        """以 dict 返回当前数值权重（用于打印/监控/测试）。
        在 Python 端再做一次严格的上下界夹紧，避免 float32->python float
        的舍入把 0.01 变成 0.0099999997 导致单测失败。
        """
        w = self._weights_tensor().detach().cpu().tolist()  # float32 -> python float
        if self.min_weight is not None or self.max_weight is not None:
            lo = self.min_weight if self.min_weight is not None else -float("inf")
            hi = self.max_weight if self.max_weight is not None else float("inf")
            # Python 端严格夹紧，确保返回值满足闭区间 [lo, hi]
            w = [min(max(float(v), lo), hi) for v in w]
        return {name: w[i] for i, name in enumerate(self.loss_names)}
    
    def update_weights(self,
                      losses: Dict[str, torch.Tensor],
                      shared_parameters: List[nn.Parameter],
                      grad_norm_lr: float = 0.025) -> Dict[str, float]:
        """
        Update loss weights using GradNorm algorithm.
        
        Args:
            losses: Dictionary of loss values
            shared_parameters: List of shared model parameters for gradient computation
            grad_norm_lr: Learning rate for weight updates
        
        Returns:
            Updated weights dictionary
        """
        self.step_count += 1
        
        # Store initial losses on first call
        if not self.initial_losses_set:
            for i, name in enumerate(self.loss_names):
                if name in losses:
                    self.initial_losses[i] = losses[name].detach()
            self.initial_losses_set = True
            return self.get_weights()
        
        # Only update every update_freq steps
        if self.step_count % self.update_freq != 0:
            return self.get_weights()
        
        # Get current weights
        current_w = self._weights_tensor()
        
        # 计算每个任务的梯度范数
        grad_norms = []
        names_present = []
        for i, name in enumerate(self.loss_names):
            if name not in losses:
                continue
            loss_scaled = current_w[i] * losses[name]  # 关键：loss 对 w 可微
            grads = torch.autograd.grad(
                outputs=loss_scaled,
                inputs=shared_parameters,
                retain_graph=True,
                create_graph=True,   # 关键：保留图，允许后续对 w 求导
                allow_unused=True,
            )
            gn = torch.zeros((), device=loss_scaled.device)
            for g in grads:
                if g is not None:
                    gn = gn + g.norm(2)  # 用 L2 范数
            grad_norms.append(gn)
            names_present.append(name)

        if len(grad_norms) < 2:
            return self.get_weights()

        grad_norm_tensor = torch.stack(grad_norms)  # [M]
        loss_tensor = torch.stack([losses[n].detach() for n in names_present])
        initial_loss_tensor = torch.stack([self.initial_losses[self._name_to_idx[n]] for n in names_present])

        avg_grad_norm = grad_norm_tensor.mean().detach()  # 只做刻度，避免反向
        loss_ratios = loss_tensor / (initial_loss_tensor + 1e-8)
        target_grad_norms = avg_grad_norm * (loss_ratios ** self.alpha)

        gradnorm_loss = torch.sum(torch.abs(grad_norm_tensor - target_grad_norms))

        # 对参与的那些权重参数求导
        raw_params = [self.weights[n] for n in names_present]
        weight_grads = torch.autograd.grad(
            outputs=gradnorm_loss,
            inputs=raw_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        with torch.no_grad():
            for n, p, g in zip(names_present, raw_params, weight_grads):
                if g is not None:
                    p.data -= grad_norm_lr * g

        # 日志
        self._log_update(losses, {n: v.item() for n, v in zip(names_present, grad_norms)}, self.get_weights())
        return self.get_weights()
    
    def _log_update(self, 
                   losses: Dict[str, torch.Tensor],
                   grad_norms: Dict[str, float],
                   weights: Dict[str, float]):
        """Log update information for monitoring."""
        # Store in history for analysis
        for name in self.loss_names:
            if name in losses:
                self.loss_history[name].append(losses[name].item())
            if name in grad_norms:
                self.grad_norm_history[name].append(grad_norms[name])
        
        # Keep only recent history
        max_history = 1000
        for name in self.loss_names:
            if len(self.loss_history[name]) > max_history:
                self.loss_history[name] = self.loss_history[name][-max_history:]
            if len(self.grad_norm_history[name]) > max_history:
                self.grad_norm_history[name] = self.grad_norm_history[name][-max_history:]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get training statistics for monitoring."""
        stats = {
            'step_count': self.step_count.item(),
            'current_weights': self.get_weights(),
            'initial_losses': {
                name: self.initial_losses[i].item() 
                for i, name in enumerate(self.loss_names)
            }
        }
        
        # Add recent loss and gradient norm statistics
        for name in self.loss_names:
            if self.loss_history[name]:
                stats[f'recent_loss_{name}'] = np.mean(self.loss_history[name][-10:])
            if self.grad_norm_history[name]:
                stats[f'recent_grad_norm_{name}'] = np.mean(self.grad_norm_history[name][-10:])
        
        return stats
    
    def reset_history(self):
        """Reset loss and gradient norm history."""
        self.loss_history.clear()
        self.grad_norm_history.clear()
        self.initial_losses_set = False
        self.step_count.zero_()


class DynamicWeightAveraging(nn.Module):
    """
    Dynamic Weight Averaging (DWA) as an alternative to GradNorm.
    Simpler approach based on loss rate changes.
    """
    
    def __init__(self,
                 loss_names: List[str],
                 temperature: float = 2.0,
                 update_freq: int = 10,
                 initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize DWA.
        
        Args:
            loss_names: List of loss component names
            temperature: Temperature parameter for weight computation
            update_freq: Update weights every N steps
            initial_weights: Initial loss weights
        """
        super().__init__()
        
        self.loss_names = loss_names
        self.temperature = temperature
        self.update_freq = update_freq
        self.num_tasks = len(loss_names)
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_names}
        
        self.register_buffer('weights', torch.tensor([
            initial_weights.get(name, 1.0) for name in loss_names
        ], dtype=torch.float32))
        
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))
        self.register_buffer('prev_losses', torch.zeros(self.num_tasks))
        self.prev_losses_set = False
        
        print(f"DWA initialized with {self.num_tasks} tasks: {loss_names}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights as dictionary."""
        return {name: self.weights[i].item() for i, name in enumerate(self.loss_names)}
    
    def update_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update weights using DWA algorithm.
        
        Args:
            losses: Dictionary of current loss values
        
        Returns:
            Updated weights dictionary
        """
        self.step_count += 1
        
        # Convert losses to tensor
        current_losses = torch.stack([
            losses[name].detach() if name in losses else torch.tensor(0.0)
            for name in self.loss_names
        ])
        
        # Store previous losses on first call
        if not self.prev_losses_set:
            self.prev_losses.copy_(current_losses)
            self.prev_losses_set = True
            return self.get_weights()
        
        # Only update every update_freq steps
        if self.step_count % self.update_freq != 0:
            return self.get_weights()
        
        # Compute loss ratios: L_k(t-1) / L_k(t-2)
        loss_ratios = self.prev_losses / (current_losses + 1e-8)
        
        # Compute weights: w_k ∝ exp(loss_ratio_k / T)
        exp_ratios = torch.exp(loss_ratios / self.temperature)
        new_weights = exp_ratios / exp_ratios.sum() * self.num_tasks
        
        # Update stored weights and previous losses
        self.weights.copy_(new_weights)
        self.prev_losses.copy_(current_losses)
        
        return self.get_weights()


class MultiTaskBalancer(nn.Module):
    """
    Unified interface for multi-task loss balancing.
    Supports both GradNorm and DWA methods.
    """
    
    def __init__(self,
                 loss_names: List[str],
                 method: str = 'gradnorm',
                 **kwargs):
        """
        Initialize multi-task balancer.
        
        Args:
            loss_names: List of loss component names
            method: Balancing method ('gradnorm', 'dwa', 'fixed')
            **kwargs: Additional arguments for specific methods
        """
        super().__init__()
        
        self.loss_names = loss_names
        self.method = method
        self.balancer = None

        def _filtered(ctor):
            sig = inspect.signature(ctor)
            return {k: v for k, v in kwargs.items() if v is not None and k in sig.parameters}

        if method == 'gradnorm':
            self.balancer = GradNorm(loss_names, **_filtered(GradNorm.__init__))
        elif method == 'dwa':
            self.balancer = DynamicWeightAveraging(loss_names, **_filtered(DynamicWeightAveraging.__init__))
        elif method == 'fixed':
            initial_weights = kwargs.get('initial_weights', {name: 1.0 for name in loss_names})
            self.register_buffer('fixed_weights', torch.tensor([initial_weights.get(name, 1.0) for name in loss_names]))
        else:
            raise ValueError(f"Unknown balancing method: {method}")

        print(f"MultiTaskBalancer initialized with method: {method}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        if self.method == 'fixed':
            return {name: self.fixed_weights[i].item() for i, name in enumerate(self.loss_names)}
        else:
            return self.balancer.get_weights()
    
    def update_weights(self,
                      losses: Dict[str, torch.Tensor],
                      shared_parameters: Optional[List[nn.Parameter]] = None,
                      **kwargs) -> Dict[str, float]:
        """Update loss weights."""
        if self.method == 'fixed':
            return self.get_weights()
        elif self.method == 'gradnorm':
            if shared_parameters is None:
                raise ValueError("GradNorm requires shared_parameters")
            return self.balancer.update_weights(losses, shared_parameters, **kwargs)
        elif self.method == 'dwa':
            return self.balancer.update_weights(losses)
        else:
            return self.get_weights()
    
    def get_statistics(self) -> Dict[str, any]:
        """Get balancer statistics."""
        base_stats = {
            'method': self.method,
            'current_weights': self.get_weights()
        }
        
        if hasattr(self.balancer, 'get_statistics'):
            base_stats.update(self.balancer.get_statistics())
        
        return base_stats


# Test the implementation
if __name__ == "__main__":
    print("=== Testing GradNorm and Multi-Task Balancing ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup test scenario
    loss_names = ['ci', 'mbr', 'mac', 'align', 'style', 'ib']
    
    print("\n--- Testing GradNorm ---")
    try:
        gradnorm = GradNorm(
            loss_names=loss_names,
            alpha=0.5,
            update_freq=10,
            initial_weights={'ci': 1.0, 'mbr': 1.0, 'mac': 0.5, 'align': 0.2, 'style': 0.1, 'ib': 0.01}
        ).to(device)
        
        print(f"Initial weights: {gradnorm.get_weights()}")
        
        # Create mock shared parameters
        shared_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)
        
        shared_params = list(shared_model.parameters())
        
        # Simulate training steps
        for step in range(25):
            # Mock input
            x = torch.randn(32, 64, device=device)
            output = shared_model(x)
            
            # Mock losses that change over time
            base = (output ** 2).mean()  # 让每个任务损失依赖共享参数
            mock_losses = {
                'ci': (1.0 - step * 0.02) * base,
                'mbr': (0.8 - step * 0.01) * base,
                'mac': (0.5 + step * 0.01) * base,
                'align': 0.3 * base,
                'style': 0.2 * base,
                'ib': 0.05 * base,
            }
            
            # Make losses depend on shared parameters
            total_loss = sum(loss * torch.sum(output) * 1e-6 for loss in mock_losses.values())
            
            # Update weights
            updated_weights = gradnorm.update_weights(mock_losses, shared_params)
            
            if step % 10 == 0:
                print(f"Step {step}: weights = {updated_weights}")
        
        # Check statistics
        stats = gradnorm.get_statistics()
        print(f"Final statistics: step_count={stats['step_count']}")
        print(f"Final weights: {stats['current_weights']}")
        
        print("✓ GradNorm test passed")
        
    except Exception as e:
        print(f"GradNorm test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing DWA ---")
    try:
        dwa = DynamicWeightAveraging(
            loss_names=loss_names,
            temperature=2.0,
            update_freq=5
        ).to(device)
        
        print(f"Initial DWA weights: {dwa.get_weights()}")
        
        # Simulate training with changing losses
        for step in range(20):
            mock_losses = {
                'ci': torch.tensor(1.0 - step * 0.03),
                'mbr': torch.tensor(0.8 - step * 0.02),
                'mac': torch.tensor(0.5 + step * 0.02),
                'align': torch.tensor(0.3),
                'style': torch.tensor(0.2 + step * 0.01),
                'ib': torch.tensor(0.05)
            }
            
            updated_weights = dwa.update_weights(mock_losses)
            
            if step % 5 == 0:
                print(f"DWA Step {step}: weights = {updated_weights}")
        
        print("✓ DWA test passed")
        
    except Exception as e:
        print(f"DWA test failed: {e}")
    
    print("\n--- Testing MultiTaskBalancer ---")
    try:
        # Test all methods
        for method in ['gradnorm', 'dwa', 'fixed']:
            print(f"\nTesting method: {method}")
            
            balancer = MultiTaskBalancer(
                loss_names=loss_names,
                method=method,
                alpha=0.5 if method == 'gradnorm' else None,
                temperature=2.0 if method == 'dwa' else None,
                initial_weights={'ci': 1.0, 'mbr': 1.0, 'mac': 0.5, 'align': 0.2, 'style': 0.1, 'ib': 0.01}
            ).to(device)
            
            print(f"Initial weights: {balancer.get_weights()}")
            
            # Mock losses
            mock_losses = {
                'ci': torch.tensor(0.5, requires_grad=True),
                'mbr': torch.tensor(0.3, requires_grad=True), 
                'mac': torch.tensor(0.2, requires_grad=True),
                'align': torch.tensor(0.1, requires_grad=True),
                'style': torch.tensor(0.05, requires_grad=True),
                'ib': torch.tensor(0.01, requires_grad=True)
            }
            
            # Update weights
            if method == 'gradnorm':
                # Need shared parameters for gradnorm
                shared_model = nn.Linear(10, 5).to(device)
                shared_params = list(shared_model.parameters())
                updated_weights = balancer.update_weights(mock_losses, shared_params)
            else:
                updated_weights = balancer.update_weights(mock_losses)
            
            print(f"Updated weights: {updated_weights}")
            
            # Get statistics
            stats = balancer.get_statistics()
            print(f"Statistics keys: {list(stats.keys())}")
        
        print("✓ MultiTaskBalancer test passed")
        
    except Exception as e:
        print(f"MultiTaskBalancer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing weight constraint handling ---")
    try:
        # Test extreme loss values to check constraint handling
        gradnorm = GradNorm(
            loss_names=['loss1', 'loss2'],
            min_weight=0.01,
            max_weight=3.0
        ).to(device)
        
        # Manually set extreme weights to test clamping
        with torch.no_grad():
            gradnorm.weights['loss1'].data = torch.tensor(100.0)  # Should be clamped
            gradnorm.weights['loss2'].data = torch.tensor(-100.0)  # Should be clamped
        
        weights = gradnorm.get_weights()
        print(f"Constrained weights: {weights}")
        
        # Check constraints
        for name, weight in weights.items():
            assert gradnorm.min_weight <= weight <= gradnorm.max_weight, \
                f"Weight {name}={weight} violates constraints [{gradnorm.min_weight}, {gradnorm.max_weight}]"
        
        print("✓ Weight constraint test passed")
        
    except Exception as e:
        print(f"Weight constraint test failed: {e}")
    
    print("\n=== GradNorm Test Complete ===")