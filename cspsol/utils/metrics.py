"""
Training metrics utilities for CSP framework.
Provides lightweight metrics computation during training for monitoring.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
import warnings


class MetricsTracker:
    """
    Lightweight metrics tracker for training monitoring.
    Computes and maintains moving averages of key metrics.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
    
    def update(self, metrics_dict: Dict[str, float]):
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric name -> value
        """
        for name, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.metrics[name].append(value)
            self.totals[name] += value
            self.counts[name] += 1
    
    def get_current(self, name: str) -> Optional[float]:
        """Get most recent value for metric."""
        if name in self.metrics and len(self.metrics[name]) > 0:
            return self.metrics[name][-1]
        return None
    
    def get_average(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """Get moving average for metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None
        
        values = list(self.metrics[name])
        if window is not None:
            values = values[-window:]
        
        return np.mean(values)
    
    def get_total_average(self, name: str) -> Optional[float]:
        """Get total average since tracking started."""
        if self.counts[name] == 0:
            return None
        return self.totals[name] / self.counts[name]
    
    def get_std(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """Get standard deviation for metric."""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return None
        
        values = list(self.metrics[name])
        if window is not None:
            values = values[-window:]
        
        return np.std(values)
    
    def get_trend(self, name: str, window: int = 20) -> Optional[str]:
        """
        Get trend direction for metric.
        
        Returns:
            'increasing', 'decreasing', 'stable', or None
        """
        if name not in self.metrics or len(self.metrics[name]) < window:
            return None
        
        values = list(self.metrics[name])[-window:]
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        relative_change = (second_half - first_half) / (abs(first_half) + 1e-8)
        
        if relative_change > 0.05:
            return 'increasing'
        elif relative_change < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.totals.clear()
        self.counts.clear()
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary of all metrics."""
        summary = {}
        
        for name in self.metrics.keys():
            summary[name] = {
                'current': self.get_current(name),
                'avg_recent': self.get_average(name, window=20),
                'avg_total': self.get_total_average(name),
                'std_recent': self.get_std(name, window=20),
                'trend': self.get_trend(name),
                'count': self.counts[name]
            }
        
        return summary


def compute_correlation_coefficient(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Pearson correlation coefficient between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Correlation coefficient
    """
    if x.numel() != y.numel() or x.numel() < 2:
        return 0.0
    
    x_flat = x.view(-1).float()
    y_flat = y.view(-1).float()
    
    # Handle edge cases
    if torch.std(x_flat) < 1e-8 or torch.std(y_flat) < 1e-8:
        return 0.0
    
    # Compute correlation
    x_centered = x_flat - torch.mean(x_flat)
    y_centered = y_flat - torch.mean(y_flat)
    
    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2) * torch.sum(y_centered ** 2))
    
    if denominator < 1e-8:
        return 0.0
    
    correlation = numerator / denominator
    return correlation.item()


def estimate_mutual_information_simple(x: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    """
    Simple mutual information estimation using histogram method.
    For training monitoring only - not for loss computation.
    
    Args:
        x: First variable
        y: Second variable
        bins: Number of histogram bins
        
    Returns:
        Estimated mutual information
    """
    try:
        x_np = x.detach().cpu().numpy().flatten()
        y_np = y.detach().cpu().numpy().flatten()
        
        if len(x_np) < 10:  # Too few samples
            return 0.0
        
        # Create histograms
        hist_x, _ = np.histogram(x_np, bins=bins, density=True)
        hist_y, _ = np.histogram(y_np, bins=bins, density=True)
        hist_xy, _, _ = np.histogram2d(x_np, y_np, bins=bins, density=True)
        
        # Convert to probabilities
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)
        p_xy = hist_xy / np.sum(hist_xy)
        
        # Compute MI
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 1e-8 and p_x[i] > 1e-8 and p_y[j] > 1e-8:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return max(0.0, mi)  # MI should be non-negative
        
    except Exception:
        return 0.0


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute total gradient norm for model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    param_count = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count == 0:
        return 0.0
    
    return total_norm ** 0.5


def compute_parameter_stats(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute statistics about model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of parameter statistics
    """
    stats = {
        'total_params': 0,
        'trainable_params': 0,
        'mean_weight': 0.0,
        'std_weight': 0.0,
        'mean_grad': 0.0,
        'std_grad': 0.0,
        'zero_grad_ratio': 0.0
    }
    
    all_weights = []
    all_grads = []
    zero_grads = 0
    total_grads = 0
    
    for param in model.parameters():
        stats['total_params'] += param.numel()
        
        if param.requires_grad:
            stats['trainable_params'] += param.numel()
            all_weights.extend(param.data.view(-1).cpu().numpy())
            
            if param.grad is not None:
                grad_flat = param.grad.data.view(-1).cpu().numpy()
                all_grads.extend(grad_flat)
                zero_grads += np.sum(np.abs(grad_flat) < 1e-8)
                total_grads += len(grad_flat)
    
    if all_weights:
        stats['mean_weight'] = np.mean(all_weights)
        stats['std_weight'] = np.std(all_weights)
    
    if all_grads:
        stats['mean_grad'] = np.mean(all_grads)
        stats['std_grad'] = np.std(all_grads)
        stats['zero_grad_ratio'] = zero_grads / total_grads if total_grads > 0 else 0.0
    
    return stats


class LossBalanceMonitor:
    """
    Monitor for loss balance in multi-task learning.
    Tracks relative magnitudes and trends of different loss components.
    """
    
    def __init__(self, loss_names: List[str], window_size: int = 50):
        """
        Initialize loss balance monitor.
        
        Args:
            loss_names: Names of loss components to track
            window_size: Window size for moving averages
        """
        self.loss_names = loss_names
        self.tracker = MetricsTracker(window_size)
        self.initial_losses = {}
        self.loss_ratios = defaultdict(list)
    
    def update(self, losses: Dict[str, float], weights: Optional[Dict[str, float]] = None):
        """
        Update loss tracking.
        
        Args:
            losses: Dictionary of loss values
            weights: Optional loss weights
        """
        # Store initial losses for normalization
        for name, value in losses.items():
            if name in self.loss_names and name not in self.initial_losses:
                if value > 1e-8:  # Avoid storing zero initial losses
                    self.initial_losses[name] = value
        
        # Update tracker
        self.tracker.update(losses)
        
        # Compute loss ratios for balance analysis
        if len(self.initial_losses) > 1:
            current_ratios = {}
            for name in self.loss_names:
                if name in losses and name in self.initial_losses:
                    current_loss = losses[name]
                    initial_loss = self.initial_losses[name]
                    ratio = current_loss / (initial_loss + 1e-8)
                    current_ratios[name] = ratio
            
            # Store ratios for trend analysis
            if current_ratios:
                for name, ratio in current_ratios.items():
                    self.loss_ratios[name].append(ratio)
                    # Keep only recent ratios
                    if len(self.loss_ratios[name]) > self.tracker.window_size:
                        self.loss_ratios[name] = self.loss_ratios[name][-self.tracker.window_size:]
    
    def get_balance_score(self) -> float:
        """
        Compute loss balance score.
        Higher score means better balance (closer to 1.0).
        """
        if len(self.loss_ratios) < 2:
            return 1.0
        
        recent_ratios = []
        for name in self.loss_names:
            if name in self.loss_ratios and len(self.loss_ratios[name]) > 0:
                recent_ratios.append(self.loss_ratios[name][-1])
        
        if len(recent_ratios) < 2:
            return 1.0
        
        # Balance score based on coefficient of variation
        mean_ratio = np.mean(recent_ratios)
        std_ratio = np.std(recent_ratios)
        
        if mean_ratio < 1e-8:
            return 1.0
        
        cv = std_ratio / mean_ratio  # Coefficient of variation
        balance_score = 1.0 / (1.0 + cv)  # Higher is better
        
        return balance_score
    
    def get_imbalance_report(self) -> Dict[str, str]:
        """
        Get report on loss imbalances.
        
        Returns:
            Dictionary with imbalance analysis
        """
        report = {}
        
        for name in self.loss_names:
            if name not in self.loss_ratios or len(self.loss_ratios[name]) < 5:
                report[name] = "insufficient_data"
                continue
            
            recent_ratios = self.loss_ratios[name][-10:]
            current_ratio = recent_ratios[-1]
            trend = np.polyfit(range(len(recent_ratios)), recent_ratios, 1)[0]
            
            # Classify loss behavior
            if current_ratio < 0.1:
                status = "very_low"
            elif current_ratio < 0.5:
                status = "low"
            elif current_ratio > 2.0:
                status = "high"
            elif current_ratio > 5.0:
                status = "very_high"
            else:
                status = "normal"
            
            # Add trend information
            if abs(trend) > 0.05:
                trend_str = "increasing" if trend > 0 else "decreasing"
            else:
                trend_str = "stable"
            
            report[name] = f"{status}_{trend_str}"
        
        return report


# Test functions
def test_metrics_tracker():
    """Test MetricsTracker functionality."""
    print("Testing MetricsTracker...")
    
    tracker = MetricsTracker(window_size=10)
    
    # Add some test data
    for i in range(15):
        metrics = {
            'loss': 1.0 - i * 0.05,
            'accuracy': 0.1 + i * 0.05,
            'lr': 1e-3 * (0.9 ** i)
        }
        tracker.update(metrics)
    
    # Test retrieval
    print(f"Current loss: {tracker.get_current('loss'):.4f}")
    print(f"Average loss (window): {tracker.get_average('loss'):.4f}")
    print(f"Total average loss: {tracker.get_total_average('loss'):.4f}")
    print(f"Loss trend: {tracker.get_trend('loss')}")
    
    summary = tracker.get_summary()
    print(f"Summary keys: {list(summary.keys())}")
    
    return tracker


def test_correlation_computation():
    """Test correlation coefficient computation."""
    print("\nTesting correlation computation...")
    
    # Test cases
    x1 = torch.randn(100)
    y1 = x1 + 0.1 * torch.randn(100)  # Strong positive correlation
    
    x2 = torch.randn(100)
    y2 = -x2 + 0.1 * torch.randn(100)  # Strong negative correlation
    
    x3 = torch.randn(100)
    y3 = torch.randn(100)  # No correlation
    
    corr1 = compute_correlation_coefficient(x1, y1)
    corr2 = compute_correlation_coefficient(x2, y2)
    corr3 = compute_correlation_coefficient(x3, y3)
    
    print(f"Positive correlation: {corr1:.4f} (expected: ~0.9)")
    print(f"Negative correlation: {corr2:.4f} (expected: ~-0.9)")
    print(f"No correlation: {corr3:.4f} (expected: ~0.0)")
    
    return [corr1, corr2, corr3]


def test_mutual_information():
    """Test mutual information estimation."""
    print("\nTesting mutual information estimation...")
    
    # Test cases
    x1 = torch.randn(1000)
    y1 = x1 + 0.1 * torch.randn(1000)  # High MI
    
    x2 = torch.randn(1000)
    y2 = torch.randn(1000)  # Low MI
    
    mi1 = estimate_mutual_information_simple(x1, y1)
    mi2 = estimate_mutual_information_simple(x2, y2)
    
    print(f"High MI case: {mi1:.4f}")
    print(f"Low MI case: {mi2:.4f}")
    
    return [mi1, mi2]


def test_loss_balance_monitor():
    """Test LossBalanceMonitor functionality."""
    print("\nTesting LossBalanceMonitor...")
    
    monitor = LossBalanceMonitor(['loss1', 'loss2', 'loss3'])
    
    # Simulate training with changing loss balance
    for i in range(50):
        losses = {
            'loss1': 1.0 * (0.9 ** i),  # Decreasing
            'loss2': 0.5,  # Constant
            'loss3': 0.1 + i * 0.01  # Increasing
        }
        monitor.update(losses)
    
    balance_score = monitor.get_balance_score()
    imbalance_report = monitor.get_imbalance_report()
    
    print(f"Balance score: {balance_score:.4f}")
    print(f"Imbalance report: {imbalance_report}")
    
    return monitor


if __name__ == "__main__":
    print("="*50)
    print("CSP Metrics Utilities Test")
    print("="*50)
    
    # Run all tests
    tracker = test_metrics_tracker()
    correlations = test_correlation_computation()
    mi_values = test_mutual_information()
    monitor = test_loss_balance_monitor()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)