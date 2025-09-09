"""
Advanced learning rate schedulers for CARL model training.
Supports phase-aware scheduling, cosine annealing with restarts, and custom warmup strategies.
Designed to work with multi-phase training (warmup1 → warmup2 → full).
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math
import warnings
import numpy as np


class WarmupScheduler(_LRScheduler):
    """
    Base class for warmup-enabled schedulers.
    Provides linear warmup followed by a main scheduling strategy.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 warmup_epochs: int = 5,
                 warmup_start_lr: float = 1e-6,
                 main_scheduler: Optional[_LRScheduler] = None,
                 last_epoch: int = -1):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer instance
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
            main_scheduler: Main scheduler to use after warmup
            last_epoch: Last epoch number
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.main_scheduler = main_scheduler
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
                for base_lr in self.base_lrs
            ]
        else:
            # Use main scheduler
            if self.main_scheduler is not None:
                # Adjust main scheduler epoch
                self.main_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
                return self.main_scheduler.get_lr()
            else:
                return self.base_lrs
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        super().step(epoch)
        # Also step main scheduler if it exists and we're past warmup
        if (self.main_scheduler is not None and 
            self.last_epoch >= self.warmup_epochs):
            self.main_scheduler.step()


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts and initial warmup.
    Combines warmup + SGDR (Stochastic Gradient Descent with Warm Restarts).
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.0,
                 max_lr: Optional[float] = None,
                 min_lr: float = 1e-6,
                 warmup_steps: int = 0,
                 gamma: float = 1.0,
                 last_epoch: int = -1):
        """
        Initialize cosine annealing with warm restarts.
        
        Args:
            optimizer: Optimizer instance
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Cycle length multiplier
            max_lr: Maximum learning rate (default: optimizer lr)
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            gamma: Decay factor for max_lr after each cycle
            last_epoch: Last epoch number
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.base_max_lrs = [max_lr or group['lr'] for group in optimizer.param_groups]
        self.max_lrs = self.base_max_lrs.copy()
        
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        """Initialize learning rates."""
        for param_group, max_lr in zip(self.optimizer.param_groups, self.max_lrs):
            param_group['lr'] = self.min_lr if self.warmup_steps > 0 else max_lr
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.step_in_cycle == -1:
            return self.base_lrs
        
        # Warmup phase
        if self.step_in_cycle < self.warmup_steps:
            warmup_factor = self.step_in_cycle / self.warmup_steps
            return [
                self.min_lr + (max_lr - self.min_lr) * warmup_factor
                for max_lr in self.max_lrs
            ]
        
        # Cosine annealing phase
        effective_step = self.step_in_cycle - self.warmup_steps
        cycle_steps = self.first_cycle_steps * (self.cycle_mult ** self.cycle)
        effective_cycle_steps = cycle_steps - self.warmup_steps
        
        if effective_step >= effective_cycle_steps:
            # Start new cycle
            self.cycle += 1
            self.step_in_cycle = 0
            self.max_lrs = [max_lr * self.gamma for max_lr in self.max_lrs]
            return self.max_lrs
        
        # Cosine decay
        cos_factor = (1 + math.cos(math.pi * effective_step / effective_cycle_steps)) / 2
        return [
            self.min_lr + (max_lr - self.min_lr) * cos_factor
            for max_lr in self.max_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self.step_in_cycle = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class PhaseAwareScheduler(_LRScheduler):
    """
    Phase-aware scheduler for multi-phase training.
    Different learning rate strategies for warmup1, warmup2, and full phases.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 phase_configs: Dict[str, Dict[str, Any]],
                 phase_epochs: Dict[str, Tuple[int, int]],
                 last_epoch: int = -1):
        """
        Initialize phase-aware scheduler.
        
        Args:
            optimizer: Optimizer instance
            phase_configs: Configuration for each phase
            phase_epochs: Epoch ranges for each phase
            last_epoch: Last epoch number
        """
        self.phase_configs = phase_configs
        self.phase_epochs = phase_epochs
        self.current_phase = 'warmup1'
        self.phase_schedulers = {}
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Build phase-specific schedulers
        self._build_phase_schedulers(optimizer)
        
        super().__init__(optimizer, last_epoch)
    
    def _build_phase_schedulers(self, optimizer: optim.Optimizer):
        """Build schedulers for each phase."""
        for phase_name, config in self.phase_configs.items():
            sched_type = config.get('type', 'constant')
            
            if sched_type == 'constant':
                scheduler = None
            elif sched_type == 'linear':
                start_lr = config.get('start_lr', self.base_lrs[0])
                end_lr = config.get('end_lr', self.base_lrs[0])
                epochs = self.phase_epochs[phase_name][1] - self.phase_epochs[phase_name][0]
                scheduler = optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=start_lr / self.base_lrs[0],
                    end_factor=end_lr / self.base_lrs[0],
                    total_iters=epochs
                )
            elif sched_type == 'cosine':
                epochs = self.phase_epochs[phase_name][1] - self.phase_epochs[phase_name][0]
                eta_min = config.get('eta_min', 1e-6)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs,
                    eta_min=eta_min
                )
            elif sched_type == 'exponential':
                gamma = config.get('gamma', 0.95)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            else:
                scheduler = None
            
            self.phase_schedulers[phase_name] = scheduler
    
    def _update_current_phase(self, epoch: int):
        """Update current phase based on epoch."""
        for phase_name, (start_epoch, end_epoch) in self.phase_epochs.items():
            if start_epoch <= epoch < end_epoch:
                if self.current_phase != phase_name:
                    self.current_phase = phase_name
                    print(f"Scheduler switched to phase: {phase_name} at epoch {epoch}")
                break
    
    def get_lr(self):
        """Get learning rate for current phase."""
        current_scheduler = self.phase_schedulers.get(self.current_phase)
        
        if current_scheduler is not None:
            return current_scheduler.get_lr()
        else:
            # Use base learning rate with phase-specific multiplier
            phase_config = self.phase_configs.get(self.current_phase, {})
            lr_mult = phase_config.get('lr_mult', 1.0)
            return [base_lr * lr_mult for base_lr in self.base_lrs]
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        self._update_current_phase(epoch)
        
        # Step current phase scheduler
        current_scheduler = self.phase_schedulers.get(self.current_phase)
        if current_scheduler is not None:
            current_scheduler.step()
        
        # Update optimizer learning rates
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class AdaptiveLRScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler based on loss dynamics.
    Reduces LR when loss plateaus, increases when loss is decreasing consistently.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 mode: str = 'min',
                 factor: float = 0.5,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: float = 1e-6,
                 max_lr: float = 1e-1,
                 increase_factor: float = 1.1,
                 increase_patience: int = 5,
                 last_epoch: int = -1):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: Optimizer instance
            mode: 'min' or 'max' for loss monitoring
            factor: Factor by which to reduce LR
            patience: Epochs to wait before reducing LR
            threshold: Threshold for measuring improvement
            threshold_mode: 'rel' or 'abs' for threshold
            cooldown: Epochs to wait before resuming normal operation
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            increase_factor: Factor by which to increase LR
            increase_patience: Epochs of improvement before increasing LR
            last_epoch: Last epoch number
        """
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.increase_patience = increase_patience
        
        self.best = None
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.mode_worse = None
        self.cooldown_counter = 0
        self.eps = 1e-8
        
        self._init_is_better(mode, threshold, threshold_mode)
        
        super().__init__(optimizer, last_epoch)
    
    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str):
        """Initialize comparison function."""
        if mode not in {'min', 'max'}:
            raise ValueError(f'mode {mode} is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f'threshold mode {threshold_mode} is unknown!')
        
        if mode == 'min':
            self.mode_worse = float('inf')
        else:
            self.mode_worse = -float('inf')
        
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
    
    def _is_better(self, a: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold
    
    def step(self, metrics: float, epoch: Optional[int] = None):
        """
        Step the scheduler with current metrics.
        
        Args:
            metrics: Current metric value (e.g., validation loss)
            epoch: Current epoch number
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.best is None:
            self.best = metrics
        
        # Check if in cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            return
        
        # Check improvement
        if self._is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
            self.num_good_epochs += 1
            
            # Consider increasing LR if consistently improving
            if self.num_good_epochs >= self.increase_patience:
                self._increase_lr()
                self.num_good_epochs = 0
        else:
            self.num_bad_epochs += 1
            self.num_good_epochs = 0
        
        # Reduce LR if no improvement
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            print(f'Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}')
    
    def _increase_lr(self):
        """Increase learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = min(old_lr * self.increase_factor, self.max_lr)
            if new_lr > old_lr:
                param_group['lr'] = new_lr
                print(f'Increasing learning rate from {old_lr:.2e} to {new_lr:.2e}')


class SchedulerFactory:
    """
    Factory for creating learning rate schedulers.
    Provides convenient interface for different scheduler types.
    """
    
    @staticmethod
    def create_scheduler(scheduler_type: str, 
                        optimizer: optim.Optimizer,
                        **kwargs) -> _LRScheduler:
        """
        Create scheduler by type.
        
        Args:
            scheduler_type: Type of scheduler to create
            optimizer: Optimizer instance
            **kwargs: Scheduler-specific arguments
            
        Returns:
            Scheduler instance
        """
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        
        elif scheduler_type == 'cosine_warmup_restarts':
            return CosineAnnealingWarmupRestarts(optimizer, **kwargs)
        
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, **kwargs)
        
        elif scheduler_type == 'multistep':
            return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        
        elif scheduler_type == 'warmup':
            main_scheduler_type = kwargs.pop('main_scheduler', 'cosine')
            main_kwargs = kwargs.pop('main_kwargs', {})
            main_scheduler = SchedulerFactory.create_scheduler(
                main_scheduler_type, optimizer, **main_kwargs
            )
            return WarmupScheduler(optimizer, main_scheduler=main_scheduler, **kwargs)
        
        elif scheduler_type == 'phase_aware':
            return PhaseAwareScheduler(optimizer, **kwargs)
        
        elif scheduler_type == 'adaptive':
            return AdaptiveLRScheduler(optimizer, **kwargs)
        
        elif scheduler_type == 'none' or scheduler_type == 'constant':
            return None
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @staticmethod
    def create_carl_scheduler(optimizer: optim.Optimizer,
                             training_config: Dict[str, Any]) -> _LRScheduler:
        """
        Create scheduler optimized for CARL training.
        
        Args:
            optimizer: Optimizer instance
            training_config: Training configuration
            
        Returns:
            Scheduler instance optimized for CARL
        """
        max_epochs = training_config.get('max_epochs', 100)
        
        # Default CARL scheduler: warmup + cosine with restarts
        scheduler_config = {
            'first_cycle_steps': max_epochs // 3,
            'cycle_mult': 1.5,
            'max_lr': training_config.get('lr', 1e-3),
            'min_lr': training_config.get('min_lr', 1e-6),
            'warmup_steps': training_config.get('warmup_epochs', 5),
            'gamma': 0.9
        }
        
        return CosineAnnealingWarmupRestarts(optimizer, **scheduler_config)


# Test implementation
if __name__ == "__main__":
    print("=== Testing CSP Learning Rate Schedulers ===")
    
    # Create test optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    print("\n--- Testing CosineAnnealingWarmupRestarts ---")
    try:
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=20,
            cycle_mult=1.0,
            max_lr=1e-3,
            min_lr=1e-6,
            warmup_steps=5
        )
        
        lrs = []
        for epoch in range(50):
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}")
        
        # Check warmup
        assert lrs[0] < lrs[4], "LR should increase during warmup"
        # Check restarts
        restart_epochs = [20, 40]
        for restart_epoch in restart_epochs:
            if restart_epoch < len(lrs):
                assert lrs[restart_epoch] > lrs[restart_epoch-1], f"LR should restart at epoch {restart_epoch}"
        
        print("✓ CosineAnnealingWarmupRestarts test passed")
        
    except Exception as e:
        print(f"✗ CosineAnnealingWarmupRestarts test failed: {e}")
    
    print("\n--- Testing PhaseAwareScheduler ---")
    try:
        # Reset optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        phase_configs = {
            'warmup1': {'type': 'linear', 'start_lr': 1e-6, 'end_lr': 1e-3},
            'warmup2': {'type': 'constant', 'lr_mult': 1.0},
            'full': {'type': 'cosine', 'eta_min': 1e-6}
        }
        
        phase_epochs = {
            'warmup1': (0, 10),
            'warmup2': (10, 20), 
            'full': (20, 50)
        }
        
        scheduler = PhaseAwareScheduler(
            optimizer,
            phase_configs=phase_configs,
            phase_epochs=phase_epochs
        )
        
        lrs = []
        phases = []
        for epoch in range(30):
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            phases.append(scheduler.current_phase)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:2d}: LR = {current_lr:.6f}, Phase = {scheduler.current_phase}")
        
        # Check phase transitions
        assert phases[5] == 'warmup1', "Should be in warmup1 at epoch 5"
        assert phases[15] == 'warmup2', "Should be in warmup2 at epoch 15"
        assert phases[25] == 'full', "Should be in full at epoch 25"
        
        print("✓ PhaseAwareScheduler test passed")
        
    except Exception as e:
        print(f"✗ PhaseAwareScheduler test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing AdaptiveLRScheduler ---")
    try:
        # Reset optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        scheduler = AdaptiveLRScheduler(
            optimizer,
            mode='min',
            patience=3,
            factor=0.5,
            increase_patience=2,
            increase_factor=1.2
        )
        
        # Simulate loss values
        losses = [1.0, 0.8, 0.6, 0.5, 0.4, 0.45, 0.5, 0.6, 0.7, 0.3, 0.2, 0.1]
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        for epoch, loss in enumerate(losses):
            scheduler.step(loss, epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch:2d}: Loss = {loss:.3f}, LR = {current_lr:.6f}")
        
        print("✓ AdaptiveLRScheduler test passed")
        
    except Exception as e:
        print(f"✗ AdaptiveLRScheduler test failed: {e}")
    
    print("\n--- Testing SchedulerFactory ---")
    try:
        # Reset optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Test different scheduler types
        scheduler_types = [
            ('cosine', {'T_max': 50}),
            ('cosine_warmup_restarts', {'first_cycle_steps': 20, 'warmup_steps': 5}),
            ('warmup', {'warmup_epochs': 5, 'main_scheduler': 'cosine', 'main_kwargs': {'T_max': 45}}),
            ('none', {})
        ]
        
        for sched_type, kwargs in scheduler_types:
            scheduler = SchedulerFactory.create_scheduler(sched_type, optimizer, **kwargs)
            
            if scheduler is not None:
                # Test a few steps
                for _ in range(5):
                    scheduler.step()
                print(f"✓ {sched_type} scheduler created and tested")
            else:
                print(f"✓ {sched_type} scheduler (None) handled correctly")
        
        # Test CARL-optimized scheduler
        training_config = {
            'max_epochs': 100,
            'lr': 1e-3,
            'min_lr': 1e-6,
            'warmup_epochs': 10
        }
        
        carl_scheduler = SchedulerFactory.create_carl_scheduler(optimizer, training_config)
        assert carl_scheduler is not None, "CARL scheduler should be created"
        
        # Test a few steps
        for _ in range(10):
            carl_scheduler.step()
        
        print("✓ SchedulerFactory and CARL scheduler test passed")
        
    except Exception as e:
        print(f"✗ SchedulerFactory test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Testing integration with real optimizer schedules ---")
    try:
        # Test realistic learning rate evolution
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=30,
            warmup_steps=5,
            max_lr=1e-3,
            min_lr=1e-6,
            cycle_mult=1.0
        )
        
        print("LR Schedule preview:")
        for epoch in range(0, 60, 5):
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:2d}: {lr:.6f}")
        
        print("✓ Integration test passed")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
    
    print("\n=== CSP Scheduler Test Complete ===")