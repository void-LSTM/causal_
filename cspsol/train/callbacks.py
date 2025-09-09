"""
Training callbacks for CARL model monitoring and intervention.
Provides early stopping, checkpointing, metrics logging, and CSP-specific monitoring.
Designed to integrate seamlessly with CSPTrainer.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import warnings
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import json
import csv
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns


class Callback(ABC):
    """
    Base class for training callbacks.
    
    Callbacks provide hooks into the training process for monitoring,
    logging, and intervention at key training events.
    """
    
    def __init__(self):
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set reference to trainer."""
        self.trainer = trainer
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each batch."""
        pass
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of validation."""
        pass
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of validation."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback with configurable patience and metrics.
    Supports both improvement and plateau-based stopping conditions.
    """
    
    def __init__(self,
                 monitor: str = 'val_total_loss',
                 patience: int = 15,
                 min_delta: float = 1e-4,
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 save_best_checkpoint: bool = True,
                 verbose: bool = True):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor for early stopping
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric optimization
            restore_best_weights: Whether to restore best weights on stop
            save_best_checkpoint: Whether to save best checkpoint
            verbose: Whether to print early stopping messages
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.save_best_checkpoint = save_best_checkpoint
        self.verbose = verbose
        
        # Initialize tracking variables
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Comparison function
        if mode == 'min':
            self.is_better = lambda current, best: current < best - self.min_delta
        else:
            self.is_better = lambda current, best: current > best + self.min_delta
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Initialize early stopping state."""
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if self.verbose:
            print(f"Early stopping monitoring '{self.monitor}' with patience {self.patience}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Check early stopping condition."""
        if logs is None or self.monitor not in logs:
            if self.verbose:
                print(f"Warning: Early stopping metric '{self.monitor}' not found in logs")
            return
        
        current_metric = logs[self.monitor]
        
        if self.is_better(current_metric, self.best_metric):
            # Improvement found
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best weights
            if self.restore_best_weights and self.trainer is not None:
                self.best_weights = {
                    name: param.clone().detach()
                    for name, param in self.trainer.model.named_parameters()
                }
            
            # Save best checkpoint
            if self.save_best_checkpoint and self.trainer is not None:
                self.trainer.save_checkpoint(
                    epoch=epoch,
                    metrics=logs,
                    is_best=True
                )
            
            if self.verbose:
                print(f"Epoch {epoch:03d}: {self.monitor} improved to {current_metric:.6f}")
        else:
            # No improvement
            self.wait += 1
            
            if self.verbose and self.wait > 0:
                print(f"Epoch {epoch:03d}: {self.monitor} did not improve from {self.best_metric:.6f} "
                      f"(patience: {self.wait}/{self.patience})")
        
        # Check if should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            
            if self.verbose:
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best {self.monitor}: {self.best_metric:.6f} at epoch {self.best_epoch}")
            
            # Restore best weights
            if self.restore_best_weights and self.best_weights is not None:
                for name, param in self.trainer.model.named_parameters():
                    param.data.copy_(self.best_weights[name])
                
                if self.verbose:
                    print("Restored best model weights")
            
            # Signal trainer to stop
            if hasattr(self.trainer, '_should_stop'):
                self.trainer._should_stop = True
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Print final early stopping summary."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"\nTraining stopped early at epoch {self.stopped_epoch}")
            print(f"Best {self.monitor}: {self.best_metric:.6f} at epoch {self.best_epoch}")


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback with flexible saving strategies.
    Supports periodic saving, best model saving, and custom conditions.
    """
    
    def __init__(self,
                 filepath: str,
                 monitor: Optional[str] = None,
                 save_best_only: bool = False,
                 save_freq: Union[int, str] = 'epoch',
                 mode: str = 'min',
                 save_weights_only: bool = False,
                 verbose: bool = True):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path pattern for saving checkpoints
            monitor: Metric to monitor for best model saving
            save_best_only: Whether to only save when monitor improves
            save_freq: Save frequency ('epoch' or integer for every N epochs)
            mode: 'min' or 'max' for metric optimization
            save_weights_only: Whether to save only model weights
            verbose: Whether to print checkpoint messages
        """
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        # Create output directory
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.epochs_since_last_save = 0
        
        # Comparison function
        if mode == 'min':
            self.is_better = lambda current, best: current < best
        else:
            self.is_better = lambda current, best: current > best
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Save checkpoint based on strategy."""
        should_save = False
        is_best = False
        
        # Check if should save based on monitor
        if self.monitor is not None and logs is not None and self.monitor in logs:
            current_metric = logs[self.monitor]
            is_best = self.is_better(current_metric, self.best_metric)
            
            if is_best:
                self.best_metric = current_metric
                should_save = True
            elif not self.save_best_only:
                should_save = self._check_save_freq(epoch)
        else:
            # No monitor specified, use frequency
            should_save = self._check_save_freq(epoch)
        
        if should_save:
            self._save_checkpoint(epoch, logs, is_best)
            self.epochs_since_last_save = 0
        else:
            self.epochs_since_last_save += 1
    
    def _check_save_freq(self, epoch: int) -> bool:
        """Check if should save based on frequency."""
        if self.save_freq == 'epoch':
            return True
        elif isinstance(self.save_freq, int):
            return (epoch + 1) % self.save_freq == 0
        else:
            return False
    
    def _save_checkpoint(self, epoch: int, logs: Optional[Dict[str, Any]], is_best: bool):
        """Save checkpoint to disk."""
        if self.trainer is None:
            return
        
        # Format filepath
        filepath = str(self.filepath).format(epoch=epoch)
        
        if self.save_weights_only:
            # Save only model weights
            torch.save(self.trainer.model.state_dict(), filepath)
        else:
            # Save full checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.trainer.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'scheduler_state_dict': (
                    self.trainer.scheduler.state_dict() 
                    if self.trainer.scheduler else None
                ),
                'metrics': logs or {},
                'best_metric': self.best_metric,
            }
            
            # Add model config if it's a CARL model
            if hasattr(self.trainer.model, 'scenario'):
                checkpoint['model_config'] = {
                    'scenario': self.trainer.model.scenario,
                    'z_dim': getattr(self.trainer.model, 'z_dim', None),
                    'feature_dims': getattr(self.trainer.model, 'feature_dims', None)
                }
            else:
                # Generic model config
                checkpoint['model_config'] = {
                    'model_type': type(self.trainer.model).__name__
                }
            
            torch.save(checkpoint, filepath)
        
        if self.verbose:
            status = " (best)" if is_best else ""
            print(f"Checkpoint saved: {filepath}{status}")


class MetricsLogger(Callback):
    """
    Metrics logging callback with support for multiple output formats.
    Logs training metrics to CSV, JSON, and optionally TensorBoard.
    """
    
    def __init__(self,
                 log_dir: str,
                 log_freq: int = 1,
                 log_batch_metrics: bool = False,
                 save_plots: bool = True):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for saving logs
            log_freq: Frequency of logging (epochs)
            log_batch_metrics: Whether to log batch-level metrics
            save_plots: Whether to save training plots
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_freq = log_freq
        self.log_batch_metrics = log_batch_metrics
        self.save_plots = save_plots
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log storage
        self.epoch_logs = []
        self.batch_logs = []
        
        # Initialize CSV files
        self.epoch_csv_path = self.log_dir / 'epoch_metrics.csv'
        self.batch_csv_path = self.log_dir / 'batch_metrics.csv'
        
        self.epoch_csv_file = None
        self.epoch_csv_writer = None
        self.batch_csv_file = None
        self.batch_csv_writer = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Initialize logging files."""
        # Open CSV files
        self.epoch_csv_file = open(self.epoch_csv_path, 'w', newline='')
        
        if self.log_batch_metrics:
            self.batch_csv_file = open(self.batch_csv_path, 'w', newline='')
        
        print(f"Metrics logging to: {self.log_dir}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Log epoch metrics."""
        if logs is None:
            return
        
        # Add epoch number and timestamp
        log_entry = {
            'epoch': epoch,
            'timestamp': time.time(),
            **logs
        }
        
        self.epoch_logs.append(log_entry)
        
        # Write to CSV
        if self.epoch_csv_writer is None:
            fieldnames = list(log_entry.keys())
            self.epoch_csv_writer = csv.DictWriter(
                self.epoch_csv_file, fieldnames=fieldnames
            )
            self.epoch_csv_writer.writeheader()
        
        self.epoch_csv_writer.writerow(log_entry)
        self.epoch_csv_file.flush()
        
        # Save JSON periodically
        if (epoch + 1) % self.log_freq == 0:
            self._save_json_logs()
        
        # Generate plots periodically
        if self.save_plots and (epoch + 1) % (self.log_freq * 5) == 0:
            self._generate_plots()
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Log batch metrics if enabled."""
        if not self.log_batch_metrics or logs is None:
            return
        
        # Add batch number and timestamp
        log_entry = {
            'epoch': getattr(self.trainer, 'current_epoch', 0),
            'batch': batch,
            'timestamp': time.time(),
            **logs
        }
        
        self.batch_logs.append(log_entry)
        
        # Write to CSV
        if self.batch_csv_writer is None:
            fieldnames = list(log_entry.keys())
            self.batch_csv_writer = csv.DictWriter(
                self.batch_csv_file, fieldnames=fieldnames
            )
            self.batch_csv_writer.writeheader()
        
        self.batch_csv_writer.writerow(log_entry)
        self.batch_csv_file.flush()
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Finalize logging."""
        # Save final JSON logs
        self._save_json_logs()
        
        # Generate final plots
        if self.save_plots:
            self._generate_plots()
        
        # Close CSV files
        if self.epoch_csv_file:
            self.epoch_csv_file.close()
        if self.batch_csv_file:
            self.batch_csv_file.close()
        
        print(f"Training logs saved to: {self.log_dir}")
    
    def _save_json_logs(self):
        """Save logs to JSON format."""
        # Save epoch logs
        epoch_json_path = self.log_dir / 'epoch_metrics.json'
        with open(epoch_json_path, 'w') as f:
            json.dump(self.epoch_logs, f, indent=2)
        
        # Save batch logs
        if self.batch_logs:
            batch_json_path = self.log_dir / 'batch_metrics.json'
            with open(batch_json_path, 'w') as f:
                json.dump(self.batch_logs, f, indent=2)
    
    def _generate_plots(self):
        """Generate training plots."""
        if not self.epoch_logs:
            return
        
        try:
            # Convert to arrays for plotting
            epochs = [log['epoch'] for log in self.epoch_logs]
            
            # Plot loss curves
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Metrics')
            
            # Training and validation loss
            train_losses = [log.get('train_total_loss', 0) for log in self.epoch_logs]
            val_losses = [log.get('val_total_loss', 0) for log in self.epoch_logs]
            
            axes[0, 0].plot(epochs, train_losses, label='Train Loss')
            axes[0, 0].plot(epochs, val_losses, label='Val Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            
            # Individual loss components
            loss_components = ['ci', 'mbr', 'mac', 'align', 'style', 'ib']
            for i, component in enumerate(loss_components[:4]):
                train_key = f'train_{component}'
                val_key = f'val_{component}'
                
                if any(train_key in log for log in self.epoch_logs):
                    train_vals = [log.get(train_key, 0) for log in self.epoch_logs]
                    val_vals = [log.get(val_key, 0) for log in self.epoch_logs]
                    
                    ax = axes[i//2 + (0 if i < 2 else 0), i%2 + (1 if i >= 2 else 1)]
                    ax.plot(epochs, train_vals, label=f'Train {component.upper()}')
                    ax.plot(epochs, val_vals, label=f'Val {component.upper()}')
                    ax.set_title(f'{component.upper()} Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


class LearningRateMonitor(Callback):
    """
    Learning rate monitoring callback.
    Tracks and logs learning rate changes throughout training.
    """
    
    def __init__(self, log_momentum: bool = False):
        """
        Initialize LR monitor.
        
        Args:
            log_momentum: Whether to also log momentum values
        """
        super().__init__()
        self.log_momentum = log_momentum
        self.lr_history = []
        self.momentum_history = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Log learning rate."""
        if self.trainer is None:
            return
        
        # Get current learning rates
        lrs = [group['lr'] for group in self.trainer.optimizer.param_groups]
        self.lr_history.append({'epoch': epoch, 'lrs': lrs})
        
        # Log momentum if requested
        if self.log_momentum:
            momentums = []
            for group in self.trainer.optimizer.param_groups:
                if 'momentum' in group:
                    momentums.append(group['momentum'])
                elif 'betas' in group:
                    momentums.append(group['betas'][0])
            
            if momentums:
                self.momentum_history.append({'epoch': epoch, 'momentums': momentums})
        
        # Add to logs
        if logs is not None:
            logs['learning_rate'] = lrs[0] if lrs else 0.0
            if self.log_momentum and momentums:
                logs['momentum'] = momentums[0]


class CSPStructureMonitor(Callback):
    """
    CSP-specific monitoring callback.
    Tracks causal structure preservation metrics and phase transitions.
    """
    
    def __init__(self,
                 monitor_interval: int = 5,
                 save_representations: bool = False):
        """
        Initialize CSP structure monitor.
        
        Args:
            monitor_interval: Epochs between structure monitoring
            save_representations: Whether to save learned representations
        """
        super().__init__()
        self.monitor_interval = monitor_interval
        self.save_representations = save_representations
        self.structure_history = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Monitor CSP structure preservation."""
        if (epoch + 1) % self.monitor_interval != 0:
            return
        
        if self.trainer is None or logs is None:
            return
        
        # Extract CSP-specific metrics
        csp_metrics = {}
        
        # Loss component analysis
        for component in ['ci', 'mbr', 'mac', 'align', 'style', 'ib']:
            train_key = f'train_{component}'
            val_key = f'val_{component}'
            
            if train_key in logs:
                csp_metrics[f'{component}_train'] = logs[train_key]
            if val_key in logs:
                csp_metrics[f'{component}_val'] = logs[val_key]
        
        # Loss weights from balancer
        if 'loss_weights' in logs:
            csp_metrics['loss_weights'] = logs['loss_weights']
        
        # Model statistics
        if hasattr(self.trainer.model, 'get_statistics'):
            model_stats = self.trainer.model.get_statistics()
            csp_metrics.update(model_stats)
        
        # Store in history
        self.structure_history.append({
            'epoch': epoch,
            'metrics': csp_metrics
        })
        
        # Print summary
        print(f"\nCSP Structure Monitor (Epoch {epoch}):")
        if 'current_phase' in csp_metrics:
            print(f"  Training phase: {csp_metrics['current_phase']}")
        
        # CI and MBR health check
        ci_val = logs.get('val_ci', 0)
        mbr_val = logs.get('val_mbr', 0)
        
        if abs(ci_val) < 0.1:
            print(f"  ✓ Conditional independence preserved (CI = {ci_val:.4f})")
        else:
            print(f"  ⚠ Conditional independence may be violated (CI = {ci_val:.4f})")
        
        if mbr_val > 0:
            print(f"  ✓ Markov boundary retained (MBR = {mbr_val:.4f})")
        else:
            print(f"  ⚠ Markov boundary may be compromised (MBR = {mbr_val:.4f})")


class CallbackContainer:
    """
    Container for managing multiple callbacks.
    Provides unified interface for callback execution.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback container.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
        self.trainer = None
    
    def set_trainer(self, trainer):
        """Set trainer reference for all callbacks."""
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)
    
    def add_callback(self, callback: Callback):
        """Add a callback to the container."""
        callback.set_trainer(self.trainer)
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Union[Callback, type]):
        """Remove a callback from the container."""
        if isinstance(callback, type):
            self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, callback)]
        else:
            self.callbacks = [cb for cb in self.callbacks if cb is not callback]
    
    # Event dispatchers
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def on_validation_begin(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_validation_begin(logs)
    
    def on_validation_end(self, logs: Optional[Dict[str, Any]] = None):
        for callback in self.callbacks:
            callback.on_validation_end(logs)


# Test implementation
if __name__ == "__main__":
    print("=== Testing CSP Training Callbacks ===")
    
    # Create mock trainer for testing
    class MockTrainer:
        def __init__(self):
            self.model = torch.nn.Linear(10, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.scheduler = None
            self.current_epoch = 0
            self._should_stop = False
        
        def save_checkpoint(self, epoch, metrics, is_best=False):
            print(f"Mock checkpoint saved for epoch {epoch}, is_best={is_best}")
    
    trainer = MockTrainer()
    
    print("\n--- Testing EarlyStopping ---")
    try:
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=True
        )
        early_stopping.set_trainer(trainer)
        
        # Simulate training with improving then worsening metrics
        early_stopping.on_train_begin()
        
        test_metrics = [
            {'val_loss': 1.0, 'train_loss': 1.2},
            {'val_loss': 0.8, 'train_loss': 1.0},  # improvement
            {'val_loss': 0.6, 'train_loss': 0.8},  # improvement
            {'val_loss': 0.7, 'train_loss': 0.7},  # no improvement (1)
            {'val_loss': 0.8, 'train_loss': 0.6},  # no improvement (2)
            {'val_loss': 0.9, 'train_loss': 0.5},  # no improvement (3) -> should stop
        ]
        
        for epoch, metrics in enumerate(test_metrics):
            early_stopping.on_epoch_end(epoch, metrics)
            if trainer._should_stop:
                print(f"Training would stop at epoch {epoch}")
                break
        
        print("✓ EarlyStopping test passed")
        
    except Exception as e:
        print(f"✗ EarlyStopping test failed: {e}")
    
    print("\n--- Testing ModelCheckpoint ---")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_cb = ModelCheckpoint(
                filepath=f"{temp_dir}/model_{{epoch:03d}}.pt",
                monitor='val_loss',
                save_best_only=False,
                save_freq=2,
                verbose=True
            )
            checkpoint_cb.set_trainer(trainer)
            
            # Simulate epochs
            for epoch in range(5):
                metrics = {'val_loss': 1.0 - epoch * 0.1, 'train_loss': 1.2 - epoch * 0.1}
                checkpoint_cb.on_epoch_end(epoch, metrics)
        
        print("✓ ModelCheckpoint test passed")
        
    except Exception as e:
        print(f"✗ ModelCheckpoint test failed: {e}")
    
    print("\n--- Testing MetricsLogger ---")
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = MetricsLogger(
                log_dir=temp_dir,
                log_freq=1,
                save_plots=False  # Disable plots for testing
            )
            logger.set_trainer(trainer)
            
            logger.on_train_begin()
            
            # Simulate training
            for epoch in range(3):
                metrics = {
                    'train_total_loss': 1.0 - epoch * 0.2,
                    'val_total_loss': 1.1 - epoch * 0.15,
                    'train_ci': 0.1 - epoch * 0.02,
                    'val_ci': 0.12 - epoch * 0.02,
                }
                logger.on_epoch_end(epoch, metrics)
            
            logger.on_train_end()
        
        print("✓ MetricsLogger test passed")
        
    except Exception as e:
        print(f"✗ MetricsLogger test failed: {e}")
    
    print("\n--- Testing LearningRateMonitor ---")
    try:
        lr_monitor = LearningRateMonitor(log_momentum=True)
        lr_monitor.set_trainer(trainer)
        
        # Simulate epochs with changing LR
        for epoch in range(3):
            # Simulate LR change
            for group in trainer.optimizer.param_groups:
                group['lr'] = 0.001 * (0.9 ** epoch)
            
            logs = {}
            lr_monitor.on_epoch_end(epoch, logs)
            print(f"Epoch {epoch}: LR logged as {logs.get('learning_rate', 'N/A')}")
        
        print("✓ LearningRateMonitor test passed")
        
    except Exception as e:
        print(f"✗ LearningRateMonitor test failed: {e}")
    
    print("\n--- Testing CSPStructureMonitor ---")
    try:
        csp_monitor = CSPStructureMonitor(monitor_interval=2)
        csp_monitor.set_trainer(trainer)
        
        # Simulate epochs with CSP metrics
        for epoch in range(5):
            metrics = {
                'val_ci': 0.05 - epoch * 0.01,
                'val_mbr': 0.8 + epoch * 0.05,
                'train_ci': 0.06 - epoch * 0.01,
                'train_mbr': 0.75 + epoch * 0.05,
                'loss_weights': {'ci': 1.0, 'mbr': 1.0, 'mac': 0.5}
            }
            csp_monitor.on_epoch_end(epoch, metrics)
        
        print("✓ CSPStructureMonitor test passed")
        
    except Exception as e:
        print(f"✗ CSPStructureMonitor test failed: {e}")
    
    print("\n--- Testing CallbackContainer ---")
    try:
        # Create callback container with multiple callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=False),
            LearningRateMonitor(),
            CSPStructureMonitor(monitor_interval=1)
        ]
        
        container = CallbackContainer(callbacks)
        container.set_trainer(trainer)
        
        # Test event dispatching
        container.on_train_begin()
        
        for epoch in range(3):
            logs = {
                'val_loss': 1.0 - epoch * 0.1,
                'val_ci': 0.05,
                'val_mbr': 0.8
            }
            container.on_epoch_begin(epoch, logs)
            container.on_epoch_end(epoch, logs)
        
        container.on_train_end()
        
        print("✓ CallbackContainer test passed")
        
    except Exception as e:
        print(f"✗ CallbackContainer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== CSP Training Callbacks Test Complete ===")