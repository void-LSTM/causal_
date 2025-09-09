"""
Training infrastructure for CSP framework.
Comprehensive training loop with callbacks and scheduling.
"""

from .loop import CSPTrainer
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    MetricsLogger,
    LearningRateMonitor,
    CSPStructureMonitor,
    CallbackContainer
)
from .sched import (
    WarmupScheduler,
    CosineAnnealingWarmupRestarts,
    PhaseAwareScheduler,
    AdaptiveLRScheduler,
    SchedulerFactory
)

__all__ = [
    # Main trainer
    'CSPTrainer',
    
    # Callbacks
    'EarlyStopping',
    'ModelCheckpoint',
    'MetricsLogger',
    'LearningRateMonitor', 
    'CSPStructureMonitor',
    'CallbackContainer',
    
    # Schedulers
    'WarmupScheduler',
    'CosineAnnealingWarmupRestarts',
    'PhaseAwareScheduler',
    'AdaptiveLRScheduler',
    'SchedulerFactory'
]

# Training utilities
def create_trainer(model, datamodule, config=None, **kwargs):
    """
    Factory function to create CSP trainer with sensible defaults.
    
    Args:
        model: CARL model instance
        datamodule: CSP data module
        config: Training configuration
        **kwargs: Additional trainer arguments
        
    Returns:
        CSPTrainer instance
    """
    if config is not None:
        # Extract configs from ExperimentConfig
        training_config = config.training.__dict__
        optimizer_config = {
            'name': 'adamw',
            'lr': training_config.get('learning_rate', 1e-3),
            'weight_decay': training_config.get('weight_decay', 1e-4)
        }
        scheduler_config = {
            'name': training_config.get('scheduler', 'cosine'),
            'warmup_epochs': training_config.get('warmup_epochs', 5)
        }
    else:
        training_config = None
        optimizer_config = None
        scheduler_config = None
    
    return CSPTrainer(
        model=model,
        datamodule=datamodule,
        training_config=training_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        **kwargs
    )

def create_default_callbacks(output_dir, monitor='val_total_loss'):
    """
    Create default callback set for training.
    
    Args:
        output_dir: Output directory for logs and checkpoints
        monitor: Metric to monitor for early stopping
        
    Returns:
        List of callback instances
    """
    callbacks = [
        EarlyStopping(monitor=monitor, patience=15, verbose=True),
        ModelCheckpoint(
            filepath=f"{output_dir}/checkpoint_{{epoch:03d}}.pt",
            monitor=monitor,
            save_best_only=False,
            save_freq=10
        ),
        MetricsLogger(log_dir=f"{output_dir}/logs"),
        LearningRateMonitor(),
        CSPStructureMonitor(monitor_interval=5)
    ]
    
    return callbacks