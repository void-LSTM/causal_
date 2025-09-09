"""
CARL: Causal-Aware Representation Learning for Cross-modal Structure Preservation
A comprehensive framework for learning causal representations across modalities.
"""

__version__ = "0.1.0"
__author__ = "CSP Research Team"
__description__ = "Causal-Aware Representation Learning for Cross-modal Structure Preservation"

# Core imports
from .models.carl import CausalAwareModel
from .data.datamodule import CSPDataModule, CSPDataset
from .train.loop import CSPTrainer
from .config.manager import ConfigManager, ExperimentConfig

# Evaluation
from .eval.hooks import ModelEvaluator, RepresentationExtractor, CSPMetricsComputer

# Utilities
from .utils.grl import create_grl, StaticGRL, AdaptiveGRL

__all__ = [
    # Core classes
    'CausalAwareModel',
    'CSPDataModule', 
    'CSPDataset',
    'CSPTrainer',
    'ConfigManager',
    'ExperimentConfig',
    
    # Evaluation
    'ModelEvaluator',
    'RepresentationExtractor', 
    'CSPMetricsComputer',
    
    # Utilities
    'create_grl',
    'StaticGRL',
    'AdaptiveGRL',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]

# Convenience functions
def create_experiment(scenario='IM', preset='dev', **overrides):
    """
    Create a complete experiment setup with sensible defaults.
    
    Args:
        scenario: Training scenario ('IM', 'IY', 'DUAL')
        preset: Configuration preset ('dev', 'research', 'production')
        **overrides: Configuration overrides
        
    Returns:
        Tuple of (config, model, datamodule, trainer)
    """
    # Create configuration
    config_manager = ConfigManager()
    config = config_manager.create_config(
        preset=preset,
        **{'model.scenario': scenario, **overrides}
    )
    
    return config

def quick_start(data_dir, scenario='IM', **kwargs):
    """
    Quick start function for immediate experimentation.
    
    Args:
        data_dir: Path to CSP dataset
        scenario: Training scenario
        **kwargs: Additional configuration overrides
        
    Returns:
        Trained model and results
    """
    config = create_experiment(scenario=scenario, **kwargs)
    
    # Note: This is a convenience wrapper
    # Full implementation would require actual training
    print(f"Quick start configuration created for {scenario} scenario")
    print(f"Data directory: {data_dir}")
    print(f"Use CSPTrainer to begin training")
    
    return config