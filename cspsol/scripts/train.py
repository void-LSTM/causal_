#!/usr/bin/env python3
"""
Main training script for CSP framework.
Provides command-line interface for training CARL models with flexible configuration.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import warnings

# Add CSP framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cspsol.config.manager import ConfigManager, ExperimentConfig
    from cspsol.data.datamodule import CSPDataModule
    from cspsol.models.carl import CausalAwareModel
    from cspsol.train.loop import CSPTrainer
    from cspsol.train.callbacks import (
        EarlyStopping, ModelCheckpoint, MetricsLogger, 
        LearningRateMonitor, CSPStructureMonitor, CallbackContainer
    )
    from cspsol.eval.hooks import ModelEvaluator
    from cspsol.utils.seed import set_seed
    from cspsol.utils.io import ensure_directory
except ImportError as e:
    print(f"Failed to import CSP framework components: {e}")
    print("Please ensure the CSP framework is properly installed.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CARL model for causal structure preservation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=['dev', 'research', 'production'],
        default='dev',
        help='Configuration preset to use'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to CSP dataset directory'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['IM', 'IY', 'DUAL'],
        help='Training scenario (auto-detect if not specified)'
    )
    
    # Model arguments
    parser.add_argument(
        '--z-dim',
        type=int,
        default=128,
        help='Latent representation dimension'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name for this experiment'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use for training (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable automatic mixed precision training'
    )
    
    # Validation and checkpointing
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip validation during training'
    )
    
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation (requires --resume)'
    )
    
    parser.add_argument(
        '--save-representations',
        action='store_true',
        help='Save learned representations during evaluation'
    )
    
    # Logging and debugging
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress most output'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (overrides quiet)'
    )
    
    # Advanced options
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Enable deterministic training (may reduce performance)'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    return parser.parse_args()


def setup_logging(args):
    """Setup logging configuration."""
    import logging
    
    if args.debug:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = getattr(logging, args.log_level.upper())
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress some noisy loggers
    if not args.debug:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)


def create_experiment_config(args):
    """Create experiment configuration from arguments."""
    config_manager = ConfigManager()
    
    # Load config from file if provided
    if args.config:
        config = ExperimentConfig.load(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        # Create config from preset
        config = config_manager.create_config(preset=args.preset)
        print(f"Created configuration from preset: {args.preset}")
    
    # Apply command line overrides
    overrides = {}
    
    if args.scenario:
        overrides['data.scenario'] = args.scenario
        overrides['model.scenario'] = args.scenario
    
    if args.z_dim != 128:  # Only override if different from default
        overrides['model.z_dim'] = args.z_dim
    
    if args.epochs:
        overrides['training.max_epochs'] = args.epochs
    
    if args.batch_size:
        overrides['data.batch_size'] = args.batch_size
    
    if args.learning_rate:
        overrides['training.learning_rate'] = args.learning_rate
    
    if args.amp:
        overrides['training.use_amp'] = True
    
    if args.deterministic:
        overrides['deterministic'] = True
    
    # Apply overrides
    if overrides:
        print(f"Applying {len(overrides)} command line overrides...")
        config_manager._apply_overrides(config, overrides)
    
    # Set data directory
    config.data.data_dir = args.data_dir
    
    # Set experiment name and output directory
    if args.experiment_name:
        config.name = args.experiment_name
    
    config.output_dir = str(Path(args.output_dir) / config.name)
    
    # Set device
    config.device = args.device
    
    return config


def setup_data_module(config, args):
    """Setup data module."""
    print("Setting up data module...")
    
    try:
        datamodule = CSPDataModule(
            data_dir=config.data.data_dir,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            scenario=config.data.scenario,
            normalize_tabular=config.data.normalize,
            val_split=config.data.val_split
        )
        
        print(f"Data module created for scenario: {datamodule.detected_scenario}")
        print(f"Train: {len(datamodule.train_dataset)} samples")
        print(f"Val: {len(datamodule.val_dataset)} samples")
        print(f"Test: {len(datamodule.test_dataset)} samples")
        
        return datamodule
        
    except Exception as e:
        print(f"Failed to create data module: {e}")
        raise


def setup_model(config, feature_dims, args):
    """Setup CARL model."""
    print("Setting up CARL model...")
    
    try:
        model = CausalAwareModel(
            scenario=config.model.scenario,
            z_dim=config.model.z_dim,
            feature_dims=feature_dims,
            loss_config=config.model.loss_config,
            encoder_config=config.model.encoder_config,
            balancer_config=config.model.balancer_config
        )
        
        # Move to device
        device = torch.device(config.device if config.device != 'auto' 
                            else 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created with {param_count:,} trainable parameters")
        print(f"Device: {device}")
        
        return model, device
        
    except Exception as e:
        print(f"Failed to create model: {e}")
        raise


def setup_callbacks(config, args):
    """Setup training callbacks."""
    callbacks = []
    
    # Early stopping
    if config.training.early_stopping_patience > 0:
        callbacks.append(EarlyStopping(
            monitor=config.training.early_stopping_metric,
            patience=config.training.early_stopping_patience,
            mode=config.training.early_stopping_mode,
            verbose=not args.quiet
        ))
    
    # Model checkpointing
    checkpoint_dir = Path(config.output_dir) / 'checkpoints'
    callbacks.append(ModelCheckpoint(
        filepath=str(checkpoint_dir / 'checkpoint_{epoch:03d}.pt'),
        monitor=config.training.early_stopping_metric,
        save_freq=args.checkpoint_freq,
        verbose=not args.quiet
    ))
    
    # Metrics logging
    log_dir = Path(config.output_dir) / 'logs'
    callbacks.append(MetricsLogger(
        log_dir=str(log_dir),
        log_freq=1,
        save_plots=True
    ))
    
    # Learning rate monitoring
    callbacks.append(LearningRateMonitor())
    
    # CSP structure monitoring
    callbacks.append(CSPStructureMonitor(
        monitor_interval=5,
        save_representations=args.save_representations
    ))
    
    return CallbackContainer(callbacks)


def run_training(model, datamodule, config, callbacks, device, args):
    """Run model training."""
    print("Starting training...")
    
    try:
        # Create trainer
        trainer = CSPTrainer(
            model=model,
            datamodule=datamodule,
            training_config=config.training.__dict__,
            device=device,
            output_dir=config.output_dir
        )
        
        # Set callbacks
        if callbacks:
            trainer.callbacks = callbacks
            callbacks.set_trainer(trainer)
        
        # Resume from checkpoint if specified
        resume_path = args.resume
        if resume_path and Path(resume_path).exists():
            print(f"Resuming from checkpoint: {resume_path}")
        
        # Run training
        history = trainer.fit(resume_from_checkpoint=resume_path)
        
        print("Training completed successfully!")
        
        return history, trainer
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise


def run_evaluation(model, datamodule, config, device, args):
    """Run model evaluation."""
    print("Running evaluation...")
    
    try:
        evaluator = ModelEvaluator(
            model=model,
            scenario=config.model.scenario,
            device=device,
            output_dir=str(Path(config.output_dir) / 'evaluation')
        )
        
        # Run evaluation on test set
        test_loader = datamodule.test_dataloader()
        results = evaluator.evaluate(
            test_loader,
            save_representations=args.save_representations,
            save_metrics=True
        )
        
        print("Evaluation completed successfully!")
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


def main():
    """Main training function."""
    print("="*60)
    print("CSP Framework - CARL Model Training")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args)
    
    try:
        # Set random seed
        if args.seed is not None:
            seed_manager = set_seed(args.seed, deterministic=args.deterministic)
            print(f"Random seed set to: {args.seed}")
        
        # Create experiment configuration
        config = create_experiment_config(args)
        
        # Ensure output directory exists
        ensure_directory(config.output_dir)
        
        # Save configuration
        config_path = Path(config.output_dir) / 'config.yaml'
        config.save(config_path)
        print(f"Configuration saved to: {config_path}")
        
        # Setup data module
        datamodule = setup_data_module(config, args)
        feature_dims = datamodule.get_feature_dims()
        
        # Setup model
        model, device = setup_model(config, feature_dims, args)
        
        # Check if evaluation only
        if args.eval_only:
            if not args.resume:
                print("Error: --eval-only requires --resume")
                sys.exit(1)
            
            # Load model checkpoint
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from: {args.resume}")
            
            # Run evaluation
            results = run_evaluation(model, datamodule, config, device, args)
            
        else:
            # Setup callbacks
            callbacks = setup_callbacks(config, args)
            
            # Run training
            history, trainer = run_training(model, datamodule, config, callbacks, device, args)
            
            # Run final evaluation if not skipped
            if not args.no_validation:
                results = run_evaluation(model, datamodule, config, device, args)
        
        print("="*60)
        print("Training completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()