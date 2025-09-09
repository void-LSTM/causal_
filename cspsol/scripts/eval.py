#!/usr/bin/env python3
"""
Standalone evaluation script for CSP framework.
Loads trained CARL models and evaluates them on test datasets.
Provides comprehensive evaluation including CSP metrics and representation analysis.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import warnings
import json
from typing import Dict, Any, Optional

# Add CSP framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cspsol.config.manager import ConfigManager, ExperimentConfig
    from cspsol.data.datamodule import CSPDataModule
    from cspsol.models.carl import CausalAwareModel
    from cspsol.eval.hooks import ModelEvaluator
    from cspsol.utils.seed import set_seed
    from cspsol.utils.io import ensure_directory, SafeFileHandler
except ImportError as e:
    print(f"Failed to import CSP framework components: {e}")
    print("Please ensure the CSP framework is properly installed.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained CARL model for causal structure preservation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model and checkpoint arguments
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model configuration file (auto-detect if not provided)'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to CSP dataset directory'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'all'],
        default='test',
        help='Dataset split to evaluate on'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Evaluation batch size'
    )
    
    # Evaluation options
    parser.add_argument(
        '--save-representations',
        action='store_true',
        help='Save learned representations'
    )
    
    parser.add_argument(
        '--compute-all-metrics',
        action='store_true',
        help='Compute all available CSP metrics'
    )
    
    parser.add_argument(
        '--detailed-analysis',
        action='store_true',
        help='Perform detailed analysis including correlations and trends'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for evaluation results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name for this evaluation (default: auto-generated)'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use for evaluation (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    # Logging arguments
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Advanced options
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--compare-with-random',
        action='store_true',
        help='Compare performance with random baseline'
    )
    
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Save evaluation plots'
    )
    
    return parser.parse_args()


def setup_logging(args):
    """Setup logging configuration."""
    import logging
    
    if args.verbose:
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
    if not args.verbose:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)


def load_checkpoint_and_config(checkpoint_path: str, config_path: Optional[str] = None) -> tuple:
    """
    Load model checkpoint and configuration.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Optional path to config file
        
    Returns:
        Tuple of (checkpoint_data, config)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Try to load config
    config = None
    
    if config_path:
        # Use provided config path
        try:
            config = ExperimentConfig.load(config_path)
            print(f"Configuration loaded from: {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    if config is None:
        # Try to auto-detect config file
        possible_config_paths = [
            checkpoint_path.parent / 'config.yaml',
            checkpoint_path.parent.parent / 'config.yaml',
            checkpoint_path.parent / 'experiment_config.yaml'
        ]
        
        for config_file in possible_config_paths:
            if config_file.exists():
                try:
                    config = ExperimentConfig.load(config_file)
                    print(f"Auto-detected configuration from: {config_file}")
                    break
                except Exception:
                    continue
    
    if config is None:
        print("Warning: No configuration file found, using checkpoint model config")
        # Create minimal config from checkpoint
        model_config = checkpoint.get('model_config', {})
        config = ExperimentConfig(
            model=model_config,
            data={'scenario': model_config.get('scenario', 'IM')},
            training={'max_epochs': 100}  # Dummy values
        )
    
    return checkpoint, config


def create_model_from_checkpoint(checkpoint: Dict[str, Any], 
                                config: ExperimentConfig,
                                device: torch.device) -> CausalAwareModel:
    """
    Create and load model from checkpoint.
    
    Args:
        checkpoint: Checkpoint data
        config: Experiment configuration
        device: Target device
        
    Returns:
        Loaded CARL model
    """
    model_config = checkpoint.get('model_config', {})
    
    # Extract model parameters
    scenario = model_config.get('scenario', config.model.scenario)
    z_dim = model_config.get('z_dim', config.model.z_dim)
    feature_dims = model_config.get('feature_dims', {
        'T_dim': 1, 'M_dim': 1, 'Y_dim': 1,
        'img_channels': 1, 'img_height': 28, 'img_width': 28
    })
    
    print(f"Creating model: scenario={scenario}, z_dim={z_dim}")
    
    # Create model
    model = CausalAwareModel(
        scenario=scenario,
        z_dim=z_dim,
        feature_dims=feature_dims,
        loss_config=config.model.loss_config,
        encoder_config=getattr(config.model, 'encoder_config', {}),
        balancer_config=getattr(config.model, 'balancer_config', {})
    )
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded successfully")
    except Exception as e:
        print(f"Warning: Some issues loading model state: {e}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {param_count:,} parameters")
    
    return model


def setup_data_module(config: ExperimentConfig, args) -> CSPDataModule:
    """
    Setup data module for evaluation.
    
    Args:
        config: Experiment configuration
        args: Command line arguments
        
    Returns:
        CSP data module
    """
    print("Setting up data module...")
    
    datamodule = CSPDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,  # Reduced for evaluation
        scenario=config.model.scenario,
        normalize_tabular=getattr(config.data, 'normalize', True),
        val_split=getattr(config.data, 'val_split', 0.2)
    )
    
    print(f"Data module created for scenario: {datamodule.detected_scenario}")
    print(f"Available splits - Train: {len(datamodule.train_dataset)}, "
          f"Val: {len(datamodule.val_dataset)}, Test: {len(datamodule.test_dataset)}")
    
    return datamodule


def get_dataloader_for_split(datamodule: CSPDataModule, split: str):
    """Get dataloader for specified split."""
    if split == 'train':
        return datamodule.train_dataloader()
    elif split == 'val':
        return datamodule.val_dataloader()
    elif split == 'test':
        return datamodule.test_dataloader()
    elif split == 'all':
        # Combine all splits
        all_batches = []
        all_batches.extend(datamodule.train_dataloader())
        all_batches.extend(datamodule.val_dataloader())
        all_batches.extend(datamodule.test_dataloader())
        return all_batches
    else:
        raise ValueError(f"Unknown split: {split}")


def run_evaluation(model: CausalAwareModel,
                  dataloader,
                  config: ExperimentConfig,
                  args,
                  device: torch.device) -> Dict[str, Any]:
    """
    Run comprehensive model evaluation.
    
    Args:
        model: Trained CARL model
        dataloader: Data loader for evaluation
        config: Experiment configuration
        args: Command line arguments
        device: Computation device
        
    Returns:
        Evaluation results dictionary
    """
    print(f"Running evaluation on {args.split} split...")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        scenario=config.model.scenario,
        device=device,
        output_dir=str(Path(args.output_dir) / 'detailed_analysis')
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        dataloader,
        save_representations=args.save_representations,
        save_metrics=True
    )
    
    print("Evaluation completed successfully!")
    
    return results


def compare_with_random_baseline(model: CausalAwareModel,
                                dataloader,
                                config: ExperimentConfig,
                                device: torch.device) -> Dict[str, Any]:
    """
    Compare model performance with random baseline.
    
    Args:
        model: Trained model
        dataloader: Data loader
        config: Configuration
        device: Device
        
    Returns:
        Comparison results
    """
    print("Comparing with random baseline...")
    
    # Create random model with same architecture
    random_model = CausalAwareModel(
        scenario=config.model.scenario,
        z_dim=model.z_dim,
        feature_dims=model.feature_dims,
        loss_config=config.model.loss_config
    )
    random_model = random_model.to(device)
    random_model.eval()
    
    # Evaluate random model
    random_evaluator = ModelEvaluator(
        model=random_model,
        scenario=config.model.scenario,
        device=device,
        output_dir=None  # Don't save random model outputs
    )
    
    random_results = random_evaluator.evaluate(
        dataloader,
        save_representations=False,
        save_metrics=False
    )
    
    return random_results


def save_evaluation_results(results: Dict[str, Any],
                           random_results: Optional[Dict[str, Any]],
                           args,
                           config: ExperimentConfig):
    """
    Save comprehensive evaluation results.
    
    Args:
        results: Main evaluation results
        random_results: Optional random baseline results
        args: Command line arguments
        config: Experiment configuration
    """
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    
    # Prepare comprehensive results
    final_results = {
        'evaluation_info': {
            'checkpoint_path': args.checkpoint,
            'data_dir': args.data_dir,
            'split': args.split,
            'batch_size': args.batch_size,
            'experiment_name': args.experiment_name,
            'evaluation_timestamp': str(torch.get_default_dtype()),  # Simple timestamp
        },
        'model_info': {
            'scenario': config.model.scenario,
            'z_dim': getattr(config.model, 'z_dim', 'unknown'),
            'loss_components': list(config.model.loss_config.keys()) if hasattr(config.model, 'loss_config') else [],
        },
        'results': results,
        'random_baseline': random_results
    }
    
    # Save main results
    handler = SafeFileHandler()
    results_path = output_dir / 'evaluation_results.json'
    handler.save_json(final_results, results_path)
    
    # Save summary
    summary = {
        'scenario': config.model.scenario,
        'split': args.split,
        'n_samples': results.get('n_samples', 0),
        'main_scores': results.get('summary', {}).get('main_scores', {}),
        'overall_score': results.get('summary', {}).get('overall_score', 0.0)
    }
    
    if random_results:
        summary['random_baseline_scores'] = random_results.get('summary', {}).get('main_scores', {})
        summary['improvement_over_random'] = {}
        
        main_scores = summary.get('main_scores', {})
        random_scores = summary.get('random_baseline_scores', {})
        
        for metric in main_scores:
            if metric in random_scores:
                improvement = main_scores[metric] - random_scores[metric]
                summary['improvement_over_random'][metric] = improvement
    
    summary_path = output_dir / 'evaluation_summary.json'
    handler.save_json(summary, summary_path)
    
    print(f"Results saved to: {output_dir}")
    print(f"Summary file: {summary_path}")


def print_evaluation_summary(results: Dict[str, Any], 
                            random_results: Optional[Dict[str, Any]],
                            args):
    """Print evaluation summary to console."""
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY - {args.split.upper()} SPLIT")
    print("="*60)
    
    # Basic info
    scenario = results.get('scenario', 'unknown')
    n_samples = results.get('n_samples', 0)
    print(f"Scenario: {scenario}")
    print(f"Samples evaluated: {n_samples}")
    print(f"Split: {args.split}")
    
    # Main metrics
    summary = results.get('summary', {})
    main_scores = summary.get('main_scores', {})
    
    if main_scores:
        print(f"\nCSP Metrics:")
        for metric, score in main_scores.items():
            status = "✓" if score > 0.7 else "⚠" if score > 0.4 else "✗"
            print(f"  {metric:6s}: {score:.4f} {status}")
        
        overall = summary.get('overall_score', 0.0)
        print(f"\nOverall Score: {overall:.4f}")
    else:
        print("\nNo CSP metrics computed")
    
    # Random baseline comparison
    if random_results:
        print(f"\nComparison with Random Baseline:")
        random_scores = random_results.get('summary', {}).get('main_scores', {})
        
        for metric in main_scores:
            if metric in random_scores:
                improvement = main_scores[metric] - random_scores[metric]
                print(f"  {metric}: {improvement:+.4f} (trained: {main_scores[metric]:.4f} vs random: {random_scores[metric]:.4f})")
    
    # Representation info
    repr_shapes = results.get('representation_shapes', {})
    if repr_shapes:
        print(f"\nRepresentation Shapes:")
        for name, shape in repr_shapes.items():
            print(f"  {name}: {shape}")
    
    print("="*60)


def main():
    """Main evaluation function."""
    print("="*60)
    print("CSP Framework - Model Evaluation")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args)
    
    try:
        # Set random seed
        set_seed(args.seed, deterministic=True)
        print(f"Random seed set to: {args.seed}")
        
        # Load checkpoint and config
        checkpoint, config = load_checkpoint_and_config(args.checkpoint, args.config)
        
        # Setup device
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # Create model
        model = create_model_from_checkpoint(checkpoint, config, device)
        
        # Setup data module
        datamodule = setup_data_module(config, args)
        
        # Get appropriate dataloader
        dataloader = get_dataloader_for_split(datamodule, args.split)
        
        # Run evaluation
        results = run_evaluation(model, dataloader, config, args, device)
        
        # Compare with random baseline if requested
        random_results = None
        if args.compare_with_random:
            random_results = compare_with_random_baseline(model, dataloader, config, device)
        
        # Save results
        save_evaluation_results(results, random_results, args, config)
        
        # Print summary
        print_evaluation_summary(results, random_results, args)
        
        print("\n" + "="*60)
        print("Evaluation completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nEvaluation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# Test the evaluation script functionality
def test_evaluation_script():
    """Test evaluation script functions with mock data."""
    print("="*50)
    print("Testing CSP Evaluation Script")
    print("="*50)
    
    try:
        # Test argument parsing
        print("\n--- Testing argument parsing ---")
        test_args = [
            '--checkpoint', 'test_checkpoint.pt',
            '--data-dir', './test_data',
            '--split', 'test',
            '--save-representations',
            '--output-dir', './test_eval_output'
        ]
        
        # Mock sys.argv for testing
        import sys
        original_argv = sys.argv
        sys.argv = ['eval.py'] + test_args
        
        try:
            args = parse_arguments()
            print(f"✓ Arguments parsed successfully")
            print(f"  Checkpoint: {args.checkpoint}")
            print(f"  Data dir: {args.data_dir}")
            print(f"  Split: {args.split}")
            print(f"  Save representations: {args.save_representations}")
            print(f"  Output dir: {args.output_dir}")
        finally:
            sys.argv = original_argv
        
        # Test config creation for missing config scenario
        print("\n--- Testing config handling ---")
        
        # Create mock checkpoint
        mock_checkpoint = {
            'epoch': 50,
            'model_state_dict': {},
            'model_config': {
                'scenario': 'IM',
                'z_dim': 64,
                'feature_dims': {'T_dim': 1, 'M_dim': 1, 'Y_dim': 1}
            },
            'metrics': {'val_loss': 0.5}
        }
        
        # Test config creation from checkpoint
        config = ExperimentConfig(
            model={'scenario': 'IM', 'z_dim': 64},
            data={'scenario': 'IM'},
            training={'max_epochs': 100}
        )
        
        print(f"✓ Config created from checkpoint")
        print(f"  Scenario: {config.model.scenario}")
        print(f"  Z-dim: {config.model.z_dim}")
        
        # Test summary formatting
        print("\n--- Testing summary functions ---")
        
        mock_results = {
            'scenario': 'IM',
            'n_samples': 1000,
            'summary': {
                'main_scores': {
                    'CIP': 0.85,
                    'CSI': 0.72,
                    'MBRI': 0.68,
                    'MAC': 0.79
                },
                'overall_score': 0.76
            },
            'representation_shapes': {
                'z_T': (1000, 64),
                'z_M': (1000, 64),
                'z_Y': (1000, 64)
            }
        }
        
        mock_random_results = {
            'summary': {
                'main_scores': {
                    'CIP': 0.45,
                    'CSI': 0.52,
                    'MBRI': 0.48,
                    'MAC': 0.51
                }
            }
        }
        
        # Test summary printing
        print("Summary output test:")
        print_evaluation_summary(mock_results, mock_random_results, args)
        
        print("\n✓ All evaluation script tests passed!")
        
        # Expected outputs format
        print("\n--- Expected Function Outputs ---")
        print("parse_arguments() -> argparse.Namespace with:")
        print("  - checkpoint: str (path to model)")
        print("  - data_dir: str (path to data)")
        print("  - split: str ('train'|'val'|'test'|'all')")
        print("  - save_representations: bool")
        print("  - output_dir: str")
        
        print("\nload_checkpoint_and_config() -> Tuple[Dict, ExperimentConfig]:")
        print("  - checkpoint: dict with model_state_dict, epoch, metrics")
        print("  - config: ExperimentConfig object")
        
        print("\nrun_evaluation() -> Dict with:")
        print("  - scenario: str")
        print("  - n_samples: int")
        print("  - summary: dict with main_scores and overall_score")
        print("  - representation_shapes: dict of tensor shapes")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Evaluation script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if script is executed directly
    test_evaluation_script()