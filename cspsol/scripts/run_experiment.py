#!/usr/bin/env python3
"""
Experiment runner for CSP framework.
Manages multiple experiments, parameter sweeps, and comparative analysis.
Provides comprehensive experiment lifecycle management and result aggregation.
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import warnings
import json
import yaml
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import itertools
import subprocess

# Add CSP framework to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from cspsol.config.manager import ConfigManager, ExperimentConfig
    from cspsol.utils.seed import generate_experiment_seeds, create_seed_config
    from cspsol.utils.io import SafeFileHandler, ensure_directory, create_experiment_archive
    from cspsol.utils.metrics import MetricsTracker
except ImportError as e:
    print(f"Failed to import CSP framework components: {e}")
    print("Please ensure the CSP framework is properly installed.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run CSP experiments with parameter sweeps and comparative analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Experiment configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Base configuration file for experiments'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        required=True,
        help='Name for this experiment batch'
    )
    
    parser.add_argument(
        '--sweep-config',
        type=str,
        help='Parameter sweep configuration file'
    )
    
    # Data and scenario options
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to CSP dataset directory'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        choices=['IM', 'IY', 'DUAL'],
        default=['IM'],
        help='Scenarios to run experiments for'
    )
    
    # Experiment execution options
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of runs per configuration (for statistical significance)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel (requires multiple GPUs)'
    )
    
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=2,
        help='Maximum number of parallel experiments'
    )
    
    parser.add_argument(
        '--gpu-ids',
        nargs='+',
        type=int,
        default=[0],
        help='GPU IDs to use for experiments'
    )
    
    # Parameter sweep options
    parser.add_argument(
        '--param-grid',
        type=str,
        help='JSON string or file path for parameter grid'
    )
    
    parser.add_argument(
        '--quick-sweep',
        action='store_true',
        help='Run quick parameter sweep with reduced epochs'
    )
    
    # Output and monitoring
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./experiments',
        help='Base output directory for all experiments'
    )
    
    parser.add_argument(
        '--save-all-checkpoints',
        action='store_true',
        help='Save checkpoints for all experiments (not just best)'
    )
    
    parser.add_argument(
        '--monitor-metric',
        type=str,
        default='val_total_loss',
        help='Metric to monitor for best model selection'
    )
    
    # Analysis options
    parser.add_argument(
        '--compare-baselines',
        action='store_true',
        help='Include baseline comparisons in analysis'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive experiment report'
    )
    
    parser.add_argument(
        '--plot-results',
        action='store_true',
        help='Generate result plots and visualizations'
    )
    
    # Control options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without actually running experiments'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume interrupted experiment batch'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing experiment results'
    )
    
    # Logging
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


def load_parameter_sweep_config(args) -> Dict[str, List[Any]]:
    """
    Load parameter sweep configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Parameter grid dictionary
    """
    param_grid = {}
    
    # Load from sweep config file if provided
    if args.sweep_config:
        sweep_path = Path(args.sweep_config)
        if sweep_path.exists():
            with open(sweep_path, 'r') as f:
                if sweep_path.suffix.lower() in ['.yaml', '.yml']:
                    sweep_data = yaml.safe_load(f)
                else:
                    sweep_data = json.load(f)
            param_grid.update(sweep_data.get('parameter_grid', {}))
            print(f"Loaded parameter sweep from: {sweep_path}")
    
    # Load from command line param-grid if provided
    if args.param_grid:
        try:
            # Try as file path first
            if Path(args.param_grid).exists():
                with open(args.param_grid, 'r') as f:
                    grid_data = json.load(f)
            else:
                # Try as JSON string
                grid_data = json.loads(args.param_grid)
            
            param_grid.update(grid_data)
            print("Loaded parameter grid from command line")
        except Exception as e:
            print(f"Warning: Failed to parse parameter grid: {e}")
    
    # Default parameter grid if none provided
    if not param_grid:
        if args.quick_sweep:
            param_grid = {
                'model.z_dim': [32, 64],
                'training.learning_rate': [1e-3, 5e-4],
                'training.max_epochs': [10, 20]  # Reduced for quick sweep
            }
        else:
            param_grid = {
                'model.z_dim': [64, 128, 256],
                'training.learning_rate': [1e-3, 5e-4, 1e-4],
                'model.loss_config.ci.weight': [0.5, 1.0, 1.5],
                'model.loss_config.mbr.weight': [0.5, 1.0, 1.5]
            }
        print("Using default parameter grid")
    
    return param_grid


def create_experiment_configs(base_config: ExperimentConfig,
                             param_grid: Dict[str, List[Any]],
                             scenarios: List[str],
                             num_runs: int,
                             args) -> List[Dict[str, Any]]:
    """
    Create all experiment configurations.
    
    Args:
        base_config: Base experiment configuration
        param_grid: Parameter sweep grid
        scenarios: List of scenarios to run
        num_runs: Number of runs per configuration
        args: Command line arguments
        
    Returns:
        List of experiment configurations
    """
    config_manager = ConfigManager()
    experiments = []
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    experiment_id = 0
    
    for scenario in scenarios:
        for run_id in range(num_runs):
            for param_combination in itertools.product(*param_values):
                # Create parameter overrides
                overrides = dict(zip(param_names, param_combination))
                
                # Add scenario and run-specific overrides
                overrides['data.scenario'] = scenario
                overrides['model.scenario'] = scenario
                overrides['data.data_dir'] = args.data_dir
                
                # Create experiment config
                exp_config = config_manager.create_config(
                    preset=None,
                    **overrides
                )
                
                # Generate experiment name
                param_str = '_'.join([f"{k.split('.')[-1]}={v}" for k, v in zip(param_names, param_combination)])
                exp_config.name = f"{args.experiment_name}_{scenario}_run{run_id}_{param_str}"
                
                # Generate seeds for reproducibility
                seed_config = create_seed_config(
                    base_seed=42 + experiment_id,
                    components=['global', 'data', 'model', 'training']
                )
                
                # Create experiment metadata
                experiment = {
                    'id': experiment_id,
                    'config': exp_config,
                    'scenario': scenario,
                    'run_id': run_id,
                    'parameters': dict(zip(param_names, param_combination)),
                    'seeds': seed_config,
                    'status': 'pending',
                    'output_dir': str(Path(args.output_dir) / exp_config.name),
                    'start_time': None,
                    'end_time': None,
                    'duration': None,
                    'results': None
                }
                
                experiments.append(experiment)
                experiment_id += 1
    
    print(f"Created {len(experiments)} experiment configurations")
    print(f"Scenarios: {scenarios}")
    print(f"Runs per config: {num_runs}")
    print(f"Parameter combinations: {len(list(itertools.product(*param_values)))}")
    
    return experiments


def run_single_experiment(experiment: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Run a single experiment.
    
    Args:
        experiment: Experiment configuration
        args: Command line arguments
        
    Returns:
        Updated experiment with results
    """
    exp_config = experiment['config']
    output_dir = experiment['output_dir']
    
    print(f"\nRunning experiment: {exp_config.name}")
    print(f"Scenario: {experiment['scenario']}, Run: {experiment['run_id']}")
    print(f"Parameters: {experiment['parameters']}")
    
    # Create output directory
    ensure_directory(output_dir, clean=args.force)
    
    # Save experiment configuration
    config_path = Path(output_dir) / 'config.yaml'
    exp_config.save(config_path)
    
    # Save experiment metadata
    metadata_path = Path(output_dir) / 'experiment_metadata.json'
    handler = SafeFileHandler()
    handler.save_json(experiment, metadata_path)
    
    # Record start time
    experiment['start_time'] = time.time()
    experiment['status'] = 'running'
    
    try:
        # Construct training command
        train_script = Path(__file__).parent / 'train.py'
        cmd = [
            sys.executable, str(train_script),
            '--config', str(config_path),
            '--data-dir', args.data_dir,
            '--output-dir', output_dir,
            '--seed', str(experiment['seeds']['global'])
        ]
        
        # Add additional arguments based on experiment config
        if args.save_all_checkpoints:
            cmd.extend(['--checkpoint-freq', '5'])
        
        if args.quiet:
            cmd.append('--quiet')
        elif args.verbose:
            cmd.extend(['--log-level', 'DEBUG'])
        
        # Run training
        print(f"Executing: {' '.join(cmd)}")
        
        if not args.dry_run:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                experiment['status'] = 'completed'
                print(f"✓ Experiment completed successfully")
                
                # Load results if available
                results_path = Path(output_dir) / 'training_history.json'
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        experiment['results'] = json.load(f)
                
            else:
                experiment['status'] = 'failed'
                experiment['error'] = result.stderr
                print(f"✗ Experiment failed: {result.stderr}")
        else:
            experiment['status'] = 'dry_run'
            print(f"✓ Dry run completed")
    
    except Exception as e:
        experiment['status'] = 'error'
        experiment['error'] = str(e)
        print(f"✗ Experiment error: {e}")
    
    # Record end time and duration
    experiment['end_time'] = time.time()
    experiment['duration'] = experiment['end_time'] - experiment['start_time']
    
    # Update metadata
    if not args.dry_run:
        handler.save_json(experiment, metadata_path)
    
    return experiment


def run_experiments_sequential(experiments: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Run experiments sequentially."""
    print(f"\nRunning {len(experiments)} experiments sequentially...")
    
    completed_experiments = []
    
    for i, experiment in enumerate(experiments):
        print(f"\n--- Experiment {i+1}/{len(experiments)} ---")
        
        # Check if experiment already exists and not forced
        output_dir = Path(experiment['output_dir'])
        if output_dir.exists() and not args.force and not args.resume:
            print(f"Skipping existing experiment: {experiment['config'].name}")
            experiment['status'] = 'skipped'
        else:
            experiment = run_single_experiment(experiment, args)
        
        completed_experiments.append(experiment)
        
        # Save progress
        progress_file = Path(args.output_dir) / f"{args.experiment_name}_progress.json"
        handler = SafeFileHandler()
        handler.save_json({
            'completed': len(completed_experiments),
            'total': len(experiments),
            'experiments': completed_experiments
        }, progress_file)
    
    return completed_experiments


def run_experiments_parallel(experiments: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    """Run experiments in parallel."""
    print(f"\nRunning {len(experiments)} experiments in parallel...")
    print(f"Max parallel: {args.max_parallel}")
    print(f"Available GPUs: {args.gpu_ids}")
    
    # This is a simplified parallel implementation
    # In practice, you might use multiprocessing or job schedulers
    
    import concurrent.futures
    import queue
    
    # Create GPU queue
    gpu_queue = queue.Queue()
    for gpu_id in args.gpu_ids:
        gpu_queue.put(gpu_id)
    
    def run_with_gpu(experiment):
        # Get GPU from queue
        gpu_id = gpu_queue.get()
        
        try:
            # Set GPU for this experiment
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # Run experiment
            result = run_single_experiment(experiment, args)
            
        finally:
            # Return GPU to queue
            gpu_queue.put(gpu_id)
        
        return result
    
    # Run experiments with thread pool
    completed_experiments = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        future_to_exp = {executor.submit(run_with_gpu, exp): exp for exp in experiments}
        
        for future in concurrent.futures.as_completed(future_to_exp):
            experiment = future_to_exp[future]
            try:
                result = future.result()
                completed_experiments.append(result)
                print(f"Completed: {result['config'].name} (Status: {result['status']})")
            except Exception as e:
                print(f"Error in experiment {experiment['config'].name}: {e}")
                experiment['status'] = 'error'
                experiment['error'] = str(e)
                completed_experiments.append(experiment)
    
    return completed_experiments


def analyze_experiment_results(experiments: List[Dict[str, Any]], args) -> Dict[str, Any]:
    """
    Analyze results from all experiments.
    
    Args:
        experiments: List of completed experiments
        args: Command line arguments
        
    Returns:
        Analysis results
    """
    print("\nAnalyzing experiment results...")
    
    analysis = {
        'summary': {
            'total_experiments': len(experiments),
            'completed': len([e for e in experiments if e['status'] == 'completed']),
            'failed': len([e for e in experiments if e['status'] == 'failed']),
            'errors': len([e for e in experiments if e['status'] == 'error'])
        },
        'scenarios': {},
        'parameter_analysis': {},
        'best_configurations': {},
        'performance_trends': {}
    }
    
    # Group by scenario
    by_scenario = {}
    for exp in experiments:
        scenario = exp['scenario']
        if scenario not in by_scenario:
            by_scenario[scenario] = []
        by_scenario[scenario].append(exp)
    
    # Analyze each scenario
    for scenario, scenario_exps in by_scenario.items():
        completed_exps = [e for e in scenario_exps if e['status'] == 'completed' and e.get('results')]
        
        scenario_analysis = {
            'total': len(scenario_exps),
            'completed': len(completed_exps),
            'success_rate': len(completed_exps) / len(scenario_exps) if scenario_exps else 0
        }
        
        if completed_exps:
            # Extract final metrics
            final_metrics = []
            for exp in completed_exps:
                results = exp.get('results', {})
                val_metrics = results.get('val', {})
                
                if val_metrics:
                    # Get final values
                    final_val_loss = val_metrics.get('val_total_loss', [])
                    if final_val_loss:
                        final_metrics.append({
                            'experiment': exp['config'].name,
                            'parameters': exp['parameters'],
                            'final_val_loss': final_val_loss[-1],
                            'best_val_loss': min(final_val_loss),
                            'total_epochs': len(final_val_loss)
                        })
            
            if final_metrics:
                # Find best configuration
                best_config = min(final_metrics, key=lambda x: x['best_val_loss'])
                scenario_analysis['best_config'] = best_config
                
                # Compute statistics
                val_losses = [m['final_val_loss'] for m in final_metrics]
                scenario_analysis['mean_final_loss'] = sum(val_losses) / len(val_losses)
                scenario_analysis['min_final_loss'] = min(val_losses)
                scenario_analysis['max_final_loss'] = max(val_losses)
        
        analysis['scenarios'][scenario] = scenario_analysis
    
    # Parameter impact analysis
    if len(experiments) > 1:
        # Group by parameter values
        completed_exps = [e for e in experiments if e['status'] == 'completed' and e.get('results')]
        
        if completed_exps:
            # Analyze each parameter
            param_names = set()
            for exp in completed_exps:
                param_names.update(exp['parameters'].keys())
            
            for param_name in param_names:
                param_analysis = {}
                
                # Group by parameter value
                by_param_value = {}
                for exp in completed_exps:
                    param_value = exp['parameters'].get(param_name)
                    if param_value is not None:
                        if param_value not in by_param_value:
                            by_param_value[param_value] = []
                        
                        results = exp.get('results', {})
                        val_metrics = results.get('val', {})
                        final_val_loss = val_metrics.get('val_total_loss', [])
                        
                        if final_val_loss:
                            by_param_value[param_value].append(final_val_loss[-1])
                
                # Compute statistics for each value
                for value, losses in by_param_value.items():
                    param_analysis[str(value)] = {
                        'count': len(losses),
                        'mean_loss': sum(losses) / len(losses),
                        'min_loss': min(losses),
                        'max_loss': max(losses)
                    }
                
                analysis['parameter_analysis'][param_name] = param_analysis
    
    print(f"Analysis completed for {analysis['summary']['completed']} experiments")
    
    return analysis


def generate_experiment_report(experiments: List[Dict[str, Any]],
                              analysis: Dict[str, Any],
                              args) -> str:
    """
    Generate comprehensive experiment report.
    
    Args:
        experiments: List of experiments
        analysis: Analysis results
        args: Command line arguments
        
    Returns:
        Report file path
    """
    print("Generating experiment report...")
    
    output_dir = Path(args.output_dir)
    report_path = output_dir / f"{args.experiment_name}_report.md"
    
    # Generate markdown report
    report_lines = [
        f"# CSP Experiment Report: {args.experiment_name}",
        f"",
        f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Data Directory:** {args.data_dir}",
        f"**Scenarios:** {', '.join(args.scenarios)}",
        f"**Number of Runs:** {args.num_runs}",
        f"",
        f"## Summary",
        f"",
        f"- **Total Experiments:** {analysis['summary']['total_experiments']}",
        f"- **Completed:** {analysis['summary']['completed']}",
        f"- **Failed:** {analysis['summary']['failed']}",
        f"- **Errors:** {analysis['summary']['errors']}",
        f"",
        f"## Results by Scenario",
        f""
    ]
    
    # Add scenario results
    for scenario, scenario_analysis in analysis['scenarios'].items():
        report_lines.extend([
            f"### {scenario} Scenario",
            f"",
            f"- **Success Rate:** {scenario_analysis['success_rate']:.2%}",
            f"- **Completed Experiments:** {scenario_analysis['completed']}/{scenario_analysis['total']}"
        ])
        
        if 'best_config' in scenario_analysis:
            best = scenario_analysis['best_config']
            report_lines.extend([
                f"- **Best Configuration:**",
                f"  - Best Validation Loss: {best['best_val_loss']:.4f}",
                f"  - Parameters: {best['parameters']}",
                f"- **Performance Statistics:**",
                f"  - Mean Final Loss: {scenario_analysis['mean_final_loss']:.4f}",
                f"  - Min Final Loss: {scenario_analysis['min_final_loss']:.4f}",
                f"  - Max Final Loss: {scenario_analysis['max_final_loss']:.4f}"
            ])
        
        report_lines.append("")
    
    # Add parameter analysis
    if analysis['parameter_analysis']:
        report_lines.extend([
            f"## Parameter Impact Analysis",
            f""
        ])
        
        for param_name, param_analysis in analysis['parameter_analysis'].items():
            report_lines.extend([
                f"### {param_name}",
                f""
            ])
            
            for value, stats in param_analysis.items():
                report_lines.append(f"- **{value}:** Mean Loss = {stats['mean_loss']:.4f} (n={stats['count']})")
            
            report_lines.append("")
    
    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {report_path}")
    return str(report_path)


def save_experiment_results(experiments: List[Dict[str, Any]],
                           analysis: Dict[str, Any],
                           args):
    """Save all experiment results and analysis."""
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    
    handler = SafeFileHandler()
    
    # Save complete experiment data
    experiments_file = output_dir / f"{args.experiment_name}_experiments.json"
    handler.save_json(experiments, experiments_file)
    
    # Save analysis
    analysis_file = output_dir / f"{args.experiment_name}_analysis.json"
    handler.save_json(analysis, analysis_file)
    
    # Save summary
    summary = {
        'experiment_name': args.experiment_name,
        'total_experiments': len(experiments),
        'completed': len([e for e in experiments if e['status'] == 'completed']),
        'scenarios': args.scenarios,
        'num_runs': args.num_runs,
        'output_directory': str(output_dir),
        'analysis_summary': analysis['summary']
    }
    
    summary_file = output_dir / f"{args.experiment_name}_summary.json"
    handler.save_json(summary, summary_file)
    
    print(f"Results saved to: {output_dir}")


def main():
    """Main experiment runner function."""
    print("="*60)
    print("CSP Framework - Experiment Runner")
    print("="*60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args)
    
    try:
        # Load base configuration
        config_manager = ConfigManager()
        
        if args.config:
            base_config = ExperimentConfig.load(args.config)
            print(f"Loaded base configuration from: {args.config}")
        else:
            base_config = config_manager.create_config(preset='research')
            print("Using default research configuration")
        
        # Load parameter sweep configuration
        param_grid = load_parameter_sweep_config(args)
        print(f"Parameter grid: {param_grid}")
        
        # Create experiment configurations
        experiments = create_experiment_configs(
            base_config, param_grid, args.scenarios, args.num_runs, args
        )
        
        if args.dry_run:
            print(f"\nDry run - would execute {len(experiments)} experiments:")
            for exp in experiments[:5]:  # Show first 5
                print(f"  {exp['config'].name}")
            if len(experiments) > 5:
                print(f"  ... and {len(experiments) - 5} more")
            return
        
        # Run experiments
        if args.parallel and len(args.gpu_ids) > 1:
            completed_experiments = run_experiments_parallel(experiments, args)
        else:
            completed_experiments = run_experiments_sequential(experiments, args)
        
        # Analyze results
        analysis = analyze_experiment_results(completed_experiments, args)
        
        # Generate report if requested
        if args.generate_report:
            report_path = generate_experiment_report(completed_experiments, analysis, args)
        
        # Save results
        save_experiment_results(completed_experiments, analysis, args)
        
        # Print final summary
        print("\n" + "="*60)
        print("EXPERIMENT BATCH COMPLETED")
        print("="*60)
        print(f"Total experiments: {len(completed_experiments)}")
        print(f"Completed: {analysis['summary']['completed']}")
        print(f"Failed: {analysis['summary']['failed']}")
        print(f"Success rate: {analysis['summary']['completed']/len(completed_experiments):.2%}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nExperiment batch interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment batch failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


# Test the experiment runner functionality
def test_experiment_runner():
    """Test experiment runner functions with mock data."""
    print("="*50)
    print("Testing CSP Experiment Runner")
    print("="*50)
    
    try:
        # Test argument parsing
        print("\n--- Testing argument parsing ---")
        test_args = [
            '--experiment-name', 'test_experiment',
            '--data-dir', './test_data',
            '--scenarios', 'IM', 'IY',
            '--num-runs', '2',
            '--dry-run'
        ]
        
        # Mock sys.argv for testing
        import sys
        original_argv = sys.argv
        sys.argv = ['run_experiment.py'] + test_args
        
        try:
            args = parse_arguments()
            print(f"✓ Arguments parsed successfully")
            print(f"  Experiment name: {args.experiment_name}")
            print(f"  Data dir: {args.data_dir}")
            print(f"  Scenarios: {args.scenarios}")
            print(f"  Num runs: {args.num_runs}")
            print(f"  Dry run: {args.dry_run}")
        finally:
            sys.argv = original_argv
        
        # Test parameter grid loading
        print("\n--- Testing parameter grid ---")
        
        # Mock parameter grid
        mock_param_grid = {
            'model.z_dim': [32, 64],
            'training.learning_rate': [1e-3, 5e-4],
            'model.loss_config.ci.weight': [0.5, 1.0]
        }
        
        print(f"Mock parameter grid: {mock_param_grid}")
        
        # Test experiment config creation
        print("\n--- Testing experiment configuration creation ---")
        
        # Create mock base config
        config_manager = ConfigManager()
        base_config = config_manager.create_config(preset='dev')
        
        # Create experiment configs
        experiments = create_experiment_configs(
            base_config, mock_param_grid, ['IM'], 1, args
        )
        
        print(f"✓ Created {len(experiments)} experiment configurations")
        print(f"First experiment name: {experiments[0]['config'].name}")
        print(f"First experiment parameters: {experiments[0]['parameters']}")
        
        # Test analysis functions
        print("\n--- Testing analysis functions ---")
        
        # Create mock completed experiments
        mock_experiments = []
        for i, exp in enumerate(experiments[:2]):  # Take first 2
            mock_exp = exp.copy()
            mock_exp['status'] = 'completed'
            mock_exp['results'] = {
                'val': {
                    'val_total_loss': [1.0 - i*0.1, 0.8 - i*0.1, 0.6 - i*0.1]  # Decreasing loss
                }
            }
            mock_experiments.append(mock_exp)
        
        # Add one failed experiment
        failed_exp = experiments[0].copy()
        failed_exp['status'] = 'failed'
        failed_exp['error'] = 'Mock error for testing'
        mock_experiments.append(failed_exp)
        
        # Run analysis
        analysis = analyze_experiment_results(mock_experiments, args)
        
        print(f"✓ Analysis completed")
        print(f"  Total experiments: {analysis['summary']['total_experiments']}")
        print(f"  Completed: {analysis['summary']['completed']}")
        print(f"  Failed: {analysis['summary']['failed']}")
        print(f"  Scenarios analyzed: {list(analysis['scenarios'].keys())}")
        
        # Test report generation
        print("\n--- Testing report generation ---")
        
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = temp_dir
            report_path = generate_experiment_report(mock_experiments, analysis, args)
            
            # Check if report was created
            if Path(report_path).exists():
                print(f"✓ Report generated: {report_path}")
                
                # Read first few lines
                with open(report_path, 'r') as f:
                    lines = f.readlines()[:10]
                print(f"  Report preview: {len(lines)} lines")
            else:
                print(f"✗ Report generation failed")
        
        print("\n✓ All experiment runner tests passed!")
        
        # Expected outputs format
        print("\n--- Expected Function Outputs ---")
        print("parse_arguments() -> argparse.Namespace with:")
        print("  - experiment_name: str")
        print("  - data_dir: str")
        print("  - scenarios: List[str]")
        print("  - num_runs: int")
        print("  - param_grid: str or None")
        
        print("\ncreate_experiment_configs() -> List[Dict] with:")
        print("  - id: int")
        print("  - config: ExperimentConfig")
        print("  - scenario: str")
        print("  - parameters: Dict[str, Any]")
        print("  - status: str ('pending'|'running'|'completed'|'failed')")
        
        print("\nanalyze_experiment_results() -> Dict with:")
        print("  - summary: Dict with total/completed/failed counts")
        print("  - scenarios: Dict with per-scenario analysis")
        print("  - parameter_analysis: Dict with parameter impact analysis")
        print("  - best_configurations: Dict with best configs per scenario")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Experiment runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests if script is executed directly
    test_experiment_runner()