#!/usr/bin/env python3
"""
Main data generation script for CSP synthetic datasets.
Orchestrates the complete pipeline from configuration to final data package.
"""

import os
import sys
import argparse
import yaml
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cspgen.scm import SCM
from cspgen.mnist_pool import MnistPool
from cspgen.imaging import generate_I_M, generate_I_Y, create_imaging_config
from cspgen.pack import export_pack
from cspgen.sanity import run_sanity_checks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate CSP synthetic datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python gen_dataset.py configs/mnist_dual.yaml
  
  # With overrides
  python gen_dataset.py --config configs/mnist_dual.yaml --seed-override 42 --n-override 2000
  
  # Only I^M images
  python gen_dataset.py configs/mnist_dual.yaml --im-only
  
  # Parameter overrides
  python gen_dataset.py configs/mnist_dual.yaml --set scm.rho=1.0 --set imaging.s_level=large
        """
    )
    
    # Main config file
    parser.add_argument("config", nargs="?", help="Path to YAML configuration file")
    parser.add_argument("--config", dest="config_file", help="Path to YAML configuration file (alternative)")
    
    # Quick overrides
    parser.add_argument("--seed-override", type=int, help="Override random seed")
    parser.add_argument("--n-override", type=int, help="Override number of samples")
    parser.add_argument("--output-override", help="Override output directory")
    
    # Image generation modes
    parser.add_argument("--im-only", action="store_true", help="Generate only I^M images")
    parser.add_argument("--iy-only", action="store_true", help="Generate only I^Y images")
    
    # General parameter overrides
    parser.add_argument("--set", action="append", dest="param_overrides", 
                       help="Override config parameters (e.g., --set scm.rho=0.5)")
    
    # Control flags
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    parser.add_argument("--no-sanity", action="store_true", help="Skip sanity checks")
    parser.add_argument("--dry-run", action="store_true", help="Parse config and print summary without generating data")
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial output")
    
    # Output options
    parser.add_argument("--png", action="store_true", default=True, help="Export PNG images (default)")
    parser.add_argument("--no-png", action="store_false", dest="png", help="Don't export PNG images")
    parser.add_argument("--npy-images", action="store_true", help="Also export images as .npy files")
    
    # Performance
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers (not implemented)")
    
    # Debug options
    parser.add_argument("--save-pairs", action="store_true", help="Save debug permutation pairs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Empty or invalid configuration file: {config_path}")
    
    return config


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply command line overrides to configuration."""
    # Create a copy to avoid modifying original
    config = config.copy()
    
    # Apply quick overrides
    if args.seed_override is not None:
        config["seed"] = args.seed_override
    
    if args.n_override is not None:
        config["n_samples"] = args.n_override
    
    if args.output_override is not None:
        if "output" not in config:
            config["output"] = {}
        config["output"]["dir"] = args.output_override
    
    # Apply image mode overrides
    if args.im_only:
        if "imaging" not in config:
            config["imaging"] = {}
        config["imaging"]["use_I_M"] = True
        config["imaging"]["use_I_Y"] = False
    
    if args.iy_only:
        if "imaging" not in config:
            config["imaging"] = {}
        config["imaging"]["use_I_M"] = False
        config["imaging"]["use_I_Y"] = True
    
    # Apply PNG settings
    if "output" not in config:
        config["output"] = {}
    config["output"]["save_png"] = args.png
    config["output"]["png_optimize"] = True
    config["output"]["save_npy_images"] = args.npy_images
    
    # Apply general parameter overrides
    if args.param_overrides:
        for override in args.param_overrides:
            if "=" not in override:
                raise ValueError(f"Invalid parameter override format: {override}. Use key=value")
            
            key, value = override.split("=", 1)
            
            # Parse value type
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif value.replace(".", "").replace("-", "").isdigit():
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            # Otherwise keep as string
            
            # Set nested key
            set_nested_key(config, key, value)
    
    return config


def set_nested_key(config: Dict[str, Any], key: str, value: Any):
    """Set a nested configuration key using dot notation."""
    keys = key.split(".")
    current = config
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    current[keys[-1]] = value


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_keys = ["n_samples"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate sample count
    if config["n_samples"] <= 0:
        raise ValueError(f"n_samples must be positive, got {config['n_samples']}")
    
    # Validate imaging configuration
    imaging_cfg = config.get("imaging", {})
    use_I_M = imaging_cfg.get("use_I_M", False)
    use_I_Y = imaging_cfg.get("use_I_Y", False)
    
    if not use_I_M and not use_I_Y:
        print("Warning: Neither I^M nor I^Y enabled. Generating tabular data only.")
    
    # Validate SCM configuration
    scm_cfg = config.get("scm", {})
    if "Y_type" in scm_cfg:
        if scm_cfg["Y_type"] not in ["cont", "bin"]:
            raise ValueError(f"Invalid Y_type: {scm_cfg['Y_type']}. Must be 'cont' or 'bin'")


def print_config_summary(config: Dict[str, Any], output_dir: str):
    """Print configuration summary."""
    print(f"\n=== Configuration Summary ===")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {config.get('seed', 'not set')}")
    print(f"Number of samples: {config['n_samples']}")
    print(f"Misc variables (q): {config.get('q_misc', 30)}")
    
    # SCM summary
    scm_cfg = config.get("scm", {})
    print(f"\nSCM configuration:")
    print(f"  Structure: α₁={scm_cfg.get('alpha1', 1.0)}, α₂={scm_cfg.get('alpha2', 1.0)}, ρ={scm_cfg.get('rho', 0.5)}, δ={scm_cfg.get('delta', 0.0)}")
    print(f"  Noise: σ_T={scm_cfg.get('sigma_T', 0.2)}, σ_M={scm_cfg.get('sigma_M', 0.2)}, σ_Y={scm_cfg.get('sigma_Y', 0.1)}")
    print(f"  Functions: h₁={scm_cfg.get('h1', 'square')}, h₂={scm_cfg.get('h2', 'tanh')}")
    print(f"  Y type: {scm_cfg.get('Y_type', 'cont')}")
    
    # Imaging summary
    imaging_cfg = config.get("imaging", {})
    use_I_M = imaging_cfg.get("use_I_M", False)
    use_I_Y = imaging_cfg.get("use_I_Y", False)
    
    print(f"\nImaging configuration:")
    print(f"  I^M enabled: {use_I_M}")
    print(f"  I^Y enabled: {use_I_Y}")
    
    if use_I_M or use_I_Y:
        print(f"  Strength level: {imaging_cfg.get('s_level', 'mid')}")
        print(f"  Transform params: θ={imaging_cfg.get('theta_deg', 25)}°, β={imaging_cfg.get('beta', 0.25)}, γ={imaging_cfg.get('gamma', 0.25)}")
        print(f"  Pixel noise: σ_pix={imaging_cfg.get('sigma_pix', 0.0)}")
        
        perm_M = imaging_cfg.get('perm_M', 0.0)
        perm_Y = imaging_cfg.get('perm_Y', 0.0)
        if perm_M > 0 or perm_Y > 0:
            print(f"  Permutations: I^M={perm_M}, I^Y={perm_Y}")
    
    # Splits summary
    splits_cfg = config.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    print(f"\nData splits: train={splits_cfg.get('train', 0.8)}, val={splits_cfg.get('val', 0.1)}, test={splits_cfg.get('test', 0.1)}")


def check_output_directory(output_dir: str, force: bool, resume: bool) -> bool:
    """Check output directory and handle existing files."""
    output_path = Path(output_dir)
    
    if output_path.exists():
        if resume:
            print(f"Resuming generation in existing directory: {output_dir}")
            return True
        elif force:
            print(f"Overwriting existing directory: {output_dir}")
            import shutil
            shutil.rmtree(output_path)
            output_path.mkdir(parents=True)
            return True
        else:
            print(f"Error: Output directory already exists: {output_dir}")
            print("Use --force to overwrite or --resume to continue")
            return False
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        return True


def generate_dataset(config: Dict[str, Any], output_dir: str, verbose: bool = False) -> bool:
    """Generate the complete dataset."""
    start_time = time.time()
    
    # Extract configuration
    seed = config.get("seed", 42)
    n_samples = config["n_samples"]
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Set seeds for reproducibility
    np.random.seed(seed)
    
    step_times = {}
    
    try:
        # Step 1: Initialize SCM
        print("\n=== Step 1: Initializing SCM ===")
        step_start = time.time()
        
        scm = SCM(config)
        step_times["scm_init"] = time.time() - step_start
        
        if verbose:
            expected = scm.get_expected_correlations()
            print(f"Expected correlation patterns: {expected}")
        
        # Step 2: Sample structural equations
        print("\n=== Step 2: Sampling structural equations ===")
        step_start = time.time()
        
        tab_core = scm.sample(n_samples, rng)
        print(f"Generated {n_samples} samples")
        print(f"Variables: {list(tab_core.keys())}")
        
        if verbose:
            for var_name, var_data in tab_core.items():
                if var_name != "W":  # W is 2D
                    print(f"  {var_name}: mean={np.mean(var_data):.4f}, std={np.std(var_data):.4f}")
                else:
                    print(f"  {var_name}: shape={var_data.shape}, mean={np.mean(var_data):.4f}")
        
        step_times["scm_sample"] = time.time() - step_start
        
        # Step 3: Compute semantic parameters
        print("\n=== Step 3: Computing semantic parameters ===")
        step_start = time.time()
        
        semantics = scm.compute_semantics(tab_core, config)
        print(f"Computed semantic parameters: {list(semantics.keys())}")
        
        if verbose:
            for sem_name, sem_data in semantics.items():
                print(f"  {sem_name}: range=[{np.min(sem_data):.4f}, {np.max(sem_data):.4f}], mean={np.mean(sem_data):.4f}")
        
        step_times["semantics"] = time.time() - step_start
        
        # Step 4: Initialize MNIST pool
        print("\n=== Step 4: Initializing MNIST pool ===")
        step_start = time.time()
        
        mnist_cfg = config.get("mnist", {})
        mnist_root = mnist_cfg.get("root", "~/.cache/csp_mnist")
        
        mnist_pool = MnistPool(root=mnist_root, rng=rng)
        print(f"MNIST pool ready with {mnist_pool.get_total_samples()} total images")
        
        if verbose:
            class_dist = mnist_pool.get_class_distribution()
            print(f"Class distribution: {class_dist}")
        
        step_times["mnist_init"] = time.time() - step_start
        
        # Step 5: Generate images
        imaging_cfg = config.get("imaging", {})
        use_I_M = imaging_cfg.get("use_I_M", False)
        use_I_Y = imaging_cfg.get("use_I_Y", False)
        
        I_M_result = None
        I_Y_result = None
        base_M = None
        base_Y = None
        
        if use_I_M:
            print("\n=== Step 5a: Generating I^M images ===")
            step_start = time.time()
            
            # Sample base images for I^M
            base_M = mnist_pool.sample_base(n_samples, class_uniform=True)
            print(f"Sampled {len(base_M['base_id'])} base images for I^M")
            
            # Create imaging configuration
            img_config = create_imaging_config(
                s_level=imaging_cfg.get("s_level", "mid"),
                theta_deg=imaging_cfg.get("theta_deg"),
                beta=imaging_cfg.get("beta"),
                gamma=imaging_cfg.get("gamma"),
                sigma_pix=imaging_cfg.get("sigma_pix", 0.0),
                perm_M=imaging_cfg.get("perm_M", 0.0)
            )
            
            if verbose:
                print(f"I^M imaging config: {img_config}")
            
            # Generate I^M images
            I_M_result = generate_I_M(
                base_M["image"],
                semantics["a_M"], 
                semantics["b_style"],
                img_config,
                rng
            )
            
            print(f"Generated I^M images: {I_M_result['image'].shape}")
            
            if verbose:
                applied = I_M_result["applied"]
                print(f"Applied transforms:")
                for param_name, param_values in applied.items():
                    if param_name != "permutation_applied" and isinstance(param_values, np.ndarray):
                        print(f"  {param_name}: mean={np.mean(param_values):.4f}, std={np.std(param_values):.4f}")
            
            step_times["imaging_M"] = time.time() - step_start
        
        if use_I_Y:
            print("\n=== Step 5b: Generating I^Y images ===")
            step_start = time.time()
            
            # Sample base images for I^Y
            base_Y = mnist_pool.sample_base(n_samples, class_uniform=True)
            print(f"Sampled {len(base_Y['base_id'])} base images for I^Y")
            
            # Create imaging configuration
            img_config = create_imaging_config(
                s_level=imaging_cfg.get("s_level", "mid"),
                theta_deg=imaging_cfg.get("theta_deg"),
                beta=imaging_cfg.get("beta"),
                gamma=imaging_cfg.get("gamma"),
                sigma_pix=imaging_cfg.get("sigma_pix", 0.0),
                perm_Y=imaging_cfg.get("perm_Y", 0.0)
            )
            
            # Generate I^Y images
            I_Y_result = generate_I_Y(
                base_Y["image"],
                semantics["a_Y"],
                semantics["b_style"],
                img_config,
                rng
            )
            
            print(f"Generated I^Y images: {I_Y_result['image'].shape}")
            
            step_times["imaging_Y"] = time.time() - step_start
        
        # Step 6: Package and export
        print("\n=== Step 6: Packaging and exporting ===")
        step_start = time.time()
        
        export_pack(
            out_dir=output_dir,
            tab_core=tab_core,
            I_M=I_M_result,
            I_Y=I_Y_result,
            base_M=base_M,
            base_Y=base_Y,
            cfg=config,
            splits=None,  # Let export_pack create splits
            rng=rng
        )
        
        step_times["export"] = time.time() - step_start
        
        # Report timing
        total_time = time.time() - start_time
        print(f"\n=== Generation completed in {total_time:.2f}s ===")
        
        if verbose:
            print("Step timings:")
            for step_name, step_time in step_times.items():
                print(f"  {step_name}: {step_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"\nError during dataset generation: {e}")
        if verbose:
            traceback.print_exc()
        return False


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine config file path
    config_path = args.config or args.config_file
    if not config_path:
        print("Error: No configuration file specified")
        print("Usage: python gen_dataset.py <config_file>")
        return 2
    
    try:
        # Load and process configuration
        print(f"Loading configuration from {config_path}")
        config = load_config(config_path)
        
        # Apply overrides
        config = apply_overrides(config, args)
        
        # Validate configuration
        validate_config(config)
        
        # Determine output directory
        output_cfg = config.get("output", {})
        output_dir = output_cfg.get("dir", f"./CSP-MNIST/cfg_{int(time.time())}")
        
        # Print configuration summary
        print_config_summary(config, output_dir)
        
        # Dry run mode
        if args.dry_run:
            print("\n=== Dry run completed ===")
            return 0
        
        # Check output directory
        if not check_output_directory(output_dir, args.force, args.resume):
            return 4
        
        # Generate dataset
        success = generate_dataset(config, output_dir, args.verbose)
        
        if not success:
            print("Dataset generation failed")
            return 3
        
        # Run sanity checks
        if not args.no_sanity:
            print("\n=== Running sanity checks ===")
            try:
                sanity_report_path = str(Path(output_dir) / "sanity_report.json")
                sanity_report = run_sanity_checks(output_dir, sanity_report_path)
                
                if sanity_report["summary"]["passed_all"]:
                    print("✓ All sanity checks passed")
                else:
                    print("⚠ Some sanity checks failed:")
                    print(f"  {sanity_report['summary']['notes']}")
                    if not args.force:
                        print("Use --force to ignore sanity check failures")
                        return 3
            
            except Exception as e:
                print(f"Warning: Sanity checks failed with error: {e}")
                if args.verbose:
                    traceback.print_exc()
                if not args.force:
                    return 3
        
        print(f"\n✓ Dataset generation completed successfully")
        print(f"Output directory: {output_dir}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 2
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())