"""
Test script for CSP configuration templates.
Validates that all YAML configurations load correctly and contain expected values.
"""

import yaml
import os
from pathlib import Path


def test_config_loading():
    """Test loading and parsing of all configuration files."""
    config_dir = Path(__file__).parent
    config_files = ['im_main.yaml', 'iy_main.yaml', 'dual.yaml']
    
    results = {}
    
    for config_file in config_files:
        config_path = config_dir / config_file
        
        try:
            # Load YAML configuration
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate basic structure
            required_sections = ['data', 'model', 'training', 'evaluation']
            missing_sections = [section for section in required_sections 
                              if section not in config]
            
            # Test scenario-specific validations
            test_config = config.get('_test_config', {})
            scenario = config['model']['scenario']
            z_dim = config['model']['z_dim']
            enabled_losses = [name for name, cfg in config['model']['loss_config'].items() 
                            if cfg.get('enabled', False)]
            
            # Validation results
            validation_results = {
                'file_exists': True,
                'yaml_valid': True,
                'missing_sections': missing_sections,
                'scenario_correct': scenario == test_config.get('expected_scenario'),
                'z_dim_correct': z_dim == test_config.get('expected_z_dim'),
                'losses_match': set(enabled_losses) == set(test_config.get('expected_losses', [])),
                'config_complete': len(missing_sections) == 0,
                'loaded_config': config
            }
            
            results[config_file] = validation_results
            
        except FileNotFoundError:
            results[config_file] = {
                'file_exists': False,
                'error': f"Configuration file {config_file} not found"
            }
        except yaml.YAMLError as e:
            results[config_file] = {
                'file_exists': True,
                'yaml_valid': False,
                'error': f"YAML parsing error: {e}"
            }
        except Exception as e:
            results[config_file] = {
                'file_exists': True,
                'yaml_valid': True,
                'error': f"Validation error: {e}"
            }
    
    return results


def test_config_compatibility():
    """Test compatibility with ConfigManager."""
    try:
        # This would normally import from our framework
        # For testing, we'll simulate the structure
        config_dir = Path(__file__).parent
        
        compatibility_results = {}
        
        for config_file in ['im_main.yaml', 'iy_main.yaml', 'dual.yaml']:
            config_path = config_dir / config_file
            
            if not config_path.exists():
                compatibility_results[config_file] = {'compatible': False, 'error': 'File not found'}
                continue
                
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Test expected structure for ConfigManager compatibility
            compatibility_checks = {
                'has_data_section': 'data' in config,
                'has_model_section': 'model' in config,
                'has_training_section': 'training' in config,
                'has_evaluation_section': 'evaluation' in config,
                'scenario_valid': config.get('model', {}).get('scenario') in ['IM', 'IY', 'DUAL'],
                'loss_config_valid': isinstance(config.get('model', {}).get('loss_config'), dict),
                'optimizer_valid': config.get('training', {}).get('optimizer') in ['adam', 'adamw', 'sgd']
            }
            
            all_compatible = all(compatibility_checks.values())
            
            compatibility_results[config_file] = {
                'compatible': all_compatible,
                'checks': compatibility_checks,
                'config_keys': list(config.keys())
            }
        
        return compatibility_results
        
    except Exception as e:
        return {'error': f"Compatibility test failed: {e}"}


def print_test_results():
    """Print comprehensive test results."""
    print("="*60)
    print("CSP Configuration Template Test Results")
    print("="*60)
    
    # Test 1: Configuration loading
    print("\n1. Configuration Loading Test:")
    print("-" * 40)
    
    loading_results = test_config_loading()
    
    for config_file, results in loading_results.items():
        print(f"\n{config_file}:")
        
        if not results.get('file_exists', False):
            print(f"  ❌ File not found: {results.get('error', 'Unknown error')}")
            continue
            
        if not results.get('yaml_valid', False):
            print(f"  ❌ YAML invalid: {results.get('error', 'Unknown error')}")
            continue
        
        # Print validation results
        print(f"  ✅ File exists and YAML is valid")
        print(f"  📋 Scenario: {results['loaded_config']['model']['scenario']}")
        print(f"  🧠 Z-dimension: {results['loaded_config']['model']['z_dim']}")
        print(f"  🔧 Enabled losses: {[name for name, cfg in results['loaded_config']['model']['loss_config'].items() if cfg.get('enabled', False)]}")
        
        if results.get('missing_sections'):
            print(f"  ⚠️  Missing sections: {results['missing_sections']}")
        else:
            print(f"  ✅ All required sections present")
        
        if results.get('error'):
            print(f"  ❌ Validation error: {results['error']}")
    
    # Test 2: Compatibility test
    print(f"\n2. ConfigManager Compatibility Test:")
    print("-" * 40)
    
    compatibility_results = test_config_compatibility()
    
    if 'error' in compatibility_results:
        print(f"❌ Compatibility test failed: {compatibility_results['error']}")
    else:
        for config_file, results in compatibility_results.items():
            print(f"\n{config_file}:")
            
            if results['compatible']:
                print(f"  ✅ Compatible with ConfigManager")
            else:
                print(f"  ❌ Compatibility issues found")
                
            for check, passed in results['checks'].items():
                status = "✅" if passed else "❌"
                print(f"    {status} {check}")
    
    # Test 3: Configuration summary
    print(f"\n3. Configuration Summary:")
    print("-" * 40)
    
    for config_file in ['im_main.yaml', 'iy_main.yaml', 'dual.yaml']:
        if config_file in loading_results and loading_results[config_file].get('yaml_valid'):
            config = loading_results[config_file]['loaded_config']
            print(f"\n{config_file}:")
            print(f"  📝 Description: {config.get('description', 'N/A')}")
            print(f"  🎯 Scenario: {config['model']['scenario']}")
            print(f"  📊 Batch size: {config['data']['batch_size']}")
            print(f"  🔄 Max epochs: {config['training']['max_epochs']}")
            print(f"  📈 Learning rate: {config['training']['learning_rate']}")
            print(f"  💾 Output dir: {config.get('output_dir', 'N/A')}")
    
    print(f"\n{'='*60}")
    print("Test completed! Check results above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    print_test_results()