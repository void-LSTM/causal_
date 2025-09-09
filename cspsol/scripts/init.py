"""
Command-line scripts for CSP framework.
Provides entry points for training, evaluation, and experimentation.
"""

# Scripts will be imported when implemented
__all__ = []

def list_scripts():
    """List available command-line scripts."""
    scripts = [
        'train.py - Main training script',
        'eval.py - Model evaluation script', 
        'run_experiment.py - Experiment runner with configuration management'
    ]
    
    print("Available CSP scripts:")
    for script in scripts:
        print(f"  - {script}")
    
    return scripts