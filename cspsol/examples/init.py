"""
Example scripts and tutorials for CSP framework.
Demonstrates usage patterns and best practices.
"""

# Import example functions when available
try:
    from .complete_example import run_complete_example
    __all__ = ['run_complete_example']
except ImportError:
    __all__ = []

def list_examples():
    """List available example scripts."""
    examples = [
        'complete_example.py - Full CARL training and evaluation pipeline',
        'quick_start.py - Simple getting started example (coming soon)',
        'advanced_config.py - Advanced configuration examples (coming soon)'
    ]
    
    print("Available CSP examples:")
    for example in examples:
        print(f"  - {example}")
    
    return examples