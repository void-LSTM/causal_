"""
Data loading and preprocessing for CSP framework.
Supports CSP synthetic datasets with flexible scenario handling.
"""

from .datamodule import CSPDataModule, CSPDataset

__all__ = [
    'CSPDataModule',
    'CSPDataset'
]

# Data utilities
def load_csp_data(data_dir, scenario=None, **kwargs):
    """
    Convenience function to load CSP data.
    
    Args:
        data_dir: Path to CSP dataset directory
        scenario: Force scenario type ('IM', 'IY', 'DUAL')
        **kwargs: Additional datamodule arguments
        
    Returns:
        CSPDataModule instance
    """
    return CSPDataModule(
        data_dir=data_dir,
        scenario=scenario,
        **kwargs
    )

def get_feature_dims_from_data(data_dir):
    """
    Extract feature dimensions from dataset metadata.
    
    Args:
        data_dir: Path to CSP dataset directory
        
    Returns:
        Dictionary of feature dimensions
    """
    # Create temporary dataset to get dimensions
    temp_dataset = CSPDataset(
        data_dir=data_dir,
        split='train',
        max_samples=1
    )
    
    return temp_dataset.get_feature_dims()