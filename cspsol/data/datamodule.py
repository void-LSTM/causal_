"""
CSP Data Module for loading and processing CSP synthetic datasets.
Supports both I^M (image as mediator) and I^Y (image as result) scenarios.
Designed for seamless integration with CSPBench evaluation framework.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from sklearn.preprocessing import StandardScaler


class CSPDataset(Dataset):
    """
    Dataset class for CSP synthetic data.
    
    Supports flexible data loading for:
    - Tabular data (T, M, Y_star, W, semantic params)
    - Images (I^M and/or I^Y)
    - Scenario detection (IM/IY/DUAL)
    - Extensible to high-dimensional tables + multi-images
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 scenario: Optional[str] = None,
                 normalize_tabular: bool = True,
                 image_transform: Optional[callable] = None,
                 max_samples: Optional[int] = None,
                 shared_scaler: Optional[StandardScaler] = None,
                 shared_valid_cols: Optional[List[int]] = None):
        """
        Initialize CSP dataset.
        
        Args:
            data_dir: Path to CSP dataset directory
            split: Data split ('train', 'val', 'test')
            scenario: Force scenario ('IM', 'IY', 'DUAL', None for auto-detect)
            normalize_tabular: Whether to normalize tabular features
            image_transform: Optional image transforms
            max_samples: Limit number of samples (for debugging)
            shared_scaler: Pre-fitted scaler for consistent normalization
            shared_valid_cols: Valid column indices for shared scaler
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize_tabular = normalize_tabular
        self.image_transform = image_transform
        self.shared_scaler = shared_scaler
        self.shared_valid_cols = shared_valid_cols
        
        # Load metadata and validate
        self._load_metadata()
        self._load_splits()
        self._detect_scenario(scenario)
        self._load_tabular_data()
        self._setup_image_paths()
        
        # Apply sample limit if specified
        if max_samples is not None:
            self.split_indices = self.split_indices[:max_samples]
        
        print(f"CSPDataset initialized: {len(self)} samples, scenario={self.scenario}, split={split}")
    
    def _load_metadata(self):
        """Load and parse metadata."""
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        # Extract key info
        self.n_total_samples = self.meta.get('n_samples', 0)
        self.use_I_M = self.meta.get('imaging', {}).get('use_I_M', False)
        self.use_I_Y = self.meta.get('imaging', {}).get('use_I_Y', False)
        
        # Extract data structure info
        self.data_structure = self.meta.get('data_structure', {})
        self.float_columns = self.data_structure.get('float_columns', [])
        self.int_columns = self.data_structure.get('int_columns', [])
    
    def _load_splits(self):
        """Load train/val/test splits."""
        splits_path = self.data_dir / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(f"Splits not found: {splits_path}")
        
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        
        if self.split not in splits:
            raise ValueError(f"Split '{self.split}' not found. Available: {list(splits.keys())}")
        
        self.split_indices = splits[self.split]
        print(f"Loaded {len(self.split_indices)} samples for split '{self.split}'")
    
    def _detect_scenario(self, force_scenario: Optional[str]):
        """Detect or set scenario type."""
        if force_scenario is not None:
            if force_scenario not in ['IM', 'IY', 'DUAL']:
                raise ValueError(f"Invalid scenario: {force_scenario}")
            self.scenario = force_scenario
        else:
            # Auto-detect based on available images
            if self.use_I_M and self.use_I_Y:
                self.scenario = 'DUAL'
            elif self.use_I_M:
                self.scenario = 'IM'
            elif self.use_I_Y:
                self.scenario = 'IY'
            else:
                self.scenario = 'TABULAR_ONLY'
        
        print(f"Scenario detected/set: {self.scenario}")
    
    def _load_tabular_data(self):
        """Load and process tabular data."""
        tab_path = self.data_dir / "tab.npz"
        if not tab_path.exists():
            raise FileNotFoundError(f"Tabular data not found: {tab_path}")
        
        # Load NPZ file
        tab_data = np.load(tab_path)
        
        # Extract arrays
        self.float_data = tab_data['float']  # (n_samples, n_float_cols)
        self.int_data = tab_data['int']      # (n_samples, n_int_cols)
        
        # Verify column consistency
        stored_float_cols = list(tab_data.get('columns_float', []))
        stored_int_cols = list(tab_data.get('columns_int', []))
        
        if stored_float_cols != self.float_columns:
            warnings.warn("Float column mismatch between metadata and data")
        if stored_int_cols != self.int_columns:
            warnings.warn("Int column mismatch between metadata and data")
        
        # Create column mappings for easy access
        self.float_col_map = {col: i for i, col in enumerate(self.float_columns)}
        self.int_col_map = {col: i for i, col in enumerate(self.int_columns)}
        
        # Setup normalization for tabular data if requested
        if self.normalize_tabular:
            self._setup_normalization()
        
        print(f"Tabular data loaded: {self.float_data.shape} float, {self.int_data.shape} int")
    
    def _setup_normalization(self):
        """Setup feature normalization using training split statistics."""
        if self.shared_scaler is not None and self.shared_valid_cols is not None:
            # Use provided scaler and valid columns
            self.float_scaler = self.shared_scaler
            self.valid_float_cols = self.shared_valid_cols
            
            # Apply normalization to valid columns only
            if self.valid_float_cols:
                all_data_valid = self.float_data[:, self.valid_float_cols]
                normalized_valid = self.float_scaler.transform(all_data_valid)
                
                # Create full normalized array (copy original, replace valid columns)
                self.float_data_normalized = self.float_data.copy()
                self.float_data_normalized[:, self.valid_float_cols] = normalized_valid
            else:
                self.float_data_normalized = self.float_data.copy()
            
            print("Using shared scaler for normalization")
            return
        
        # Get training indices for computing statistics
        splits_path = self.data_dir / "splits.json"
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        train_indices = splits['train']
        
        if len(train_indices) == 0:
            warnings.warn("No training samples found for normalization, using all data")
            train_indices = list(range(len(self.float_data)))
        
        # Compute normalization statistics on training data
        train_float_data = self.float_data[train_indices]
        
        # Filter out constant columns and NaN columns to avoid sklearn warnings
        valid_cols = []
        for i in range(train_float_data.shape[1]):
            col_data = train_float_data[:, i]
            # Check if column is not all NaN and has some variation
            if not np.all(np.isnan(col_data)) and np.nanstd(col_data) > 1e-8:
                valid_cols.append(i)
            else:
                col_name = self.float_columns[i] if i < len(self.float_columns) else f"col_{i}"
                print(f"Warning: Skipping constant/NaN column {col_name}")
        
        # Initialize scaler with robust parameters
        self.float_scaler = StandardScaler(with_mean=True, with_std=True)
        
        # Fit scaler on valid columns only
        if valid_cols:
            self.valid_float_cols = valid_cols
            train_data_valid = train_float_data[:, valid_cols]
            self.float_scaler.fit(train_data_valid)
            
            # Apply normalization to all data (valid columns only)
            all_data_valid = self.float_data[:, valid_cols]
            normalized_valid = self.float_scaler.transform(all_data_valid)
            
            # Create full normalized array (copy original, replace valid columns)
            self.float_data_normalized = self.float_data.copy()
            self.float_data_normalized[:, valid_cols] = normalized_valid
        else:
            warnings.warn("No valid columns found for normalization, using original data")
            self.float_data_normalized = self.float_data.copy()
            self.valid_float_cols = []
        
        print("Feature normalization setup completed")
    
    def _setup_image_paths(self):
        """Setup image file paths and verify existence."""
        self.img_M_dir = self.data_dir / "img_M" if self.use_I_M else None
        self.img_Y_dir = self.data_dir / "img_Y" if self.use_I_Y else None
        
        # Verify image directories exist
        if self.use_I_M and not self.img_M_dir.exists():
            raise FileNotFoundError(f"I^M image directory not found: {self.img_M_dir}")
        if self.use_I_Y and not self.img_Y_dir.exists():
            raise FileNotFoundError(f"I^Y image directory not found: {self.img_Y_dir}")
        
        # Count available images
        if self.use_I_M:
            img_M_files = list(self.img_M_dir.glob("*.png"))
            print(f"Found {len(img_M_files)} I^M images")
        
        if self.use_I_Y:
            img_Y_files = list(self.img_Y_dir.glob("*.png"))
            print(f"Found {len(img_Y_files)} I^Y images")
    
    def _get_float_column(self, col_name: str, indices: List[int]) -> np.ndarray:
        """Get float column data for given indices."""
        if col_name not in self.float_col_map:
            raise KeyError(f"Float column '{col_name}' not found")
        
        col_idx = self.float_col_map[col_name]
        if self.normalize_tabular:
            return self.float_data_normalized[indices, col_idx]
        else:
            return self.float_data[indices, col_idx]
    
    def _get_int_column(self, col_name: str, indices: List[int]) -> np.ndarray:
        """Get int column data for given indices."""
        if col_name not in self.int_col_map:
            raise KeyError(f"Int column '{col_name}' not found")
        
        col_idx = self.int_col_map[col_name]
        return self.int_data[indices, col_idx]
    
    def _load_image(self, img_dir: Path, sample_idx: int) -> np.ndarray:
        """Load single image and convert to array."""
        img_path = img_dir / f"{sample_idx:06d}.png"
        
        if not img_path.exists():
            # Return black image if file missing
            warnings.warn(f"Image not found: {img_path}")
            return np.zeros((28, 28), dtype=np.float32)
        
        # Load image
        img = Image.open(img_path).convert('L')  # Grayscale
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        
        # Apply transforms if specified
        if self.image_transform is not None:
            img_array = self.image_transform(img_array)
        
        return img_array
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.split_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample.
        
        Returns:
            Dictionary containing tensors for T, M, Y_star, images, and metadata
        """
        # Get global sample index
        sample_idx = self.split_indices[idx]
        
        # Extract tabular data
        sample = {}
        
        # Core variables: T, M, Y_star
        sample['T'] = torch.tensor(self._get_float_column('T', [sample_idx])[0], dtype=torch.float32)
        sample['M'] = torch.tensor(self._get_float_column('M', [sample_idx])[0], dtype=torch.float32)
        sample['Y_star'] = torch.tensor(self._get_float_column('Y_star', [sample_idx])[0], dtype=torch.float32)
        
        # Semantic parameters
        for param in ['a_M', 'a_Y', 'b_style']:
            if param in self.float_col_map:
                sample[param] = torch.tensor(self._get_float_column(param, [sample_idx])[0], dtype=torch.float32)
            else:
                sample[param] = torch.tensor(0.0, dtype=torch.float32)  # Default value
        
        # Class and subject IDs
        for id_col in ['subject_id', 'class_id_M', 'class_id_Y']:
            if id_col in self.int_col_map:
                sample[id_col] = torch.tensor(self._get_int_column(id_col, [sample_idx])[0], dtype=torch.long)
            else:
                sample[id_col] = torch.tensor(-1, dtype=torch.long)  # Missing marker
        
        # Load images
        sample['I_M'] = None
        sample['I_Y'] = None
        
        if self.use_I_M and self.img_M_dir is not None:
            img_M = self._load_image(self.img_M_dir, sample_idx)
            sample['I_M'] = torch.tensor(img_M[None, :, :], dtype=torch.float32)  # Add channel dim
        
        if self.use_I_Y and self.img_Y_dir is not None:
            img_Y = self._load_image(self.img_Y_dir, sample_idx)
            sample['I_Y'] = torch.tensor(img_Y[None, :, :], dtype=torch.float32)  # Add channel dim
        
        # Add scenario info
        sample['scenario'] = self.scenario
        
        return sample
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for model initialization."""
        dims = {
            'T_dim': 1,  # T is scalar
            'M_dim': 1,  # M is scalar
            'Y_dim': 1,  # Y_star is scalar
            'img_channels': 1,
            'img_height': 28,
            'img_width': 28,
        }
        
        # Add W dimensions if present
        W_cols = [col for col in self.float_columns if col.startswith('W_')]
        dims['W_dim'] = len(W_cols)
        
        return dims
    
    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """Compute class weights for balanced training."""
        weights = {}
        
        # Get training split indices
        splits_path = self.data_dir / "splits.json"
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        train_indices = splits['train']
        
        if len(train_indices) == 0:
            return weights
        
        # Compute class weights for M and Y if available
        for class_col in ['class_id_M', 'class_id_Y']:
            if class_col in self.int_col_map:
                class_ids = self._get_int_column(class_col, train_indices)
                valid_mask = class_ids >= 0  # Exclude missing values (-1)
                
                if np.any(valid_mask):
                    valid_classes = class_ids[valid_mask]
                    unique_classes, counts = np.unique(valid_classes, return_counts=True)
                    
                    # Compute inverse frequency weights
                    weights_array = 1.0 / counts
                    weights_array = weights_array / weights_array.sum() * len(unique_classes)
                    
                    # Create weight tensor
                    weight_tensor = torch.zeros(max(unique_classes) + 1)
                    weight_tensor[unique_classes] = torch.tensor(weights_array, dtype=torch.float32)
                    weights[class_col] = weight_tensor
        
        return weights


class CSPDataModule:
    """
    Data module for CSP datasets.
    Handles train/val/test splits and provides DataLoaders.
    """
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 scenario: Optional[str] = None,
                 normalize_tabular: bool = True,
                 max_samples: Optional[int] = None,
                 val_split: float = 0.2):
        """
        Initialize CSP data module.
        
        Args:
            data_dir: Path to CSP dataset directory
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            scenario: Force scenario type
            normalize_tabular: Whether to normalize tabular features
            max_samples: Limit samples per split (for debugging)
            val_split: Fraction of training data to use as validation if val set is empty
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scenario = scenario
        self.normalize_tabular = normalize_tabular
        self.max_samples = max_samples
        self.val_split = val_split
        
        # Initialize datasets
        self.setup()
    
    def setup(self):
        """Setup train/val/test datasets."""
        # Check if validation set is empty and handle it
        splits_path = Path(self.data_dir) / "splits.json"
        with open(splits_path, 'r') as f:
            original_splits = json.load(f)
        
        # Handle empty validation set
        if len(original_splits.get('val', [])) == 0:
            print(f"Warning: Validation set is empty. Creating validation set from {self.val_split:.1%} of training data")
            
            train_indices = original_splits['train']
            np.random.seed(42)  # For reproducible split
            train_indices_copy = train_indices.copy()
            np.random.shuffle(train_indices_copy)
            
            val_size = int(len(train_indices_copy) * self.val_split)
            new_val_indices = train_indices_copy[:val_size]
            new_train_indices = train_indices_copy[val_size:]
            
            # Create modified splits
            self._use_temp_splits = True
            self._temp_splits = {
                'train': new_train_indices,
                'val': new_val_indices,
                'test': original_splits['test']
            }
        else:
            self._use_temp_splits = False
        
        # First create a training dataset to get the scaler and valid columns
        if self._use_temp_splits:
            # Create temporary dataset with modified train split
            temp_train = CSPDataset(
                data_dir=self.data_dir,
                split='train',  # Will load original splits first
                scenario=self.scenario,
                normalize_tabular=self.normalize_tabular,
                max_samples=None
            )
            # Override with new training indices
            temp_train.split_indices = self._temp_splits['train']
            
            # Re-run normalization with correct indices
            if self.normalize_tabular:
                temp_train._setup_normalization()
        else:
            temp_train = CSPDataset(
                data_dir=self.data_dir,
                split='train',
                scenario=self.scenario,
                normalize_tabular=self.normalize_tabular,
                max_samples=None
            )
        
        # Get the fitted scaler and valid columns
        fitted_scaler = temp_train.float_scaler if self.normalize_tabular else None
        valid_cols = getattr(temp_train, 'valid_float_cols', None) if self.normalize_tabular else None
        
        # Create datasets with shared scaler and valid columns
        common_kwargs = {
            'data_dir': self.data_dir,
            'scenario': self.scenario,
            'normalize_tabular': self.normalize_tabular,
            'max_samples': self.max_samples,
            'shared_scaler': fitted_scaler,
            'shared_valid_cols': valid_cols
        }
        
        # Create final datasets
        if self._use_temp_splits:
            self.train_dataset = self._create_dataset_with_custom_split('train', **common_kwargs)
            self.val_dataset = self._create_dataset_with_custom_split('val', **common_kwargs)
            self.test_dataset = CSPDataset(split='test', **common_kwargs)
        else:
            self.train_dataset = CSPDataset(split='train', **common_kwargs)
            self.val_dataset = CSPDataset(split='val', **common_kwargs)
            self.test_dataset = CSPDataset(split='test', **common_kwargs)
        
        # Store scenario from first dataset
        self.detected_scenario = self.train_dataset.scenario
        
        print(f"Data module setup complete. Scenario: {self.detected_scenario}")
        print(f"Final split sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def _create_dataset_with_custom_split(self, split_name: str, **kwargs):
        """Create dataset with custom split indices."""
        # Create dataset normally first
        dataset = CSPDataset(split='train', **kwargs)  # Use train to load all data
        
        # Override split indices
        dataset.split = split_name
        dataset.split_indices = self._temp_splits[split_name]
        
        # Apply max_samples limit if specified
        if self.max_samples is not None:
            dataset.split_indices = dataset.split_indices[:self.max_samples]
        
        return dataset
    
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        collated = {}
        
        # Get all keys from first sample
        sample_keys = batch[0].keys()
        
        for key in sample_keys:
            if key == 'scenario':
                # Keep scenario as string
                collated[key] = batch[0][key]
            elif any(batch[i][key] is None for i in range(len(batch))):
                # Handle None values (e.g., missing images)
                non_none_items = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
                if non_none_items:
                    collated[key] = torch.stack(non_none_items)
                else:
                    collated[key] = None
            else:
                # Stack tensors normally
                collated[key] = torch.stack([batch[i][key] for i in range(len(batch))])
        
        return collated
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions from training dataset."""
        return self.train_dataset.get_feature_dims()
    
    def get_class_weights(self) -> Dict[str, torch.Tensor]:
        """Get class weights from training dataset.""" 
        return self.train_dataset.get_class_weights()


# # Test the implementation
# if __name__ == "__main__":
#     print("=== Testing CSP Data Module (Fixed Version) ===")
    
#     # Test with CSP-MNIST dataset
#     data_dir = "csp_synth/CSP-MNIST/cfg_dual_42"
    
#     print(f"\n--- Testing CSPDataset ---")
#     try:
#         # Test individual dataset
#         dataset = CSPDataset(
#             data_dir=data_dir,
#             split='train',
#             scenario=None,  # Auto-detect
#             normalize_tabular=True,
#             max_samples=10  # Small sample for testing
#         )
        
#         print(f"Dataset length: {len(dataset)}")
#         print(f"Scenario: {dataset.scenario}")
#         print(f"Feature dims: {dataset.get_feature_dims()}")
        
#         # Test single sample
#         sample = dataset[0]
#         print(f"\nSample keys: {list(sample.keys())}")
        
#         for key, value in sample.items():
#             if isinstance(value, torch.Tensor):
#                 print(f"{key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.4f}, {value.max():.4f}]")
#             else:
#                 print(f"{key}: {value}")
        
#     except Exception as e:
#         print(f"Dataset test failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print(f"\n--- Testing CSPDataModule ---")
#     try:
#         # Test data module with validation split handling
#         datamodule = CSPDataModule(
#             data_dir=data_dir,
#             batch_size=4,
#             num_workers=0,  # No multiprocessing for testing
#             max_samples=10,
#             val_split=0.2  # Use 20% of training for validation
#         )
        
#         print(f"Detected scenario: {datamodule.detected_scenario}")
#         print(f"Feature dims: {datamodule.get_feature_dims()}")
        
#         # Test dataloaders
#         train_loader = datamodule.train_dataloader()
#         val_loader = datamodule.val_dataloader()
#         test_loader = datamodule.test_dataloader()
        
#         print(f"Train loader: {len(train_loader)} batches")
#         print(f"Val loader: {len(val_loader)} batches") 
#         print(f"Test loader: {len(test_loader)} batches")
        
#         # Test single batch
#         batch = next(iter(train_loader))
#         print(f"\nBatch keys: {list(batch.keys())}")
        
#         for key, value in batch.items():
#             if isinstance(value, torch.Tensor):
#                 print(f"{key}: shape={value.shape}, dtype={value.dtype}")
#             else:
#                 print(f"{key}: {value}")
        
#         print("\n✓ All data module tests passed!")
        
#     except Exception as e:
#         print(f"DataModule test failed: {e}")
#         import traceback
#         traceback.print_exc()
    
#     print("\n=== CSP Data Module Test Complete ===")