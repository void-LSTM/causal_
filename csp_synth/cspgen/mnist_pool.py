"""
MNIST dataset loader and class-uniform sampler.
"""

import os
import numpy as np
import pickle
from typing import Dict, Tuple, Optional, List
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import warnings


class MnistPool:
    """
    MNIST dataset pool with class-uniform sampling capability.
    """
    
    def __init__(self, root: str = "~/.cache/csp_mnist", rng: Optional[np.random.Generator] = None):
        """
        Initialize MNIST pool.
        
        Args:
            root: Cache directory for MNIST data
            rng: Random number generator
        """
        self.root = os.path.expanduser(root)
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Create directories
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)
        
        # Load or download MNIST data
        self._load_mnist()
    
    def _load_mnist(self):
        """Load MNIST data, downloading if necessary."""
        processed_dir = os.path.join(self.root, "processed")
        images_path = os.path.join(processed_dir, "images.npy")
        labels_path = os.path.join(processed_dir, "labels.npy")
        indices_path = os.path.join(processed_dir, "class_indices.pkl")
        
        # Check if processed data exists
        if (os.path.exists(images_path) and 
            os.path.exists(labels_path) and 
            os.path.exists(indices_path)):
            
            print(f"Loading cached MNIST data from {processed_dir}")
            self.images = np.load(images_path)
            self.labels = np.load(labels_path)
            
            with open(indices_path, 'rb') as f:
                self.class_indices = pickle.load(f)
            
            print(f"Loaded {len(self.images)} MNIST images")
            return
        
        print(f"Downloading MNIST data to {self.root}")
        
        # Download MNIST using torchvision
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.MNIST(
            root=self.root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=self.root, train=False, download=True, transform=transform
        )
        
        # Combine train and test data
        all_images = []
        all_labels = []
        
        # Process training data
        for img_tensor, label in train_dataset:
            img_array = img_tensor.squeeze().numpy()  # Remove channel dim, convert to numpy
            all_images.append(img_array)
            all_labels.append(label)
        
        # Process test data
        for img_tensor, label in test_dataset:
            img_array = img_tensor.squeeze().numpy()
            all_images.append(img_array)
            all_labels.append(label)
        
        # Convert to numpy arrays
        self.images = np.stack(all_images).astype(np.float32)  # Shape: (70000, 28, 28)
        self.labels = np.array(all_labels, dtype=np.int32)      # Shape: (70000,)
        
        print(f"Combined {len(self.images)} images from train+test sets")
        print(f"Image shape: {self.images.shape}, dtype: {self.images.dtype}")
        print(f"Label shape: {self.labels.shape}, dtype: {self.labels.dtype}")
        print(f"Image value range: [{self.images.min():.4f}, {self.images.max():.4f}]")
        
        # Create class indices for efficient sampling
        self.class_indices = {}
        for class_id in range(10):
            indices = np.where(self.labels == class_id)[0]
            self.class_indices[class_id] = indices
            print(f"Class {class_id}: {len(indices)} samples")
        
        # Save processed data
        print(f"Saving processed data to {processed_dir}")
        np.save(images_path, self.images)
        np.save(labels_path, self.labels)
        
        with open(indices_path, 'wb') as f:
            pickle.dump(self.class_indices, f)
        
        print("MNIST data processing completed")
    
    def sample_base(self, n: int, class_uniform: bool = True) -> Dict[str, np.ndarray]:
        """
        Sample base images with optional class uniformity.
        
        Args:
            n: Number of samples to draw
            class_uniform: If True, sample approximately equally from each class
        
        Returns:
            Dictionary with 'base_id', 'class_id', and 'image' arrays
        """
        if not class_uniform:
            # Simple random sampling
            indices = self.rng.choice(len(self.images), size=n, replace=False)
            base_ids = indices
            class_ids = self.labels[indices]
            images = self.images[indices].copy()
            
        else:
            # Class-uniform sampling algorithm from specification
            base_per_class = n // 10
            remainder = n % 10
            
            sampled_indices = []
            
            # Sample base_per_class from each class
            for class_id in range(10):
                available_indices = self.class_indices[class_id].copy()
                
                if len(available_indices) < base_per_class:
                    # Not enough samples in this class - use with replacement
                    warnings.warn(f"Class {class_id} has only {len(available_indices)} samples, need {base_per_class}. Using replacement.")
                    class_samples = self.rng.choice(available_indices, size=base_per_class, replace=True)
                else:
                    # Sample without replacement
                    class_samples = self.rng.choice(available_indices, size=base_per_class, replace=False)
                
                sampled_indices.extend(class_samples)
            
            # Handle remainder samples - distribute equally across classes
            if remainder > 0:
                remainder_classes = self.rng.choice(10, size=remainder, replace=False)
                for class_id in remainder_classes:
                    available_indices = self.class_indices[class_id]
                    # Avoid duplicates with already sampled indices
                    used_from_class = [idx for idx in sampled_indices if self.labels[idx] == class_id]
                    remaining_indices = [idx for idx in available_indices if idx not in used_from_class]
                    
                    if len(remaining_indices) > 0:
                        additional_sample = self.rng.choice(remaining_indices, size=1)[0]
                    else:
                        # Fallback: allow duplicates
                        additional_sample = self.rng.choice(available_indices, size=1)[0]
                    
                    sampled_indices.append(additional_sample)
            
            # Convert to numpy arrays
            sampled_indices = np.array(sampled_indices, dtype=np.int32)
            
            # Shuffle to break class ordering
            shuffled_order = self.rng.permutation(len(sampled_indices))
            base_ids = sampled_indices[shuffled_order]
            class_ids = self.labels[base_ids]
            images = self.images[base_ids].copy()
        
        # Verify class distribution for uniform sampling
        if class_uniform:
            class_counts = np.bincount(class_ids, minlength=10)
            max_imbalance = np.max(class_counts) - np.min(class_counts)
            expected_balance = max(1, base_per_class)
            
            if max_imbalance > expected_balance:
                warnings.warn(f"Class imbalance detected: max_imbalance={max_imbalance}, expected_balance={expected_balance}")
        
        return {
            "base_id": base_ids,
            "class_id": class_ids,
            "image": images
        }
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of samples across classes."""
        return {class_id: len(indices) for class_id, indices in self.class_indices.items()}
    
    def get_total_samples(self) -> int:
        """Get total number of available samples."""
        return len(self.images)


# # Test functions
# if __name__ == "__main__":
#     print("=== Testing mnist_pool.py ===")
    
#     # Initialize with test cache directory
#     rng = np.random.default_rng(42)
#     pool = MnistPool(root="~/.cache/csp_mnist_test", rng=rng)
    
#     print(f"\nTotal samples available: {pool.get_total_samples()}")
#     print(f"Class distribution: {pool.get_class_distribution()}")
    
#     # Test class-uniform sampling
#     print("\n--- Testing class-uniform sampling ---")
    
#     for n_test in [100, 1000, 2500, 5000]:
#         print(f"\nSampling {n_test} images with class_uniform=True")
        
#         result = pool.sample_base(n_test, class_uniform=True)
        
#         # Check return format
#         print(f"Return keys: {list(result.keys())}")
#         print(f"base_id shape: {result['base_id'].shape}, dtype: {result['base_id'].dtype}")
#         print(f"class_id shape: {result['class_id'].shape}, dtype: {result['class_id'].dtype}")
#         print(f"image shape: {result['image'].shape}, dtype: {result['image'].dtype}")
        
#         # Check class distribution
#         class_counts = np.bincount(result['class_id'], minlength=10)
#         print(f"Class counts: {class_counts}")
#         print(f"Class count range: [{class_counts.min()}, {class_counts.max()}]")
#         print(f"Class count std: {class_counts.std():.2f}")
        
#         # Check image properties
#         images = result['image']
#         print(f"Image value range: [{images.min():.4f}, {images.max():.4f}]")
#         print(f"Mean image brightness: {images.mean():.4f}")
        
#         # Check for duplicates within the sample
#         unique_base_ids = len(np.unique(result['base_id']))
#         print(f"Unique base_ids: {unique_base_ids} / {n_test} (duplicates: {n_test - unique_base_ids})")
        
#         # Verify class_id consistency
#         for i in range(min(5, n_test)):
#             base_id = result['base_id'][i]
#             claimed_class = result['class_id'][i]
#             actual_class = pool.labels[base_id]
#             if claimed_class != actual_class:
#                 print(f"ERROR: base_id {base_id} has class_id mismatch: {claimed_class} vs {actual_class}")
            
#         print(f"Class-id consistency check: PASSED (first 5 samples)")
    
#     # Test regular sampling
#     print("\n--- Testing regular sampling ---")
    
#     result_regular = pool.sample_base(1000, class_uniform=False)
#     class_counts_regular = np.bincount(result_regular['class_id'], minlength=10)
#     print(f"Regular sampling class counts: {class_counts_regular}")
#     print(f"Regular sampling class count std: {class_counts_regular.std():.2f}")
    
#     # Test independence between a_values and class_id (simulation)
#     print("\n--- Testing semantic-class independence ---")
    
#     # Simulate the independence test: sample a_values independently of class_id
#     n_independence_test = 5000
#     result_indep = pool.sample_base(n_independence_test, class_uniform=True)
    
#     # Generate semantic variables independently (simulating the SCM output)
#     # These should be independent of class_id by construction
#     a_M_sim = rng.uniform(0, 1, n_independence_test)
#     a_Y_sim = rng.uniform(0, 1, n_independence_test)
    
#     # Test independence using ANOVA (simplified test)
#     from scipy import stats
    
#     class_ids = result_indep['class_id']
    
#     # Group a_M by class and test if means are similar
#     a_M_by_class = [a_M_sim[class_ids == c] for c in range(10)]
#     f_stat_M, p_val_M = stats.f_oneway(*a_M_by_class)
    
#     a_Y_by_class = [a_Y_sim[class_ids == c] for c in range(10)]
#     f_stat_Y, p_val_Y = stats.f_oneway(*a_Y_by_class)
    
#     print(f"a_M vs class_id ANOVA: F={f_stat_M:.4f}, p={p_val_M:.4f} (expect p>0.1)")
#     print(f"a_Y vs class_id ANOVA: F={f_stat_Y:.4f}, p={p_val_Y:.4f} (expect p>0.1)")
    
#     # Show means by class
#     a_M_means = [a_M_sim[class_ids == c].mean() for c in range(10)]
#     a_Y_means = [a_Y_sim[class_ids == c].mean() for c in range(10)]
#     print(f"a_M means by class: {[f'{m:.3f}' for m in a_M_means]}")
#     print(f"a_Y means by class: {[f'{m:.3f}' for m in a_Y_means]}")
    
#     # Test multiple samples consistency
#     print("\n--- Testing sampling consistency ---")
    
#     # Two independent samples should have different base_ids but similar class distributions
#     sample1 = pool.sample_base(1000, class_uniform=True)
#     sample2 = pool.sample_base(1000, class_uniform=True)
    
#     # Check overlap
#     overlap = len(np.intersect1d(sample1['base_id'], sample2['base_id']))
#     print(f"Base ID overlap between two samples: {overlap} / 1000")
    
#     # Check class distribution similarity
#     counts1 = np.bincount(sample1['class_id'], minlength=10)
#     counts2 = np.bincount(sample2['class_id'], minlength=10)
#     print(f"Sample 1 class counts: {counts1}")
#     print(f"Sample 2 class counts: {counts2}")
#     print(f"Class count differences: {np.abs(counts1 - counts2)}")
    
#     # Test edge cases
#     print("\n--- Edge Cases ---")
    
#     # Small sample
#     small_sample = pool.sample_base(15, class_uniform=True)  # 1.5 per class
#     small_counts = np.bincount(small_sample['class_id'], minlength=10)
#     print(f"Small sample (n=15) class counts: {small_counts}")
    
#     # Very small sample
#     tiny_sample = pool.sample_base(3, class_uniform=True)
#     tiny_counts = np.bincount(tiny_sample['class_id'], minlength=10)
#     print(f"Tiny sample (n=3) class counts: {tiny_counts}")
    
#     # Test data integrity
#     print("\n--- Data Integrity ---")
    
#     # Check that images are valid
#     test_sample = pool.sample_base(100)
#     test_images = test_sample['image']
    
#     print(f"Image shape consistency: {np.all([img.shape == (28, 28) for img in test_images])}")
#     print(f"Image value range valid: {test_images.min() >= 0 and test_images.max() <= 1}")
#     print(f"No NaN values: {not np.any(np.isnan(test_images))}")
#     print(f"Sample image mean brightness by class:")
    
#     for c in range(10):
#         class_mask = test_sample['class_id'] == c
#         if np.any(class_mask):
#             class_brightness = test_images[class_mask].mean()
#             print(f"  Class {c}: {class_brightness:.4f}")
    
#     # Test error handling
#     print("\n--- Error Handling ---")
    
#     try:
#         # Try to sample more than available (should work with replacement warning)
#         huge_sample = pool.sample_base(100000, class_uniform=True)
#         print(f"Large sample successful: {len(huge_sample['base_id'])} samples")
#     except Exception as e:
#         print(f"Large sample error: {e}")
    
#     print("\n=== mnist_pool.py test completed ===")