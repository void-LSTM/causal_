"""
Data packaging and export module.
Combines tabular data, images, and metadata into final data package.
"""

import os
import json
import numpy as np
from typing import Dict, Union, Optional, List, Tuple, Any
from PIL import Image
import warnings
from pathlib import Path
import hashlib


def create_tabular_data(tab_core: Dict[str, np.ndarray],
                       semantics: Dict[str, np.ndarray],
                       base_M: Optional[Dict[str, np.ndarray]] = None,
                       base_Y: Optional[Dict[str, np.ndarray]] = None,
                       n_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Create structured tabular data arrays.
    
    Args:
        tab_core: Core variables from SCM (T, M, Y_star, W, S_style, C)
        semantics: Semantic parameters (a_M, a_Y, b_style)
        base_M: Base image info for I^M (base_id, class_id)
        base_Y: Base image info for I^Y (base_id, class_id)
        n_samples: Number of samples (for validation)
    
    Returns:
        Tuple of (float_data, int_data, float_columns, int_columns)
    """
    if n_samples is None:
        n_samples = len(tab_core["T"])
    
    # Prepare float columns
    float_columns = ["T", "M", "Y_star"]
    float_data_list = [
        tab_core["T"],
        tab_core["M"], 
        tab_core["Y_star"]
    ]
    
    # Add W columns
    W = tab_core["W"]
    if W.shape[1] > 0:
        for i in range(W.shape[1]):
            float_columns.append(f"W_{i+1}")
            float_data_list.append(W[:, i])
    
    # Add other float variables
    for var in ["S_style", "C"]:
        float_columns.append(var)
        float_data_list.append(tab_core[var])
    
    # Add semantic parameters
    for var in ["a_M", "a_Y", "b_style"]:
        float_columns.append(var)
        if var in semantics:
            float_data_list.append(semantics[var])
        else:
            # Fill with NaN if not available
            float_data_list.append(np.full(n_samples, np.nan, dtype=np.float32))
    
    # Prepare int columns
    int_columns = ["subject_id"]
    int_data_list = [np.arange(n_samples, dtype=np.int32)]  # subject_id = 0, 1, 2, ...
    
    # Add base image info
    for suffix, base_info in [("M", base_M), ("Y", base_Y)]:
        for var in ["base_id", "class_id"]:
            col_name = f"{var}_{suffix}"
            int_columns.append(col_name)
            
            if base_info is not None and var in base_info:
                int_data_list.append(base_info[var].astype(np.int32))
            else:
                # Fill with -1 if not available
                int_data_list.append(np.full(n_samples, -1, dtype=np.int32))
    
    # Stack data
    float_data = np.column_stack(float_data_list).astype(np.float32)
    int_data = np.column_stack(int_data_list).astype(np.int32)
    
    return float_data, int_data, float_columns, int_columns


def create_splits(n_samples: int, 
                 class_ids_M: Optional[np.ndarray] = None,
                 class_ids_Y: Optional[np.ndarray] = None,
                 splits_config: Dict[str, float] = None,
                 rng: np.random.Generator = None) -> Dict[str, List[int]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        n_samples: Total number of samples
        class_ids_M: Class IDs for I^M stratification
        class_ids_Y: Class IDs for I^Y stratification  
        splits_config: Split ratios {'train': 0.8, 'val': 0.1, 'test': 0.1}
        rng: Random number generator
    
    Returns:
        Dictionary with split indices
    """
    if splits_config is None:
        splits_config = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Determine stratification variable
    stratify_by = None
    if class_ids_M is not None and np.any(class_ids_M >= 0):
        stratify_by = class_ids_M
    elif class_ids_Y is not None and np.any(class_ids_Y >= 0):
        stratify_by = class_ids_Y
    
    if stratify_by is None:
        # No stratification - simple random split
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        n_train = int(n_samples * splits_config["train"])
        n_val = int(n_samples * splits_config["val"])
        
        splits = {
            "train": indices[:n_train].tolist(),
            "val": indices[n_train:n_train + n_val].tolist(),
            "test": indices[n_train + n_val:].tolist()
        }
    
    else:
        # Stratified split
        splits = {"train": [], "val": [], "test": []}
        
        # Process each class
        unique_classes = np.unique(stratify_by[stratify_by >= 0])  # Exclude -1 (missing)
        
        for class_id in unique_classes:
            class_indices = np.where(stratify_by == class_id)[0]
            n_class = len(class_indices)
            
            # Shuffle class indices
            rng.shuffle(class_indices)
            
            # Split within class
            n_train_class = int(n_class * splits_config["train"])
            n_val_class = int(n_class * splits_config["val"])
            
            splits["train"].extend(class_indices[:n_train_class].tolist())
            splits["val"].extend(class_indices[n_train_class:n_train_class + n_val_class].tolist())
            splits["test"].extend(class_indices[n_train_class + n_val_class:].tolist())
        
        # Handle samples with missing class info (-1)
        missing_indices = np.where(stratify_by < 0)[0]
        if len(missing_indices) > 0:
            rng.shuffle(missing_indices)
            
            n_train_missing = int(len(missing_indices) * splits_config["train"])
            n_val_missing = int(len(missing_indices) * splits_config["val"])
            
            splits["train"].extend(missing_indices[:n_train_missing].tolist())
            splits["val"].extend(missing_indices[n_train_missing:n_train_missing + n_val_missing].tolist())
            splits["test"].extend(missing_indices[n_train_missing + n_val_missing:].tolist())
        
        # Shuffle final splits
        for split_name in splits:
            rng.shuffle(splits[split_name])
    
    return splits


def save_images(images: np.ndarray,
               output_dir: str,
               subdir: str,
               png_optimize: bool = True) -> int:
    """
    Save images as PNG files.
    
    Args:
        images: Image array, shape (n, H, W), values in [0, 1]
        output_dir: Output directory path
        subdir: Subdirectory name ('img_M' or 'img_Y')
        png_optimize: Whether to optimize PNG compression
    
    Returns:
        Number of images saved
    """
    img_dir = os.path.join(output_dir, subdir)
    os.makedirs(img_dir, exist_ok=True)
    
    n_images = len(images)
    
    for i in range(n_images):
        # Convert to uint8
        img_uint8 = np.clip(np.round(images[i] * 255), 0, 255).astype(np.uint8)
        
        # Create PIL image
        img_pil = Image.fromarray(img_uint8, mode='L')  # Grayscale
        
        # Save with proper filename
        filename = f"{i:06d}.png"
        filepath = os.path.join(img_dir, filename)
        
        img_pil.save(filepath, optimize=png_optimize)
    
    return n_images


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """
    Compute hash of configuration for reproducibility tracking.
    
    Args:
        cfg: Configuration dictionary
    
    Returns:
        Hash string (first 12 characters of SHA256)
    """
    # Convert config to stable string representation
    config_str = json.dumps(cfg, sort_keys=True, default=str)
    
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:12]


def export_pack(out_dir: str,
               tab_core: Dict[str, np.ndarray],
               I_M: Optional[Dict[str, np.ndarray]] = None,
               I_Y: Optional[Dict[str, np.ndarray]] = None,
               base_M: Optional[Dict[str, np.ndarray]] = None,
               base_Y: Optional[Dict[str, np.ndarray]] = None,
               cfg: Dict[str, Any] = None,
               splits: Optional[Dict[str, List[int]]] = None,
               rng: np.random.Generator = None) -> None:
    """
    Export complete data package.
    
    Args:
        out_dir: Output directory path
        tab_core: Core tabular variables from SCM
        I_M: I^M images and metadata (optional)
        I_Y: I^Y images and metadata (optional)
        base_M: Base image info for I^M (optional)
        base_Y: Base image info for I^Y (optional)
        cfg: Full configuration dictionary
        splits: Pre-computed splits (optional)
        rng: Random number generator
    """
    if cfg is None:
        cfg = {}
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    n_samples = len(tab_core["T"])
    print(f"Exporting data package to {out_dir}")
    print(f"Number of samples: {n_samples}")
    
    # Compute semantic parameters if available
    semantics = {}
    if I_M is not None or I_Y is not None:
        from .semantics import g_sem, g_style
        
        # Get semantic mapping configuration
        g_sem_cfg = cfg.get('g_sem', {})
        g_style_cfg = cfg.get('g_style', {})
        
        g_sem_method = g_sem_cfg.get('method', 'gauss_cdf')
        g_style_method = g_style_cfg.get('method', 'gauss_cdf')
        clip_quantiles = g_sem_cfg.get('clip_quantiles', (0.005, 0.995))
        
        # Compute semantic parameters
        if I_M is not None:
            semantics["a_M"] = g_sem(tab_core["M"], method=g_sem_method, clip_quantiles=clip_quantiles)
        
        if I_Y is not None:
            semantics["a_Y"] = g_sem(tab_core["Y_star"], method=g_sem_method, clip_quantiles=clip_quantiles)
        
        semantics["b_style"] = g_style(tab_core["S_style"], method=g_style_method)
    
    # Create tabular data
    float_data, int_data, float_columns, int_columns = create_tabular_data(
        tab_core, semantics, base_M, base_Y, n_samples
    )
    
    print(f"Tabular data shapes: float={float_data.shape}, int={int_data.shape}")
    
    # Create splits if not provided
    if splits is None:
        splits_config = cfg.get('splits', {"train": 0.8, "val": 0.1, "test": 0.1})
        
        class_ids_M = base_M["class_id"] if base_M is not None else None
        class_ids_Y = base_Y["class_id"] if base_Y is not None else None
        
        splits = create_splits(n_samples, class_ids_M, class_ids_Y, splits_config, rng)
    
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Save tabular data
    tab_path = os.path.join(out_dir, "tab.npz")
    np.savez_compressed(
        tab_path,
        float=float_data,
        int=int_data,
        columns_float=float_columns,
        columns_int=int_columns
    )
    print(f"Saved tabular data: {tab_path}")
    
    # Save images
    n_images_M = 0
    n_images_Y = 0
    
    if I_M is not None and "image" in I_M:
        n_images_M = save_images(I_M["image"], out_dir, "img_M", 
                                 png_optimize=cfg.get('output', {}).get('png_optimize', True))
        print(f"Saved {n_images_M} I^M images")
    
    if I_Y is not None and "image" in I_Y:
        n_images_Y = save_images(I_Y["image"], out_dir, "img_Y",
                                 png_optimize=cfg.get('output', {}).get('png_optimize', True))
        print(f"Saved {n_images_Y} I^Y images")
    
    # Create metadata
    meta = create_metadata(cfg, tab_core, I_M, I_Y, base_M, base_Y, 
                          n_samples, n_images_M, n_images_Y, 
                          float_columns, int_columns)
    
    # Save metadata
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved metadata: {meta_path}")
    
    # Save splits
    splits_path = os.path.join(out_dir, "splits.json")
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits: {splits_path}")
    
    print(f"Data package export completed: {out_dir}")


def create_metadata(cfg: Dict[str, Any],
                   tab_core: Dict[str, np.ndarray],
                   I_M: Optional[Dict[str, np.ndarray]],
                   I_Y: Optional[Dict[str, np.ndarray]],
                   base_M: Optional[Dict[str, np.ndarray]],
                   base_Y: Optional[Dict[str, np.ndarray]],
                   n_samples: int,
                   n_images_M: int,
                   n_images_Y: int,
                   float_columns: List[str],
                   int_columns: List[str]) -> Dict[str, Any]:
    """
    Create comprehensive metadata dictionary.
    
    Args:
        cfg: Configuration dictionary
        tab_core: Core tabular data
        I_M: I^M data
        I_Y: I^Y data
        base_M: Base M info
        base_Y: Base Y info
        n_samples: Number of samples
        n_images_M: Number of I^M images
        n_images_Y: Number of I^Y images
        float_columns: Float column names
        int_columns: Int column names
    
    Returns:
        Metadata dictionary
    """
    meta = {
        "seed": cfg.get('seed', 0),
        "n_samples": n_samples,
        "q_misc": cfg.get('q_misc', 30),
        "splits": cfg.get('splits', {"train": 0.8, "val": 0.1, "test": 0.1}),
        
        "scm": cfg.get('scm', {}),
        "imaging": cfg.get('imaging', {}),
        "g_sem": cfg.get('g_sem', {}),
        "g_style": cfg.get('g_style', {}),
        "mnist": cfg.get('mnist', {}),
        "ci": cfg.get('ci', {}),
        "mi": cfg.get('mi', {}),
        "output": cfg.get('output', {}),
        
        "_resolved_cfg_hash": compute_config_hash(cfg),
        
        "data_structure": {
            "tabular_format": "npz",
            "float_columns": float_columns,
            "int_columns": int_columns,
            "n_float_cols": len(float_columns),
            "n_int_cols": len(int_columns),
            "images_format": "png",
            "n_images_M": n_images_M,
            "n_images_Y": n_images_Y
        },
        
        "truth_assertions": {
            "IM": {
                "TM": False,  # T ⊥ M should be rejected (dependent)
                "MY*": False,  # M ⊥ Y* should be rejected (dependent)
                "TY*|M": True if cfg.get('scm', {}).get('delta', 0) == 0 else False  # T ⊥ Y* | M
            },
            "IY": {
                "TM": False,  # T ⊥ M should be rejected (dependent)
                "MphiY": False,  # M ⊥ φ(I^Y) should be rejected (dependent)
                "TphiY|M": True if cfg.get('scm', {}).get('delta', 0) == 0 else False  # T ⊥ φ(I^Y) | M
            }
        },
        
        "notes": ""
    }
    
    # Add dataset statistics
    stats = compute_dataset_statistics(tab_core, I_M, I_Y, base_M, base_Y)
    meta["dataset_statistics"] = stats
    
    # Add notes based on configuration
    notes = []
    if cfg.get('scm', {}).get('delta', 0) > 0:
        notes.append(f"delta={cfg['scm']['delta']} > 0, expect TY|M not independent")
    
    imaging_cfg = cfg.get('imaging', {})
    if imaging_cfg.get('perm_M', 0) > 0:
        notes.append(f"I^M permutation: {imaging_cfg['perm_M']}")
    if imaging_cfg.get('perm_Y', 0) > 0:
        notes.append(f"I^Y permutation: {imaging_cfg['perm_Y']}")
    
    meta["notes"] = "; ".join(notes)
    
    return meta


def compute_dataset_statistics(tab_core: Dict[str, np.ndarray],
                              I_M: Optional[Dict[str, np.ndarray]],
                              I_Y: Optional[Dict[str, np.ndarray]],
                              base_M: Optional[Dict[str, np.ndarray]],
                              base_Y: Optional[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """
    Compute dataset statistics for metadata.
    
    Args:
        tab_core: Core tabular data
        I_M: I^M data
        I_Y: I^Y data
        base_M: Base M info
        base_Y: Base Y info
    
    Returns:
        Statistics dictionary
    """
    stats = {}
    
    # Tabular statistics
    for var in ["T", "M", "Y_star"]:
        if var in tab_core:
            data = tab_core[var]
            stats[var] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "nan_count": int(np.sum(np.isnan(data)))
            }
    
    # W statistics
    if "W" in tab_core and tab_core["W"].shape[1] > 0:
        W = tab_core["W"]
        stats["W"] = {
            "shape": W.shape,
            "mean": float(np.mean(W)),
            "std": float(np.std(W)),
            "nan_count": int(np.sum(np.isnan(W)))
        }
    
    # Class distribution statistics
    for suffix, base_info in [("M", base_M), ("Y", base_Y)]:
        if base_info is not None and "class_id" in base_info:
            class_ids = base_info["class_id"]
            valid_classes = class_ids[class_ids >= 0]
            
            if len(valid_classes) > 0:
                unique_classes, counts = np.unique(valid_classes, return_counts=True)
                stats[f"class_distribution_{suffix}"] = {
                    "classes": unique_classes.tolist(),
                    "counts": counts.tolist(),
                    "n_valid": len(valid_classes),
                    "n_missing": int(np.sum(class_ids < 0))
                }
    
    # Image statistics
    for suffix, img_data in [("M", I_M), ("Y", I_Y)]:
        if img_data is not None and "image" in img_data:
            images = img_data["image"]
            stats[f"images_{suffix}"] = {
                "shape": images.shape,
                "mean_brightness": float(np.mean(images)),
                "std_brightness": float(np.std(images)),
                "min_value": float(np.min(images)),
                "max_value": float(np.max(images))
            }
    
    return stats


# # Test functions
# if __name__ == "__main__":
#     print("=== Testing pack.py ===")
    
#     # Setup test data
#     rng = np.random.default_rng(42)
#     n_test = 1000
    
#     print(f"Creating test data with n={n_test}")
    
#     # Create mock tabular data
#     tab_core = {
#         "T": rng.normal(0, 1, n_test).astype(np.float32),
#         "M": rng.normal(0, 1, n_test).astype(np.float32),
#         "Y_star": rng.normal(0, 1, n_test).astype(np.float32),
#         "W": rng.normal(0, 1, (n_test, 5)).astype(np.float32),  # 5 misc variables
#         "S_style": rng.normal(0, 1, n_test).astype(np.float32),
#         "C": np.full(n_test, np.nan, dtype=np.float32)  # Not used
#     }
    
#     print("Created mock tabular data")
    
#     # Create mock semantic parameters
#     semantics = {
#         "a_M": rng.uniform(0, 1, n_test).astype(np.float32),
#         "a_Y": rng.uniform(0, 1, n_test).astype(np.float32),
#         "b_style": rng.uniform(0, 1, n_test).astype(np.float32)
#     }
    
#     print("Created mock semantic parameters")
    
#     # Create mock base image info
#     base_M = {
#         "base_id": rng.integers(0, 70000, n_test).astype(np.int32),
#         "class_id": rng.integers(0, 10, n_test).astype(np.int32)
#     }
    
#     base_Y = {
#         "base_id": rng.integers(0, 70000, n_test).astype(np.int32),
#         "class_id": rng.integers(0, 10, n_test).astype(np.int32)
#     }
    
#     print("Created mock base image info")
    
#     # Create mock images
#     H, W = 28, 28
#     I_M_images = rng.random((n_test, H, W)).astype(np.float32)
#     I_Y_images = rng.random((n_test, H, W)).astype(np.float32)
    
#     I_M = {
#         "image": I_M_images,
#         "applied": {
#             "degrees": rng.uniform(-25, 25, n_test),
#             "brightness_delta": rng.uniform(-0.2, 0.2, n_test),
#             "contrast_gain": rng.uniform(0.5, 1.5, n_test),
#             "noise_sigma": rng.uniform(0, 0.1, n_test)
#         }
#     }
    
#     I_Y = {
#         "image": I_Y_images,
#         "applied": {
#             "degrees": rng.uniform(-45, 45, n_test),
#             "brightness_delta": rng.uniform(-0.3, 0.3, n_test),
#             "contrast_gain": rng.uniform(0.3, 1.7, n_test),
#             "noise_sigma": rng.uniform(0, 0.05, n_test)
#         }
#     }
    
#     print("Created mock images")
    
#     # Test tabular data creation
#     print("\n--- Testing tabular data creation ---")
    
#     float_data, int_data, float_columns, int_columns = create_tabular_data(
#         tab_core, semantics, base_M, base_Y, n_test
#     )
    
#     print(f"Float data shape: {float_data.shape}")
#     print(f"Int data shape: {int_data.shape}")
#     print(f"Float columns ({len(float_columns)}): {float_columns}")
#     print(f"Int columns ({len(int_columns)}): {int_columns}")
    
#     # Check data integrity
#     print(f"Float data range: [{float_data.min():.4f}, {float_data.max():.4f}]")
#     print(f"Int data range: [{int_data.min()}, {int_data.max()}]")
#     print(f"NaN count in float data: {np.sum(np.isnan(float_data))}")
    
#     # Test splits creation
#     print("\n--- Testing splits creation ---")
    
#     # Test with stratification
#     splits_strat = create_splits(n_test, base_M["class_id"], base_Y["class_id"], 
#                                 {"train": 0.8, "val": 0.1, "test": 0.1}, rng)
    
#     print(f"Stratified splits: train={len(splits_strat['train'])}, val={len(splits_strat['val'])}, test={len(splits_strat['test'])}")
    
#     # Check class distribution in each split
#     for split_name, indices in splits_strat.items():
#         if len(indices) > 0:
#             split_classes = base_M["class_id"][indices]
#             class_counts = np.bincount(split_classes, minlength=10)
#             print(f"  {split_name} class distribution: {class_counts}")
    
#     # Test without stratification
#     splits_simple = create_splits(n_test, None, None, 
#                                  {"train": 0.7, "val": 0.15, "test": 0.15}, rng)
    
#     print(f"Simple splits: train={len(splits_simple['train'])}, val={len(splits_simple['val'])}, test={len(splits_simple['test'])}")
    
#     # Test configuration hash
#     print("\n--- Testing configuration hash ---")
    
#     test_cfg = {
#         "seed": 42,
#         "scm": {"alpha1": 1.0, "rho": 0.5},
#         "imaging": {"s_level": "mid"}
#     }
    
#     hash1 = compute_config_hash(test_cfg)
#     hash2 = compute_config_hash(test_cfg)  # Should be identical
    
#     # Slightly different config
#     test_cfg_diff = test_cfg.copy()
#     test_cfg_diff["scm"]["rho"] = 0.6
#     hash3 = compute_config_hash(test_cfg_diff)
    
#     print(f"Hash 1: {hash1}")
#     print(f"Hash 2: {hash2}")
#     print(f"Hash 3 (different): {hash3}")
#     print(f"Hash consistency: {hash1 == hash2}")
#     print(f"Hash sensitivity: {hash1 != hash3}")
    
#     # Test metadata creation
#     print("\n--- Testing metadata creation ---")
    
#     cfg_full = {
#         "seed": 42,
#         "n_samples": n_test,
#         "q_misc": 5,
#         "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
#         "scm": {
#             "alpha1": 1.0, "alpha2": 1.0, "rho": 0.5, "delta": 0.0,
#             "sigma_T": 0.2, "sigma_M": 0.2, "sigma_Y": 0.1,
#             "h1": "square", "h2": "tanh", "Y_type": "cont"
#         },
#         "imaging": {
#             "use_I_M": True, "use_I_Y": True, "s_level": "mid",
#             "theta_deg": 25, "beta": 0.25, "gamma": 0.25, "sigma_pix": 0.1,
#             "perm_M": 0.0, "perm_Y": 0.0
#         }
#     }
    
#     meta = create_metadata(cfg_full, tab_core, I_M, I_Y, base_M, base_Y,
#                           n_test, n_test, n_test, float_columns, int_columns)
    
#     print(f"Metadata keys: {list(meta.keys())}")
#     print(f"Truth assertions: {meta['truth_assertions']}")
#     print(f"Data structure: {meta['data_structure']}")
#     print(f"Config hash: {meta['_resolved_cfg_hash']}")
    
#     # Test dataset statistics
#     stats = meta["dataset_statistics"]
#     print(f"\nDataset statistics keys: {list(stats.keys())}")
#     if "T" in stats:
#         print(f"T statistics: {stats['T']}")
#     if "class_distribution_M" in stats:
#         print(f"Class distribution M: {stats['class_distribution_M']}")
    
#     # Test full export (to temporary directory)
#     print("\n--- Testing full export ---")
    
#     import tempfile
    
#     with tempfile.TemporaryDirectory() as temp_dir:
#         test_output_dir = os.path.join(temp_dir, "test_pack")
        
#         export_pack(
#             out_dir=test_output_dir,
#             tab_core=tab_core,
#             I_M=I_M,
#             I_Y=I_Y,
#             base_M=base_M,
#             base_Y=base_Y,
#             cfg=cfg_full,
#             splits=splits_strat,
#             rng=rng
#         )
        
#         # Verify exported files
#         print(f"\nExported files:")
#         for root, dirs, files in os.walk(test_output_dir):
#             level = root.replace(test_output_dir, '').count(os.sep)
#             indent = ' ' * 2 * level
#             print(f"{indent}{os.path.basename(root)}/")
#             sub_indent = ' ' * 2 * (level + 1)
#             for file in files[:5]:  # Show first 5 files in each dir
#                 print(f"{sub_indent}{file}")
#             if len(files) > 5:
#                 print(f"{sub_indent}... and {len(files) - 5} more files")
        
#         # Verify file contents
#         tab_path = os.path.join(test_output_dir, "tab.npz")
#         if os.path.exists(tab_path):
#             tab_data = np.load(tab_path)
#             print(f"\nTabular data verification:")
#             print(f"  Keys: {list(tab_data.keys())}")
#             print(f"  Float data shape: {tab_data['float'].shape}")
#             print(f"  Int data shape: {tab_data['int'].shape}")
#             print(f"  Float columns: {len(tab_data['columns_float'])}")
#             print(f"  Int columns: {len(tab_data['columns_int'])}")
        
#         # Check image directories
#         img_M_dir = os.path.join(test_output_dir, "img_M")
#         img_Y_dir = os.path.join(test_output_dir, "img_Y")
        
#         if os.path.exists(img_M_dir):
#             n_imgs_M = len([f for f in os.listdir(img_M_dir) if f.endswith('.png')])
#             print(f"  I^M images: {n_imgs_M}")
        
#         if os.path.exists(img_Y_dir):
#             n_imgs_Y = len([f for f in os.listdir(img_Y_dir) if f.endswith('.png')])
#             print(f"  I^Y images: {n_imgs_Y}")
        
#         # Check metadata
#         meta_path = os.path.join(test_output_dir, "meta.json")
#         if os.path.exists(meta_path):
#             with open(meta_path, 'r') as f:
#                 meta_loaded = json.load(f)
#             print(f"  Metadata loaded successfully, {len(meta_loaded)} keys")
        
#         # Check splits
#         splits_path = os.path.join(test_output_dir, "splits.json")
#         if os.path.exists(splits_path):
#             with open(splits_path, 'r') as f:
#                 splits_loaded = json.load(f)
#             print(f"  Splits loaded: {[f'{k}: {len(v)}' for k, v in splits_loaded.items()]}")
    
#     # Test edge cases
#     print("\n--- Testing edge cases ---")
    
#     # Test with missing I^Y
#     print("Testing export without I^Y")
#     with tempfile.TemporaryDirectory() as temp_dir:
#         test_dir = os.path.join(temp_dir, "no_IY")
#         export_pack(
#             out_dir=test_dir,
#             tab_core=tab_core,
#             I_M=I_M,
#             I_Y=None,  # No I^Y
#             base_M=base_M,
#             base_Y=None,
#             cfg=cfg_full,
#             rng=rng
#         )
#         print(f"  Export without I^Y successful")
    
#     # Test with missing I^M
#     print("Testing export without I^M")
#     with tempfile.TemporaryDirectory() as temp_dir:
#         test_dir = os.path.join(temp_dir, "no_IM")
#         export_pack(
#             out_dir=test_dir,
#             tab_core=tab_core,
#             I_M=None,  # No I^M
#             I_Y=I_Y,
#             base_M=None,
#             base_Y=base_Y,
#             cfg=cfg_full,
#             rng=rng
#         )
#         print(f"  Export without I^M successful")
    
#     # Test with no images at all
#     print("Testing export with tabular data only")
#     with tempfile.TemporaryDirectory() as temp_dir:
#         test_dir = os.path.join(temp_dir, "tabular_only")
#         export_pack(
#             out_dir=test_dir,
#             tab_core=tab_core,
#             I_M=None,
#             I_Y=None,
#             base_M=None,
#             base_Y=None,
#             cfg=cfg_full,
#             rng=rng
#         )
#         print(f"  Tabular-only export successful")
    
#     print("\n=== pack.py test completed ===")