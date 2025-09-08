"""
Image generation for I^M (image as mediator) and I^Y (image as result) modalities.
"""

import numpy as np
from typing import Dict, Union, Optional, Tuple
from .transforms import batch_transform, get_transform_params
import warnings


def generate_I_M(base_imgs: np.ndarray,
                 a_M: np.ndarray,
                 b: np.ndarray,
                 cfg_img: Dict[str, Union[float, bool]],
                 rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Generate I^M images (image as mediator).
    
    Args:
        base_imgs: Base MNIST images, shape (n, H, W), values in [0, 1]
        a_M: Semantic amplitudes for M, shape (n,), values in [0, 1]
        b: Style parameters, shape (n,), values in [0, 1]
        cfg_img: Image configuration dictionary
        rng: Random number generator
    
    Returns:
        Dictionary with 'image' and 'applied' keys
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n, H, W = base_imgs.shape
    
    # Validate inputs
    assert len(a_M) == n, f"a_M length {len(a_M)} != n_images {n}"
    assert len(b) == n, f"b length {len(b)} != n_images {n}"
    assert np.all((a_M >= 0) & (a_M <= 1)), "a_M values must be in [0, 1]"
    assert np.all((b >= 0) & (b <= 1)), "b values must be in [0, 1]"
    
    # Apply transformations using a_M as semantic parameter
    transformed_imgs, applied_params = batch_transform(
        base_imgs, a_M, b, cfg_img, rng
    )
    
    # Handle permutation if specified
    perm_M = cfg_img.get('perm_M', 0.0)
    if perm_M > 0:
        transformed_imgs, perm_info = apply_permutation(
            transformed_imgs, perm_M, rng
        )
        applied_params['permutation_applied'] = perm_info
    else:
        applied_params['permutation_applied'] = None
    
    return {
        "image": transformed_imgs,
        "applied": applied_params
    }


def generate_I_Y(base_imgs: np.ndarray,
                 a_Y: np.ndarray,
                 b: np.ndarray,
                 cfg_img: Dict[str, Union[float, bool]],
                 rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Generate I^Y images (image as result).
    
    Args:
        base_imgs: Base MNIST images, shape (n, H, W), values in [0, 1]
        a_Y: Semantic amplitudes for Y, shape (n,), values in [0, 1]
        b: Style parameters, shape (n,), values in [0, 1]
        cfg_img: Image configuration dictionary
        rng: Random number generator
    
    Returns:
        Dictionary with 'image' and 'applied' keys
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n, H, W = base_imgs.shape
    
    # Validate inputs
    assert len(a_Y) == n, f"a_Y length {len(a_Y)} != n_images {n}"
    assert len(b) == n, f"b length {len(b)} != n_images {n}"
    assert np.all((a_Y >= 0) & (a_Y <= 1)), "a_Y values must be in [0, 1]"
    assert np.all((b >= 0) & (b <= 1)), "b values must be in [0, 1]"
    
    # Apply transformations using a_Y as semantic parameter
    transformed_imgs, applied_params = batch_transform(
        base_imgs, a_Y, b, cfg_img, rng
    )
    
    # Handle permutation if specified
    perm_Y = cfg_img.get('perm_Y', 0.0)
    if perm_Y > 0:
        transformed_imgs, perm_info = apply_permutation(
            transformed_imgs, perm_Y, rng
        )
        applied_params['permutation_applied'] = perm_info
    else:
        applied_params['permutation_applied'] = None
    
    return {
        "image": transformed_imgs,
        "applied": applied_params
    }


def apply_permutation(images: np.ndarray, 
                     perm_ratio: float, 
                     rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    """
    Apply permutation to break semantic-image correspondence.
    
    Args:
        images: Input images, shape (n, H, W)
        perm_ratio: Fraction of images to permute (0.0 to 1.0)
        rng: Random number generator
    
    Returns:
        Tuple of (permuted_images, permutation_info)
    """
    n = len(images)
    n_perm = int(n * perm_ratio)
    
    if n_perm == 0:
        return images.copy(), {"n_permuted": 0, "perm_pairs": []}
    
    # Select indices to permute
    perm_indices = rng.choice(n, size=n_perm, replace=False)
    
    # Create permuted version
    permuted_images = images.copy()
    
    # Shuffle the selected indices
    shuffled_indices = perm_indices.copy()
    rng.shuffle(shuffled_indices)
    
    # Apply permutation
    perm_pairs = []
    for i, (orig_idx, new_idx) in enumerate(zip(perm_indices, shuffled_indices)):
        if orig_idx != new_idx:  # Only record actual swaps
            permuted_images[orig_idx] = images[new_idx]
            perm_pairs.append((int(orig_idx), int(new_idx)))
    
    perm_info = {
        "n_permuted": len(perm_pairs),
        "perm_pairs": perm_pairs,
        "perm_ratio_actual": len(perm_pairs) / n
    }
    
    return permuted_images, perm_info


def create_imaging_config(s_level: str = "mid",
                         theta_deg: Optional[float] = None,
                         beta: Optional[float] = None,
                         gamma: Optional[float] = None,
                         sigma_pix: float = 0.0,
                         perm_M: float = 0.0,
                         perm_Y: float = 0.0) -> Dict[str, float]:
    """
    Create imaging configuration dictionary.
    
    Args:
        s_level: Strength level ('small', 'mid', 'large')
        theta_deg: Rotation angle override
        beta: Brightness strength override
        gamma: Contrast strength override
        sigma_pix: Pixel noise strength
        perm_M: Permutation ratio for I^M
        perm_Y: Permutation ratio for I^Y
    
    Returns:
        Configuration dictionary
    """
    # Get base parameters from s_level
    base_params = get_transform_params(s_level)
    
    # Override with specific values if provided
    cfg = {
        'theta_deg': theta_deg if theta_deg is not None else base_params['theta_deg'],
        'beta': beta if beta is not None else base_params['beta'],
        'gamma': gamma if gamma is not None else base_params['gamma'],
        'sigma_pix': sigma_pix,
        'perm_M': perm_M,
        'perm_Y': perm_Y
    }
    
    return cfg


def validate_imaging_inputs(base_imgs: np.ndarray,
                           a_values: np.ndarray,
                           b_values: np.ndarray) -> None:
    """
    Validate inputs for imaging functions.
    
    Args:
        base_imgs: Base images array
        a_values: Semantic amplitude values
        b_values: Style parameter values
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Check base images
    if base_imgs.ndim != 3:
        raise ValueError(f"base_imgs must be 3D (n, H, W), got shape {base_imgs.shape}")
    
    if not (base_imgs.min() >= 0 and base_imgs.max() <= 1):
        raise ValueError(f"base_imgs values must be in [0, 1], got range [{base_imgs.min():.4f}, {base_imgs.max():.4f}]")
    
    n = len(base_imgs)
    
    # Check semantic amplitudes
    if len(a_values) != n:
        raise ValueError(f"a_values length {len(a_values)} != n_images {n}")
    
    if not (a_values.min() >= 0 and a_values.max() <= 1):
        raise ValueError(f"a_values must be in [0, 1], got range [{a_values.min():.4f}, {a_values.max():.4f}]")
    
    # Check style parameters
    if len(b_values) != n:
        raise ValueError(f"b_values length {len(b_values)} != n_images {n}")
    
    if not (b_values.min() >= 0 and b_values.max() <= 1):
        raise ValueError(f"b_values must be in [0, 1], got range [{b_values.min():.4f}, {b_values.max():.4f}]")


def compute_imaging_statistics(images: np.ndarray,
                              applied_params: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute statistics for generated images.
    
    Args:
        images: Generated images, shape (n, H, W)
        applied_params: Applied transformation parameters
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_images': len(images),
        'mean_brightness': float(images.mean()),
        'std_brightness': float(images.std()),
        'min_value': float(images.min()),
        'max_value': float(images.max()),
    }
    
    # Add transformation statistics
    if 'degrees' in applied_params:
        degrees = applied_params['degrees']
        stats['rotation_mean'] = float(degrees.mean())
        stats['rotation_std'] = float(degrees.std())
        stats['rotation_range'] = [float(degrees.min()), float(degrees.max())]
    
    if 'brightness_delta' in applied_params:
        brightness = applied_params['brightness_delta']
        stats['brightness_delta_mean'] = float(brightness.mean())
        stats['brightness_delta_std'] = float(brightness.std())
    
    if 'contrast_gain' in applied_params:
        contrast = applied_params['contrast_gain']
        stats['contrast_gain_mean'] = float(contrast.mean())
        stats['contrast_gain_std'] = float(contrast.std())
    
    if 'noise_sigma' in applied_params:
        noise = applied_params['noise_sigma']
        stats['noise_sigma_mean'] = float(noise.mean())
        stats['noise_sigma_std'] = float(noise.std())
    
    return stats


# # Test functions
# if __name__ == "__main__":
#     print("=== Testing imaging.py ===")
    
#     # Setup test data
#     rng = np.random.default_rng(42)
    
#     # Create synthetic base images (simulating MNIST)
#     n_test = 100
#     H, W = 28, 28
    
#     # Generate diverse base images
#     base_images = np.zeros((n_test, H, W))
    
#     # Image types: circles, rectangles, lines, noise
#     for i in range(n_test):
#         img_type = i % 4
        
#         if img_type == 0:  # Circle
#             center_y, center_x = H//2 + rng.integers(-5, 6), W//2 + rng.integers(-5, 6)
#             radius = rng.integers(6, 12)
#             for y in range(H):
#                 for x in range(W):
#                     if (y - center_y)**2 + (x - center_x)**2 <= radius**2:
#                         base_images[i, y, x] = 0.8
        
#         elif img_type == 1:  # Rectangle
#             y1, y2 = sorted(rng.integers(5, H-5, 2))
#             x1, x2 = sorted(rng.integers(5, W-5, 2))
#             base_images[i, y1:y2, x1:x2] = 0.9
        
#         elif img_type == 2:  # Line
#             if rng.random() > 0.5:  # Horizontal
#                 y = rng.integers(5, H-5)
#                 base_images[i, y:y+2, 5:W-5] = 1.0
#             else:  # Vertical
#                 x = rng.integers(5, W-5)
#                 base_images[i, 5:H-5, x:x+2] = 1.0
        
#         else:  # Noise pattern
#             base_images[i] = rng.random((H, W)) * 0.6
    
#     print(f"Generated {n_test} base images")
#     print(f"Base images shape: {base_images.shape}")
#     print(f"Base images range: [{base_images.min():.4f}, {base_images.max():.4f}]")
#     print(f"Base images mean brightness: {base_images.mean():.4f}")
    
#     # Generate semantic parameters
#     a_M = rng.uniform(0, 1, n_test)
#     a_Y = rng.uniform(0, 1, n_test)
#     b_style = rng.uniform(0, 1, n_test)
    
#     print(f"\nSemantic parameters:")
#     print(f"a_M: range=[{a_M.min():.3f}, {a_M.max():.3f}], mean={a_M.mean():.3f}")
#     print(f"a_Y: range=[{a_Y.min():.3f}, {a_Y.max():.3f}], mean={a_Y.mean():.3f}")
#     print(f"b_style: range=[{b_style.min():.3f}, {b_style.max():.3f}], mean={b_style.mean():.3f}")
    
#     # Test configuration creation
#     print("\n--- Testing configuration creation ---")
    
#     for s_level in ['small', 'mid', 'large']:
#         cfg = create_imaging_config(s_level=s_level, sigma_pix=0.05)
#         print(f"{s_level}: {cfg}")
    
#     # Test I^M generation
#     print("\n--- Testing I^M generation ---")
    
#     cfg_IM = create_imaging_config(s_level='mid', sigma_pix=0.1)
#     result_IM = generate_I_M(base_images, a_M, b_style, cfg_IM, rng)
    
#     print(f"I^M result keys: {list(result_IM.keys())}")
#     print(f"I^M images shape: {result_IM['image'].shape}")
#     print(f"I^M applied params keys: {list(result_IM['applied'].keys())}")
    
#     # Check I^M statistics
#     stats_IM = compute_imaging_statistics(result_IM['image'], result_IM['applied'])
#     print(f"I^M statistics: {stats_IM}")
    
#     # Test I^Y generation
#     print("\n--- Testing I^Y generation ---")
    
#     cfg_IY = create_imaging_config(s_level='large', sigma_pix=0.05)
#     result_IY = generate_I_Y(base_images, a_Y, b_style, cfg_IY, rng)
    
#     print(f"I^Y result keys: {list(result_IY.keys())}")
#     print(f"I^Y images shape: {result_IY['image'].shape}")
#     print(f"I^Y applied params keys: {list(result_IY['applied'].keys())}")
    
#     # Check I^Y statistics
#     stats_IY = compute_imaging_statistics(result_IY['image'], result_IY['applied'])
#     print(f"I^Y statistics: {stats_IY}")
    
#     # Test semantic effects
#     print("\n--- Testing semantic effects ---")
    
#     # Test with extreme semantic values
#     a_extreme = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
#     b_fixed = np.full(5, 0.5)
#     base_subset = base_images[:5]
    
#     result_extreme = generate_I_M(base_subset, a_extreme, b_fixed, cfg_IM, rng)
    
#     print("Semantic amplitude effects on I^M:")
#     for i, a in enumerate(a_extreme):
#         orig_brightness = base_subset[i].mean()
#         new_brightness = result_extreme['image'][i].mean()
#         rotation = result_extreme['applied']['degrees'][i]
#         brightness_delta = result_extreme['applied']['brightness_delta'][i]
#         contrast_gain = result_extreme['applied']['contrast_gain'][i]
        
#         print(f"  a={a:.2f}: {orig_brightness:.3f}→{new_brightness:.3f}, rot={rotation:+.1f}°, bright={brightness_delta:+.3f}, contrast={contrast_gain:.3f}")
    
#     # Test style effects
#     print("\nStyle parameter effects:")
    
#     a_fixed = np.full(5, 0.7)
#     b_extreme = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
#     result_style = generate_I_M(base_subset, a_fixed, b_extreme, cfg_IM, rng)
    
#     for i, b in enumerate(b_extreme):
#         orig_std = base_subset[i].std()
#         new_std = result_style['image'][i].std()
#         noise_sigma = result_style['applied']['noise_sigma'][i]
        
#         print(f"  b={b:.2f}: std {orig_std:.3f}→{new_std:.3f}, noise_σ={noise_sigma:.4f}")
    
#     # Test permutation
#     print("\n--- Testing permutation ---")
    
#     cfg_perm = create_imaging_config(s_level='mid', perm_M=0.2, perm_Y=0.15)
    
#     result_perm_M = generate_I_M(base_images[:20], a_M[:20], b_style[:20], cfg_perm, rng)
#     result_perm_Y = generate_I_Y(base_images[:20], a_Y[:20], b_style[:20], cfg_perm, rng)
    
#     perm_info_M = result_perm_M['applied']['permutation_applied']
#     perm_info_Y = result_perm_Y['applied']['permutation_applied']
    
#     print(f"I^M permutation: {perm_info_M}")
#     print(f"I^Y permutation: {perm_info_Y}")
    
#     # Test without permutation
#     cfg_no_perm = create_imaging_config(s_level='mid', perm_M=0.0, perm_Y=0.0)
#     result_no_perm = generate_I_M(base_images[:10], a_M[:10], b_style[:10], cfg_no_perm, rng)
#     print(f"No permutation: {result_no_perm['applied']['permutation_applied']}")
    
#     # Test value range preservation
#     print("\n--- Testing value range preservation ---")
    
#     # Test with extreme transformations
#     cfg_extreme = {
#         'theta_deg': 90.0,  # Large rotation
#         'beta': 0.8,        # Large brightness
#         'gamma': 1.0,       # Large contrast
#         'sigma_pix': 0.3,   # Large noise
#         'perm_M': 0.0,
#         'perm_Y': 0.0
#     }
    
#     a_max = np.ones(10)  # Maximum semantic amplitude
#     b_max = np.ones(10)  # Maximum style
    
#     result_max = generate_I_M(base_images[:10], a_max, b_max, cfg_extreme, rng)
#     extreme_images = result_max['image']
    
#     print(f"Extreme transformation range: [{extreme_images.min():.4f}, {extreme_images.max():.4f}]")
#     print(f"Value range preserved: {extreme_images.min() >= 0 and extreme_images.max() <= 1}")
    
#     # Test consistency between I^M and I^Y
#     print("\n--- Testing I^M vs I^Y consistency ---")
    
#     # Same parameters should produce similar transformations
#     cfg_consistent = create_imaging_config(s_level='mid', sigma_pix=0.05)
#     a_same = a_M[:10]
    
#     result_M_cons = generate_I_M(base_images[:10], a_same, b_style[:10], cfg_consistent, rng)
#     result_Y_cons = generate_I_Y(base_images[:10], a_same, b_style[:10], cfg_consistent, rng)
    
#     # Parameters should be identical (since same inputs)
#     degrees_diff = np.abs(result_M_cons['applied']['degrees'] - result_Y_cons['applied']['degrees']).max()
#     brightness_diff = np.abs(result_M_cons['applied']['brightness_delta'] - result_Y_cons['applied']['brightness_delta']).max()
    
#     print(f"Parameter consistency (should be ~0): degrees_diff={degrees_diff:.6f}, brightness_diff={brightness_diff:.6f}")
    
#     # Test error handling
#     print("\n--- Testing error handling ---")
    
#     try:
#         # Wrong number of semantic parameters
#         generate_I_M(base_images[:10], a_M[:5], b_style[:10], cfg_IM, rng)
#         print("ERROR: Should have raised assertion error")
#     except AssertionError as e:
#         print(f"Correctly caught assertion: a_M length mismatch")
    
#     try:
#         # Invalid semantic range
#         a_invalid = np.array([0.5, 1.5, 0.3])  # 1.5 > 1.0
#         generate_I_M(base_images[:3], a_invalid, b_style[:3], cfg_IM, rng)
#         print("ERROR: Should have raised assertion error")
#     except AssertionError as e:
#         print(f"Correctly caught assertion: a_M range invalid")
    
#     try:
#         # Invalid image dimensions
#         base_2d = base_images[0]  # 2D instead of 3D
#         generate_I_M(base_2d, a_M[:1], b_style[:1], cfg_IM, rng)
#         print("ERROR: Should have raised error")
#     except Exception as e:
#         print(f"Correctly caught error: {type(e).__name__}")
    
#     # Test batch processing efficiency
#     print("\n--- Testing batch processing ---")
    
#     import time
    
#     # Large batch test
#     n_large = 1000
#     base_large = rng.random((n_large, H, W))
#     a_large = rng.uniform(0, 1, n_large)
#     b_large = rng.uniform(0, 1, n_large)
    
#     start_time = time.time()
#     result_large = generate_I_M(base_large, a_large, b_large, cfg_IM, rng)
#     end_time = time.time()
    
#     processing_time = end_time - start_time
#     rate = n_large / processing_time
    
#     print(f"Processed {n_large} images in {processing_time:.3f}s ({rate:.1f} images/sec)")
#     print(f"Output shape: {result_large['image'].shape}")
#     print(f"Memory usage reasonable: {result_large['image'].nbytes / (1024**2):.1f} MB")
    
#     print("\n=== imaging.py test completed ===")