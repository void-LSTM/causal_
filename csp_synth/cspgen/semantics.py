"""
Semantic mapping functions: convert numerical values to semantic parameters and extract features.
"""

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from typing import Union, Tuple, Dict, List
import warnings


def g_sem(x: np.ndarray, method: str = "gauss_cdf", clip_quantiles: Tuple[float, float] = (0.005, 0.995)) -> np.ndarray:
    """
    Map numerical values to semantic amplitudes in [0, 1].
    
    Args:
        x: Input array, shape (n,)
        method: Mapping method ('gauss_cdf' or 'minmax')
        clip_quantiles: Quantile clipping for minmax method, default (0.5%, 99.5%)
    
    Returns:
        Semantic amplitudes a in [0, 1], shape (n,)
    """
    x = np.asarray(x).flatten()
    eps = 1e-8
    
    if method == "gauss_cdf":
        # Use statistics from entire dataset
        mu = np.mean(x)
        sigma = np.std(x)
        
        # Apply Gaussian CDF: Φ((x-μ)/(σ+ε))
        z_scores = (x - mu) / (sigma + eps)
        a = stats.norm.cdf(z_scores)
        
    elif method == "minmax":
        # Quantile clipping first
        q_low, q_high = clip_quantiles
        x_min_clip = np.percentile(x, q_low * 100)
        x_max_clip = np.percentile(x, q_high * 100)
        x_clipped = np.clip(x, x_min_clip, x_max_clip)
        
        # Min-max normalization
        x_min = np.min(x_clipped)
        x_max = np.max(x_clipped)
        a = (x_clipped - x_min) / (x_max - x_min + eps)
        
    else:
        raise ValueError(f"Unknown g_sem method: {method}")
    
    # Ensure [0, 1] range
    a = np.clip(a, 0.0, 1.0)
    return a


def g_style(s: np.ndarray, method: str = "gauss_cdf") -> np.ndarray:
    """
    Map style variables to style parameters in [0, 1].
    
    Args:
        s: Style variable array, shape (n,), typically S_style ~ N(0,1)
        method: Mapping method, default 'gauss_cdf'
    
    Returns:
        Style parameters b in [0, 1], shape (n,)
    """
    s = np.asarray(s).flatten()
    eps = 1e-8
    
    if method == "gauss_cdf":
        # For standard normal input, can use direct CDF
        # But for consistency, compute statistics from data
        mu_s = np.mean(s)
        sigma_s = np.std(s)
        
        z_scores = (s - mu_s) / (sigma_s + eps)
        b = stats.norm.cdf(z_scores)
        
    elif method == "sigmoid":
        # Alternative: b = 1/(1+exp(-s))
        b = 1.0 / (1.0 + np.exp(-s))
        
    else:
        raise ValueError(f"Unknown g_style method: {method}")
    
    # Ensure [0, 1] range
    b = np.clip(b, 0.0, 1.0)
    return b


def phi_readout_batch(images: np.ndarray, 
                     features: Tuple[str, ...] = ("brightness", "axis_angle", "stroke_density")) -> Dict[str, np.ndarray]:
    """
    Extract semantic features from batch of images.
    
    Args:
        images: Image batch, shape (n, H, W), values in [0, 1]
        features: Tuple of feature names to extract
    
    Returns:
        Dictionary with feature arrays, each shape (n,)
    """
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(f"Expected 3D image batch (n, H, W), got shape {images.shape}")
    
    n, H, W = images.shape
    result = {}
    
    for feature in features:
        if feature == "brightness":
            # Simple brightness: mean pixel value
            result["brightness"] = np.mean(images.reshape(n, -1), axis=1)
            
        elif feature == "axis_angle":
            # Principal axis angle via PCA
            angles = []
            for i in range(n):
                img = images[i]
                
                # Threshold to binary
                try:
                    threshold = threshold_otsu(img)
                    binary = img > threshold
                except:
                    # Fallback if Otsu fails
                    binary = img > 0.5
                
                # Get foreground coordinates
                coords = np.column_stack(np.where(binary))
                
                if len(coords) < 2:
                    # Not enough points for PCA
                    angles.append(0.0)
                    continue
                
                # PCA to find principal axis
                try:
                    pca = PCA(n_components=2)
                    pca.fit(coords)
                    
                    # First principal component
                    v = pca.components_[0]  # (dy, dx) in image coordinates
                    
                    # Convert to angle: arctan2(dy, dx)
                    angle = np.arctan2(v[0], v[1])
                    angles.append(angle)
                    
                except:
                    # PCA failed
                    angles.append(0.0)
            
            result["axis_angle"] = np.array(angles)
            
        elif feature == "stroke_density":
            # Foreground density
            densities = []
            for i in range(n):
                img = images[i]
                
                # Threshold to binary
                try:
                    threshold = threshold_otsu(img)
                    binary = img > threshold
                except:
                    binary = img > 0.5
                
                # Density = fraction of foreground pixels
                density = np.mean(binary.astype(float))
                densities.append(density)
            
            result["stroke_density"] = np.array(densities)
            
        else:
            raise ValueError(f"Unknown feature: {feature}")
    
    return result


# # Test functions
# if __name__ == "__main__":
#     print("=== Testing semantics.py ===")
    
#     # Set up test data
#     rng = np.random.default_rng(42)
#     n = 1000
    
#     # Test g_sem function
#     print("\n--- Testing g_sem ---")
    
#     # Generate various distributions
#     x_normal = rng.normal(0, 1, n)
#     x_uniform = rng.uniform(-2, 3, n)
#     x_skewed = rng.exponential(1, n) - 1  # Shifted exponential
    
#     print(f"x_normal: mean={x_normal.mean():.4f}, std={x_normal.std():.4f}, range=[{x_normal.min():.4f}, {x_normal.max():.4f}]")
#     print(f"x_uniform: mean={x_uniform.mean():.4f}, std={x_uniform.std():.4f}, range=[{x_uniform.min():.4f}, {x_uniform.max():.4f}]")
#     print(f"x_skewed: mean={x_skewed.mean():.4f}, std={x_skewed.std():.4f}, range=[{x_skewed.min():.4f}, {x_skewed.max():.4f}]")
    
#     # Test gauss_cdf method
#     a_normal_gauss = g_sem(x_normal, method="gauss_cdf")
#     a_uniform_gauss = g_sem(x_uniform, method="gauss_cdf")
#     a_skewed_gauss = g_sem(x_skewed, method="gauss_cdf")
    
#     print(f"\ng_sem(gauss_cdf) on normal: mean={a_normal_gauss.mean():.4f}, std={a_normal_gauss.std():.4f}, range=[{a_normal_gauss.min():.4f}, {a_normal_gauss.max():.4f}]")
#     print(f"g_sem(gauss_cdf) on uniform: mean={a_uniform_gauss.mean():.4f}, std={a_uniform_gauss.std():.4f}, range=[{a_uniform_gauss.min():.4f}, {a_uniform_gauss.max():.4f}]")
#     print(f"g_sem(gauss_cdf) on skewed: mean={a_skewed_gauss.mean():.4f}, std={a_skewed_gauss.std():.4f}, range=[{a_skewed_gauss.min():.4f}, {a_skewed_gauss.max():.4f}]")
    
#     # Test minmax method
#     a_normal_minmax = g_sem(x_normal, method="minmax")
#     a_uniform_minmax = g_sem(x_uniform, method="minmax")
#     a_skewed_minmax = g_sem(x_skewed, method="minmax")
    
#     print(f"\ng_sem(minmax) on normal: mean={a_normal_minmax.mean():.4f}, std={a_normal_minmax.std():.4f}, range=[{a_normal_minmax.min():.4f}, {a_normal_minmax.max():.4f}]")
#     print(f"g_sem(minmax) on uniform: mean={a_uniform_minmax.mean():.4f}, std={a_uniform_minmax.std():.4f}, range=[{a_uniform_minmax.min():.4f}, {a_uniform_minmax.max():.4f}]")
#     print(f"g_sem(minmax) on skewed: mean={a_skewed_minmax.mean():.4f}, std={a_skewed_minmax.std():.4f}, range=[{a_skewed_minmax.min():.4f}, {a_skewed_minmax.max():.4f}]")
    
#     # Test g_style function
#     print("\n--- Testing g_style ---")
    
#     s_style = rng.normal(0, 1, n)  # Standard normal style variable
#     print(f"s_style: mean={s_style.mean():.4f}, std={s_style.std():.4f}, range=[{s_style.min():.4f}, {s_style.max():.4f}]")
    
#     b_gauss = g_style(s_style, method="gauss_cdf")
#     b_sigmoid = g_style(s_style, method="sigmoid")
    
#     print(f"g_style(gauss_cdf): mean={b_gauss.mean():.4f}, std={b_gauss.std():.4f}, range=[{b_gauss.min():.4f}, {b_gauss.max():.4f}]")
#     print(f"g_style(sigmoid): mean={b_sigmoid.mean():.4f}, std={b_sigmoid.std():.4f}, range=[{b_sigmoid.min():.4f}, {b_sigmoid.max():.4f}]")
    
#     # Test phi_readout_batch function
#     print("\n--- Testing phi_readout_batch ---")
    
#     # Create synthetic images
#     n_imgs = 10
#     H, W = 28, 28
    
#     # Simple test images
#     test_images = np.zeros((n_imgs, H, W))
    
#     # Image 0: uniform brightness
#     test_images[0] = 0.5
    
#     # Image 1: bright image
#     test_images[1] = 0.8
    
#     # Image 2: dark image
#     test_images[2] = 0.2
    
#     # Image 3: horizontal line (should have specific axis angle)
#     test_images[3, H//2-1:H//2+2, W//4:3*W//4] = 1.0
    
#     # Image 4: vertical line
#     test_images[4, H//4:3*H//4, W//2-1:W//2+2] = 1.0
    
#     # Image 5: diagonal line
#     for i in range(min(H, W)):
#         if i < H and i < W:
#             test_images[5, i, i] = 1.0
    
#     # Images 6-9: random patterns
#     for i in range(6, n_imgs):
#         test_images[i] = rng.random((H, W))
    
#     print(f"Test images shape: {test_images.shape}")
#     print(f"Test images value range: [{test_images.min():.4f}, {test_images.max():.4f}]")
    
#     # Extract features
#     features = phi_readout_batch(test_images)
    
#     print(f"\nExtracted features:")
#     for feature_name, feature_values in features.items():
#         print(f"{feature_name}: shape={feature_values.shape}, mean={feature_values.mean():.4f}, std={feature_values.std():.4f}")
#         print(f"  values: {feature_values}")
    
#     # Test specific feature subsets
#     brightness_only = phi_readout_batch(test_images, features=("brightness",))
#     print(f"\nBrightness only: {list(brightness_only.keys())}")
    
#     # Test edge cases
#     print("\n--- Edge Cases ---")
    
#     # Test with extreme values
#     x_extreme = np.array([-1000, -1, 0, 1, 1000])
#     a_extreme = g_sem(x_extreme, method="gauss_cdf")
#     print(f"Extreme values: {x_extreme}")
#     print(f"g_sem(extreme): {a_extreme}")
    
#     # Test with constant values
#     x_constant = np.ones(100)
#     a_constant = g_sem(x_constant, method="gauss_cdf")
#     print(f"Constant input: std={x_constant.std():.6f}")
#     print(f"g_sem(constant): mean={a_constant.mean():.4f}, std={a_constant.std():.6f}")
    
#     # Test with single image
#     single_img = test_images[:1]
#     single_features = phi_readout_batch(single_img)
#     print(f"Single image features shapes: {[f'{k}: {v.shape}' for k, v in single_features.items()]}")
    
#     # Test error handling
#     try:
#         g_sem(x_normal, method="unknown")
#         print("ERROR: Should have raised ValueError")
#     except ValueError as e:
#         print(f"Correctly caught g_sem error: {e}")
    
#     try:
#         g_style(s_style, method="unknown")
#         print("ERROR: Should have raised ValueError")
#     except ValueError as e:
#         print(f"Correctly caught g_style error: {e}")
    
#     try:
#         phi_readout_batch(test_images.reshape(-1), features=("brightness",))
#         print("ERROR: Should have raised ValueError")
#     except ValueError as e:
#         print(f"Correctly caught phi_readout error: {e}")
    
#     try:
#         phi_readout_batch(test_images, features=("unknown_feature",))
#         print("ERROR: Should have raised ValueError")
#     except ValueError as e:
#         print(f"Correctly caught feature error: {e}")
    
#     print("\n=== semantics.py test completed ===")