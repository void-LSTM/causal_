"""
Image transformation functions for semantic manipulation.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Union, Optional, Tuple
import warnings


def apply_rotation(img: np.ndarray, 
                  degrees: float, 
                  fill_value: float = 0.5) -> np.ndarray:
    """
    Apply rotation transformation to image.
    
    Args:
        img: Input image, shape (H, W), values in [0, 1]
        degrees: Rotation angle in degrees (positive = counterclockwise)
        fill_value: Fill value for empty areas after rotation
    
    Returns:
        Rotated image, same shape as input
    """
    if abs(degrees) < 1e-6:
        return img.copy()
    
    rotated = ndimage.rotate(
        img, 
        degrees, 
        reshape=False,     # Keep original shape
        order=1,          # Bilinear interpolation
        mode="constant",  # Fill with constant value
        cval=fill_value,  # Fill value
        prefilter=False   # No pre-filtering
    )
    
    # Ensure [0, 1] range
    rotated = np.clip(rotated, 0.0, 1.0)
    return rotated


def apply_brightness(img: np.ndarray, 
                    beta: float, 
                    a: float) -> np.ndarray:
    """
    Apply brightness adjustment.
    Formula: I = clip(I0 + beta * (a - 0.5), 0, 1)
    
    Args:
        img: Input image, shape (H, W), values in [0, 1]
        beta: Brightness adjustment strength
        a: Semantic amplitude in [0, 1]
    
    Returns:
        Brightness-adjusted image
    """
    brightness_delta = beta * (a - 0.5)
    adjusted = img + brightness_delta
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return adjusted


def apply_contrast(img: np.ndarray, 
                  gamma: float, 
                  a: float) -> np.ndarray:
    """
    Apply contrast adjustment around image mean.
    Formula: gain = 1 + gamma * (2*a - 1)
             I = clip(mu + gain * (I0 - mu), 0, 1)
    
    Args:
        img: Input image, shape (H, W), values in [0, 1]
        gamma: Contrast adjustment strength
        a: Semantic amplitude in [0, 1]
    
    Returns:
        Contrast-adjusted image
    """
    # Image mean as center point
    mu = np.mean(img)
    
    # Contrast gain
    gain = 1.0 + gamma * (2.0 * a - 1.0)
    
    # Apply contrast around mean
    adjusted = mu + gain * (img - mu)
    adjusted = np.clip(adjusted, 0.0, 1.0)
    return adjusted


def apply_noise(img: np.ndarray, 
               sigma_pix: float, 
               b: float, 
               rng: np.random.Generator) -> np.ndarray:
    """
    Apply pixel noise based on style parameter.
    Formula: sigma = sigma_pix * b
             I = clip(I_sem + Normal(0, sigma^2), 0, 1)
    
    Args:
        img: Input image, shape (H, W), values in [0, 1]
        sigma_pix: Base noise strength
        b: Style parameter in [0, 1]
        rng: Random number generator
    
    Returns:
        Noise-added image
    """
    if sigma_pix <= 0 or b <= 0:
        return img.copy()
    
    # Noise standard deviation
    sigma = sigma_pix * b
    
    # Generate noise
    noise = rng.normal(0.0, sigma, size=img.shape)
    
    # Add noise and clip
    noisy = img + noise
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


def get_transform_params(s_level: str) -> Dict[str, float]:
    """
    Get transform parameters for different strength levels.
    
    Args:
        s_level: Strength level ('small', 'mid', 'large')
    
    Returns:
        Dictionary with theta_deg, beta, gamma parameters
    """
    params_map = {
        'small': {'theta_deg': 10.0, 'beta': 0.10, 'gamma': 0.10},
        'mid':   {'theta_deg': 25.0, 'beta': 0.25, 'gamma': 0.25},
        'large': {'theta_deg': 45.0, 'beta': 0.40, 'gamma': 0.40},
    }
    
    if s_level not in params_map:
        raise ValueError(f"Unknown s_level: {s_level}. Must be one of {list(params_map.keys())}")
    
    return params_map[s_level]


def compute_rotation_angle(a: float, theta_deg: float) -> float:
    """
    Compute rotation angle from semantic amplitude.
    Formula: degrees = theta_deg * (2*a - 1)
    
    Args:
        a: Semantic amplitude in [0, 1]
        theta_deg: Maximum rotation angle
    
    Returns:
        Rotation angle in degrees
    """
    return theta_deg * (2.0 * a - 1.0)


def apply_transform_sequence(img: np.ndarray,
                           a: float,
                           b: float,
                           cfg_img: Dict[str, Union[float, bool]],
                           rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply complete transformation sequence to image.
    Order: rotation → brightness → contrast → noise
    
    Args:
        img: Input image, shape (H, W), values in [0, 1]
        a: Semantic amplitude in [0, 1]
        b: Style parameter in [0, 1]
        cfg_img: Image configuration dictionary
        rng: Random number generator
    
    Returns:
        Tuple of (transformed_image, applied_params)
    """
    # Get parameters
    theta_deg = cfg_img.get('theta_deg', 0.0)
    beta = cfg_img.get('beta', 0.25)
    gamma = cfg_img.get('gamma', 0.25)
    sigma_pix = cfg_img.get('sigma_pix', 0.0)
    
    # Track applied parameters
    applied = {}
    
    # Start with input image
    result = img.copy()
    
    # 1. Rotation (if enabled)
    if theta_deg > 0:
        degrees = compute_rotation_angle(a, theta_deg)
        result = apply_rotation(result, degrees)
        applied['degrees'] = degrees
    else:
        applied['degrees'] = 0.0
    
    # 2. Brightness
    brightness_delta = beta * (a - 0.5)
    result = apply_brightness(result, beta, a)
    applied['brightness_delta'] = brightness_delta
    
    # 3. Contrast
    contrast_gain = 1.0 + gamma * (2.0 * a - 1.0)
    result = apply_contrast(result, gamma, a)
    applied['contrast_gain'] = contrast_gain
    
    # 4. Noise
    noise_sigma = sigma_pix * b
    result = apply_noise(result, sigma_pix, b, rng)
    applied['noise_sigma'] = noise_sigma
    
    return result, applied


def batch_transform(images: np.ndarray,
                   a_values: np.ndarray,
                   b_values: np.ndarray,
                   cfg_img: Dict[str, Union[float, bool]],
                   rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply transformations to a batch of images.
    
    Args:
        images: Input images, shape (n, H, W), values in [0, 1]
        a_values: Semantic amplitudes, shape (n,)
        b_values: Style parameters, shape (n,)
        cfg_img: Image configuration dictionary
        rng: Random number generator
    
    Returns:
        Tuple of (transformed_images, applied_params_dict)
    """
    n, H, W = images.shape
    
    # Initialize output arrays
    transformed = np.zeros_like(images)
    applied_params = {
        'degrees': np.zeros(n, dtype=np.float32),
        'brightness_delta': np.zeros(n, dtype=np.float32),
        'contrast_gain': np.zeros(n, dtype=np.float32),
        'noise_sigma': np.zeros(n, dtype=np.float32),
    }
    
    # Transform each image
    for i in range(n):
        img_transformed, params = apply_transform_sequence(
            images[i], a_values[i], b_values[i], cfg_img, rng
        )
        transformed[i] = img_transformed
        
        # Store parameters
        for key, value in params.items():
            applied_params[key][i] = value
    
    return transformed, applied_params


# # Test functions
# if __name__ == "__main__":
#     print("=== Testing transforms.py ===")
    
#     # Setup test data
#     rng = np.random.default_rng(42)
    
#     # Create test images
#     H, W = 28, 28
#     n_test = 5
    
#     # Test image 1: uniform gray
#     img_uniform = np.full((H, W), 0.5)
    
#     # Test image 2: gradient
#     img_gradient = np.linspace(0, 1, H)[:, np.newaxis] * np.ones((1, W))
    
#     # Test image 3: checkerboard pattern
#     img_checker = np.zeros((H, W))
#     for i in range(H):
#         for j in range(W):
#             if (i // 4 + j // 4) % 2 == 0:
#                 img_checker[i, j] = 1.0
    
#     # Test image 4: circle
#     center = H // 2
#     img_circle = np.zeros((H, W))
#     for i in range(H):
#         for j in range(W):
#             if (i - center)**2 + (j - center)**2 <= (H//4)**2:
#                 img_circle[i, j] = 1.0
    
#     # Test image 5: random noise
#     img_random = rng.random((H, W))
    
#     test_images = np.stack([img_uniform, img_gradient, img_checker, img_circle, img_random])
    
#     print(f"Test images shape: {test_images.shape}")
#     print(f"Test images value range: [{test_images.min():.4f}, {test_images.max():.4f}]")
    
#     # Test parameter mapping
#     print("\n--- Testing parameter mapping ---")
    
#     for s_level in ['small', 'mid', 'large']:
#         params = get_transform_params(s_level)
#         print(f"{s_level}: {params}")
    
#     # Test individual transforms
#     print("\n--- Testing individual transforms ---")
    
#     test_img = img_circle.copy()
#     a_test = 0.8  # High semantic amplitude
#     b_test = 0.6  # Medium style parameter
    
#     print(f"Original image: mean={test_img.mean():.4f}, std={test_img.std():.4f}")
    
#     # Test rotation
#     degrees_test = compute_rotation_angle(a_test, 45.0)
#     img_rotated = apply_rotation(test_img, degrees_test)
#     print(f"Rotation ({degrees_test:.1f}°): mean={img_rotated.mean():.4f}, std={img_rotated.std():.4f}")
    
#     # Test brightness
#     img_bright = apply_brightness(test_img, 0.3, a_test)
#     print(f"Brightness (β=0.3, a={a_test}): mean={img_bright.mean():.4f}, std={img_bright.std():.4f}")
    
#     # Test contrast
#     img_contrast = apply_contrast(test_img, 0.5, a_test)
#     print(f"Contrast (γ=0.5, a={a_test}): mean={img_contrast.mean():.4f}, std={img_contrast.std():.4f}")
    
#     # Test noise
#     img_noisy = apply_noise(test_img, 0.1, b_test, rng)
#     print(f"Noise (σ_pix=0.1, b={b_test}): mean={img_noisy.mean():.4f}, std={img_noisy.std():.4f}")
    
#     # Test transform sequence
#     print("\n--- Testing transform sequence ---")
    
#     cfg_test = {
#         'theta_deg': 25.0,
#         'beta': 0.25,
#         'gamma': 0.25,
#         'sigma_pix': 0.1
#     }
    
#     img_seq, params_seq = apply_transform_sequence(test_img, a_test, b_test, cfg_test, rng)
#     print(f"Sequential transform result: mean={img_seq.mean():.4f}, std={img_seq.std():.4f}")
#     print(f"Applied parameters: {params_seq}")
    
#     # Test semantic amplitude effects
#     print("\n--- Testing semantic amplitude effects ---")
    
#     a_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
#     b_fixed = 0.5
    
#     for i, a in enumerate(a_values):
#         img_transformed, params = apply_transform_sequence(test_img, a, b_fixed, cfg_test, rng)
        
#         degrees = params['degrees']
#         brightness = params['brightness_delta']
#         contrast = params['contrast_gain']
        
#         print(f"a={a:.2f}: rotation={degrees:+.1f}°, brightness={brightness:+.3f}, contrast={contrast:.3f}, mean={img_transformed.mean():.4f}")
    
#     # Test style parameter effects
#     print("\n--- Testing style parameter effects ---")
    
#     a_fixed = 0.7
#     b_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
#     for i, b in enumerate(b_values):
#         img_transformed, params = apply_transform_sequence(test_img, a_fixed, b, cfg_test, rng)
        
#         noise_sigma = params['noise_sigma']
        
#         print(f"b={b:.2f}: noise_sigma={noise_sigma:.4f}, mean={img_transformed.mean():.4f}, std={img_transformed.std():.4f}")
    
#     # Test batch transformation
#     print("\n--- Testing batch transformation ---")
    
#     n_batch = len(test_images)
#     a_batch = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
#     b_batch = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
#     transformed_batch, params_batch = batch_transform(test_images, a_batch, b_batch, cfg_test, rng)
    
#     print(f"Batch input shape: {test_images.shape}")
#     print(f"Batch output shape: {transformed_batch.shape}")
#     print(f"Batch params shapes: {[f'{k}: {v.shape}' for k, v in params_batch.items()]}")
    
#     # Check individual image transformations
#     for i in range(n_batch):
#         orig_mean = test_images[i].mean()
#         trans_mean = transformed_batch[i].mean()
#         degrees = params_batch['degrees'][i]
#         brightness = params_batch['brightness_delta'][i]
        
#         print(f"  Image {i}: {orig_mean:.3f} → {trans_mean:.3f}, rot={degrees:+.1f}°, bright={brightness:+.3f}")
    
#     # Test edge cases
#     print("\n--- Testing edge cases ---")
    
#     # Zero parameters
#     cfg_zero = {'theta_deg': 0.0, 'beta': 0.0, 'gamma': 0.0, 'sigma_pix': 0.0}
#     img_zero, params_zero = apply_transform_sequence(test_img, 0.5, 0.5, cfg_zero, rng)
    
#     print(f"Zero parameters: difference from original = {np.abs(img_zero - test_img).max():.6f}")
#     print(f"Zero params: {params_zero}")
    
#     # Extreme values
#     img_extreme, params_extreme = apply_transform_sequence(test_img, 1.0, 1.0, cfg_test, rng)
#     print(f"Extreme values (a=1, b=1): mean={img_extreme.mean():.4f}, std={img_extreme.std():.4f}")
#     print(f"Extreme params: {params_extreme}")
    
#     # Value range preservation
#     print(f"Value range preservation: [{img_extreme.min():.4f}, {img_extreme.max():.4f}]")
    
#     # Test monotonicity of brightness
#     print("\n--- Testing monotonicity ---")
    
#     a_mono = np.linspace(0, 1, 11)
#     brightness_deltas = []
#     rotation_angles = []
#     contrast_gains = []
    
#     for a in a_mono:
#         _, params = apply_transform_sequence(test_img, a, 0.5, cfg_test, rng)
#         brightness_deltas.append(params['brightness_delta'])
#         rotation_angles.append(params['degrees'])
#         contrast_gains.append(params['contrast_gain'])
    
#     # Check monotonicity
#     brightness_mono = np.all(np.diff(brightness_deltas) >= 0)
#     rotation_mono = np.all(np.diff(rotation_angles) >= 0)
#     contrast_mono = np.all(np.diff(contrast_gains) >= 0)
    
#     print(f"Brightness monotonic: {brightness_mono}")
#     print(f"Rotation monotonic: {rotation_mono}")
#     print(f"Contrast monotonic: {contrast_mono}")
    
#     # Test error handling
#     print("\n--- Testing error handling ---")
    
#     try:
#         get_transform_params('unknown')
#         print("ERROR: Should have raised ValueError")
#     except ValueError as e:
#         print(f"Correctly caught error: {e}")
    
#     # Test with invalid image values (should still work due to clipping)
#     img_invalid = test_img + 2.0  # Values > 1
#     img_clipped, _ = apply_transform_sequence(img_invalid, 0.5, 0.5, cfg_test, rng)
#     print(f"Invalid input range handled: output range=[{img_clipped.min():.4f}, {img_clipped.max():.4f}]")
    
#     print("\n=== transforms.py test completed ===")