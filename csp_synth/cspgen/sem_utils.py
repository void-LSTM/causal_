"""
Semantic utilities: nonlinear functions and noise distributions for SCM.
"""

import numpy as np
from typing import Union, Tuple


def h_square(x: np.ndarray) -> np.ndarray:
    """
    Square nonlinearity: h(x) = x^2
    
    Args:
        x: Input array, shape (n,)
    
    Returns:
        x^2, same shape as input
    """
    return np.square(x)


def h_sin(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Sine nonlinearity: h(x) = sin(gamma * x)
    
    Args:
        x: Input array, shape (n,)
        gamma: Frequency parameter, default 1.0
    
    Returns:
        sin(gamma * x), same shape as input
    """
    return np.sin(gamma * x)


def h_tanh(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Hyperbolic tangent nonlinearity: h(x) = tanh(gamma * x)
    
    Args:
        x: Input array, shape (n,)
        gamma: Scaling parameter, default 1.0
    
    Returns:
        tanh(gamma * x), same shape as input
    """
    return np.tanh(gamma * x)


def get_nonlinear_func(name: str, gamma: float = 1.0):
    """
    Get nonlinear function by name.
    
    Args:
        name: Function name ('square', 'sin', 'tanh')
        gamma: Parameter for sin/tanh functions
    
    Returns:
        Callable function
    """
    if name == "square":
        return h_square
    elif name == "sin":
        return lambda x: h_sin(x, gamma)
    elif name == "tanh":
        return lambda x: h_tanh(x, gamma)
    else:
        raise ValueError(f"Unknown nonlinear function: {name}")


def sample_gaussian_noise(shape: Union[int, Tuple[int, ...]], 
                         sigma: float, 
                         rng: np.random.Generator) -> np.ndarray:
    """
    Sample Gaussian noise: N(0, sigma^2)
    
    Args:
        shape: Output shape
        sigma: Standard deviation
        rng: Random number generator
    
    Returns:
        Gaussian noise array
    """
    return rng.normal(0.0, sigma, size=shape)


def sample_laplace_noise(shape: Union[int, Tuple[int, ...]], 
                        scale: float, 
                        rng: np.random.Generator) -> np.ndarray:
    """
    Sample Laplace noise: Laplace(0, scale)
    
    Args:
        shape: Output shape
        scale: Scale parameter (like sigma for Gaussian)
        rng: Random number generator
    
    Returns:
        Laplace noise array
    """
    return rng.laplace(0.0, scale, size=shape)


def sample_student_t_noise(shape: Union[int, Tuple[int, ...]], 
                          df: float, 
                          scale: float, 
                          rng: np.random.Generator) -> np.ndarray:
    """
    Sample Student's t noise: scale * t(df)
    
    Args:
        shape: Output shape
        df: Degrees of freedom
        scale: Scale parameter
        rng: Random number generator
    
    Returns:
        Student's t noise array
    """
    return scale * rng.standard_t(df, size=shape)


def get_noise_sampler(noise_type: str, **kwargs):
    """
    Get noise sampling function by type.
    
    Args:
        noise_type: 'gaussian', 'laplace', or 'student_t'
        **kwargs: Parameters for the noise distribution
    
    Returns:
        Callable function (shape, rng) -> noise_array
    """
    if noise_type == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        return lambda shape, rng: sample_gaussian_noise(shape, sigma, rng)
    elif noise_type == "laplace":
        scale = kwargs.get("scale", 1.0)
        return lambda shape, rng: sample_laplace_noise(shape, scale, rng)
    elif noise_type == "student_t":
        df = kwargs.get("df", 3.0)
        scale = kwargs.get("scale", 1.0)
        return lambda shape, rng: sample_student_t_noise(shape, df, scale, rng)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# # Test functions
# if __name__ == "__main__":
#     print("=== Testing sem_utils.py ===")
    
#     # Set up test data
#     rng = np.random.default_rng(42)
#     n = 1000
#     x = rng.normal(0, 1, n)  # Standard normal input
    
#     print(f"Input x: shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
#     print(f"Input x range: [{x.min():.4f}, {x.max():.4f}]")
    
#     # Test nonlinear functions
#     print("\n--- Nonlinear Functions ---")
    
#     # Test square
#     y_square = h_square(x)
#     print(f"h_square(x): shape={y_square.shape}, mean={y_square.mean():.4f}, std={y_square.std():.4f}")
#     print(f"h_square(x) range: [{y_square.min():.4f}, {y_square.max():.4f}]")
    
#     # Test sin
#     y_sin = h_sin(x, gamma=1.0)
#     print(f"h_sin(x, γ=1.0): shape={y_sin.shape}, mean={y_sin.mean():.4f}, std={y_sin.std():.4f}")
#     print(f"h_sin(x, γ=1.0) range: [{y_sin.min():.4f}, {y_sin.max():.4f}]")
    
#     # Test tanh
#     y_tanh = h_tanh(x, gamma=1.0)
#     print(f"h_tanh(x, γ=1.0): shape={y_tanh.shape}, mean={y_tanh.mean():.4f}, std={y_tanh.std():.4f}")
#     print(f"h_tanh(x, γ=1.0) range: [{y_tanh.min():.4f}, {y_tanh.max():.4f}]")
    
#     # Test function getter
#     print("\n--- Function Getter ---")
#     h1 = get_nonlinear_func("square")
#     h2 = get_nonlinear_func("sin", gamma=2.0)
#     h3 = get_nonlinear_func("tanh", gamma=0.5)
    
#     y1 = h1(x[:5])
#     y2 = h2(x[:5])
#     y3 = h3(x[:5])
    
#     print(f"get_nonlinear_func('square') on x[:5]: {y1}")
#     print(f"get_nonlinear_func('sin', γ=2.0) on x[:5]: {y2}")
#     print(f"get_nonlinear_func('tanh', γ=0.5) on x[:5]: {y3}")
    
#     # Test noise functions
#     print("\n--- Noise Functions ---")
    
#     # Test Gaussian noise
#     noise_gauss = sample_gaussian_noise(n, sigma=0.2, rng=rng)
#     print(f"Gaussian noise (σ=0.2): shape={noise_gauss.shape}, mean={noise_gauss.mean():.4f}, std={noise_gauss.std():.4f}")
    
#     # Test Laplace noise
#     noise_laplace = sample_laplace_noise(n, scale=0.2, rng=rng)
#     print(f"Laplace noise (scale=0.2): shape={noise_laplace.shape}, mean={noise_laplace.mean():.4f}, std={noise_laplace.std():.4f}")
    
#     # Test Student's t noise
#     noise_t = sample_student_t_noise(n, df=3.0, scale=0.2, rng=rng)
#     print(f"Student's t noise (df=3, scale=0.2): shape={noise_t.shape}, mean={noise_t.mean():.4f}, std={noise_t.std():.4f}")
    
#     # Test noise getter
#     print("\n--- Noise Getter ---")
#     noise_sampler1 = get_noise_sampler("gaussian", sigma=0.1)
#     noise_sampler2 = get_noise_sampler("laplace", scale=0.15)
#     noise_sampler3 = get_noise_sampler("student_t", df=5.0, scale=0.1)
    
#     n1 = noise_sampler1(100, rng)
#     n2 = noise_sampler2(100, rng)
#     n3 = noise_sampler3(100, rng)
    
#     print(f"Gaussian sampler (σ=0.1): mean={n1.mean():.4f}, std={n1.std():.4f}")
#     print(f"Laplace sampler (scale=0.15): mean={n2.mean():.4f}, std={n2.std():.4f}")
#     print(f"Student's t sampler (df=5, scale=0.1): mean={n3.mean():.4f}, std={n3.std():.4f}")
    
#     # Test edge cases
#     print("\n--- Edge Cases ---")
    
#     # Test with zero input
#     zeros = np.zeros(5)
#     print(f"h_square(zeros): {h_square(zeros)}")
#     print(f"h_sin(zeros): {h_sin(zeros)}")
#     print(f"h_tanh(zeros): {h_tanh(zeros)}")
    
#     # Test with different shapes
#     x_2d = rng.normal(0, 1, (10, 3))
#     noise_2d = sample_gaussian_noise((10, 3), sigma=0.1, rng=rng)
#     print(f"2D input shape: {x_2d.shape}")
#     print(f"2D noise shape: {noise_2d.shape}")
    
#     # Test error handling
#     try:
#         get_nonlinear_func("unknown")
#         print("ERROR: Should have raised ValueError")
#     except ValueError as e:
#         print(f"Correctly caught error: {e}")
    
#     try:
#         get_noise_sampler("unknown")
#         print("ERROR: Should have raised ValueError")
#     except ValueError as e:
#         print(f"Correctly caught error: {e}")
    
#     print("\n=== sem_utils.py test completed ===")