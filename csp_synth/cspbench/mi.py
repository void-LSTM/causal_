"""
Mutual Information (MI) and Conditional Mutual Information (CMI) estimation module.
Supports kNN-based KSG estimator and MINE neural estimator.
"""

import numpy as np
from typing import Dict, Union, Optional, Tuple, Any
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Import MI estimation libraries
try:
    from npeet import entropy_estimators as ee
    NPEET_AVAILABLE = True
except ImportError:
    NPEET_AVAILABLE = False
    warnings.warn("NPEET not available. Install with: pip install npeet")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. MINE estimator will not be available.")


def standardize_data(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Standardize arrays to zero mean and unit variance.
    
    Args:
        *arrays: Variable number of numpy arrays to standardize
    
    Returns:
        Tuple of standardized arrays
    """
    standardized = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        
        scaler = StandardScaler()
        arr_std = scaler.fit_transform(arr)
        
        # Return to original shape if it was 1D
        if arr.shape[1] == 1:
            arr_std = arr_std.flatten()
        
        standardized.append(arr_std)
    
    return tuple(standardized)


def knn_mutual_info(X: np.ndarray, 
                   Y: np.ndarray, 
                   k: int = 10,
                   standardize: bool = True) -> float:
    """
    Estimate mutual information using k-nearest neighbors (KSG estimator).
    
    Args:
        X: First variable, shape (n,) or (n, d1)
        Y: Second variable, shape (n,) or (n, d2)
        k: Number of nearest neighbors
        standardize: Whether to standardize variables
    
    Returns:
        Estimated mutual information in nats
    """
    if not NPEET_AVAILABLE:
        raise ImportError("NPEET not available. Install with: pip install npeet")
    
    # Prepare data
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n = len(X)
    assert len(Y) == n, f"X and Y must have same length: {len(X)} vs {len(Y)}"
    
    # Standardize if requested
    if standardize:
        X, Y = standardize_data(X, Y)
    
    try:
        # Use NPEET's mutual information estimator
        mi_estimate = ee.mi(X, Y, k=k)
        return float(mi_estimate)
        
    except Exception as e:
        warnings.warn(f"KNN MI estimation failed: {e}")
        return np.nan


def knn_conditional_mutual_info(X: np.ndarray,
                               Y: np.ndarray,
                               Z: np.ndarray,
                               k: int = 10,
                               standardize: bool = True) -> float:
    """
    Estimate conditional mutual information using difference method.
    I(X;Y|Z) = I(X;[Y,Z]) - I(X;Z)
    
    Args:
        X: First variable, shape (n,) or (n, d1)
        Y: Second variable, shape (n,) or (n, d2)
        Z: Conditioning variable, shape (n,) or (n, d3)
        k: Number of nearest neighbors
        standardize: Whether to standardize variables
    
    Returns:
        Estimated conditional mutual information in nats
    """
    # Prepare data
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    
    n = len(X)
    assert len(Y) == n and len(Z) == n, f"All variables must have same length: {len(X)}, {len(Y)}, {len(Z)}"
    
    # Standardize if requested
    if standardize:
        X, Y, Z = standardize_data(X, Y, Z)
    
    # Concatenate Y and Z for joint computation
    YZ = np.column_stack([Y, Z])
    
    try:
        # I(X; [Y,Z])
        mi_X_YZ = knn_mutual_info(X, YZ, k=k, standardize=False)  # Already standardized
        
        # I(X; Z)
        mi_X_Z = knn_mutual_info(X, Z, k=k, standardize=False)
        
        # I(X; Y | Z) = I(X; [Y,Z]) - I(X; Z)
        cmi_estimate = mi_X_YZ - mi_X_Z
        
        return float(cmi_estimate)
        
    except Exception as e:
        warnings.warn(f"KNN CMI estimation failed: {e}")
        return np.nan


def knn_entropy(X: np.ndarray, 
               k: int = 10,
               standardize: bool = True) -> float:
    """
    Estimate entropy using k-nearest neighbors.
    Note: This is a simplified implementation as NPEET entropy API varies.
    """
    if not NPEET_AVAILABLE:
        raise ImportError("NPEET not available. Install with: pip install npeet")
    
    # Prepare data
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Standardize if requested
    if standardize:
        X = standardize_data(X)[0]
        X = X.reshape(-1, 1)
    
    try:
        # Try different NPEET entropy API calls
        try:
            entropy_estimate = ee.entropy(X, k=k)
        except TypeError:
            # Alternative API
            entropy_estimate = ee.entropy(X.flatten(), k=k)
        
        return float(entropy_estimate)
        
    except Exception as e:
        warnings.warn(f"KNN entropy estimation failed: {e}")
        # Fallback: estimate using MI with uniform distribution
        try:
            # Simple fallback using differential entropy approximation
            return float(np.log(X.max() - X.min() + 1e-8))
        except:
            return np.nan

class MINEEstimator:
    """
    Mutual Information Neural Estimation (MINE) estimator.
    """
    
    def __init__(self, 
                 input_dim_x: int,
                 input_dim_y: int,
                 hidden_dim: int = 128,
                 lr: float = 1e-3):
        """
        Initialize MINE estimator.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # Neural network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim_x + input_dim_y, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
    
    def estimate_mi(self, 
                   X: np.ndarray, 
                   Y: np.ndarray,
                   epochs: int = 1000,
                   batch_size: int = 256,
                   standardize: bool = True) -> float:
        """
        Estimate mutual information using MINE.
        """
        # Prepare data
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        
        n = len(X)
        assert len(Y) == n, f"X and Y must have same length: {len(X)} vs {len(Y)}"
        
        # Standardize if requested
        if standardize:
            X, Y = standardize_data(X, Y)
            # Ensure proper shape after standardization
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
        
        # Convert to torch tensors with explicit 2D shape
        X_tensor = torch.FloatTensor(X).view(n, -1)  # Ensure 2D
        Y_tensor = torch.FloatTensor(Y).view(n, -1)  # Ensure 2D
        
        print(f"MINE tensor shapes: X={X_tensor.shape}, Y={Y_tensor.shape}")
        
        # Training loop
        mi_estimates = []
        
        for epoch in range(epochs):
            # Sample batch
            batch_size_actual = min(batch_size, n)
            batch_indices = np.random.choice(n, size=batch_size_actual, replace=False)
            X_batch = X_tensor[batch_indices]
            Y_batch = Y_tensor[batch_indices]
            
            # Joint samples
            joint_batch = torch.cat([X_batch, Y_batch], dim=1)
            
            # Marginal samples (shuffle Y)
            shuffle_indices = torch.randperm(len(Y_batch))
            Y_shuffled = Y_batch[shuffle_indices]
            marginal_batch = torch.cat([X_batch, Y_shuffled], dim=1)
            
            # Forward pass
            joint_scores = self.net(joint_batch)
            marginal_scores = self.net(marginal_batch)
            
            # MINE loss
            mi_estimate = torch.mean(joint_scores) - torch.log(torch.mean(torch.exp(marginal_scores)))
            loss = -mi_estimate
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            mi_estimates.append(mi_estimate.item())
            
            # Early stopping if converged
            if epoch > 100 and len(mi_estimates) > 50:
                recent_estimates = mi_estimates[-50:]
                if np.std(recent_estimates) < 0.01:
                    break
        
        # Return final estimate
        return float(np.mean(mi_estimates[-10:]))  # Average last 10 estimates


def mutual_info(X: np.ndarray,
               Y: np.ndarray,
               method: str = "knn",
               k: int = 10,
               standardize: bool = True,
               **kwargs) -> float:
    """
    Unified interface for mutual information estimation.
    
    Args:
        X: First variable
        Y: Second variable
        method: Estimation method ('knn' or 'mine')
        k: Number of nearest neighbors (for knn method)
        standardize: Whether to standardize variables
        **kwargs: Additional parameters for specific methods
    
    Returns:
        Estimated mutual information
    """
    if method == "knn":
        return knn_mutual_info(X, Y, k=k, standardize=standardize)
    
    elif method == "mine":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for MINE estimator")
        
        # Determine input dimensions
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        if X.ndim == 1:
            input_dim_x = 1
        else:
            input_dim_x = X.shape[1]
        
        if Y.ndim == 1:
            input_dim_y = 1
        else:
            input_dim_y = Y.shape[1]
        
        # Extract MINE-specific parameters
        mine_kwargs = {}
        estimate_kwargs = {}
        
        for key, value in kwargs.items():
            if key in ['hidden_dim', 'lr']:
                mine_kwargs[key] = value
            elif key in ['epochs', 'batch_size']:
                estimate_kwargs[key] = value
        
        # Initialize and train MINE estimator
        mine = MINEEstimator(input_dim_x, input_dim_y, **mine_kwargs)
        return mine.estimate_mi(X, Y, standardize=standardize, **estimate_kwargs)
    
    else:
        raise ValueError(f"Unknown MI estimation method: {method}")

def conditional_mutual_info(X: np.ndarray,
                           Y: np.ndarray,
                           Z: np.ndarray,
                           method: str = "knn",
                           k: int = 10,
                           standardize: bool = True,
                           **kwargs) -> float:
    """
    Unified interface for conditional mutual information estimation.
    
    Args:
        X: First variable
        Y: Second variable
        Z: Conditioning variable
        method: Estimation method (currently only 'knn' supported)
        k: Number of nearest neighbors
        standardize: Whether to standardize variables
        **kwargs: Additional parameters
    
    Returns:
        Estimated conditional mutual information
    """
    if method == "knn":
        return knn_conditional_mutual_info(X, Y, Z, k=k, standardize=standardize)
    else:
        raise ValueError(f"Unknown CMI estimation method: {method}. Currently only 'knn' is supported.")


def get_default_mi_config() -> Dict[str, Any]:
    """Get default configuration for MI estimation."""
    return {
        "method": "knn",
        "k": 10,
        "standardize": True
    }


# Test functions
if __name__ == "__main__":
    print("=== Testing mi.py ===")
    
    # Setup test data
    rng = np.random.default_rng(42)
    n = 2000
    
    print(f"Generating test data with n={n}")
    print(f"NPEET available: {NPEET_AVAILABLE}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    
    # Test Case 1: Independent variables (MI ≈ 0)
    print("\n--- Test Case 1: Independent variables ---")
    X_indep = rng.normal(0, 1, n)
    Y_indep = rng.normal(0, 1, n)
    
    print(f"X_indep: mean={X_indep.mean():.4f}, std={X_indep.std():.4f}")
    print(f"Y_indep: mean={Y_indep.mean():.4f}, std={Y_indep.std():.4f}")
    print(f"Empirical correlation: {np.corrcoef(X_indep, Y_indep)[0,1]:.4f}")
    
    if NPEET_AVAILABLE:
        mi_indep = mutual_info(X_indep, Y_indep, method="knn", k=10)
        print(f"MI(X_indep, Y_indep) = {mi_indep:.4f} nats (expected ≈ 0)")
    
    # Test Case 2: Linearly dependent variables
    print("\n--- Test Case 2: Linearly dependent variables ---")
    X_dep = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    Y_dep = 0.8 * X_dep + noise
    
    print(f"X_dep: mean={X_dep.mean():.4f}, std={X_dep.std():.4f}")
    print(f"Y_dep: mean={Y_dep.mean():.4f}, std={Y_dep.std():.4f}")
    print(f"Empirical correlation: {np.corrcoef(X_dep, Y_dep)[0,1]:.4f}")
    
    if NPEET_AVAILABLE:
        mi_dep = mutual_info(X_dep, Y_dep, method="knn", k=10)
        print(f"MI(X_dep, Y_dep) = {mi_dep:.4f} nats (expected > 0)")
        
        # Theoretical MI for Gaussian case
        correlation = np.corrcoef(X_dep, Y_dep)[0,1]
        mi_theoretical = -0.5 * np.log(1 - correlation**2)
        print(f"Theoretical MI (Gaussian assumption) = {mi_theoretical:.4f} nats")
    
    # Test Case 3: Nonlinear dependence
    print("\n--- Test Case 3: Nonlinear dependence ---")
    X_nonlin = rng.uniform(-2, 2, n)
    Y_nonlin = X_nonlin**2 + 0.5 * rng.normal(0, 1, n)
    
    print(f"X_nonlin: mean={X_nonlin.mean():.4f}, std={X_nonlin.std():.4f}")
    print(f"Y_nonlin: mean={Y_nonlin.mean():.4f}, std={Y_nonlin.std():.4f}")
    print(f"Empirical correlation: {np.corrcoef(X_nonlin, Y_nonlin)[0,1]:.4f}")
    
    if NPEET_AVAILABLE:
        mi_nonlin = mutual_info(X_nonlin, Y_nonlin, method="knn", k=10)
        print(f"MI(X_nonlin, Y_nonlin) = {mi_nonlin:.4f} nats (expected > linear case)")
    
    # Test Case 4: Conditional mutual information
    print("\n--- Test Case 4: Conditional mutual information ---")
    
    # Create T → M → Y chain
    T = rng.normal(0, 1, n)
    M = 0.7 * T + 0.3 * rng.normal(0, 1, n)
    Y_cond = 0.8 * M + 0.2 * rng.normal(0, 1, n)  # Y depends only on M
    Y_direct = 0.6 * M + 0.3 * T + 0.2 * rng.normal(0, 1, n)  # Y depends on both M and T
    
    print(f"T: mean={T.mean():.4f}, std={T.std():.4f}")
    print(f"M: mean={M.mean():.4f}, std={M.std():.4f}")
    print(f"Y_cond: mean={Y_cond.mean():.4f}, std={Y_cond.std():.4f}")
    print(f"Y_direct: mean={Y_direct.mean():.4f}, std={Y_direct.std():.4f}")
    
    if NPEET_AVAILABLE:
        # Unconditional mutual informations
        mi_TM = mutual_info(T, M, method="knn", k=10)
        mi_MY_cond = mutual_info(M, Y_cond, method="knn", k=10)
        mi_TY_cond = mutual_info(T, Y_cond, method="knn", k=10)
        mi_TY_direct = mutual_info(T, Y_direct, method="knn", k=10)
        
        print(f"MI(T, M) = {mi_TM:.4f} nats")
        print(f"MI(M, Y_cond) = {mi_MY_cond:.4f} nats")
        print(f"MI(T, Y_cond) = {mi_TY_cond:.4f} nats")
        print(f"MI(T, Y_direct) = {mi_TY_direct:.4f} nats")
        
        # Conditional mutual informations
        cmi_TY_given_M_cond = conditional_mutual_info(T, Y_cond, M, method="knn", k=10)
        cmi_TY_given_M_direct = conditional_mutual_info(T, Y_direct, M, method="knn", k=10)
        
        print(f"CMI(T; Y_cond | M) = {cmi_TY_given_M_cond:.4f} nats (expected ≈ 0)")
        print(f"CMI(T; Y_direct | M) = {cmi_TY_given_M_direct:.4f} nats (expected > 0)")
    
    # Test Case 5: Entropy estimation
    print("\n--- Test Case 5: Entropy estimation ---")

    if NPEET_AVAILABLE:
        try:
            # Gaussian entropy: H(X) = 0.5 * log(2πeσ²)
            X_gauss = rng.normal(0, 1, n)
            entropy_gauss = knn_entropy(X_gauss, k=10)
            entropy_theoretical = 0.5 * np.log(2 * np.pi * np.e)  # For σ=1
            
            print(f"Gaussian entropy (estimated): {entropy_gauss:.4f} nats")
            print(f"Gaussian entropy (theoretical): {entropy_theoretical:.4f} nats")
            
            # Uniform entropy: H(X) = log(b-a)
            X_uniform = rng.uniform(0, 2, n)
            entropy_uniform = knn_entropy(X_uniform, k=10)
            entropy_uniform_theoretical = np.log(2)  # log(2-0)
            
            print(f"Uniform entropy (estimated): {entropy_uniform:.4f} nats")
            print(f"Uniform entropy (theoretical): {entropy_uniform_theoretical:.4f} nats")
            
        except Exception as e:
            print(f"Entropy estimation test failed: {e}")
            print("Skipping entropy tests - using fallback estimates")
        
    # Test Case 6: Different k values
    print("\n--- Test Case 6: Effect of k parameter ---")
    
    if NPEET_AVAILABLE:
        k_values = [5, 10, 15, 20]
        mi_estimates = []
        
        for k in k_values:
            mi_k = mutual_info(X_dep, Y_dep, method="knn", k=k)
            mi_estimates.append(mi_k)
            print(f"k={k}: MI = {mi_k:.4f} nats")
        
        print(f"MI estimate std across k values: {np.std(mi_estimates):.4f}")
    
    # Test Case 7: MINE estimator (if available)
    if TORCH_AVAILABLE and NPEET_AVAILABLE:
        print("\n--- Test Case 7: MINE estimator ---")
        
        try:
            # Use smaller dataset for MINE
            n_mine = 500
            X_mine = X_dep[:n_mine]
            Y_mine = Y_dep[:n_mine]
            
            # Compare kNN and MINE
            mi_knn = mutual_info(X_mine, Y_mine, method="knn", k=10)
            print(f"MI (kNN): {mi_knn:.4f} nats")
            
            # MINE with reduced epochs for testing
            mi_mine = mutual_info(X_mine, Y_mine, method="mine", epochs=100, hidden_dim=64, batch_size=64)
            print(f"MI (MINE): {mi_mine:.4f} nats")
            print(f"Difference: {abs(mi_knn - mi_mine):.4f} nats")
            
        except Exception as e:
            print(f"MINE test failed: {e}")
            print("Skipping MINE test")
        
    # Test Case 8: Multivariate variables
    print("\n--- Test Case 8: Multivariate variables ---")
    
    if NPEET_AVAILABLE:
        # 2D variables
        X_multi = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n)
        Y_multi = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n)
        
        print(f"X_multi shape: {X_multi.shape}")
        print(f"Y_multi shape: {Y_multi.shape}")
        
        mi_multi_indep = mutual_info(X_multi, Y_multi, method="knn", k=10)
        print(f"MI(X_multi, Y_multi) independent = {mi_multi_indep:.4f} nats")
        
        # Dependent multivariate
        Y_multi_dep = X_multi @ np.array([[0.8, 0.2], [0.3, 0.7]]) + 0.3 * rng.normal(0, 1, (n, 2))
        mi_multi_dep = mutual_info(X_multi, Y_multi_dep, method="knn", k=10)
        print(f"MI(X_multi, Y_multi_dep) dependent = {mi_multi_dep:.4f} nats")
    
    # Test Case 9: Edge cases and error handling
    print("\n--- Test Case 9: Edge cases ---")
    
    if NPEET_AVAILABLE:
        # Small sample size
        X_small = rng.normal(0, 1, 50)
        Y_small = rng.normal(0, 1, 50)
        mi_small = mutual_info(X_small, Y_small, method="knn", k=5)
        print(f"Small sample MI (n=50): {mi_small:.4f} nats")
        
        # Constant variable
        X_const = np.ones(n)
        Y_const_test = rng.normal(0, 1, n)
        
        try:
            mi_const = mutual_info(X_const, Y_const_test, method="knn", k=10)
            print(f"Constant variable MI: {mi_const:.4f} nats")
        except Exception as e:
            print(f"Constant variable handling: {type(e).__name__}: {e}")
    
    # Test error handling
    print("\n--- Test Case 10: Error handling ---")
    
    try:
        # Mismatched lengths
        mutual_info(X_indep[:100], Y_indep[:200])
        print("ERROR: Should have raised assertion error")
    except Exception as e:
        print(f"Correctly caught error: {type(e).__name__}")
    
    try:
        # Unknown method
        mutual_info(X_indep, Y_indep, method="unknown")
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    try:
        # MINE without PyTorch
        if not TORCH_AVAILABLE:
            mutual_info(X_indep, Y_indep, method="mine")
            print("ERROR: Should have raised ImportError")
    except ImportError as e:
        print(f"Correctly caught error: MINE without PyTorch")
    
    # Test performance
    print("\n--- Performance test ---")
    
    if NPEET_AVAILABLE:
        import time
        
        # Large dataset test
        n_large = 5000
        X_large = rng.normal(0, 1, n_large)
        Y_large = 0.5 * X_large + 0.5 * rng.normal(0, 1, n_large)
        
        start_time = time.time()
        mi_large = mutual_info(X_large, Y_large, method="knn", k=10)
        end_time = time.time()
        
        print(f"Large dataset (n={n_large}): {end_time - start_time:.3f}s, MI = {mi_large:.4f}")
        
        # CMI performance test
        Z_large = rng.normal(0, 1, n_large)
        
        start_time = time.time()
        cmi_large = conditional_mutual_info(X_large, Y_large, Z_large, method="knn", k=10)
        end_time = time.time()
        
        print(f"Large CMI (n={n_large}): {end_time - start_time:.3f}s, CMI = {cmi_large:.4f}")
    
    # Test configuration
    print("\n--- Configuration test ---")
    
    config = get_default_mi_config()
    print(f"Default config: {config}")
    
    print("\n=== mi.py test completed ===")