"""
Conditional Independence (CI) testing module.
Supports HSIC, KCI, and Tigramite CMI-kNN methods.
"""

import numpy as np
from typing import Dict, Union, Optional, Tuple, Any
import warnings

# Import testing libraries
try:
    # 新版本tigramite的导入路径
    from tigramite.independence_tests.cmiknn import CMIknn
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    warnings.warn("Tigramite not available. Install with: pip install tigramite")

try:
    from hyppo.independence import HSIC
    HYPPO_AVAILABLE = True
except ImportError:
    HYPPO_AVAILABLE = False
    warnings.warn("Hyppo not available. Install with: pip install hyppo")

try:
    from causallearn.utils.cit import CIT
    CAUSALLEARN_AVAILABLE = True
except ImportError:
    CAUSALLEARN_AVAILABLE = False
    warnings.warn("Causal-learn not available. Install with: pip install causal-learn")

from sklearn.preprocessing import StandardScaler


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


def test_CI_tigramite_cmi(X: np.ndarray, 
                         Y: np.ndarray, 
                         Z: Optional[np.ndarray] = None,
                         alpha: float = 0.05,
                         knn: float = 0.2,
                         shuffles: int = 200,
                         standardize: bool = True) -> Dict[str, Any]:
    """
    Test conditional independence using Tigramite's CMI-kNN method.
    """
    if not TIGRAMITE_AVAILABLE:
        raise ImportError("Tigramite not available")
    
    # Prepare data
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    n = len(X)
    assert len(Y) == n, f"X and Y must have same length"
    
    # Handle conditioning variable
    if Z is not None:
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        assert len(Z) == n, f"Z must have same length as X,Y"
        
        # Combine data: [X, Y, Z]
        data = np.column_stack([X, Y, Z]).astype(np.float64)
        
        # xyz as numpy array of integers - this is what Tigramite expects
        xyz = np.array([0, 1, 2], dtype=np.int32)
        
    else:
        # Unconditional test
        data = np.column_stack([X, Y]).astype(np.float64)
        xyz = np.array([0, 1], dtype=np.int32)
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data).astype(np.float64)
    
    # Initialize CMI test
    try:
        cmi_test = CMIknn(
            knn=knn,
            shuffle_neighbors=5,
            significance='shuffle_test',
            transform='ranks',
            workers=1
        )
        
        # Call the method - just return whatever it gives us
        result = cmi_test.get_dependence_measure(data, xyz)
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            test_stat, p_value = result
        elif hasattr(result, '__iter__') and len(result) >= 2:
            test_stat, p_value = result[0], result[1]
        else:
            # If it's a single value, treat as test statistic with no p-value
            test_stat = float(result)
            p_value = np.nan
        
        reject = p_value < alpha if not np.isnan(p_value) else False
        
        return {
            "method": "tigramite_cmi_knn",
            "stat": float(test_stat),
            "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            "reject": bool(reject),
            "alpha": alpha,
            "n": n,
            "knn": knn,
            "shuffles": "N/A"
        }
        
    except Exception as e:
        warnings.warn(f"Tigramite failed: {e}. Using fallback method.")
        # Use reliable scipy fallback
        return test_CI_fallback_scipy(X.flatten(), Y.flatten(), 
                                    Z.flatten() if Z is not None else None, alpha)


def test_CI_fallback_scipy(X: np.ndarray, 
                          Y: np.ndarray, 
                          Z: Optional[np.ndarray] = None,
                          alpha: float = 0.05) -> Dict[str, Any]:
    """
    Reliable fallback CI test using scipy.
    """
    from scipy.stats import spearmanr, pearsonr
    
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()
    n = len(X)
    
    if Z is not None:
        # Conditional independence via partial correlation
        Z = np.asarray(Z).flatten()
        
        # Calculate pairwise correlations
        rxy, _ = spearmanr(X, Y)
        rxz, _ = spearmanr(X, Z) 
        ryz, _ = spearmanr(Y, Z)
        
        # Partial correlation formula
        numerator = rxy - rxz * ryz
        denominator = np.sqrt((1 - rxz**2) * (1 - ryz**2))
        
        if abs(denominator) < 1e-10:
            partial_corr = 0.0
        else:
            partial_corr = numerator / denominator
        
        # T-test for partial correlation
        if abs(partial_corr) >= 0.9999:
            p_value = 0.0 if abs(partial_corr) > 0.9999 else 1.0
        else:
            t_stat = partial_corr * np.sqrt((n - 3) / (1 - partial_corr**2))
            from scipy.stats import t
            p_value = 2 * (1 - t.cdf(np.abs(t_stat), df=n-3))
        
        return {
            "method": "scipy_partial_spearman",
            "stat": float(partial_corr),
            "p_value": float(p_value),
            "reject": bool(p_value < alpha),
            "alpha": alpha,
            "n": n
        }
    else:
        # Simple unconditional test
        stat, p_value = spearmanr(X, Y)
        
        return {
            "method": "scipy_spearman", 
            "stat": float(stat),
            "p_value": float(p_value),
            "reject": bool(p_value < alpha),
            "alpha": alpha,
            "n": len(X)
        }

def test_CI_hsic(X: np.ndarray, 
                Y: np.ndarray,
                alpha: float = 0.05,
                reps: int = 1000,
                standardize: bool = True) -> Dict[str, Any]:
    """
    Test unconditional independence using HSIC.
    
    Args:
        X: First variable, shape (n,) or (n, d1)
        Y: Second variable, shape (n,) or (n, d2)
        alpha: Significance level
        reps: Number of permutation repetitions
        standardize: Whether to standardize variables
    
    Returns:
        Dictionary with test results
    """
    if not HYPPO_AVAILABLE:
        raise ImportError("Hyppo not available. Install with: pip install hyppo")
    
    # Prepare data
    X = np.asarray(X).squeeze()
    Y = np.asarray(Y).squeeze()
    
    n = len(X)
    assert len(Y) == n, f"X and Y must have same length: {len(X)} vs {len(Y)}"
    
    # Standardize if requested
    if standardize:
        X, Y = standardize_data(X, Y)
    
    # Initialize HSIC test
    hsic_test = HSIC()
    
    try:
        # Perform test
        stat, p_value = hsic_test.test(X, Y, reps=reps, workers=1)
        
    except Exception as e:
        warnings.warn(f"HSIC test failed: {e}")
        return {
            "method": "hsic",
            "stat": np.nan,
            "p_value": np.nan,
            "reject": False,
            "alpha": alpha,
            "error": str(e),
            "n": n,
            "reps": reps
        }
    
    # Determine rejection
    reject = p_value < alpha
    
    return {
        "method": "hsic", 
        "stat": float(stat),
        "p_value": float(p_value),
        "reject": bool(reject),
        "alpha": alpha,
        "n": n,
        "reps": reps
    }


def test_CI_kci(X: np.ndarray,
               Y: np.ndarray, 
               Z: Optional[np.ndarray] = None,
               alpha: float = 0.05,
               standardize: bool = True) -> Dict[str, Any]:
    """
    Test conditional independence using KCI from causal-learn.
    """
    if not CAUSALLEARN_AVAILABLE:
        raise ImportError("Causal-learn not available")
    
    # Prepare data
    X = np.asarray(X, dtype=np.float64).flatten()
    Y = np.asarray(Y, dtype=np.float64).flatten()
    
    n = len(X)
    assert len(Y) == n, f"X and Y must have same length: {len(X)} vs {len(Y)}"
    
    if Z is not None:
        Z = np.asarray(Z, dtype=np.float64).flatten()
        assert len(Z) == n, f"Z must have same length as X,Y: {len(Z)} vs {n}"
    
    # Standardize if requested
    if standardize:
        if Z is not None:
            X, Y, Z = standardize_data(X, Y, Z)
        else:
            X, Y = standardize_data(X, Y)
    
    try:
        # Initialize KCI test with data - causal-learn expects data at initialization
        if Z is not None:
            # Conditional test: stack all variables as columns
            data_matrix = np.column_stack([X, Y, Z])
            cit = CIT(data_matrix, method='kci')
            # Test X ⊥ Y | Z: indices 0, 1, conditioning on [2]
            p_value = cit(0, 1, [2])
        else:
            # Unconditional test
            data_matrix = np.column_stack([X, Y])
            cit = CIT(data_matrix, method='kci')
            # Test X ⊥ Y: indices 0, 1, no conditioning
            p_value = cit(0, 1, [])
        
        # KCI doesn't return test statistic directly
        test_stat = np.nan
        
    except Exception as e:
        warnings.warn(f"KCI test failed: {e}")
        return {
            "method": "kci",
            "stat": np.nan,
            "p_value": np.nan,
            "reject": False,
            "alpha": alpha,
            "error": str(e),
            "n": n
        }
    
    # Determine rejection
    reject = p_value < alpha
    
    return {
        "method": "kci",
        "stat": float(test_stat),
        "p_value": float(p_value),
        "reject": bool(reject),
        "alpha": alpha,
        "n": n
    }


def test_CI(X: np.ndarray,
           Y: np.ndarray,
           Z: Optional[np.ndarray] = None,
           method: str = "tigramite_cmi_knn",
           alpha: float = 0.05,
           **kwargs) -> Dict[str, Any]:
    """
    Unified interface for conditional independence testing.
    
    Args:
        X: First variable
        Y: Second variable
        Z: Conditioning variable (None for unconditional test)
        method: Test method ('tigramite_cmi_knn', 'hsic', 'kci')
        alpha: Significance level
        **kwargs: Additional parameters for specific methods
    
    Returns:
        Dictionary with test results
    """
    if method == "tigramite_cmi_knn":
        return test_CI_tigramite_cmi(X, Y, Z, alpha=alpha, **kwargs)
    elif method == "hsic":
        if Z is not None:
            raise ValueError("HSIC only supports unconditional independence tests")
        return test_CI_hsic(X, Y, alpha=alpha, **kwargs)
    elif method == "kci":
        return test_CI_kci(X, Y, Z, alpha=alpha, **kwargs)
    else:
        raise ValueError(f"Unknown CI test method: {method}")


# 在get_default_ci_config()中修改：
def get_default_ci_config() -> Dict[str, Any]:
    return {
        "method": "kci",  # 改为KCI作为主要方法
        "alpha": 0.05,
        "standardize": True
    }


# Test functions
if __name__ == "__main__":
    print("=== Testing ci.py ===")
    
    # Setup test data
    rng = np.random.default_rng(42)
    n = 2000
    
    print(f"Generating test data with n={n}")
    
    # Test Case 1: Independent variables
    print("\n--- Test Case 1: Independent variables ---")
    X_indep = rng.normal(0, 1, n)
    Y_indep = rng.normal(0, 1, n)
    
    print(f"X_indep: mean={X_indep.mean():.4f}, std={X_indep.std():.4f}")
    print(f"Y_indep: mean={Y_indep.mean():.4f}, std={Y_indep.std():.4f}")
    print(f"Empirical correlation: {np.corrcoef(X_indep, Y_indep)[0,1]:.4f}")
    
    # Test Case 2: Dependent variables
    print("\n--- Test Case 2: Dependent variables ---")
    X_dep = rng.normal(0, 1, n)
    Y_dep = 0.7 * X_dep + 0.3 * rng.normal(0, 1, n)
    
    print(f"X_dep: mean={X_dep.mean():.4f}, std={X_dep.std():.4f}")
    print(f"Y_dep: mean={Y_dep.mean():.4f}, std={Y_dep.std():.4f}")
    print(f"Empirical correlation: {np.corrcoef(X_dep, Y_dep)[0,1]:.4f}")
    
    # Test Case 3: Conditional independence (T ⊥ Y | M)
    print("\n--- Test Case 3: Conditional independence ---")
    T = rng.normal(0, 1, n)
    M = 0.8 * T + 0.4 * rng.normal(0, 1, n)
    Y_cond = 0.6 * M + 0.3 * rng.normal(0, 1, n)  # Y depends only on M, not directly on T
    
    print(f"T: mean={T.mean():.4f}, std={T.std():.4f}")
    print(f"M: mean={M.mean():.4f}, std={M.std():.4f}")
    print(f"Y_cond: mean={Y_cond.mean():.4f}, std={Y_cond.std():.4f}")
    print(f"Correlations: T-M={np.corrcoef(T, M)[0,1]:.3f}, M-Y={np.corrcoef(M, Y_cond)[0,1]:.3f}, T-Y={np.corrcoef(T, Y_cond)[0,1]:.3f}")
    
    # Test Case 4: Conditional dependence (T ↛ Y | M with direct effect)
    print("\n--- Test Case 4: Conditional dependence ---")
    Y_direct = 0.6 * M + 0.2 * T + 0.3 * rng.normal(0, 1, n)  # Y depends on both M and T
    print(f"Y_direct: mean={Y_direct.mean():.4f}, std={Y_direct.std():.4f}")
    print(f"Correlations: T-Y_direct={np.corrcoef(T, Y_direct)[0,1]:.3f}")
    
    # Test available methods
    print("\n--- Testing available methods ---")
    print(f"Tigramite available: {TIGRAMITE_AVAILABLE}")
    print(f"Hyppo available: {HYPPO_AVAILABLE}")
    print(f"Causal-learn available: {CAUSALLEARN_AVAILABLE}")
    
    # Test HSIC (unconditional)
    if HYPPO_AVAILABLE:
        print("\n--- Testing HSIC ---")
        
        # Independent case (should accept H0: independence)
        result_hsic_indep = test_CI_hsic(X_indep, Y_indep, alpha=0.05, reps=500)
        print(f"HSIC independent: {result_hsic_indep}")
        
        # Dependent case (should reject H0: independence)
        result_hsic_dep = test_CI_hsic(X_dep, Y_dep, alpha=0.05, reps=500)
        print(f"HSIC dependent: {result_hsic_dep}")
        
        # Test T-M dependence
        result_hsic_TM = test_CI_hsic(T, M, alpha=0.05, reps=500)
        print(f"HSIC T-M: {result_hsic_TM}")
    
    # Test Tigramite CMI
    if TIGRAMITE_AVAILABLE:
        print("\n--- Testing Tigramite CMI ---")
        
        # Unconditional independence
        result_cmi_indep = test_CI_tigramite_cmi(X_indep, Y_indep, alpha=0.05, knn=10, shuffles=200)
        print(f"CMI independent: {result_cmi_indep}")
        
        # Unconditional dependence
        result_cmi_dep = test_CI_tigramite_cmi(X_dep, Y_dep, alpha=0.05, knn=10, shuffles=200)
        print(f"CMI dependent: {result_cmi_dep}")
        
        # Conditional independence T ⊥ Y | M
        result_cmi_cond_indep = test_CI_tigramite_cmi(T, Y_cond, M, alpha=0.05, knn=10, shuffles=200)
        print(f"CMI conditional independent (T⊥Y|M): {result_cmi_cond_indep}")
        
        # Conditional dependence T ↛ Y | M  
        result_cmi_cond_dep = test_CI_tigramite_cmi(T, Y_direct, M, alpha=0.05, knn=10, shuffles=200)
        print(f"CMI conditional dependent (T↛Y|M): {result_cmi_cond_dep}")
    
    # Test KCI
    if CAUSALLEARN_AVAILABLE:
        print("\n--- Testing KCI ---")
        
        try:
            # Unconditional independence
            result_kci_indep = test_CI_kci(X_indep, Y_indep, alpha=0.05)
            print(f"KCI independent: {result_kci_indep}")
            
            # Conditional independence
            result_kci_cond = test_CI_kci(T, Y_cond, M, alpha=0.05)
            print(f"KCI conditional (T⊥Y|M): {result_kci_cond}")
        except Exception as e:
            print(f"KCI test failed: {e}")
    
    # Test unified interface
    print("\n--- Testing unified interface ---")
    
    if TIGRAMITE_AVAILABLE:
        # Test with default configuration
        config = get_default_ci_config()
        print(f"Default config: {config}")
        
        # Unconditional test
        result_unified_uncond = test_CI(T, M, method="tigramite_cmi_knn", alpha=0.05, knn=10, shuffles=100)
        print(f"Unified unconditional (T-M): {result_unified_uncond}")
        
        # Conditional test
        result_unified_cond = test_CI(T, Y_cond, M, method="tigramite_cmi_knn", alpha=0.05, knn=10, shuffles=100)
        print(f"Unified conditional (T⊥Y|M): {result_unified_cond}")
    
    # Test error handling
    print("\n--- Testing error handling ---")
    
    try:
        # Mismatched lengths
        test_CI_tigramite_cmi(X_indep[:100], Y_indep[:200])
        print("ERROR: Should have raised assertion error")
    except Exception as e:
        print(f"Correctly caught error: {type(e).__name__}: {e}")
    
    try:
        # HSIC with conditioning variable (not supported)
        test_CI(X_indep, Y_indep, T, method="hsic")
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    try:
        # Unknown method
        test_CI(X_indep, Y_indep, method="unknown")
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    # Test performance and accuracy
    print("\n--- Testing performance ---")
    
    if TIGRAMITE_AVAILABLE:
        import time
        
        # Performance test
        start_time = time.time()
        result_perf = test_CI_tigramite_cmi(T, Y_cond, M, knn=10, shuffles=100)
        end_time = time.time()
        
        print(f"Performance test (n={n}, shuffles=100): {end_time - start_time:.3f}s")
        print(f"Result: p={result_perf['p_value']:.4f}, reject={result_perf['reject']}")
        
        # Accuracy test with known ground truth
        print("\n--- Accuracy validation ---")
        
        # Create data with known conditional independence
        n_acc = 5000
        T_acc = rng.normal(0, 1, n_acc)
        M_acc = T_acc + 0.5 * rng.normal(0, 1, n_acc)
        Y_acc = M_acc + 0.3 * rng.normal(0, 1, n_acc)  # Y ⊥ T | M
        
        # Multiple tests to check Type I error rate
        n_tests = 10
        rejections = 0
        
        for i in range(n_tests):
            result_acc = test_CI_tigramite_cmi(T_acc, Y_acc, M_acc, alpha=0.05, knn=10, shuffles=100)
            if result_acc['reject']:
                rejections += 1
        
        type1_error_rate = rejections / n_tests
        print(f"Type I error rate over {n_tests} tests: {type1_error_rate:.2f} (expected ≈ 0.05)")
    
    print("\n=== ci.py test completed ===")