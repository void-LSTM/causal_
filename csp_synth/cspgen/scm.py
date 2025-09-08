"""
Structural Causal Model (SCM) implementation for T → M → Y* chain.
"""

import numpy as np
from typing import Dict, Optional, Any
from .sem_utils import get_nonlinear_func, get_noise_sampler
from .semantics import g_sem, g_style


class SCM:
    """
    Structural Causal Model implementing:
    T = ε_T
    M = α₁*T + ρ*h₁(T) + ε_M
    Y* = α₂*M + ρ*h₂(M) + δ*T + ε_Y
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        """
        Initialize SCM with configuration.
        
        Args:
            cfg: Configuration dictionary with 'scm' section
        """
        self.cfg = cfg.get('scm', {})
        
        # Structural parameters
        self.alpha1 = self.cfg.get('alpha1', 1.0)
        self.alpha2 = self.cfg.get('alpha2', 1.0)
        self.rho = self.cfg.get('rho', 0.5)
        self.delta = self.cfg.get('delta', 0.0)
        
        # Noise parameters
        self.sigma_T = self.cfg.get('sigma_T', 0.2)
        self.sigma_M = self.cfg.get('sigma_M', 0.2)
        self.sigma_Y = self.cfg.get('sigma_Y', 0.1)
        
        # Nonlinear functions
        self.h1_name = self.cfg.get('h1', 'square')
        self.h2_name = self.cfg.get('h2', 'tanh')
        self.gamma1 = self.cfg.get('gamma1', 1.0)
        self.gamma2 = self.cfg.get('gamma2', 1.0)
        
        # Y type
        self.Y_type = self.cfg.get('Y_type', 'cont')  # 'cont' or 'bin'
        
        # Misc variables
        self.q_misc = cfg.get('q_misc', 30)
        
        # Setup nonlinear functions
        self.h1 = get_nonlinear_func(self.h1_name, self.gamma1)
        self.h2 = get_nonlinear_func(self.h2_name, self.gamma2)
        
        # Setup noise samplers
        self.noise_T = get_noise_sampler('gaussian', sigma=self.sigma_T)
        self.noise_M = get_noise_sampler('gaussian', sigma=self.sigma_M)
        self.noise_Y = get_noise_sampler('gaussian', sigma=self.sigma_Y)
        
        print(f"SCM initialized:")
        print(f"  Structure: α₁={self.alpha1}, α₂={self.alpha2}, ρ={self.rho}, δ={self.delta}")
        print(f"  Noise: σ_T={self.sigma_T}, σ_M={self.sigma_M}, σ_Y={self.sigma_Y}")
        print(f"  Functions: h₁={self.h1_name}(γ={self.gamma1}), h₂={self.h2_name}(γ={self.gamma2})")
        print(f"  Y type: {self.Y_type}, q_misc: {self.q_misc}")
    
    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """
        Sample from the SCM.
        
        Args:
            n: Number of samples
            rng: Random number generator
        
        Returns:
            Dictionary with sampled variables
        """
        # Sample structural equations
        T, M, Y_star = self._sample_structural(n, rng)
        
        # Sample miscellaneous variables W
        W = self._sample_misc_variables(n, rng)
        
        # Sample style variable
        S_style = self._sample_style_variable(n, rng)
        
        # Sample confounders (placeholder, set to NaN if not used)
        C = np.full(n, np.nan, dtype=np.float32)
        
        # Prepare result dictionary
        result = {
            "T": T.astype(np.float32),
            "M": M.astype(np.float32), 
            "Y_star": Y_star.astype(np.float32),
            "W": W.astype(np.float32),
            "S_style": S_style.astype(np.float32),
            "C": C.astype(np.float32),
        }
        
        # Add binary Y if requested
        if self.Y_type == 'bin':
            Y_star_bin = self._binarize_Y(Y_star, rng)
            result["Y_star_bin"] = Y_star_bin.astype(np.int8)
        
        return result
    
    def _sample_structural(self, n: int, rng: np.random.Generator) -> tuple:
        """Sample the main structural equations T → M → Y*."""
        
        # T = ε_T
        epsilon_T = self.noise_T(n, rng)
        T = epsilon_T
        
        # M = α₁*T + ρ*h₁(T) + ε_M
        epsilon_M = self.noise_M(n, rng)
        linear_T = self.alpha1 * T
        nonlinear_T = self.rho * self.h1(T)
        M = linear_T + nonlinear_T + epsilon_M
        
        # Y* = α₂*M + ρ*h₂(M) + δ*T + ε_Y
        epsilon_Y = self.noise_Y(n, rng)
        linear_M = self.alpha2 * M
        nonlinear_M = self.rho * self.h2(M)
        leak_T = self.delta * T
        Y_star = linear_M + nonlinear_M + leak_T + epsilon_Y
        
        return T, M, Y_star
    
    def _sample_misc_variables(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample miscellaneous variables W."""
        if self.q_misc > 0:
            W = rng.normal(0, 1, size=(n, self.q_misc))
        else:
            W = np.empty((n, 0))  # Empty array with correct first dimension
        return W
    
    def _sample_style_variable(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample style variable S_style ~ N(0,1)."""
        return rng.normal(0, 1, size=n)
    
    def _binarize_Y(self, Y_star: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Convert continuous Y* to binary via logistic model.
        P(Y=1) = sigmoid(Y*)
        """
        prob = 1.0 / (1.0 + np.exp(-Y_star))
        Y_bin = rng.binomial(1, prob, size=len(Y_star))
        return Y_bin
    
    def compute_semantics(self, sample_dict: Dict[str, np.ndarray], 
                         cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Compute semantic parameters a_M, a_Y, b_style from sampled variables.
        
        Args:
            sample_dict: Dictionary from sample() method
            cfg: Full configuration including g_sem/g_style settings
        
        Returns:
            Dictionary with semantic parameters
        """
        # Extract variables
        M = sample_dict["M"]
        Y_star = sample_dict["Y_star"]
        S_style = sample_dict["S_style"]
        
        # Get semantic mapping configurations
        g_sem_cfg = cfg.get('g_sem', {})
        g_style_cfg = cfg.get('g_style', {})
        
        g_sem_method = g_sem_cfg.get('method', 'gauss_cdf')
        g_style_method = g_style_cfg.get('method', 'gauss_cdf')
        clip_quantiles = g_sem_cfg.get('clip_quantiles', (0.005, 0.995))
        
        # Compute semantic amplitudes
        a_M = g_sem(M, method=g_sem_method, clip_quantiles=clip_quantiles)
        a_Y = g_sem(Y_star, method=g_sem_method, clip_quantiles=clip_quantiles)
        
        # Compute style parameter
        b_style = g_style(S_style, method=g_style_method)
        
        return {
            "a_M": a_M.astype(np.float32),
            "a_Y": a_Y.astype(np.float32),
            "b_style": b_style.astype(np.float32)
        }
    
    def get_expected_correlations(self) -> Dict[str, float]:
        """
        Compute expected correlations in the SCM for validation.
        
        Returns:
            Dictionary with expected correlation patterns
        """
        # These are approximate expected correlations based on the linear components
        # For validation purposes
        
        expected = {}
        
        # T-M correlation (should be positive due to α₁ > 0)
        expected["T_M"] = "positive"
        
        # M-Y correlation (should be positive due to α₂ > 0)
        expected["M_Y"] = "positive"
        
        # T-Y correlation
        if self.delta == 0:
            expected["T_Y"] = "mediated_only"  # Only through M
        else:
            expected["T_Y"] = "positive_direct"  # Direct + mediated
        
        # Conditional independence
        if self.delta == 0:
            expected["T_indep_Y_given_M"] = True
        else:
            expected["T_indep_Y_given_M"] = False
        
        return expected


# # Test functions
# if __name__ == "__main__":
#     print("=== Testing scm.py ===")
    
#     # Setup test configuration
#     test_cfg = {
#         'scm': {
#             'alpha1': 1.0,
#             'alpha2': 1.0,
#             'rho': 0.5,
#             'delta': 0.0,
#             'sigma_T': 0.2,
#             'sigma_M': 0.2,
#             'sigma_Y': 0.1,
#             'h1': 'square',
#             'h2': 'tanh',
#             'gamma1': 1.0,
#             'gamma2': 1.0,
#             'Y_type': 'cont'
#         },
#         'q_misc': 30,
#         'g_sem': {
#             'method': 'gauss_cdf',
#             'clip_quantiles': [0.005, 0.995]
#         },
#         'g_style': {
#             'method': 'gauss_cdf'
#         }
#     }
    
#     rng = np.random.default_rng(42)
    
#     # Test SCM initialization
#     print("\n--- Testing SCM initialization ---")
#     scm = SCM(test_cfg)
    
#     expected = scm.get_expected_correlations()
#     print(f"Expected correlation patterns: {expected}")
    
#     # Test basic sampling
#     print("\n--- Testing basic sampling ---")
    
#     n_test = 5000
#     sample = scm.sample(n_test, rng)
    
#     print(f"Sample keys: {list(sample.keys())}")
#     print(f"Sample shapes and types:")
#     for key, value in sample.items():
#         print(f"  {key}: shape={value.shape}, dtype={value.dtype}, mean={value.mean():.4f}, std={value.std():.4f}")
    
#     # Test semantic computation
#     print("\n--- Testing semantic computation ---")
    
#     semantics = scm.compute_semantics(sample, test_cfg)
#     print(f"Semantic keys: {list(semantics.keys())}")
#     print(f"Semantic shapes and ranges:")
#     for key, value in semantics.items():
#         print(f"  {key}: shape={value.shape}, range=[{value.min():.4f}, {value.max():.4f}], mean={value.mean():.4f}")
    
#     # Test correlations
#     print("\n--- Testing correlations ---")
    
#     T = sample["T"]
#     M = sample["M"] 
#     Y_star = sample["Y_star"]
    
#     corr_TM = np.corrcoef(T, M)[0, 1]
#     corr_MY = np.corrcoef(M, Y_star)[0, 1]
#     corr_TY = np.corrcoef(T, Y_star)[0, 1]
    
#     print(f"Correlation T-M: {corr_TM:.4f} (expected: positive)")
#     print(f"Correlation M-Y*: {corr_MY:.4f} (expected: positive)")
#     print(f"Correlation T-Y*: {corr_TY:.4f} (expected: mediated, δ={scm.delta})")
    
#     # Test nonlinear effects
#     print("\n--- Testing nonlinear effects ---")
    
#     # Compare linear vs total effects
#     T_sample = T[:1000]
#     M_sample = M[:1000]
#     Y_sample = Y_star[:1000]
    
#     # Linear component of M given T
#     M_linear = scm.alpha1 * T_sample
#     M_nonlinear = scm.rho * scm.h1(T_sample)
    
#     print(f"M linear component std: {M_linear.std():.4f}")
#     print(f"M nonlinear component std: {M_nonlinear.std():.4f}")
#     print(f"M noise component std: {scm.sigma_M:.4f}")
#     print(f"Total M std: {M_sample.std():.4f}")
    
#     # Test binary Y option
#     print("\n--- Testing binary Y ---")
    
#     test_cfg_bin = test_cfg.copy()
#     test_cfg_bin['scm']['Y_type'] = 'bin'
    
#     scm_bin = SCM(test_cfg_bin)
#     sample_bin = scm_bin.sample(1000, rng)
    
#     Y_star_bin = sample_bin["Y_star"]
#     Y_bin = sample_bin["Y_star_bin"]
    
#     print(f"Binary Y available: {'Y_star_bin' in sample_bin}")
#     print(f"Y_star (continuous) range: [{Y_star_bin.min():.4f}, {Y_star_bin.max():.4f}]")
#     print(f"Y_star_bin (binary) values: {np.unique(Y_bin)}")
#     print(f"Y_star_bin mean (prob of 1): {Y_bin.mean():.4f}")
    
#     # Test relationship between Y_star and Y_bin
#     prob_theoretical = 1.0 / (1.0 + np.exp(-Y_star_bin))
#     print(f"Theoretical P(Y=1): mean={prob_theoretical.mean():.4f}, std={prob_theoretical.std():.4f}")
    
#     # Test different configurations
#     print("\n--- Testing different configurations ---")
    
#     configs_to_test = [
#         {"name": "High nonlinearity", "rho": 1.0},
#         {"name": "No nonlinearity", "rho": 0.0},
#         {"name": "With leakage", "delta": 0.1},
#         {"name": "Different functions", "h1": "sin", "h2": "square"},
#         {"name": "High noise", "sigma_M": 0.4, "sigma_Y": 0.2}
#     ]
    
#     for config_test in configs_to_test:
#         print(f"\n{config_test['name']}:")
        
#         cfg_variant = test_cfg.copy()
#         cfg_variant['scm'].update({k: v for k, v in config_test.items() if k != 'name'})
        
#         scm_variant = SCM(cfg_variant)
#         sample_variant = scm_variant.sample(2000, rng)
        
#         T_v = sample_variant["T"]
#         M_v = sample_variant["M"]
#         Y_v = sample_variant["Y_star"]
        
#         corr_TM_v = np.corrcoef(T_v, M_v)[0, 1]
#         corr_MY_v = np.corrcoef(M_v, Y_v)[0, 1]
#         corr_TY_v = np.corrcoef(T_v, Y_v)[0, 1]
        
#         print(f"  Correlations: T-M={corr_TM_v:.3f}, M-Y={corr_MY_v:.3f}, T-Y={corr_TY_v:.3f}")
#         print(f"  Std devs: T={T_v.std():.3f}, M={M_v.std():.3f}, Y={Y_v.std():.3f}")
    
#     # Test edge cases
#     print("\n--- Testing edge cases ---")
    
#     # Small sample
#     small_sample = scm.sample(10, rng)
#     print(f"Small sample (n=10) shapes: {[f'{k}: {v.shape}' for k, v in small_sample.items()]}")
    
#     # Zero misc variables
#     cfg_no_misc = test_cfg.copy()
#     cfg_no_misc['q_misc'] = 0
#     scm_no_misc = SCM(cfg_no_misc)
#     sample_no_misc = scm_no_misc.sample(100, rng)
#     print(f"No misc variables W shape: {sample_no_misc['W'].shape}")
    
#     # Test semantic independence (should be maintained)
#     print("\n--- Testing semantic independence simulation ---")
    
#     # This simulates the test that will be done in sanity checking
#     sample_large = scm.sample(5000, rng)
#     semantics_large = scm.compute_semantics(sample_large, test_cfg)
    
#     # Generate simulated class_ids (uniform random, independent of semantics)
#     rng_class = np.random.default_rng(123)  # Different seed
#     class_ids_sim = rng_class.integers(0, 10, size=5000)
    
#     # Test independence using ANOVA
#     from scipy import stats
    
#     a_M = semantics_large["a_M"]
#     a_Y = semantics_large["a_Y"]
    
#     a_M_by_class = [a_M[class_ids_sim == c] for c in range(10)]
#     a_Y_by_class = [a_Y[class_ids_sim == c] for c in range(10)]
    
#     f_stat_M, p_val_M = stats.f_oneway(*a_M_by_class)
#     f_stat_Y, p_val_Y = stats.f_oneway(*a_Y_by_class)
    
#     print(f"Semantic-class independence test:")
#     print(f"  a_M vs class: F={f_stat_M:.4f}, p={p_val_M:.4f} (expect p>0.1)")
#     print(f"  a_Y vs class: F={f_stat_Y:.4f}, p={p_val_Y:.4f} (expect p>0.1)")
    
#     print("\n=== scm.py test completed ===")