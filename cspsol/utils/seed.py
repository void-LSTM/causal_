"""
Random seed management utilities for CSP framework.
Ensures reproducible experiments across different libraries and hardware.
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any, List
import warnings
import hashlib
import json


class SeedManager:
    """
    Comprehensive random seed management for reproducible experiments.
    Handles seeding for Python, NumPy, PyTorch, and system randomness.
    """
    
    def __init__(self, base_seed: int = 42):
        """
        Initialize seed manager.
        
        Args:
            base_seed: Base seed for generating derived seeds
        """
        self.base_seed = base_seed
        self.derived_seeds = {}
        self.seeding_history = []
        self.current_state = None
    
    def set_global_seed(self, seed: Optional[int] = None) -> int:
        """
        Set global random seed for all libraries.
        
        Args:
            seed: Seed value (uses base_seed if None)
            
        Returns:
            Actually used seed value
        """
        if seed is None:
            seed = self.base_seed
        
        # Record seeding action
        self.seeding_history.append({
            'action': 'set_global_seed',
            'seed': seed,
            'timestamp': self._get_timestamp()
        })
        
        # Set seeds for all libraries
        self._set_python_seed(seed)
        self._set_numpy_seed(seed)
        self._set_torch_seed(seed)
        self._set_system_seed(seed)
        
        # Store current state
        self.current_state = self._capture_state()
        
        return seed
    
    def set_deterministic_mode(self, enabled: bool = True):
        """
        Enable/disable deterministic computation mode.
        
        Args:
            enabled: Whether to enable deterministic mode
        """
        if torch.cuda.is_available():
            # PyTorch deterministic mode
            torch.backends.cudnn.deterministic = enabled
            torch.backends.cudnn.benchmark = not enabled
            
            if enabled:
                # Use deterministic algorithms when possible
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    # Fallback for older PyTorch versions
                    pass
        
        # Record action
        self.seeding_history.append({
            'action': 'set_deterministic_mode',
            'enabled': enabled,
            'timestamp': self._get_timestamp()
        })
    
    def generate_derived_seed(self, component: str, offset: int = 0) -> int:
        """
        Generate derived seed for specific component.
        
        Args:
            component: Component name (e.g., 'data_loader', 'model_init')
            offset: Additional offset for variation
            
        Returns:
            Derived seed value
        """
        # Create deterministic hash from base seed and component
        hash_input = f"{self.base_seed}_{component}_{offset}"
        hash_object = hashlib.md5(hash_input.encode())
        derived_seed = int(hash_object.hexdigest()[:8], 16) % (2**31)
        
        # Store derived seed
        self.derived_seeds[component] = derived_seed
        
        # Record action
        self.seeding_history.append({
            'action': 'generate_derived_seed',
            'component': component,
            'offset': offset,
            'derived_seed': derived_seed,
            'timestamp': self._get_timestamp()
        })
        
        return derived_seed
    
    def seed_component(self, component: str, offset: int = 0) -> int:
        """
        Set random seed for specific component.
        
        Args:
            component: Component name
            offset: Additional offset
            
        Returns:
            Used seed value
        """
        derived_seed = self.generate_derived_seed(component, offset)
        self.set_global_seed(derived_seed)
        return derived_seed
    
    def create_rng(self, component: str, offset: int = 0) -> np.random.Generator:
        """
        Create isolated random number generator for component.
        
        Args:
            component: Component name
            offset: Additional offset
            
        Returns:
            NumPy random generator
        """
        derived_seed = self.generate_derived_seed(component, offset)
        rng = np.random.default_rng(derived_seed)
        
        # Record action
        self.seeding_history.append({
            'action': 'create_rng',
            'component': component,
            'offset': offset,
            'seed': derived_seed,
            'timestamp': self._get_timestamp()
        })
        
        return rng
    
    def save_state(self, filepath: str):
        """
        Save current random state to file.
        
        Args:
            filepath: Path to save state
        """
        state_data = {
            'base_seed': self.base_seed,
            'derived_seeds': self.derived_seeds,
            'seeding_history': self.seeding_history,
            'current_state': self.current_state,
            'python_state': random.getstate(),
            'numpy_state': {
                'bit_generator': np.random.get_state()[0],
                'state': np.random.get_state()[1].tolist(),
                'pos': int(np.random.get_state()[2]),
                'has_gauss': int(np.random.get_state()[3]),
                'cached_gaussian': float(np.random.get_state()[4])
            },
            'torch_state': torch.get_rng_state().tolist()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save random state: {e}")
    
    def load_state(self, filepath: str) -> bool:
        """
        Load random state from file.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            Success status
        """
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            # Restore basic attributes
            self.base_seed = state_data['base_seed']
            self.derived_seeds = state_data['derived_seeds']
            self.seeding_history = state_data['seeding_history']
            self.current_state = state_data['current_state']
            
            # Restore library states
            random.setstate(tuple(state_data['python_state']))
            
            numpy_state = state_data['numpy_state']
            np.random.set_state((
                numpy_state['bit_generator'],
                np.array(numpy_state['state'], dtype=np.uint32),
                numpy_state['pos'],
                numpy_state['has_gauss'],
                numpy_state['cached_gaussian']
            ))
            
            torch.set_rng_state(torch.tensor(state_data['torch_state'], dtype=torch.uint8))
            
            return True
            
        except Exception as e:
            print(f"Failed to load random state: {e}")
            return False
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current seeding state.
        
        Returns:
            State summary dictionary
        """
        return {
            'base_seed': self.base_seed,
            'derived_seeds_count': len(self.derived_seeds),
            'derived_seeds': dict(self.derived_seeds),
            'seeding_actions': len(self.seeding_history),
            'last_action': self.seeding_history[-1] if self.seeding_history else None,
            'deterministic_mode': {
                'cudnn_deterministic': getattr(torch.backends.cudnn, 'deterministic', None),
                'cudnn_benchmark': getattr(torch.backends.cudnn, 'benchmark', None)
            }
        }
    
    def _set_python_seed(self, seed: int):
        """Set Python random seed."""
        random.seed(seed)
    
    def _set_numpy_seed(self, seed: int):
        """Set NumPy random seed."""
        np.random.seed(seed)
    
    def _set_torch_seed(self, seed: int):
        """Set PyTorch random seed."""
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _set_system_seed(self, seed: int):
        """Set system-level random seed."""
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _capture_state(self) -> Dict[str, Any]:
        """Capture current random state."""
        state = {
            'python_state_type': type(random.getstate()).__name__,
            'numpy_state_type': type(np.random.get_state()).__name__,
            'torch_state_shape': torch.get_rng_state().shape
        }
        
        if torch.cuda.is_available():
            state['torch_cuda_state_shape'] = torch.cuda.get_rng_state().shape
        
        return state
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import time
        return str(time.time())


def set_seed(seed: int = 42, deterministic: bool = True) -> SeedManager:
    """
    Quick function to set global seed with sensible defaults.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode
        
    Returns:
        SeedManager instance for further control
    """
    manager = SeedManager(seed)
    manager.set_global_seed(seed)
    
    if deterministic:
        manager.set_deterministic_mode(True)
    
    return manager


def generate_experiment_seeds(base_seed: int = 42, 
                            num_runs: int = 5,
                            components: Optional[List[str]] = None) -> Dict[str, List[int]]:
    """
    Generate seeds for multiple experimental runs.
    
    Args:
        base_seed: Base seed for generation
        num_runs: Number of experimental runs
        components: List of component names
        
    Returns:
        Dictionary mapping components to seed lists
    """
    if components is None:
        components = ['data', 'model', 'training', 'evaluation']
    
    manager = SeedManager(base_seed)
    seeds = {}
    
    for component in components:
        component_seeds = []
        for run in range(num_runs):
            seed = manager.generate_derived_seed(f"{component}_run", run)
            component_seeds.append(seed)
        seeds[component] = component_seeds
    
    return seeds


def verify_reproducibility(seed: int = 42, 
                          num_trials: int = 3) -> Dict[str, bool]:
    """
    Verify that seeding produces reproducible results.
    
    Args:
        seed: Seed to test
        num_trials: Number of trials to run
        
    Returns:
        Dictionary with reproducibility test results
    """
    results = {
        'python_random': True,
        'numpy_random': True,
        'torch_random': True,
        'torch_cuda_random': True if torch.cuda.is_available() else None
    }
    
    # Store initial values for comparison
    initial_values = {}
    
    for trial in range(num_trials):
        # Set seed
        manager = SeedManager(seed)
        manager.set_global_seed(seed)
        manager.set_deterministic_mode(True)
        
        # Generate test values
        python_val = random.random()
        numpy_val = np.random.random()
        torch_val = torch.rand(1).item()
        
        cuda_val = None
        if torch.cuda.is_available():
            cuda_val = torch.cuda.FloatTensor(1).random_().item()
        
        if trial == 0:
            # Store initial values
            initial_values = {
                'python': python_val,
                'numpy': numpy_val,
                'torch': torch_val,
                'cuda': cuda_val
            }
        else:
            # Compare with initial values
            if abs(python_val - initial_values['python']) > 1e-10:
                results['python_random'] = False
            
            if abs(numpy_val - initial_values['numpy']) > 1e-10:
                results['numpy_random'] = False
            
            if abs(torch_val - initial_values['torch']) > 1e-10:
                results['torch_random'] = False
            
            if cuda_val is not None and initial_values['cuda'] is not None:
                if abs(cuda_val - initial_values['cuda']) > 1e-10:
                    results['torch_cuda_random'] = False
    
    return results


def create_seed_config(base_seed: int = 42, 
                      components: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Create comprehensive seed configuration for experiment.
    
    Args:
        base_seed: Base seed value
        components: List of components needing seeds
        
    Returns:
        Seed configuration dictionary
    """
    if components is None:
        components = [
            'global', 'data_loading', 'data_augmentation',
            'model_initialization', 'training', 'validation',
            'evaluation', 'random_sampling'
        ]
    
    manager = SeedManager(base_seed)
    seed_config = {'base_seed': base_seed}
    
    for component in components:
        if component == 'global':
            seed_config[component] = base_seed
        else:
            seed_config[component] = manager.generate_derived_seed(component)
    
    return seed_config


# Test functions
def test_seed_manager():
    """Test SeedManager functionality."""
    print("Testing SeedManager...")
    
    # Create seed manager
    manager = SeedManager(base_seed=123)
    
    # Test global seeding
    used_seed = manager.set_global_seed(456)
    print(f"Global seed set: {used_seed}")
    
    # Test derived seed generation
    data_seed = manager.generate_derived_seed('data_loader')
    model_seed = manager.generate_derived_seed('model_init')
    
    print(f"Data seed: {data_seed}")
    print(f"Model seed: {model_seed}")
    print(f"Seeds are different: {data_seed != model_seed}")
    
    # Test deterministic mode
    manager.set_deterministic_mode(True)
    
    # Test RNG creation
    rng = manager.create_rng('test_component')
    random_value = rng.random()
    print(f"RNG created, sample value: {random_value:.6f}")
    
    # Test state summary
    summary = manager.get_state_summary()
    print(f"State summary keys: {list(summary.keys())}")
    print(f"Derived seeds count: {summary['derived_seeds_count']}")
    
    return manager


def test_reproducibility_verification():
    """Test reproducibility verification."""
    print("\nTesting reproducibility verification...")
    
    # Test reproducibility
    repro_results = verify_reproducibility(seed=789, num_trials=3)
    
    print("Reproducibility test results:")
    for lib, is_repro in repro_results.items():
        if is_repro is not None:
            status = "✓" if is_repro else "✗"
            print(f"  {lib}: {status} {'Reproducible' if is_repro else 'Not reproducible'}")
        else:
            print(f"  {lib}: N/A (not available)")
    
    return repro_results


def test_experiment_seeds():
    """Test experiment seed generation."""
    print("\nTesting experiment seed generation...")
    
    # Generate seeds for multiple runs
    exp_seeds = generate_experiment_seeds(
        base_seed=999,
        num_runs=3,
        components=['data', 'model', 'training']
    )
    
    print("Generated experiment seeds:")
    for component, seeds in exp_seeds.items():
        print(f"  {component}: {seeds}")
    
    # Verify seeds are different
    all_seeds = [seed for seeds in exp_seeds.values() for seed in seeds]
    unique_seeds = set(all_seeds)
    
    print(f"Total seeds: {len(all_seeds)}")
    print(f"Unique seeds: {len(unique_seeds)}")
    print(f"All seeds unique: {len(all_seeds) == len(unique_seeds)}")
    
    return exp_seeds


def test_seed_config_creation():
    """Test seed configuration creation."""
    print("\nTesting seed configuration creation...")
    
    # Create seed config
    seed_config = create_seed_config(
        base_seed=111,
        components=['global', 'data', 'model', 'training']
    )
    
    print("Seed configuration:")
    for component, seed in seed_config.items():
        print(f"  {component}: {seed}")
    
    # Test quick set_seed function
    manager = set_seed(seed=222, deterministic=True)
    print(f"Quick seed set with base: {manager.base_seed}")
    
    return seed_config


def test_state_persistence():
    """Test state saving and loading."""
    print("\nTesting state persistence...")
    
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        state_file = f.name
    
    try:
        # Create manager and generate some state
        manager1 = SeedManager(base_seed=333)
        manager1.set_global_seed()
        manager1.generate_derived_seed('test1')
        manager1.generate_derived_seed('test2')
        
        # Generate some random values
        val1_before = random.random()
        val2_before = np.random.random()
        
        # Save state
        manager1.save_state(state_file)
        
        # Modify state
        random.seed(999)
        np.random.seed(999)
        
        # Create new manager and load state
        manager2 = SeedManager()
        load_success = manager2.load_state(state_file)
        
        # Generate values after loading
        val1_after = random.random()
        val2_after = np.random.random()
        
        print(f"State save/load success: {load_success}")
        print(f"Base seed restored: {manager2.base_seed == 333}")
        print(f"Derived seeds restored: {len(manager2.derived_seeds) == 2}")
        
        # Clean up
        os.unlink(state_file)
        
        return load_success
        
    except Exception as e:
        print(f"State persistence test failed: {e}")
        return False


if __name__ == "__main__":
    print("="*50)
    print("CSP Seed Management Test")
    print("="*50)
    
    # Run all tests
    manager = test_seed_manager()
    repro_results = test_reproducibility_verification()
    exp_seeds = test_experiment_seeds()
    seed_config = test_seed_config_creation()
    state_test = test_state_persistence()
    
    print("\n" + "="*50)
    print("All seed management tests completed!")
    print("="*50)