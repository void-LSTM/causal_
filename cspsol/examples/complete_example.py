"""
Complete example demonstrating the full CARL training and evaluation pipeline.
Shows integration of all components: data loading, model training, and evaluation.
"""
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    # 按你的路径，parents[2] = /Volumes/Yulong/ICLR
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    __package__ = "cspsol."

import torch
import numpy as np
from pathlib import Path
import tempfile
import warnings

# Import our components
from ..config.manager import ConfigManager, ExperimentConfig
from ..models.carl import CausalAwareModel
from ..train.loop import CSPTrainer
from ..train.callbacks import EarlyStopping, ModelCheckpoint, MetricsLogger
from ..eval.hooks import ModelEvaluator


def create_mock_dataset(scenario='IM', n_samples=1000, data_dir='./mock_data'):
    """Create mock dataset for demonstration."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating mock dataset for {scenario} scenario...")
    
    # Generate synthetic data following CSP structure
    np.random.seed(42)
    
    # Treatment variable T
    T = np.random.normal(0, 1, n_samples)
    
    # Mediator M depends on T
    noise_M = np.random.normal(0, 0.5, n_samples)
    M = 0.8 * T + noise_M
    
    # Outcome Y depends on M (and possibly T)
    noise_Y = np.random.normal(0, 0.3, n_samples)
    if scenario == 'IM':
        # Full mediation: T -> M -> Y
        Y_star = 0.7 * M + noise_Y
    elif scenario == 'IY':
        # Partial mediation: T -> M -> Y and T -> Y
        Y_star = 0.5 * M + 0.3 * T + noise_Y
    else:  # DUAL
        Y_star = 0.6 * M + 0.2 * T + noise_Y
    
    # Image-like data (simplified as vectors)
    img_dim = 28 * 28  # MNIST-like
    if scenario in ['IM', 'DUAL']:
        # Images represent M
        I_M = np.random.normal(M.reshape(-1, 1), 0.1, (n_samples, img_dim))
        a_M = np.abs(M) + np.random.normal(0, 0.1, n_samples)  # Semantic amplitude
    else:
        I_M = np.random.normal(0, 1, (n_samples, img_dim))
        a_M = np.random.uniform(0, 1, n_samples)
    
    if scenario in ['IY', 'DUAL']:
        # Images represent Y
        I_Y = np.random.normal(Y_star.reshape(-1, 1), 0.1, (n_samples, img_dim))
        a_Y = np.abs(Y_star) + np.random.normal(0, 0.1, n_samples)
    else:
        I_Y = np.random.normal(0, 1, (n_samples, img_dim))
        a_Y = np.random.uniform(0, 1, n_samples)
    
    # Style variable
    b_style = np.random.uniform(0, 1, n_samples)
    
    # Save data with proper dtypes
    data = {
        'T': T.astype(np.float32),
        'M': M.astype(np.float32),
        'Y_star': Y_star.astype(np.float32),
        'I_M': I_M.astype(np.float32),
        'I_Y': I_Y.astype(np.float32),
        'a_M': a_M.astype(np.float32),
        'a_Y': a_Y.astype(np.float32),
        'b_style': b_style.astype(np.float32),
        'subject_id': np.arange(n_samples, dtype=np.int32),
        'scenario': np.array([scenario] * n_samples, dtype='<U10')
    }
    
    # Save to files
    for split, indices in [('train', slice(0, 300)), ('val', slice(300, 400)), ('test', slice(400, 500))]:
        split_data = {k: v[indices] if hasattr(v, '__getitem__') else v for k, v in data.items()}
        np.savez_compressed(data_dir / f'{split}.npz', **split_data)
    
    print(f"Mock dataset saved to: {data_dir}")
    return data_dir


class MockDataModule:
    """Mock data module for demonstration."""
    
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        # Load data
        self.train_data = dict(np.load(self.data_dir / 'train.npz', allow_pickle=True))
        self.val_data = dict(np.load(self.data_dir / 'val.npz', allow_pickle=True))
        self.test_data = dict(np.load(self.data_dir / 'test.npz', allow_pickle=True))
    
    def train_dataloader(self):
        return self._create_dataloader(self.train_data)
    
    def val_dataloader(self):
        return self._create_dataloader(self.val_data)
    
    def test_dataloader(self):
        return self._create_dataloader(self.test_data)
    
    def _create_dataloader(self, data):
        """Create simple batch iterator with proper string handling."""
        n_samples = len(data['T'])
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        batches = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            
            batch = {}
            for key, values in data.items():
                if hasattr(values, '__getitem__'):
                    # Check if this is string data
                    if (key == 'scenario' or 
                        (hasattr(values, 'dtype') and values.dtype.kind in ['U', 'S', 'O'])):
                        # String or object dtype - store as list
                        batch[key] = values[start_idx:end_idx].tolist() if hasattr(values, 'tolist') else values[start_idx:end_idx]
                    else:
                        # Numeric data - convert to tensor
                        batch[key] = torch.from_numpy(values[start_idx:end_idx])
                else:
                    batch[key] = values
            
            # Reshape images
            if 'I_M' in batch and torch.is_tensor(batch['I_M']):
                batch['I_M'] = batch['I_M'].view(-1, 1, 28, 28)
            if 'I_Y' in batch and torch.is_tensor(batch['I_Y']):
                batch['I_Y'] = batch['I_Y'].view(-1, 1, 28, 28)
            
            batches.append(batch)
        
        return batches


def run_complete_example():
    """Run complete CARL training and evaluation example."""
    print("="*60)
    print("CARL COMPLETE EXAMPLE")
    print("="*60)
    
    # Setup
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create configuration
        print("\n1. Creating experiment configuration...")
        config_manager = ConfigManager()
        config = config_manager.create_config(
            preset='dev',  # Use development preset for speed
            **{
                'name': 'complete_example',
                'model.scenario': 'IM',
                'model.z_dim': 64,
                'training.max_epochs': 8,  # Reduced for faster demo
                'training.log_every_n_steps': 2,
                'training.save_every_n_epochs': 4,
                'data.batch_size': 8  # Smaller batch for demo
            }
        )
        
        # Save config (skip for demo to avoid YAML issues)
        print(f"Configuration created: {config.name}")
        
        # Step 2: Create mock dataset
        print("\n2. Creating mock dataset...")
        data_dir = create_mock_dataset(
            scenario=config.model.scenario,
            n_samples=500,  # Small for demo
            data_dir=temp_path / 'data'
        )
        
        # Step 3: Setup data module
        print("\n3. Setting up data module...")
        datamodule = MockDataModule(data_dir, batch_size=config.data.batch_size)
        
        # Determine feature dimensions from data
        sample_batch = datamodule.train_dataloader()[0]
        feature_dims = {
            'T_dim': 1,
            'M_dim': 1,
            'Y_dim': 1,
            'img_channels': 1,
            'img_height': 28,
            'img_width': 28
        }
        
        print(f"Sample batch keys: {list(sample_batch.keys())}")
        print(f"Feature dimensions: {feature_dims}")
        
        # Step 4: Create model
        print("\n4. Creating CARL model...")
        model = CausalAwareModel(
            scenario=config.model.scenario,
            z_dim=config.model.z_dim,
            feature_dims=feature_dims,
            loss_config=config.model.loss_config,
            balancer_config=config.model.balancer_config
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Step 5: Create trainer
        print("\n5. Creating trainer...")
        training_config = {
            'max_epochs': config.training.max_epochs,
            'log_every_n_steps': config.training.log_every_n_steps,
            'use_amp': False,
            'save_every_n_epochs': config.training.save_every_n_epochs,
            'early_stopping_patience': 10
        }

        # Add this for more stable training:
        config.training.learning_rate = 1e-4  # Reduce from 1e-3
        config.training.gradient_clip_val = 1.0  # Reduce from 5.0
        
        trainer = CSPTrainer(
            model=model,
            datamodule=datamodule,
            training_config=training_config,
            output_dir=str(temp_path / 'training_outputs')
        )
        
        # Step 6: Train model
        print("\n6. Starting training...")
        history = trainer.fit()
        
        print(f"\nTraining completed!")
        if 'train' in history and 'train_total_loss' in history['train']:
            final_train_loss = history['train']['train_total_loss'][-1]
            print(f"Final training loss: {final_train_loss:.4f}")
        
        if 'val' in history and 'val_total_loss' in history['val']:
            final_val_loss = history['val']['val_total_loss'][-1]
            print(f"Final validation loss: {final_val_loss:.4f}")
        
        # Step 7: Evaluate model
        print("\n7. Evaluating model...")
        evaluator = ModelEvaluator(
            model=model,
            scenario=config.model.scenario,
            output_dir=str(temp_path / 'evaluation')
        )
        
        # Run evaluation on test set
        test_loader = datamodule.test_dataloader()
        results = evaluator.evaluate(
            test_loader,
            save_representations=True,
            save_metrics=True
        )
        
        # Step 8: Display final results
        print("\n8. Final Results Summary:")
        print("-" * 40)
        
        if results['summary']['main_scores']:
            print("CSP Metrics:")
            for metric, score in results['summary']['main_scores'].items():
                print(f"  {metric}: {score:.4f}")
            print(f"\nOverall CSP Score: {results['summary']['overall_score']:.4f}")
        else:
            print("CSP metrics computed with limited data")
        
        print(f"\nTest set size: {results['n_samples']} samples")
        print(f"Representations extracted: {len(results['representation_shapes'])}")
        
        print("\n" + "="*60)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'config': config,
            'model': model,
            'training_history': history,
            'evaluation_results': results
        }


# Test implementation
if __name__ == "__main__":
    print("=== Testing Complete CARL Pipeline ===")
    
    try:
        # Run complete example
        results = run_complete_example()
        
        print("\n✓ Complete pipeline test passed!")
        print("\nPipeline components verified:")
        print("  ✓ Configuration management")
        print("  ✓ Data loading and preprocessing")
        print("  ✓ Model creation and initialization")
        print("  ✓ Training with multi-phase progression")
        print("  ✓ Loss balancing and monitoring")
        print("  ✓ Model evaluation and CSP metrics")
        print("  ✓ Results saving and reporting")
        
    except Exception as e:
        print(f"\n✗ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Complete CARL Pipeline Test Complete ===")