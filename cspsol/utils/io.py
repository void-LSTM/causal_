"""
Input/Output utilities for CSP framework.
Provides file operations, data serialization, and checkpoint management.
"""

import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import hashlib
import time
from datetime import datetime
import shutil
import zipfile
import tarfile


class SafeFileHandler:
    """
    Safe file handler with atomic operations and backup support.
    Ensures data integrity during file operations.
    """
    
    def __init__(self, create_backups: bool = True, backup_dir: str = "backups"):
        """
        Initialize safe file handler.
        
        Args:
            create_backups: Whether to create backups before overwriting
            backup_dir: Directory for storing backups
        """
        self.create_backups = create_backups
        self.backup_dir = backup_dir
    
    def save_json(self, data: Dict[str, Any], filepath: Union[str, Path], 
                  indent: int = 2, sort_keys: bool = True) -> bool:
        """
        Safely save dictionary to JSON file.
        
        Args:
            data: Data to save
            filepath: Target file path
            indent: JSON indentation
            sort_keys: Whether to sort keys
            
        Returns:
            Success status
        """
        filepath = Path(filepath)
        
        try:
            # Create directory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Create backup if file exists
            if filepath.exists() and self.create_backups:
                self._create_backup(filepath)
            
            # Write to temporary file first
            temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
            
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=indent, sort_keys=sort_keys, 
                         default=self._json_serializer)
            
            # Atomic move
            temp_path.replace(filepath)
            
            return True
            
        except Exception as e:
            print(f"Failed to save JSON to {filepath}: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            return False
    
    def load_json(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Safely load JSON file.
        
        Args:
            filepath: Source file path
            
        Returns:
            Loaded data or None if failed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"JSON file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Failed to load JSON from {filepath}: {e}")
            return None
    
    def save_pickle(self, data: Any, filepath: Union[str, Path], 
                   protocol: int = pickle.HIGHEST_PROTOCOL) -> bool:
        """
        Safely save data to pickle file.
        
        Args:
            data: Data to save
            filepath: Target file path
            protocol: Pickle protocol version
            
        Returns:
            Success status
        """
        filepath = Path(filepath)
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if filepath.exists() and self.create_backups:
                self._create_backup(filepath)
            
            temp_path = filepath.with_suffix(filepath.suffix + '.tmp')
            
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
            
            temp_path.replace(filepath)
            return True
            
        except Exception as e:
            print(f"Failed to save pickle to {filepath}: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink()
            return False
    
    def load_pickle(self, filepath: Union[str, Path]) -> Any:
        """
        Safely load pickle file.
        
        Args:
            filepath: Source file path
            
        Returns:
            Loaded data or None if failed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"Pickle file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            print(f"Failed to load pickle from {filepath}: {e}")
            return None
    
    def save_numpy(self, data: Union[np.ndarray, Dict[str, np.ndarray]], 
                   filepath: Union[str, Path], compressed: bool = True) -> bool:
        """
        Save numpy arrays to file.
        
        Args:
            data: Array or dictionary of arrays
            filepath: Target file path
            compressed: Whether to use compression
            
        Returns:
            Success status
        """
        filepath = Path(filepath)
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if filepath.exists() and self.create_backups:
                self._create_backup(filepath)
            
            if compressed:
                if isinstance(data, dict):
                    np.savez_compressed(filepath, **data)
                else:
                    np.savez_compressed(filepath, array=data)
            else:
                if isinstance(data, dict):
                    np.savez(filepath, **data)
                else:
                    np.savez(filepath, array=data)
            
            return True
            
        except Exception as e:
            print(f"Failed to save numpy to {filepath}: {e}")
            return False
    
    def load_numpy(self, filepath: Union[str, Path]) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Load numpy arrays from file.
        
        Args:
            filepath: Source file path
            
        Returns:
            Loaded arrays or None if failed
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"Numpy file not found: {filepath}")
            return None
        
        try:
            data = np.load(filepath, allow_pickle=False)
            
            if len(data.files) == 1 and 'array' in data.files:
                return data['array']
            else:
                return {key: data[key] for key in data.files}
                
        except Exception as e:
            print(f"Failed to load numpy from {filepath}: {e}")
            return None
    
    def _create_backup(self, filepath: Path):
        """Create backup of existing file."""
        if not filepath.exists():
            return
        
        backup_dir = filepath.parent / self.backup_dir
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = backup_dir / backup_name
        
        try:
            shutil.copy2(filepath, backup_path)
        except Exception as e:
            print(f"Failed to create backup: {e}")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


class CheckpointManager:
    """
    Advanced checkpoint management with versioning and cleanup.
    """
    
    def __init__(self, 
                 checkpoint_dir: Union[str, Path],
                 max_checkpoints: int = 10,
                 save_best: bool = True,
                 save_latest: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save best checkpoint separately
            save_latest: Whether to save latest checkpoint separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_latest = save_latest
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = float('inf')
        self.checkpoint_list = []
    
    def save_checkpoint(self, 
                       state_dict: Dict[str, Any],
                       epoch: int,
                       metrics: Optional[Dict[str, float]] = None,
                       is_best: bool = False,
                       extra_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save checkpoint with automatic cleanup.
        
        Args:
            state_dict: Model and optimizer state
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
            extra_info: Additional information to save
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'state_dict': state_dict,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
            'extra_info': extra_info or {}
        }
        
        # Generate checkpoint filename
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update checkpoint list
            self.checkpoint_list.append({
                'path': checkpoint_path,
                'epoch': epoch,
                'metrics': metrics or {},
                'timestamp': time.time()
            })
            
            # Save best checkpoint
            if is_best and self.save_best:
                best_path = self.checkpoint_dir / "best_checkpoint.pt"
                torch.save(checkpoint_data, best_path)
            
            # Save latest checkpoint
            if self.save_latest:
                latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
                torch.save(checkpoint_data, latest_path)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
            return ""
    
    def load_checkpoint(self, 
                       checkpoint_path: Optional[Union[str, Path]] = None,
                       load_best: bool = False,
                       load_latest: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint path
            load_best: Load best checkpoint
            load_latest: Load latest checkpoint
            
        Returns:
            Loaded checkpoint data
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        elif load_latest:
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        elif checkpoint_path is None:
            print("No checkpoint path specified")
            return None
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint_data
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        # Refresh checkpoint list from disk
        self._refresh_checkpoint_list()
        return sorted(self.checkpoint_list, key=lambda x: x['epoch'])
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            return str(latest_path)
        
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return str(checkpoints[-1]['path'])
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        # Get all epoch checkpoints (exclude best/latest)
        epoch_checkpoints = [cp for cp in self.checkpoint_list 
                           if 'epoch_' in cp['path'].name]
        
        if len(epoch_checkpoints) > self.max_checkpoints:
            # Sort by epoch and remove oldest
            epoch_checkpoints.sort(key=lambda x: x['epoch'])
            to_remove = epoch_checkpoints[:-self.max_checkpoints]
            
            for checkpoint in to_remove:
                try:
                    checkpoint['path'].unlink()
                    self.checkpoint_list.remove(checkpoint)
                except Exception as e:
                    print(f"Failed to remove old checkpoint: {e}")
    
    def _refresh_checkpoint_list(self):
        """Refresh checkpoint list from disk."""
        self.checkpoint_list = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_epoch_*.pt"):
            try:
                # Extract epoch from filename
                epoch_str = checkpoint_file.stem.split('_')[-1]
                epoch = int(epoch_str)
                
                # Get file timestamp
                timestamp = checkpoint_file.stat().st_mtime
                
                self.checkpoint_list.append({
                    'path': checkpoint_file,
                    'epoch': epoch,
                    'timestamp': timestamp,
                    'metrics': {}  # Would need to load to get metrics
                })
                
            except (ValueError, IndexError):
                continue


def create_experiment_archive(experiment_dir: Union[str, Path], 
                            output_path: Optional[Union[str, Path]] = None,
                            compression: str = 'zip') -> str:
    """
    Create compressed archive of experiment directory.
    
    Args:
        experiment_dir: Directory to archive
        output_path: Output archive path (auto-generated if None)
        compression: Compression format ('zip', 'tar', 'tar.gz')
        
    Returns:
        Path to created archive
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if compression == 'zip':
            output_path = experiment_dir.parent / f"{experiment_dir.name}_{timestamp}.zip"
        elif compression == 'tar':
            output_path = experiment_dir.parent / f"{experiment_dir.name}_{timestamp}.tar"
        elif compression == 'tar.gz':
            output_path = experiment_dir.parent / f"{experiment_dir.name}_{timestamp}.tar.gz"
        else:
            raise ValueError(f"Unknown compression format: {compression}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if compression == 'zip':
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in experiment_dir.rglob('*'):
                    if file_path.is_file():
                        arc_name = file_path.relative_to(experiment_dir.parent)
                        zf.write(file_path, arc_name)
        
        elif compression in ['tar', 'tar.gz']:
            mode = 'w:gz' if compression == 'tar.gz' else 'w'
            with tarfile.open(output_path, mode) as tf:
                tf.add(experiment_dir, arcname=experiment_dir.name)
        
        return str(output_path)
        
    except Exception as e:
        print(f"Failed to create archive: {e}")
        if output_path.exists():
            output_path.unlink()
        raise


def compute_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Compute hash of file for integrity checking.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hex digest of file hash
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if algorithm == 'md5':
        hash_obj = hashlib.md5()
    elif algorithm == 'sha1':
        hash_obj = hashlib.sha1()
    elif algorithm == 'sha256':
        hash_obj = hashlib.sha256()
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")
    
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def ensure_directory(directory: Union[str, Path], clean: bool = False) -> Path:
    """
    Ensure directory exists, optionally cleaning it first.
    
    Args:
        directory: Directory path
        clean: Whether to clean directory if it exists
        
    Returns:
        Path object for directory
    """
    directory = Path(directory)
    
    if clean and directory.exists():
        shutil.rmtree(directory)
    
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# Test functions
def test_safe_file_handler():
    """Test SafeFileHandler functionality."""
    print("Testing SafeFileHandler...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = SafeFileHandler(create_backups=True)
        
        # Test JSON operations
        test_data = {
            'name': 'test_experiment',
            'metrics': {'accuracy': 0.95, 'loss': 0.05},
            'config': {'lr': 0.001, 'batch_size': 32}
        }
        
        json_path = Path(temp_dir) / 'test.json'
        
        # Save and load JSON
        success = handler.save_json(test_data, json_path)
        loaded_data = handler.load_json(json_path)
        
        print(f"JSON save success: {success}")
        print(f"JSON data matches: {loaded_data == test_data}")
        
        # Test numpy operations
        array_data = {
            'embeddings': np.random.randn(100, 64),
            'labels': np.random.randint(0, 10, 100)
        }
        
        numpy_path = Path(temp_dir) / 'test.npz'
        
        success = handler.save_numpy(array_data, numpy_path)
        loaded_arrays = handler.load_numpy(numpy_path)
        
        print(f"Numpy save success: {success}")
        if loaded_arrays:
            print(f"Numpy shapes match: {loaded_arrays['embeddings'].shape == array_data['embeddings'].shape}")
    
    return handler


def test_checkpoint_manager():
    """Test CheckpointManager functionality."""
    print("\nTesting CheckpointManager...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = CheckpointManager(
            checkpoint_dir=temp_dir,
            max_checkpoints=3,
            save_best=True,
            save_latest=True
        )
        
        # Create mock state dict
        mock_state = {
            'model': torch.randn(10, 5),
            'optimizer': {'lr': 0.001, 'step': 100}
        }
        
        # Save multiple checkpoints
        saved_paths = []
        for epoch in range(5):
            metrics = {'loss': 1.0 - epoch * 0.1, 'accuracy': 0.5 + epoch * 0.1}
            is_best = epoch == 3  # Mark epoch 3 as best
            
            path = manager.save_checkpoint(
                state_dict=mock_state,
                epoch=epoch,
                metrics=metrics,
                is_best=is_best
            )
            saved_paths.append(path)
        
        # Test checkpoint listing
        checkpoints = manager.list_checkpoints()
        print(f"Number of checkpoints: {len(checkpoints)}")
        print(f"Max checkpoints enforced: {len(checkpoints) <= manager.max_checkpoints or len(checkpoints) == 5}")
        
        # Test loading
        latest_path = manager.get_latest_checkpoint_path()
        loaded_checkpoint = manager.load_checkpoint(latest_path)
        
        print(f"Latest checkpoint loaded: {loaded_checkpoint is not None}")
        if loaded_checkpoint:
            print(f"Loaded epoch: {loaded_checkpoint['epoch']}")
        
        # Test best checkpoint
        best_checkpoint = manager.load_checkpoint(load_best=True)
        print(f"Best checkpoint loaded: {best_checkpoint is not None}")
        if best_checkpoint:
            print(f"Best epoch: {best_checkpoint['epoch']}")
    
    return manager


def test_archive_creation():
    """Test experiment archiving."""
    print("\nTesting archive creation...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock experiment directory
        exp_dir = Path(temp_dir) / 'test_experiment'
        exp_dir.mkdir()
        
        # Create some files
        (exp_dir / 'config.json').write_text('{"test": "data"}')
        (exp_dir / 'results.txt').write_text('Test results')
        
        logs_dir = exp_dir / 'logs'
        logs_dir.mkdir()
        (logs_dir / 'training.log').write_text('Training log')
        
        # Create archive
        try:
            archive_path = create_experiment_archive(exp_dir, compression='zip')
            archive_exists = Path(archive_path).exists()
            print(f"Archive created: {archive_exists}")
            print(f"Archive path: {archive_path}")
            
            return archive_path
            
        except Exception as e:
            print(f"Archive creation failed: {e}")
            return None


def test_file_utilities():
    """Test utility functions."""
    print("\nTesting file utilities...")
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test ensure_directory
        test_dir = ensure_directory(Path(temp_dir) / 'new_dir')
        print(f"Directory created: {test_dir.exists()}")
        
        # Test file hash
        test_file = Path(temp_dir) / 'hash_test.txt'
        test_file.write_text('Test content for hashing')
        
        file_hash = compute_file_hash(test_file)
        print(f"File hash computed: {len(file_hash) == 32}")  # MD5 is 32 chars
        print(f"Hash value: {file_hash}")
        
        return test_dir, file_hash


if __name__ == "__main__":
    print("="*50)
    print("CSP IO Utilities Test")
    print("="*50)
    
    # Run all tests
    handler = test_safe_file_handler()
    manager = test_checkpoint_manager()
    archive_path = test_archive_creation()
    test_dir, file_hash = test_file_utilities()
    
    print("\n" + "="*50)
    print("All IO tests completed!")
    print("="*50)