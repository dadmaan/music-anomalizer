"""Checkpoint management system for model artifacts."""

import os
import glob
from typing import Dict, Optional, List
from pathlib import Path
from pydantic import BaseModel


class CheckpointConfig(BaseModel):
    """Configuration for model checkpoints."""
    base_dir: str = "./checkpoints"
    naming_pattern: str = "{model_name}-{dataset_name}-{stage}-epoch={epoch}-val_loss={val_loss:.2f}.ckpt"
    
    
class CheckpointRegistry:
    """Registry for managing model checkpoint paths."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.base_dir = Path(config.base_dir)
    
    def get_experiment_checkpoints(self, experiment_name: str) -> Dict[str, Dict[str, str]]:
        """Get all checkpoints for a specific experiment."""
        experiment_dir = self.base_dir / "loop_benchmark" / experiment_name.upper()
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        checkpoints = {}
        
        # Scan for checkpoint files
        for model_dir in experiment_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                
                # Find AE and SVDD checkpoints
                ae_files = list(model_dir.glob("*-AE-epoch=*.ckpt"))
                svdd_files = list(model_dir.glob("*-DSVDD-epoch=*.ckpt"))
                
                # Group by dataset
                datasets = set()
                for f in ae_files + svdd_files:
                    # Extract dataset name from filename
                    parts = f.stem.split('-')
                    if len(parts) >= 3:
                        dataset_name = '-'.join(parts[1:-2])  # Extract dataset part
                        datasets.add(dataset_name)
                
                for dataset in datasets:
                    key = f"{model_name}-{dataset}"
                    checkpoints[key] = {}
                    
                    # Find best AE checkpoint for this dataset
                    ae_pattern = f"*{dataset}*-AE-epoch=*.ckpt"
                    ae_matches = list(model_dir.glob(ae_pattern))
                    if ae_matches:
                        # Get checkpoint with lowest validation loss
                        best_ae = min(ae_matches, key=lambda x: self._extract_val_loss(x.name))
                        checkpoints[key][f"{key}-AE"] = str(best_ae)
                    
                    # Find best SVDD checkpoint for this dataset  
                    svdd_pattern = f"*{dataset}*-DSVDD-epoch=*.ckpt"
                    svdd_matches = list(model_dir.glob(svdd_pattern))
                    if svdd_matches:
                        # Get checkpoint with lowest validation loss
                        best_svdd = min(svdd_matches, key=lambda x: self._extract_val_loss(x.name))
                        checkpoints[key][f"{key}-DSVDD"] = str(best_svdd)
        
        return checkpoints
    
    def _extract_val_loss(self, filename: str) -> float:
        """Extract validation loss from checkpoint filename."""
        try:
            # Extract val_loss=X.XX part
            val_loss_part = filename.split('val_loss=')[1].split('.ckpt')[0]
            return float(val_loss_part)
        except (IndexError, ValueError):
            return float('inf')  # Return high value if parsing fails
    
    def get_model_checkpoints(self, experiment_name: str, model_name: str, dataset_name: str) -> Dict[str, str]:
        """Get specific model checkpoints for a dataset."""
        all_checkpoints = self.get_experiment_checkpoints(experiment_name)
        key = f"{model_name}-{dataset_name}"
        
        if key not in all_checkpoints:
            available_keys = list(all_checkpoints.keys())
            raise KeyError(f"Checkpoint not found for {key}. Available: {available_keys}")
        
        return all_checkpoints[key]
    
    def validate_checkpoints(self, checkpoint_paths: Dict[str, str]) -> bool:
        """Validate that all checkpoint files exist."""
        for name, path in checkpoint_paths.items():
            if not os.path.exists(path):
                print(f"Warning: Checkpoint not found: {path} (for {name})")
                return False
        return True


def get_checkpoint_registry() -> CheckpointRegistry:
    """Get default checkpoint registry."""
    config = CheckpointConfig()
    return CheckpointRegistry(config)