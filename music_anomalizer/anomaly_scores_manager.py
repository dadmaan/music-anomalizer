#!/usr/bin/env python3

"""
Anomaly Scores Manager

This module provides a centralized manager for handling anomaly score files in the
Streamlit application. It automatically detects missing files and computes them
using the compute_anomaly_scores script when needed.

Features:
- Auto-detection of missing anomaly score files
- Smart computation with progress feedback
- Configuration-aware file management
- Error handling and validation
- Streamlit integration with progress bars
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Any
from datetime import datetime

from music_anomalizer.config import load_experiment_config


class AnomalyScoresManager:
    """Manages anomaly score files for the Streamlit application."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the manager with base directory.
        
        Args:
            base_dir: Base directory of the project. If None, auto-detect.
        """
        if base_dir is None:
            # Auto-detect base directory (go up from this file to project root)
            # This file is at: /usr/src/app/music_anomalizer/anomaly_scores_manager.py
            # We want to get: /usr/src/app
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.output_dir = self.base_dir / 'output'
        self.scripts_dir = self.base_dir / 'music_anomalizer' / 'scripts'
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('anomaly_scores_manager')
    
    def get_scores_path(self, model_type: str, config_name: str = 'exp2_deeper', 
                       network_key: str = 'AEwRES') -> Path:
        """Get the path for anomaly scores file.
        
        Args:
            model_type: Type of model ('bass' or 'guitar')
            config_name: Name of experiment configuration
            network_key: Network architecture key
            
        Returns:
            Path to the anomaly scores file
        """
        # Include network key in filename to support different networks
        return self.output_dir / f'anomaly_scores_{model_type}_{network_key}.pkl'
    
    def check_scores_exist(self, model_type: str, config_name: str = 'exp2_deeper',
                          network_key: str = 'AEwRES') -> Tuple[bool, Optional[str]]:
        """Check if anomaly scores file exists and is valid.
        
        Args:
            model_type: Type of model ('bass' or 'guitar')
            config_name: Name of experiment configuration
            network_key: Network architecture key
            
        Returns:
            Tuple of (exists: bool, error_message: Optional[str])
        """
        scores_path = self.get_scores_path(model_type, config_name, network_key)
        
        # Check if file exists
        if not scores_path.exists():
            return False, f"Anomaly scores file not found: {scores_path}"
        
        # Check if file is readable and has valid content
        try:
            with open(scores_path, 'rb') as f:
                scores = pickle.load(f)
            
            # Basic validation
            if not isinstance(scores, list):
                return False, "Invalid file format: expected list of scores"
            
            if len(scores) == 0:
                return False, "Empty scores file"
            
            # Check if each score entry has required fields
            required_fields = ['file_id', 'file_path', 'anomaly_score']
            for i, score in enumerate(scores[:5]):  # Check first 5 entries
                if not isinstance(score, dict):
                    return False, f"Invalid score entry at index {i}: expected dict"
                
                for field in required_fields:
                    if field not in score:
                        return False, f"Missing field '{field}' in score entry at index {i}"
            
            self.logger.debug(f"Anomaly scores file validated: {scores_path} ({len(scores)} entries)")
            return True, None
            
        except Exception as e:
            return False, f"Error reading scores file: {str(e)}"
    
    def validate_prerequisites(self, model_type: str, config_name: str,
                              network_key: str = 'AEwRES') -> Tuple[bool, Optional[str]]:
        """Validate that all prerequisites for computing scores exist.
        
        Args:
            model_type: Type of model ('bass' or 'guitar')
            config_name: Name of experiment configuration
            network_key: Network architecture key
            
        Returns:
            Tuple of (valid: bool, error_message: Optional[str])
        """
        try:
            # Check if config exists
            config = load_experiment_config(config_name)
            
            # Get model configuration - import dynamically to avoid circular imports
            from music_anomalizer.scripts.compute_anomaly_scores import (
                get_model_choices_from_config
            )
            model_choices = get_model_choices_from_config(config, network_key)
            
            if model_type not in model_choices:
                return False, f"Unknown model type: {model_type}. Available: {list(model_choices.keys())}"
            
            model_choice = model_choices[model_type]
            dataset_name = model_choice['dataset_name']
            
            # Get dataset paths from configuration
            dataset_key = f'HTSAT_base_musicradar_{model_type}'
            if dataset_key not in config.dataset_paths:
                return False, f"Dataset path not found in config: {dataset_key}"

            # Build dataset paths from config
            dataset_path_str = config.dataset_paths[dataset_key]
            # Handle relative paths from config
            if dataset_path_str.startswith('./'):
                dataset_path_str = dataset_path_str[2:]  # Remove ./ prefix

            dataset_path = self.base_dir / dataset_path_str
            # Assume index file follows naming convention (add _index before .pkl)
            dataset_index_path = dataset_path.with_name(
                dataset_path.stem + '_index.pkl'
            )

            # Validate datasets - import dynamically to avoid circular imports
            from music_anomalizer.utils import validate_dataset
            is_valid, error_msg, _ = validate_dataset(
                dataset_path, dataset_index_path
            )
            if not is_valid:
                return False, f"Dataset validation failed: {error_msg}"

            # Check for model checkpoints
            checkpoint_dir = (
                self.base_dir / 'checkpoints' / 'loop_benchmark' /
                config_name.upper() / model_choice['model_key']
            )

            ae_checkpoints = list(
                checkpoint_dir.glob(f"*{dataset_name}*AE*.ckpt")
            )
            svdd_checkpoints = list(
                checkpoint_dir.glob(f"*{dataset_name}*DSVDD*.ckpt")
            )

            if not ae_checkpoints:
                return False, (
                    f"No AutoEncoder checkpoint found for {dataset_name} "
                    f"in {checkpoint_dir}"
                )
            if not svdd_checkpoints:
                return False, (
                    f"No DeepSVDD checkpoint found for {dataset_name} "
                    f"in {checkpoint_dir}"
                )

            return True, None

        except Exception as e:
            return False, f"Prerequisites validation failed: {str(e)}"
    
    def compute_missing_scores(
        self,
        model_type: str,
        config_name: str = 'exp2_deeper',
        network_key: str = 'AEwRES',
        progress_callback: Optional[Callable[[str, float], None]] = None,
        force_recompute: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """Compute anomaly scores for the specified model type.
        
        Args:
            model_type: Type of model ('bass' or 'guitar')
            config_name: Name of experiment configuration
            network_key: Network architecture key
            progress_callback: Optional callback for progress updates (message, progress_0_to_1)
            force_recompute: Force recomputation even if file exists
            
        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        def update_progress(message: str, progress: float = 0.0):
            """Helper to update progress with logging."""
            self.logger.info(f" {message}")
            if progress_callback:
                progress_callback(message, progress)
        
        try:
            scores_path = self.get_scores_path(model_type, config_name, network_key)
            
            # Check if already exists and we're not forcing recompute
            if not force_recompute:
                exists, error = self.check_scores_exist(model_type, config_name, network_key)
                if exists:
                    update_progress(f"Scores already exist for {model_type}", 1.0)
                    return True, None
            
            update_progress(
                f"Starting anomaly score computation for {model_type}...", 0.1
            )

            # Validate prerequisites
            update_progress("Validating prerequisites...", 0.2)
            valid, error_msg = self.validate_prerequisites(model_type, config_name, network_key)
            if not valid:
                return False, error_msg

            # Initialize device - import dynamically to avoid circular imports
            update_progress("Initializing compute device...", 0.3)
            from music_anomalizer.utils import initialize_device
            device = initialize_device()
            device_str = str(device)  # Convert torch.device to string
            self.logger.info(f"Using device: {device_str}")

            # Compute scores using the script function - import dynamically
            update_progress(
                "Computing anomaly scores (this may take a while)...", 0.4
            )
            
            try:
                from music_anomalizer.scripts.compute_anomaly_scores import (
                    compute_anomaly_scores
                )
                results = compute_anomaly_scores(
                    model_type=model_type,
                    config_name=config_name,
                    network_key=network_key,
                    device=device_str,
                    output_path=scores_path
                )

                update_progress("Validating computed scores...", 0.9)

                # Validate the computed file
                exists, error = self.check_scores_exist(model_type, config_name, network_key)
                if not exists:
                    return False, (
                        f"Computation completed but file validation failed: "
                        f"{error}"
                    )

                update_progress(
                    f"Successfully computed {len(results)} anomaly scores "
                    f"for {model_type}", 1.0
                )
                return True, None

            except Exception as e:
                return False, f"Score computation failed: {str(e)}"
                
        except Exception as e:
            self.logger.error(
                f"Unexpected error in compute_missing_scores: {str(e)}"
            )
            return False, f"Unexpected error: {str(e)}"
    
    def ensure_scores_exist(
        self,
        model_type: str,
        config_name: str = 'exp2_deeper',
        network_key: str = 'AEwRES',
        progress_callback: Optional[Callable[[str, float], None]] = None,
        force_recompute: bool = False
    ) -> Tuple[bool, Optional[str], Optional[Path]]:
        """Ensure anomaly scores exist, computing them if necessary.

        Args:
            model_type: Type of model ('bass' or 'guitar')
            config_name: Name of experiment configuration
            network_key: Network architecture key
            progress_callback: Optional callback for progress updates
            force_recompute: Force recomputation even if file exists

        Returns:
            Tuple of (success: bool, error_message: Optional[str],
                     scores_path: Optional[Path])
        """
        scores_path = self.get_scores_path(model_type, config_name, network_key)
        
        # Check if scores already exist
        if not force_recompute:
            exists, error = self.check_scores_exist(model_type, config_name, network_key)
            if exists:
                if progress_callback:
                    progress_callback(
                        f"Anomaly scores found for {model_type}", 1.0
                    )
                return True, None, scores_path

        # Need to compute scores
        success, error = self.compute_missing_scores(
            model_type, config_name, network_key, progress_callback, force_recompute
        )
        
        if success:
            return True, None, scores_path
        else:
            return False, error, None
    
    def load_scores(
        self,
        model_type: str,
        config_name: str = 'exp2_deeper',
        network_key: str = 'AEwRES',
        auto_compute: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """Load anomaly scores, computing them if necessary.

        Args:
            model_type: Type of model ('bass' or 'guitar')
            config_name: Name of experiment configuration
            network_key: Network architecture key
            auto_compute: Whether to automatically compute missing scores
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (scores: Optional[List[Dict]],
                     error_message: Optional[str])
        """
        # Ensure scores exist
        if auto_compute:
            success, error, scores_path = self.ensure_scores_exist(
                model_type, config_name, network_key, progress_callback
            )
            if not success:
                return None, error
        else:
            scores_path = self.get_scores_path(model_type, config_name, network_key)
            exists, error = self.check_scores_exist(
                model_type, config_name, network_key
            )
            if not exists:
                return None, error
        
        # Load the scores
        try:
            with open(scores_path, 'rb') as f:
                scores = pickle.load(f)
            return scores, None
        except Exception as e:
            return None, f"Error loading scores: {str(e)}"
    
    def get_scores_info(
        self,
        model_type: str,
        config_name: str = 'exp2_deeper',
        network_key: str = 'AEwRES'
    ) -> Dict[str, Any]:
        """Get information about anomaly scores file.

        Args:
            model_type: Type of model ('bass' or 'guitar')
            config_name: Name of experiment configuration
            network_key: Network architecture key

        Returns:
            Dictionary with file information
        """
        scores_path = self.get_scores_path(model_type, config_name, network_key)
        
        info = {
            'path': str(scores_path),
            'exists': scores_path.exists(),
            'size': None,
            'num_scores': None,
            'last_modified': None,
            'valid': False,
            'error': None
        }
        
        if scores_path.exists():
            try:
                # Get file stats
                stat = scores_path.stat()
                info['size'] = stat.st_size
                info['last_modified'] = datetime.fromtimestamp(stat.st_mtime)
                
                # Check validity and count
                valid, error = self.check_scores_exist(model_type, config_name, network_key)
                info['valid'] = valid
                info['error'] = error

                if valid:
                    with open(scores_path, 'rb') as f:
                        scores = pickle.load(f)
                    info['num_scores'] = len(scores)

            except Exception as e:
                info['error'] = f"Error reading file info: {str(e)}"

        return info


# Global instance for easy import
_manager_instance = None


def get_anomaly_scores_manager() -> AnomalyScoresManager:
    """Get the global AnomalyScoresManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = AnomalyScoresManager()
    return _manager_instance
