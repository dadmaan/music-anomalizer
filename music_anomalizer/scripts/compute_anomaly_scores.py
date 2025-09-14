#!/usr/bin/env python3

"""
Compute Anomaly Scores Script

This script computes anomaly scores for all files in training datasets using trained
DeepSVDD models. It processes both bass and guitar models, computing distance-based
anomaly scores for each audio file in the respective datasets.

The script loads pre-trained AutoEncoder and DeepSVDD model checkpoints, processes
embeddings through the models, and saves ranked results based on anomaly scores.

Usage:
    python music_anomalizer/scripts/compute_anomaly_scores.py
    python music_anomalizer/scripts/compute_anomaly_scores.py --config exp2_deeper --device cuda
    python music_anomalizer/scripts/compute_anomaly_scores.py --model-type bass --output results/bass_scores.pkl
    python music_anomalizer/scripts/compute_anomaly_scores.py --dry-run --log-level DEBUG
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from tqdm import tqdm

from music_anomalizer.config import load_experiment_config
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.utils import (
    load_pickle, initialize_device, setup_logging, validate_dataset
)

# Default configuration
DEFAULT_CONFIG = 'exp1'
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent


def get_model_choices_from_config(config, network_key: str = 'AEwRES') -> Dict[str, Dict[str, Any]]:
    """Get model choices based on configuration and selected network."""
    return {
        'bass': {
            'model_key': network_key,
            'dataset_name': 'HTSAT_base_musicradar_bass',
        },
        'guitar': {
            'model_key': network_key, 
            'dataset_name': 'HTSAT_base_musicradar_guitar',
        }
    }


def load_and_validate_data(dataset_path: Path, dataset_index_path: Path) -> Tuple[Any, List, Dict]:
    """Load and validate dataset and index files.
    
    Args:
        dataset_path: Path to dataset pickle file
        dataset_index_path: Path to dataset index pickle file
    
    Returns:
        Tuple of (data, data_index, validation_info)
    
    Raises:
        ValueError: If data loading or validation fails
    """
    logger = logging.getLogger('compute_anomaly_scores')
    
    try:
        logger.info(" Loading dataset...")
        data = load_pickle(str(dataset_path))
        data_index = load_pickle(str(dataset_index_path))
        
        # Validate data structure
        if not hasattr(data, 'shape'):
            raise ValueError("Dataset does not have expected shape attribute")
            
        if len(data_index) == 0:
            raise ValueError("Dataset index is empty")
            
        if len(data) != len(data_index):
            logger.warning(f" Data length ({len(data)}) != Index length ({len(data_index)})")
        
        validation_info = {
            'data_shape': data.shape,
            'num_samples': len(data),
            'num_indices': len(data_index),
            'feature_dim': data.shape[-1] if len(data.shape) > 1 else 1
        }
        
        logger.info(" Dataset loaded successfully")
        logger.info(f"    Shape: {validation_info['data_shape']}")
        logger.info(f"    Samples: {validation_info['num_samples']}")
        logger.info(f"    Features: {validation_info['feature_dim']}")
        
        return data, data_index, validation_info
        
    except Exception as e:
        logger.error(f" Failed to load dataset: {e}")
        raise ValueError(f"Dataset loading failed: {e}") from e


def initialize_anomaly_detector(model_config: Dict, svdd_config: Dict, 
                              ae_checkpoint: Path, svdd_checkpoint: Path,
                              device: torch.device) -> AnomalyDetector:
    """Initialize and load anomaly detector models.
    
    Args:
        model_config: AutoEncoder model configuration
        svdd_config: DeepSVDD configuration
        ae_checkpoint: Path to AutoEncoder checkpoint
        svdd_checkpoint: Path to DeepSVDD checkpoint
        device: Torch device to use
    
    Returns:
        Initialized AnomalyDetector instance
    
    Raises:
        ValueError: If model initialization fails
    """
    logger = logging.getLogger('compute_anomaly_scores')
    
    try:
        logger.info(" Initializing anomaly detector...")
        
        # Validate checkpoint files exist
        if not ae_checkpoint.exists():
            raise ValueError(f"AutoEncoder checkpoint not found: {ae_checkpoint}")
        if not svdd_checkpoint.exists():
            raise ValueError(f"DeepSVDD checkpoint not found: {svdd_checkpoint}")
        
        detector = AnomalyDetector(
            configs=[model_config, svdd_config],
            checkpoint_paths=[str(ae_checkpoint), str(svdd_checkpoint)],
            device=device
        )
        
        logger.info(" Loading model checkpoints...")
        detector.load_models()
        logger.info(" Models loaded successfully")
        
        return detector
        
    except Exception as e:
        logger.error(f" Failed to initialize anomaly detector: {e}")
        raise ValueError(f"Model initialization failed: {e}") from e

def compute_anomaly_scores(model_type: str, config_name: str = DEFAULT_CONFIG,
                         network_key: str = 'AEwRES', device: torch.device = None, 
                         output_path: Optional[Path] = None) -> List[Dict]:
    """Compute anomaly scores for all files in the training dataset.
    
    Args:
        model_type: Type of model ('bass' or 'guitar')
        config_name: Name of the experiment configuration
        device: Torch device to use
        output_path: Custom output path for results
    
    Returns:
        List of dictionaries containing file information and anomaly scores
    
    Raises:
        ValueError: If computation fails
    """
    logger = logging.getLogger('compute_anomaly_scores')
    
    if device is None:
        device = str(initialize_device())
    elif isinstance(device, torch.device):
        device = str(device)
    
    try:
        logger.info(f" Computing anomaly scores for {model_type} model...")
        
        # Load configuration
        logger.info(f" Loading configuration: {config_name}")
        config = load_experiment_config(config_name)
        
        # Get model configuration
        model_choices = get_model_choices_from_config(config, network_key)
        if model_type not in model_choices:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_choices.keys())}")
        
        model_choice = model_choices[model_type]
        model_config = config.networks[model_choice['model_key']].model_dump()
        svdd_config = config.deepSVDD.model_dump()
        
        # Get dataset paths from configuration
        dataset_name = model_choice['dataset_name']
        dataset_key = f'HTSAT_base_musicradar_{model_type}'
        if dataset_key not in config.dataset_paths:
            raise ValueError(f"Dataset path not found in config: {dataset_key}")
        
        # Build dataset paths from config
        dataset_path_str = config.dataset_paths[dataset_key]
        # Handle relative paths from config
        if dataset_path_str.startswith('./'):
            dataset_path_str = dataset_path_str[2:]  # Remove ./ prefix
        
        dataset_path = PROJECT_ROOT / dataset_path_str
        # Assume index file follows naming convention (add _index before .pkl)
        dataset_index_path = dataset_path.with_name(dataset_path.stem + '_index.pkl')
        
        # Validate datasets
        is_valid, error_msg, _ = validate_dataset(dataset_path, dataset_index_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Load data
        data, data_index, validation_info = load_and_validate_data(dataset_path, dataset_index_path)
        
        # Update model config with data dimensions
        if 'num_features' not in model_config or model_config['num_features'] is None:
            model_config['num_features'] = validation_info['feature_dim']
            logger.info(f" Set num_features to {model_config['num_features']}")
        
        # Build checkpoint paths (simplified fallback approach)
        checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'loop_benchmark' / config_name.upper() / model_choice['model_key']
        
        # Try to find existing checkpoints with pattern matching
        ae_checkpoints = list(checkpoint_dir.glob(f"*{dataset_name}*AE*.ckpt"))
        svdd_checkpoints = list(checkpoint_dir.glob(f"*{dataset_name}*DSVDD*.ckpt"))
        
        if not ae_checkpoints:
            raise ValueError(f"No AutoEncoder checkpoint found for {dataset_name} in {checkpoint_dir}")
        if not svdd_checkpoints:
            raise ValueError(f"No DeepSVDD checkpoint found for {dataset_name} in {checkpoint_dir}")
        
        # Use first matching checkpoint (could be improved with best validation loss selection)
        ae_checkpoint = ae_checkpoints[0]
        svdd_checkpoint = svdd_checkpoints[0]
        
        logger.info(f" Using AE checkpoint: {ae_checkpoint.name}")
        logger.info(f" Using SVDD checkpoint: {svdd_checkpoint.name}")
        
        # Initialize detector
        detector = initialize_anomaly_detector(model_config, svdd_config, ae_checkpoint, svdd_checkpoint, device)
        
        # Process files
        results = []
        failed_count = 0
        
        logger.info(f" Processing {len(data)} files...")
        for idx in tqdm(range(len(data)), desc=f"Computing {model_type} scores"):
            try:
                # Get anomaly score (distance from center) for current file
                file_data = data[idx:idx+1]  # Get single file data
                loop_data = detector.get_loop_score(file_data)
                _, dist = loop_data[0]
                
                file_path = data_index[idx][1] if idx < len(data_index) else f"unknown_{idx}"
                
                results.append({
                    'file_id': idx,
                    'file_path': file_path,
                    'anomaly_score': float(dist),
                })
                
            except Exception as e:
                failed_count += 1
                logger.debug(f" Error processing file {idx}: {e}")
                
                file_path = data_index[idx][1] if idx < len(data_index) else f"unknown_{idx}"
                results.append({
                    'file_id': idx,
                    'file_path': file_path,
                    'anomaly_score': float('inf'),
                    'error': str(e)
                })
        
        # Sort results by anomaly score (lowest first = most typical)
        results.sort(key=lambda x: x['anomaly_score'])
        
        # Save results
        if output_path is None:
            output_path = BASE_DIR / f'anomaly_scores_{model_type}.pkl'
        
        logger.info(f" Saving results to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Display summary
        success_count = len(results) - failed_count
        logger.info(f" Completed processing: {success_count}/{len(results)} successful")
        if failed_count > 0:
            logger.warning(f" Failed files: {failed_count}")
        
        logger.info(" Top 5 most typical loops (lowest anomaly scores):")
        for i, result in enumerate(results[:5]):
            if result['anomaly_score'] != float('inf'):
                logger.info(f"   {i+1}. {Path(result['file_path']).name} - Score: {result['anomaly_score']:.6f}")
        
        return results
        
    except Exception as e:
        logger.error(f" Anomaly score computation failed: {e}")
        raise ValueError(f"Computation failed for {model_type}: {e}") from e

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with comprehensive options.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Compute anomaly scores for trained DeepSVDD models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                                    # Process both bass and guitar with defaults
  %(prog)s --model-type bass --device cuda   # Process only bass model on GPU
  %(prog)s --config exp1 --output results/   # Use different config and output directory
  %(prog)s --dry-run --log-level DEBUG       # Validate setup without processing
        """
    )
    
    parser.add_argument(
        '--config', 
        default=DEFAULT_CONFIG,
        help=f'Experiment configuration name (default: {DEFAULT_CONFIG})'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['bass', 'guitar', 'both'],
        default='both',
        help='Model type to process (default: both)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Compute device (default: auto)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory or file path (default: script directory)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and datasets without processing'
    )
    
    parser.add_argument(
        '--network',
        help='Network type to use (default: AEwRES)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def validate_configuration(args: argparse.Namespace) -> bool:
    """Validate command-line arguments and configuration.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        True if configuration is valid
    
    Raises:
        ValueError: If configuration validation fails
    """
    logger = logging.getLogger('compute_anomaly_scores')
    
    try:
        logger.info(" Validating configuration...")
        
        # Validate config exists
        try:
            config = load_experiment_config(args.config)
            logger.debug(f" Configuration '{args.config}' loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load configuration '{args.config}': {e}")
        
        # Validate model types
        config = load_experiment_config(args.config)
        model_choices = get_model_choices_from_config(config, 'AEwRES')  # Use default network for validation
        if args.model_type != 'both' and args.model_type not in model_choices:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        # Validate output path
        if args.output:
            if args.output.suffix == '.pkl':
                # Specific file path
                args.output.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Directory path
                args.output.mkdir(parents=True, exist_ok=True)
        
        logger.info(" Configuration validation successful")
        return True
        
    except Exception as e:
        logger.error(f" Configuration validation failed: {e}")
        raise


def display_execution_summary(results: Dict[str, List[Dict]], failed_models: List[str]):
    """Display comprehensive execution summary.
    
    Args:
        results: Dictionary mapping model types to their results
        failed_models: List of models that failed processing
    """
    logger = logging.getLogger('compute_anomaly_scores')
    
    logger.info("\\n" + "="*60)
    logger.info(" EXECUTION SUMMARY")
    logger.info("="*60)
    
    total_processed = sum(len(result_list) for result_list in results.values())
    total_successful = sum(
        len([r for r in result_list if r['anomaly_score'] != float('inf')])
        for result_list in results.values()
    )
    
    logger.info(f" Models processed: {len(results) + len(failed_models)}")
    logger.info(f" Successful models: {len(results)}")
    if failed_models:
        logger.info(f" Failed models: {len(failed_models)} ({', '.join(failed_models)})")
    
    logger.info(f" Total files processed: {total_processed}")
    logger.info(f" Successful computations: {total_successful}")
    
    if total_processed > 0:
        success_rate = (total_successful / total_processed) * 100
        logger.info(f" Success rate: {success_rate:.1f}%")
    
    for model_type, result_list in results.items():
        model_successful = len([r for r in result_list if r['anomaly_score'] != float('inf')])
        logger.info(f"    {model_type}: {model_successful}/{len(result_list)} files")
    
    logger.info("="*60)


def main() -> int:
    """Main execution function with comprehensive error handling.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    try:
        logger.info(" Starting anomaly score computation...")
        logger.info(f" Configuration: {args.config}")
        logger.info(f" Model type: {args.model_type}")
        logger.info(f" Device: {args.device}")
        
        # Validate configuration
        validate_configuration(args)
        
        if args.dry_run:
            logger.info(" Dry-run completed successfully - all validations passed")
            return 0
        
        # Initialize device
        device = initialize_device(args.device)
        
        # Determine models to process
        if args.model_type == 'both':
            models_to_process = ['bass', 'guitar']
        else:
            models_to_process = [args.model_type]
        
        # Process models
        results = {}
        failed_models = []
        
        for model_type in models_to_process:
            try:
                logger.info(f"\\n Processing {model_type} model...")
                
                # Determine output path
                if args.output:
                    if args.output.suffix == '.pkl':
                        output_path = args.output
                    else:
                        output_path = args.output / f'anomaly_scores_{model_type}.pkl'
                else:
                    output_path = None
                
                # Compute scores
                model_results = compute_anomaly_scores(
                    model_type=model_type,
                    config_name=args.config,
                    network_key=args.network if args.network else 'AEwRES',
                    device=device,
                    output_path=output_path
                )
                
                results[model_type] = model_results
                logger.info(f" {model_type} model completed successfully")
                
            except Exception as e:
                failed_models.append(model_type)
                logger.error(f" {model_type} model failed: {e}")
                if args.log_level == 'DEBUG':
                    logger.exception("Full traceback:")
        
        # Display summary
        display_execution_summary(results, failed_models)
        
        # Determine exit code
        if failed_models:
            logger.warning(f" Some models failed: {', '.join(failed_models)}")
            return 1 if len(results) == 0 else 0  # Partial success
        else:
            logger.info("🎉 All models processed successfully!")
            return 0
    
    except KeyboardInterrupt:
        logger.warning("\\n Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f" Execution failed: {e}")
        if args.log_level == 'DEBUG':
            logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
