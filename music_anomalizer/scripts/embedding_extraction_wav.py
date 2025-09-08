#!/usr/bin/env python3

"""
Audio Embedding Extraction Script

This script extracts embeddings from audio files using the CLAP (Contrastive Language-Audio Pre-training) and HTSAT (Hierarchical Token-Semantic Audio Transformer) model.
It processes WAV files using concurrent processing for efficient handling of large datasets and saves
embeddings and index data as pickle files with flexible output naming and organization.

The script supports configurable CLAP model variants, device selection, concurrent processing parameters,
and comprehensive error handling with progress tracking and detailed execution summaries.

Usage:
    python music_anomalizer/scripts/embedding_extraction_wav.py
    python music_anomalizer/scripts/embedding_extraction_wav.py --dataset data/dataset/guitar --output data/embeddings
    python music_anomalizer/scripts/embedding_extraction_wav.py --device cuda --workers 16 --model HTSAT-base
    python music_anomalizer/scripts/embedding_extraction_wav.py --checkpoint path/to/clap_model.pt --dry-run
"""

import argparse
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
from tqdm import tqdm

from music_anomalizer.preprocessing.extract_embed import (
    load_audio, get_features, prepare_input_dict, prepare_audio_as_tensor, get_audio_branch
)
from music_anomalizer.utils import (
    PickleHandler, setup_logging, initialize_device, validate_directory_path,
    validate_file_path, display_execution_summary
)

try:
    import laion_clap
except ImportError:
    laion_clap = None

# Default configuration
DEFAULT_CONFIG = {
    'extract_features': True,
    'audio_model': 'HTSAT-base',
    'checkpoint_path': 'checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt',
    'max_workers': 8,
    'enable_fusion': False
}

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent.parent






def validate_dataset_path(dataset_path: Path) -> Tuple[bool, str]:
    """Validate dataset directory exists and contains audio files.
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    logger = logging.getLogger('embedding_extraction')
    
    if not dataset_path.exists():
        return False, f"Dataset directory not found: {dataset_path}"
    
    if not dataset_path.is_dir():
        return False, f"Dataset path is not a directory: {dataset_path}"
    
    # Check for audio files
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(dataset_path.glob(f"**/{ext}")))
    
    if not audio_files:
        return False, f"No audio files found in {dataset_path} (searched for {', '.join(audio_extensions)})"
    
    logger.debug(f" Found {len(audio_files)} audio files in dataset")
    return True, ""


def validate_checkpoint_path(checkpoint_path: Path) -> Tuple[bool, str]:
    """Validate CLAP model checkpoint file exists.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not checkpoint_path.exists():
        return False, f"Checkpoint file not found: {checkpoint_path}"
    
    if not checkpoint_path.is_file():
        return False, f"Checkpoint path is not a file: {checkpoint_path}"
    
    # Check file size (should be substantial for a model)
    file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
    if file_size < 1:  # Less than 1 MB is suspicious for a model
        return False, f"Checkpoint file seems too small: {file_size:.1f} MB"
    
    return True, ""


def initialize_clap_model(checkpoint_path: str, audio_model: str, device: torch.device,
                         enable_fusion: bool = False) -> Any:
    """Initialize CLAP model with checkpoint validation.
    
    Args:
        checkpoint_path: Path to CLAP model checkpoint
        audio_model: CLAP audio model variant
        device: Torch device to use
        enable_fusion: Whether to enable fusion in CLAP model
    
    Returns:
        Initialized CLAP model
    
    Raises:
        ValueError: If model initialization fails
    """
    logger = logging.getLogger('embedding_extraction')
    
    if laion_clap is None:
        raise ValueError("laion_clap library not available. Please install it to use CLAP models.")
    
    try:
        logger.info(" Initializing CLAP model...")
        logger.info(f"    Model variant: {audio_model}")
        logger.info(f"    Fusion enabled: {enable_fusion}")
        
        # Validate checkpoint
        checkpoint_path_obj = Path(checkpoint_path)
        is_valid, error_msg = validate_checkpoint_path(checkpoint_path_obj)
        if not is_valid:
            raise ValueError(error_msg)
        
        file_size = checkpoint_path_obj.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"    Checkpoint size: {file_size:.1f} MB")
        
        # Initialize model
        clap_model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model)
        
        logger.info(" Loading checkpoint...")
        clap_model.load_ckpt(checkpoint_path)
        
        # Move to device
        clap_model.to(device)
        clap_model.eval()
        
        logger.info(" CLAP model initialized successfully")
        return clap_model
        
    except Exception as e:
        logger.error(f" Failed to initialize CLAP model: {e}")
        raise ValueError(f"CLAP model initialization failed: {e}") from e


def get_audio_files(dataset_path: Path) -> List[Path]:
    """Get list of audio files from dataset directory.
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        List of audio file paths
    """
    logger = logging.getLogger('embedding_extraction')
    
    # Search for common audio formats
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        files = list(dataset_path.glob(f"**/{ext}"))
        audio_files.extend(files)
        if files:
            logger.debug(f" Found {len(files)} {ext.replace('*', '')} files")
    
    # Sort for consistent ordering
    audio_files.sort()
    
    logger.info(f"🎵 Total audio files found: {len(audio_files)}")
    return audio_files


def extract_embedding(model: Any, audio_path: Path, device: torch.device, 
                     extract_features: bool = True, clap_model: Any = None) -> Optional[np.ndarray]:
    """Extract embedding from a single audio file using CLAP model.
    
    This function processes audio data through the CLAP model to generate embeddings.
    It supports both feature extraction preprocessing and direct audio tensor processing.
    
    Args:
        model: The CLAP audio branch model
        audio_path: Path to the audio file
        device: Device to run the model on
        extract_features: Whether to extract features or use direct audio tensor
        clap_model: Full CLAP model (needed for feature extraction)
        
    Returns:
        Audio embedding array or None if error occurs
    """
    logger = logging.getLogger('embedding_extraction')
    
    try:
        audio_path_str = str(audio_path)
        
        if extract_features:
            if clap_model is None:
                raise ValueError("Full CLAP model required for feature extraction")
                
            audio_data = load_audio(audio_path_str, expand_dim=True)
            audio_input = get_features(audio_data, clap_model)
            audio_tensor = prepare_input_dict(audio_input, device)
        else:
            audio_data = load_audio(audio_path_str, expand_dim=False)
            audio_tensor = prepare_audio_as_tensor(audio_data, device)
            
        # Get model predictions
        model.eval()
        with torch.no_grad():
            output_dict = model(audio_tensor, device)
        
        if 'embedding' not in output_dict:
            raise ValueError("Model output does not contain 'embedding' key")
            
        audio_embed = output_dict['embedding'].detach().cpu().numpy()
        
        # Validate embedding shape
        if audio_embed.size == 0:
            raise ValueError("Empty embedding generated")
            
        # Optimize memory usage
        del audio_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return audio_embed
        
    except Exception as e:
        logger.debug(f" Error processing file {audio_path.name}: {str(e)}")
        return None


def worker(args: Tuple[Path, Any, torch.device, bool, Any]) -> Tuple[Path, Optional[np.ndarray]]:
    """Worker function for concurrent processing of a single audio file.
    
    Args:
        args: Tuple containing (path, model, device, extract_features, clap_model)
        
    Returns:
        Tuple of (path, embedding) or (path, None) if error occurs
    """
    path, model, device, extract_features, clap_model = args
    embedding = extract_embedding(model, path, device, extract_features, clap_model)
    return path, embedding


def create_output_filename(dataset_path: Path, model_name: str, num_files: int,
                          custom_name: Optional[str] = None) -> str:
    """Create standardized output filename for embeddings.
    
    Args:
        dataset_path: Path to dataset directory
        model_name: Name of the CLAP model variant
        num_files: Number of files processed
        custom_name: Optional custom name component
        
    Returns:
        Standardized filename
    """
    if custom_name:
        base_name = custom_name
    else:
        dataset_name = dataset_path.name
        base_name = f"{model_name}_{dataset_name}"
    
    return f"{base_name}_embeddings.pkl"


def run_process(audio_paths: List[Path], model: Any, device: torch.device, 
               output_file: Path, extract_features: bool = True, max_workers: int = 8,
               clap_model: Any = None) -> Dict[str, Any]:
    """Process audio files to extract embeddings using concurrent processing.
    
    This function orchestrates the concurrent processing of audio files to extract embeddings
    and saves the results with comprehensive error tracking and progress monitoring.

    Args:
        audio_paths: List of audio file paths
        model: The CLAP model used to extract embeddings  
        device: Device to run the model on
        output_file: Path to the output pickle file to save embeddings
        extract_features: Whether to extract features during processing
        max_workers: Maximum number of concurrent workers
        clap_model: Full CLAP model (needed for feature extraction)
        
    Returns:
        Dictionary with processing statistics
    """
    logger = logging.getLogger('embedding_extraction')
    
    index_file = output_file.parent / (output_file.stem + '_index.pkl')
    
    logger.info(f" Processing {len(audio_paths)} audio files...")
    logger.info(f"    Workers: {max_workers}")
    logger.info(f"    Extract features: {extract_features}")
    logger.info(f"    Output: {output_file}")
    logger.info(f"    Index: {index_file}")
    
    processed_data = []
    index_data = []
    failed_files = []
    
    try:
        # Prepare arguments for workers
        worker_args = [(path, model, device, extract_features, clap_model) for path in audio_paths]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(worker, args) for args in worker_args]
            
            # Process completed tasks with progress bar
            for future in tqdm(futures, desc="Extracting embeddings", unit="file"):
                try:
                    path, emb = future.result(timeout=60)  # 60 second timeout per file
                    if emb is not None:
                        processed_data.append(emb.flatten())
                        # Store original index and path
                        original_index = audio_paths.index(path)
                        index_data.append((original_index, str(path)))
                    else:
                        failed_files.append(path)
                        
                except Exception as e:
                    logger.debug(f" Worker exception: {e}")
                    failed_files.append(path if 'path' in locals() else "unknown")
        
        # Generate processing statistics
        stats = {
            'total_files': len(audio_paths),
            'successful': len(processed_data),
            'failed': len(failed_files),
            'success_rate': (len(processed_data) / len(audio_paths)) * 100 if audio_paths else 0
        }
        
        # Save results if any successful processing occurred
        if processed_data:
            logger.info(f" Saving {len(processed_data)} embeddings...")
            
            # Save embeddings
            embeddings_array = np.array(processed_data)
            pickle_handler = PickleHandler(str(output_file))
            pickle_handler.dump_data(embeddings_array)
            
            # Save index
            index_handler = PickleHandler(str(index_file))
            index_handler.dump_data(index_data)
            
            logger.info(f" Embeddings saved: {output_file}")
            logger.info(f" Index saved: {index_file}")
            logger.info(f"    Shape: {embeddings_array.shape}")
            
            stats['output_files'] = [str(output_file), str(index_file)]
            stats['embedding_shape'] = embeddings_array.shape
            
        else:
            logger.warning(" No valid embeddings to save")
            stats['output_files'] = []
        
        # Report failed files if any
        if failed_files:
            logger.warning(f" Failed to process {len(failed_files)} files")
            if logger.level <= logging.DEBUG:
                for failed_file in failed_files[:5]:  # Show first 5 failures
                    logger.debug(f"    {failed_file}")
                if len(failed_files) > 5:
                    logger.debug(f"   ... and {len(failed_files) - 5} more")
        
        logger.info(f" Processing completed: {stats['successful']}/{stats['total_files']} successful ({stats['success_rate']:.1f}%)")
        
        return stats
        
    except Exception as e:
        logger.error(f" Processing failed: {e}")
        raise ValueError(f"Embedding extraction failed: {e}") from e


def process_dataset(dataset_path: Path, output_dir: Path, clap_model: Any, device: torch.device, 
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a complete dataset and extract embeddings with comprehensive validation.
    
    Args:
        dataset_path: Path to dataset directory
        output_dir: Output directory for embeddings
        clap_model: Initialized CLAP model
        device: Torch device to use
        config: Processing configuration
        
    Returns:
        Processing statistics and results
    """
    logger = logging.getLogger('embedding_extraction')
    
    try:
        logger.info(f"🎵 Processing dataset: {dataset_path}")
        
        # Validate dataset
        is_valid, error_msg = validate_dataset_path(dataset_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Get audio files
        audio_files = get_audio_files(dataset_path)
        if not audio_files:
            raise ValueError(f"No audio files found in {dataset_path}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f" Output directory: {output_dir}")
        
        # Generate output filename
        output_filename = create_output_filename(
            dataset_path, 
            config['audio_model'],
            len(audio_files),
            config.get('output_name')
        )
        output_file = output_dir / output_filename
        
        # Get audio branch from model for actual processing
        audio_branch = get_audio_branch(clap_model)
        
        # Process files
        stats = run_process(
            audio_files, 
            audio_branch, 
            device, 
            output_file,
            config['extract_features'], 
            config['max_workers'],
            clap_model
        )
        
        # Add dataset information to stats
        stats.update({
            'dataset_path': str(dataset_path),
            'output_dir': str(output_dir),
            'model_variant': config['audio_model'],
            'extract_features': config['extract_features']
        })
        
        logger.info(" Dataset processing completed successfully")
        return stats
        
    except Exception as e:
        logger.error(f" Dataset processing failed: {e}")
        raise ValueError(f"Dataset processing failed: {e}") from e


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with comprehensive options.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract embeddings from audio files using CLAP model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s                                    # Basic usage with defaults
  %(prog)s --dataset data/dataset/guitar --output data/embeddings
  %(prog)s --device cuda --workers 16 --model HTSAT-base  
  %(prog)s --checkpoint path/to/clap_model.pt --dry-run
  %(prog)s --no-features --output-name custom_embeddings
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('data/dataset/bass'),
        help='Path to dataset directory (default: data/dataset/bass)'
    )
    
    parser.add_argument(
        '--output',
        type=Path, 
        default=Path('data/embeddings'),
        help='Output directory for embeddings (default: data/embeddings)'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for processing (default: auto)'
    )
    
    parser.add_argument(
        '--model',
        default='HTSAT-base',
        help='CLAP audio model variant (default: HTSAT-base)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt'),
        help='Path to CLAP model checkpoint (default: checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of worker threads (default: 8)'
    )
    
    parser.add_argument(
        '--no-features',
        action='store_true',
        help='Skip feature extraction preprocessing'
    )
    
    parser.add_argument(
        '--output-name',
        help='Custom name for output files (default: auto-generated)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and dataset without processing'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def validate_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate command-line arguments and prepare configuration.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Validated configuration dictionary
    
    Raises:
        ValueError: If configuration validation fails
    """
    logger = logging.getLogger('embedding_extraction')
    
    try:
        logger.info("⚙️ Validating configuration...")
        
        # Validate dataset path
        is_valid, error_msg = validate_dataset_path(args.dataset)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Validate checkpoint if not in dry-run mode
        if not args.dry_run:
            is_valid, error_msg = validate_checkpoint_path(args.checkpoint)
            if not is_valid:
                raise ValueError(error_msg)
        
        # Validate output directory (create if needed)
        if not args.output.exists():
            args.output.mkdir(parents=True, exist_ok=True)
        
        # Prepare configuration
        config = DEFAULT_CONFIG.copy()
        config.update({
            'audio_model': args.model,
            'checkpoint_path': str(args.checkpoint),
            'max_workers': args.workers,
            'extract_features': not args.no_features,
            'output_name': args.output_name
        })
        
        logger.info(" Configuration validation successful")
        return config
        
    except Exception as e:
        logger.error(f" Configuration validation failed: {e}")
        raise


def display_execution_summary(stats: Dict[str, Any]):
    """Display comprehensive execution summary.
    
    Args:
        stats: Processing statistics dictionary
    """
    logger = logging.getLogger('embedding_extraction')
    
    logger.info("\\n" + "="*60)
    logger.info(" EXECUTION SUMMARY")
    logger.info("="*60)
    
    logger.info(f" Dataset: {Path(stats['dataset_path']).name}")
    logger.info(f" Model: {stats['model_variant']}")
    logger.info(f" Feature extraction: {stats['extract_features']}")
    
    logger.info(f" Total files: {stats['total_files']}")
    logger.info(f" Successful: {stats['successful']}")
    logger.info(f" Failed: {stats['failed']}")
    logger.info(f" Success rate: {stats['success_rate']:.1f}%")
    
    if 'embedding_shape' in stats:
        logger.info(f" Embedding shape: {stats['embedding_shape']}")
    
    if 'output_files' in stats and stats['output_files']:
        logger.info(" Output files:")
        for output_file in stats['output_files']:
            logger.info(f"    {Path(output_file).name}")
    
    logger.info("="*60)


def main() -> int:
    """Main execution function with comprehensive error handling.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    
    try:
        logger.info(" Starting audio embedding extraction...")
        logger.info(f" Dataset: {args.dataset}")
        logger.info(f" Output: {args.output}")
        logger.info(f" Device: {args.device}")
        logger.info(f" Model: {args.model}")
        
        # Validate configuration
        config = validate_configuration(args)
        
        if args.dry_run:
            logger.info(" Dry-run completed successfully - all validations passed")
            return 0
        
        # Initialize device
        device = initialize_device(args.device)
        
        # Initialize CLAP model
        logger.info(" Initializing CLAP model...")
        clap_model = initialize_clap_model(
            config['checkpoint_path'], 
            config['audio_model'], 
            device,
            config['enable_fusion']
        )
        
        # Process dataset
        logger.info(" Starting dataset processing...")
        stats = process_dataset(args.dataset, args.output, clap_model, device, config)
        
        # Display summary
        display_execution_summary(stats)
        
        # Determine exit code based on success rate
        if stats['successful'] == 0:
            logger.error(" No files processed successfully")
            return 1
        elif stats['failed'] > 0:
            logger.warning(f" Partial success: {stats['failed']} files failed")
            return 0  # Still consider success if some files processed
        else:
            logger.info(" All files processed successfully!")
            return 0
        
    except KeyboardInterrupt:
        logger.warning("\\n Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f" Execution failed: {e}")
        if args.log_level == 'DEBUG':
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
