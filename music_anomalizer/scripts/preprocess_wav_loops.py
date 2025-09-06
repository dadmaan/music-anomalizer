"""
Audio Preprocessing Script for WAV Loops

This script preprocesses audio files by:
1. Loading audio with specified sample rate and mono conversion
2. Adjusting audio length to a target duration through padding, repeating, or truncating
3. Encoding audio tags using MultiLabelBinarizer
4. Saving processed data with labels as a pickle file
5. Using multiprocessing for efficient processing of large datasets

The script provides comprehensive CLI interface, validation, and error handling.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import time

import librosa
import yaml
import json
import numpy as np
import pickle
import multiprocessing
from tqdm import tqdm

from music_anomalizer.utils import construct_file_path, setup_logging
from music_anomalizer.config import load_config
from sklearn.preprocessing import MultiLabelBinarizer




def validate_metadata_file(metadata_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate and load metadata file with comprehensive checks.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Loaded metadata dictionary
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        json.JSONDecodeError: If metadata file is invalid JSON
        ValueError: If metadata structure is invalid
    """
    metadata_path = Path(metadata_path)
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    if metadata_path.stat().st_size == 0:
        raise ValueError(f"Metadata file is empty: {metadata_path}")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in metadata file: {e}", 
                                   metadata_path.read_text(), e.pos)
    
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dictionary, got {type(metadata)}")
    
    if not metadata:
        raise ValueError("Metadata dictionary is empty")
    
    # Validate metadata structure
    sample_key = next(iter(metadata.keys()))
    sample_item = metadata[sample_key]
    
    required_fields = ['file_path', 'audio_tags']
    for field in required_fields:
        if field not in sample_item:
            raise ValueError(f"Missing required field '{field}' in metadata")
    
    return metadata


def validate_audio_files(metadata: Dict[str, Any]) -> Tuple[int, int]:
    """
    Validate that audio files referenced in metadata exist.
    
    Args:
        metadata: Metadata dictionary with file paths
        
    Returns:
        Tuple of (existing_files, total_files)
    """
    total_files = len(metadata)
    existing_files = 0
    missing_files = []
    
    for key, info in metadata.items():
        file_path = Path(info['file_path'])
        if file_path.exists():
            existing_files += 1
        else:
            missing_files.append(str(file_path))
    
    if missing_files:
        logging.warning(f"Found {len(missing_files)} missing audio files")
        if len(missing_files) <= 5:
            for missing_file in missing_files:
                logging.warning(f"  Missing: {missing_file}")
        else:
            logging.warning(f"  First 5 missing files:")
            for missing_file in missing_files[:5]:
                logging.warning(f"    {missing_file}")
            logging.warning(f"  ... and {len(missing_files) - 5} more")
    
    return existing_files, total_files


def encode_audio_tags(tags: List[List[str]], logger: logging.Logger) -> np.ndarray:
    """
    Encode audio tags using MultiLabelBinarizer with enhanced logging.
    
    Args:
        tags: List of audio tags to encode
        logger: Logger instance for output
        
    Returns:
        Encoded tags as binary matrix
        
    Raises:
        ValueError: If tags list is empty or invalid
    """
    if not tags:
        raise ValueError("Tags list is empty")
    
    if not all(isinstance(tag_list, list) for tag_list in tags):
        raise ValueError("All tag entries must be lists")
    
    try:
        mlb = MultiLabelBinarizer()
        encoder = mlb.fit(tags)
        encoded_tags = encoder.transform(tags)
        
        logger.info(f"📊 Encoded {len(tags)} tag entries")
        logger.info(f"📊 Number of unique classes: {len(mlb.classes_)}")
        logger.debug(f"📊 Classes: {list(mlb.classes_)}")
        
        return encoded_tags
        
    except Exception as e:
        raise ValueError(f"Failed to encode audio tags: {e}")


def process_audio_length(wav: np.ndarray, sec_wav: float, target_length: int, 
                        sr: int, only_pad: bool, logger: logging.Logger) -> np.ndarray:
    """
    Adjusts the length of the audio time series to match the target length.
    
    Args:
        wav: The audio time series
        sec_wav: The original length of the audio in seconds
        target_length: The desired length of the audio in seconds
        sr: The sampling rate of the audio
        only_pad: Only pad shorter audio files with zero
        logger: Logger instance for debug output
        
    Returns:
        The processed audio time series of exactly target_length seconds
        
    Raises:
        ValueError: If input parameters are invalid
    """
    if target_length <= 0:
        raise ValueError(f"Target length must be positive, got {target_length}")
    
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")
    
    if len(wav) == 0:
        raise ValueError("Audio array is empty")
    
    current_length_samples = int(sec_wav * sr)
    target_length_samples = int(target_length * sr)
    
    logger.debug(f"🎵 Processing audio: {sec_wav:.2f}s -> {target_length}s")
    logger.debug(f"🎵 Samples: {current_length_samples} -> {target_length_samples}")
    
    if current_length_samples < target_length_samples:
        if only_pad:
            # Only pad with zeros
            pad_length = target_length_samples - len(wav)
            extended_wav = np.pad(wav, (0, pad_length), 'constant')
            logger.debug(f"🎵 Padded with {pad_length} zero samples")
        else:
            # Calculate how many full repeats are needed
            repeat_count = target_length_samples // current_length_samples
            # Repeat the audio
            extended_wav = np.tile(wav, repeat_count)
            logger.debug(f"🎵 Repeated audio {repeat_count} times")
            
            # If still not enough samples, pad the remaining
            if len(extended_wav) < target_length_samples:
                pad_length = target_length_samples - len(extended_wav)
                extended_wav = np.pad(extended_wav, (0, pad_length), 'constant')
                logger.debug(f"🎵 Added final padding of {pad_length} samples")
    else:
        # If the audio is longer than the target, just truncate it
        extended_wav = wav[:target_length_samples]
        logger.debug(f"🎵 Truncated to {target_length_samples} samples")
    
    # Ensure exact length
    if len(extended_wav) != target_length_samples:
        extended_wav = extended_wav[:target_length_samples]
        logger.debug(f"🎵 Final adjustment to exact target length")
    
    return extended_wav


def process_audio_file(args: Tuple[str, np.ndarray, Dict[str, Any]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Function to load and process a single audio file with comprehensive error handling.
    
    Args:
        args: Tuple containing (path, label, config)
        
    Returns:
        Tuple of (processed_audio, label) or None if error occurs
    """
    try:
        path, label, config = args
        
        # Validate path exists
        if not Path(path).exists():
            logging.warning(f"⚠️ File not found: {path}")
            return None
        
        # Extract config parameters with validation
        sample_rate = config.get('sample_rate', 22050)
        target_length = config.get('target_audio_length', 30)
        mono = config.get('mono', True)
        only_pad = config.get('only_pad', False)
        
        # Load audio file with error handling
        try:
            audio, loaded_sr = librosa.load(path, sr=sample_rate, mono=mono)
        except Exception as load_error:
            logging.warning(f"⚠️ Failed to load {path}: {load_error}")
            return None
        
        if len(audio) == 0:
            logging.warning(f"⚠️ Empty audio file: {path}")
            return None
        
        # Get duration and process length
        sec_wav = librosa.get_duration(y=audio, sr=sample_rate)
        
        if sec_wav <= 0:
            logging.warning(f"⚠️ Invalid duration {sec_wav}s for file: {path}")
            return None
        
        # Create a logger for this process (multiprocessing safe)
        logger = logging.getLogger(f"worker_{os.getpid()}")
        
        try:
            audio = process_audio_length(audio, sec_wav, target_length, sample_rate, only_pad, logger)
        except Exception as process_error:
            logging.warning(f"⚠️ Failed to process audio length for {path}: {process_error}")
            return None
        
        # Final length validation
        expected_samples = int(sample_rate * target_length)
        if len(audio) != expected_samples:
            audio = audio[:expected_samples]  # Final truncation if needed
        
        return (audio, label)
    
    except Exception as e:
        # Extract path safely for error message
        try:
            path = args[0] if args and len(args) > 0 else "unknown"
        except:
            path = "unknown"
        
        logging.error(f"❌ Error processing file {path}: {e}")
        return None


def run_process(meta_info: Dict[str, Any], config: Dict[str, Any], 
               output_dir: str = 'data', num_workers: Optional[int] = None,
               logger: logging.Logger = None) -> Tuple[bool, Optional[str], Dict[str, int]]:
    """
    Preprocess audio files using multiprocessing and save them with labels as a pickle file.
    
    Args:
        meta_info: Audio files metadata
        config: Configuration parameters
        output_dir: Directory to save the pickle file
        num_workers: Number of worker processes to use (None for auto-detect)
        logger: Logger instance
        
    Returns:
        Tuple of (success, output_file_path, statistics)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Display configuration
    config_details = "\n".join(f"  {key}: {value}" for key, value in config.items())
    logger.info(f"🔧 Preprocessing configuration:\n{config_details}")
    
    # Validate metadata structure
    if not meta_info:
        raise ValueError("Metadata is empty")
    
    # Extract paths and tags
    audio_paths = []
    audio_tags = []
    
    for key, info in meta_info.items():
        if 'file_path' not in info or 'audio_tags' not in info:
            logger.warning(f"⚠️ Skipping invalid metadata entry: {key}")
            continue
        
        audio_paths.append(info['file_path'])
        # Handle both string and list audio_tags
        tags = info['audio_tags']
        if isinstance(tags, str):
            audio_tags.append([tags])
        elif isinstance(tags, list):
            audio_tags.append(tags)
        else:
            logger.warning(f"⚠️ Invalid audio_tags format for {key}: {type(tags)}")
            audio_tags.append(['unknown'])
    
    if not audio_paths:
        raise ValueError("No valid audio files found in metadata")
    
    logger.info(f"📁 Found {len(audio_paths)} audio files to process")
    
    # Encode tags
    try:
        labels = encode_audio_tags(audio_tags, logger)
    except Exception as e:
        logger.error(f"❌ Failed to encode audio tags: {e}")
        return False, None, {'total': len(audio_paths), 'processed': 0, 'failed': len(audio_paths)}
    
    # Prepare arguments for multiprocessing
    args = [(path, label, config) for path, label in zip(audio_paths, labels)]
    
    # Determine number of workers
    if num_workers is None or num_workers == -1:
        num_workers = min(os.cpu_count() or 1, len(audio_paths))
    
    num_workers = max(1, min(num_workers, len(audio_paths)))
    logger.info(f"🔄 Using {num_workers} worker processes")
    
    # Process files with progress tracking
    logger.info(f"🎵 Processing {len(args)} audio files...")
    start_time = time.time()
    
    try:
        if num_workers == 1:
            # Single-threaded processing with progress bar
            processed_data = []
            for arg in tqdm(args, desc="Processing audio files"):
                result = process_audio_file(arg)
                processed_data.append(result)
        else:
            # Multi-threaded processing
            with multiprocessing.Pool(num_workers) as pool:
                processed_data = pool.map(process_audio_file, args)
    except Exception as e:
        logger.error(f"❌ Multiprocessing failed: {e}")
        return False, None, {'total': len(args), 'processed': 0, 'failed': len(args)}
    
    processing_time = time.time() - start_time
    
    # Filter out None results and calculate statistics
    valid_data = [item for item in processed_data if item is not None]
    
    total_files = len(processed_data)
    processed_files = len(valid_data)
    failed_files = total_files - processed_files
    
    statistics = {
        'total': total_files,
        'processed': processed_files,
        'failed': failed_files,
        'processing_time': processing_time
    }
    
    logger.info(f"📊 Processing complete: {processed_files}/{total_files} successful ({processing_time:.2f}s)")
    
    if failed_files > 0:
        logger.warning(f"⚠️ {failed_files} files failed to process")
    
    if not valid_data:
        logger.error("❌ No valid data to save")
        return False, None, statistics
    
    # Save processed data
    try:
        output_path = construct_file_path(config, output_dir)
        output_file = os.path.join(output_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'wb') as f:
            pickle.dump(valid_data, f)
        
        file_size = os.path.getsize(output_file)
        logger.info(f"💾 Data saved to {output_file} ({file_size / (1024*1024):.2f} MB)")
        
        return True, output_file, statistics
        
    except Exception as e:
        logger.error(f"❌ Failed to save processed data: {e}")
        return False, None, statistics


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments with comprehensive help and validation.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Audio Preprocessing Script for WAV Loops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Basic preprocessing with defaults
  python preprocess_wav_loops.py
  
  # Use custom configuration and metadata files
  python preprocess_wav_loops.py --config custom_config.yaml --metadata custom_meta.json
  
  # Specify output directory and worker count
  python preprocess_wav_loops.py --output-dir processed_data --workers 8
  
  # Validate configuration without processing
  python preprocess_wav_loops.py --dry-run --log-level DEBUG
  
  # Single-threaded processing with verbose logging
  python preprocess_wav_loops.py --workers 1 --log-level INFO
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/audio_preprocessing_config.yaml',
        help='Path to audio preprocessing configuration file (default: configs/audio_preprocessing_config.yaml)'
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        default='output/metadata/LP_meta_info_v2_processed_balanced_i.json',
        help='Path to metadata JSON file (default: output/metadata/LP_meta_info_v2_processed_balanced_i.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for processed data (default: data)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes (default: auto-detect CPU count)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and metadata without processing'
    )
    
    return parser.parse_args()


def validate_configuration(args: argparse.Namespace, logger: logging.Logger) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate configuration files and return loaded data.
    
    Args:
        args: Command-line arguments
        logger: Logger instance
        
    Returns:
        Tuple of (config, metadata)
        
    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If configuration is invalid
    """
    # Validate and load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        if config_path.suffix.lower() == '.yaml':
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Try loading as audio preprocessing config using load_config
            config = load_config(config_path)
    except Exception as e:
        raise ValueError(f"Failed to load configuration: {e}")
    
    # Validate configuration structure
    required_config_keys = ['sample_rate', 'target_audio_length', 'mono']
    missing_keys = [key for key in required_config_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Validate and load metadata
    metadata = validate_metadata_file(args.metadata)
    
    logger.info(f"✅ Configuration loaded from {config_path}")
    logger.info(f"✅ Metadata loaded from {args.metadata}")
    
    return config, metadata


def display_execution_summary(success: bool, output_file: Optional[str], 
                            statistics: Dict[str, int], logger: logging.Logger) -> None:
    """
    Display comprehensive execution summary with statistics.
    
    Args:
        success: Whether processing was successful
        output_file: Path to output file if successful
        statistics: Processing statistics
        logger: Logger instance
    """
    logger.info("\n" + "="*60)
    logger.info("📋 PREPROCESSING EXECUTION SUMMARY")
    logger.info("="*60)
    
    if success:
        logger.info(f"✅ Status: SUCCESSFUL")
        if output_file:
            logger.info(f"📁 Output File: {output_file}")
    else:
        logger.info(f"❌ Status: FAILED")
    
    logger.info(f"📊 Total Files: {statistics.get('total', 0)}")
    logger.info(f"✅ Processed: {statistics.get('processed', 0)}")
    logger.info(f"❌ Failed: {statistics.get('failed', 0)}")
    
    if 'processing_time' in statistics:
        logger.info(f"⏱️ Processing Time: {statistics['processing_time']:.2f} seconds")
    
    success_rate = (statistics.get('processed', 0) / max(statistics.get('total', 1), 1)) * 100
    logger.info(f"📈 Success Rate: {success_rate:.1f}%")
    
    logger.info("="*60)


def main() -> int:
    """
    Main function to run the preprocessing pipeline with comprehensive CLI interface.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        logger = setup_logging(args.log_level)
        
        logger.info("🎵 Audio Preprocessing Script for WAV Loops")
        logger.info(f"📋 Arguments: {vars(args)}")
        
        # Validate configuration and metadata
        try:
            config, metadata = validate_configuration(args, logger)
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return 1
        
        # Validate audio files exist
        existing_files, total_files = validate_audio_files(metadata)
        logger.info(f"📁 Audio files: {existing_files}/{total_files} exist")
        
        if existing_files == 0:
            logger.error("❌ No audio files found")
            return 1
        
        # Dry-run mode
        if args.dry_run:
            logger.info("🔍 Dry-run mode: Configuration and metadata validation complete")
            logger.info(f"✅ Would process {existing_files} files with {args.workers or 'auto'} workers")
            return 0
        
        # Run preprocessing
        success, output_file, statistics = run_process(
            metadata, config, args.output_dir, args.workers, logger
        )
        
        # Display summary
        display_execution_summary(success, output_file, statistics, logger)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug(f"Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
