#!/usr/bin/env python3

"""
Audio Data Preparation Script for DeepSVDD Training

This script provides a user-friendly interface for preparing audio data for DeepSVDD model training.
It calls the embedding extraction script to convert WAV/MP3/FLAC/M4A audio files into embeddings
that can be used for anomaly detection training.

The script automatically handles:
- Audio format validation and conversion
- CLAP model initialization and embedding extraction
- Output organization with proper naming conventions
- Progress tracking and error reporting
- Validation of prerequisites and dependencies

Usage:
    python prepare_data.py --audio-dir path/to/audio/files
    python prepare_data.py --audio-dir data/bass --output-dir data/embeddings --model-name bass_model
    python prepare_data.py --audio-dir data/guitar --device cuda --workers 16
    python prepare_data.py --help  # Show all available options
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
from music_anomalizer.utils import setup_logging


def validate_audio_directory(audio_dir: Path) -> bool:
    """Validate that the audio directory exists and contains supported audio files.
    
    Args:
        audio_dir (Path): Path to audio directory
        
    Returns:
        bool: True if directory is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if not audio_dir.exists():
        logger.error(f" Audio directory does not exist: {audio_dir}")
        return False
    
    if not audio_dir.is_dir():
        logger.error(f" Audio path is not a directory: {audio_dir}")
        return False
    
    # Check for supported audio files
    supported_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    
    for ext in supported_extensions:
        audio_files.extend(list(audio_dir.glob(f"**/*{ext}")))
        audio_files.extend(list(audio_dir.glob(f"**/*{ext.upper()}")))
    
    if not audio_files:
        logger.error(f" No supported audio files found in {audio_dir}")
        logger.info(f"   Supported formats: {', '.join(supported_extensions)}")
        return False
    
    logger.info(f" Found {len(audio_files)} audio files in {audio_dir}")
    return True


def validate_dependencies() -> bool:
    """Validate that required dependencies are available.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        logger.debug(" PyTorch available")
    except ImportError:
        logger.error(" PyTorch not available. Please install pytorch.")
        return False
    
    try:
        import laion_clap
        logger.debug(" LAION-CLAP available")
    except ImportError:
        logger.error(" LAION-CLAP not available. Please install laion_clap.")
        return False
    
    # Check if embedding extraction script exists
    script_path = Path(__file__).parent / "music_anomalizer" / "scripts" / "embedding_extraction_wav.py"
    if not script_path.exists():
        logger.error(f" Embedding extraction script not found: {script_path}")
        return False
    
    logger.debug(" All dependencies validated")
    return True


def validate_checkpoint_availability(checkpoint_path: Optional[Path] = None) -> bool:
    """Validate that CLAP model checkpoint is available.
    
    Args:
        checkpoint_path (Optional[Path]): Custom checkpoint path
        
    Returns:
        bool: True if checkpoint is available, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    if checkpoint_path is None:
        # Default checkpoint path
        checkpoint_path = Path("checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt")
    
    if not checkpoint_path.exists():
        logger.error(f" CLAP model checkpoint not found: {checkpoint_path}")
        logger.info("   Please ensure the CLAP model checkpoint is downloaded.")
        logger.info("   You can specify a custom checkpoint path with --checkpoint")
        return False
    
    # Check file size (CLAP models should be substantial)
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    if file_size_mb < 10:  # Less than 10 MB is suspicious for a CLAP model
        logger.error(f" Checkpoint file seems too small: {file_size_mb:.1f} MB")
        return False
    
    logger.info(f" CLAP checkpoint validated: {checkpoint_path} ({file_size_mb:.1f} MB)")
    return True


def generate_output_name(audio_dir: Path, model_name: Optional[str] = None) -> str:
    """Generate output name for embeddings based on audio directory and model.
    
    Args:
        audio_dir (Path): Audio directory path
        model_name (Optional[str]): Optional custom model name
        
    Returns:
        str: Generated output name
    """
    if model_name:
        return model_name
    else:
        # Use directory name as base
        return audio_dir.name


def prepare_audio_data(
    audio_dir: Path,
    output_dir: Path = Path("data/embeddings"),
    model_name: Optional[str] = None,
    device: str = "auto",
    model_variant: str = "HTSAT-base",
    checkpoint_path: Optional[Path] = None,
    workers: int = 8,
    extract_features: bool = True,
    dry_run: bool = False,
    log_level: str = "INFO"
) -> bool:
    """Prepare audio data by extracting embeddings using the embedding extraction script.
    
    Args:
        audio_dir (Path): Directory containing audio files
        output_dir (Path): Directory to save embeddings
        model_name (Optional[str]): Custom name for output files
        device (str): Device to use ('auto', 'cpu', 'cuda')
        model_variant (str): CLAP model variant
        checkpoint_path (Optional[Path]): Custom checkpoint path
        workers (int): Number of worker threads
        extract_features (bool): Whether to extract features
        dry_run (bool): Validate setup without processing
        log_level (str): Logging level
        
    Returns:
        bool: True if preparation successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Prepare arguments for embedding extraction script
    script_path = Path(__file__).parent / "music_anomalizer" / "scripts" / "embedding_extraction_wav.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset", str(audio_dir),
        "--output", str(output_dir),
        "--device", device,
        "--model", model_variant,
        "--workers", str(workers),
        "--log-level", log_level
    ]
    
    # Add optional arguments
    if checkpoint_path:
        cmd.extend(["--checkpoint", str(checkpoint_path)])
    
    if model_name:
        cmd.extend(["--output-name", model_name])
    
    if not extract_features:
        cmd.append("--no-features")
    
    if dry_run:
        cmd.append("--dry-run")
    
    logger.info(" Starting audio data preparation...")
    logger.info(f"   Audio directory: {audio_dir}")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Model variant: {model_variant}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Workers: {workers}")
    if model_name:
        logger.info(f"   Model name: {model_name}")
    
    try:
        # Run embedding extraction script
        logger.debug(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        logger.info(" Audio data preparation completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f" Audio data preparation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f" Audio data preparation failed: {e}")
        return False


def main():
    """Main function with argument parsing and preparation orchestration."""
    parser = argparse.ArgumentParser(
        description="Prepare audio data for DeepSVDD training by extracting embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py --audio-dir data/bass_loops
  python prepare_data.py --audio-dir data/guitar --output-dir embeddings --model-name guitar_model
  python prepare_data.py --audio-dir data/drums --device cuda --workers 16
  python prepare_data.py --audio-dir data/bass --dry-run --log-level DEBUG
  python prepare_data.py --audio-dir data/synth --model-variant HTSAT-base --extract-features
  python prepare_data.py --audio-dir data/vocals --checkpoint path/to/custom_clap.pt

Supported Audio Formats:
  - WAV (.wav)
  - MP3 (.mp3)  
  - FLAC (.flac)
  - M4A (.m4a)

Output:
  The script generates two files:
  - {model_name}_embeddings.pkl     # Audio embeddings for training
  - {model_name}_embeddings_index.pkl # File index mapping

Next Steps:
  After running this script, use the generated embeddings with:
  - python train.py --dataset {output_dir}/{model_name}_embeddings.pkl --network AE
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing audio files to process"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory to save embeddings (default: data/embeddings)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom name for output files (default: audio directory name)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-variant",
        type=str,
        default="HTSAT-base",
        help="CLAP model variant (default: HTSAT-base)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to CLAP model checkpoint (default: checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt)"
    )
    
    # Processing configuration
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for processing (default: auto)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)"
    )
    parser.add_argument(
        "--no-extract-features",
        action="store_true",
        help="Skip feature extraction preprocessing"
    )
    
    # Utility options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without processing audio files"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(" Audio Data Preparation for DeepSVDD Training")
    logger.info("=" * 50)
    
    # Validate dependencies
    logger.info(" Validating dependencies...")
    if not validate_dependencies():
        logger.error(" Dependency validation failed")
        sys.exit(1)
    
    # Validate audio directory
    logger.info(" Validating audio directory...")
    if not validate_audio_directory(args.audio_dir):
        logger.error(" Audio directory validation failed")
        sys.exit(1)
    
    # Validate checkpoint
    logger.info(" Validating CLAP checkpoint...")
    if not validate_checkpoint_availability(args.checkpoint):
        logger.error(" Checkpoint validation failed")
        sys.exit(1)
    
    # Generate model name if not provided
    model_name = args.model_name or generate_output_name(args.audio_dir)
    logger.info(f" Using model name: {model_name}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f" Output directory: {args.output_dir}")
    
    if args.dry_run:
        logger.info(" Dry-run validation completed successfully!")
        logger.info("   All prerequisites validated. You can now run without --dry-run to process audio files.")
        sys.exit(0)
    
    # Prepare audio data
    logger.info("=" * 50)
    success = prepare_audio_data(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        model_name=model_name,
        device=args.device,
        model_variant=args.model_variant,
        checkpoint_path=args.checkpoint,
        workers=args.workers,
        extract_features=not args.no_extract_features,
        dry_run=False,
        log_level=args.log_level
    )
    
    if success:
        logger.info("=" * 50)
        logger.info(" Audio data preparation completed successfully!")
        logger.info("")
        logger.info(" Generated files:")
        logger.info(f"   - {args.output_dir}/{model_name}_embeddings.pkl")
        logger.info(f"   - {args.output_dir}/{model_name}_embeddings_index.pkl")
        logger.info("")
        logger.info(" Next steps:")
        logger.info(f"   1. Train a model: python train.py --dataset {args.output_dir}/{model_name}_embeddings.pkl --network AE")
        logger.info("   2. Or use train_models.py for multi-model training")
        logger.info("")
        logger.info(" Available network types: AE, AEwRES, Baseline, DeepAE, CompactAE")
        sys.exit(0)
    else:
        logger.error(" Audio data preparation failed")
        logger.info(" Try running with --log-level DEBUG for more detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()