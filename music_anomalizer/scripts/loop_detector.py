"""
Loop Detection Script

Detects if a WAV file is a loop using anomaly detection models.

Usage:
    python loop_detector.py audio.wav --model bass --network AE --clap-checkpoint /path/to/clap.pt
    python loop_detector.py audio.wav --model guitar --network AEwRES --clap-checkpoint /path/to/clap.pt --threshold 0.5
    python loop_detector.py audio.wav --model bass --network DeepAE --clap-checkpoint /path/to/clap.pt --device cuda
    python loop_detector.py audio.wav --model guitar --dry-run --clap-checkpoint /path/to/clap.pt
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from music_anomalizer.config import load_experiment_config
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.preprocessing.wav2embed import Wav2Embedding
from music_anomalizer.utils import setup_logging


def get_checkpoint_paths(network_type, model_type, clap_checkpoint):
    """Get checkpoint paths for the model."""
    script_dir = Path(__file__).parent.parent.parent
    
    # Use network-specific checkpoint directories
    base_path = (script_dir / 'checkpoints' / 'loop_benchmark' /
                 'EXP2_DEEPER' / network_type)

    # Find available checkpoints by pattern
    ae_pattern = f'{network_type}-HTSAT_base_musicradar_{model_type}-AE-*.ckpt'
    svdd_pattern = f'{network_type}-HTSAT_base_musicradar_{model_type}-DSVDD-*.ckpt'
    
    ae_files = list(base_path.glob(ae_pattern))
    svdd_files = list(base_path.glob(svdd_pattern))
    
    if not ae_files:
        raise FileNotFoundError(
            f"No AE checkpoint found matching pattern: {base_path / ae_pattern}")
    if not svdd_files:
        raise FileNotFoundError(
            f"No SVDD checkpoint found matching pattern: {base_path / svdd_pattern}")
    
    # Use the first matching file for each type
    ae_checkpoint = str(ae_files[0])
    svdd_checkpoint = str(svdd_files[0])

    paths = {
        'clap': clap_checkpoint,
        model_type: {
            'ae': ae_checkpoint,
            'svdd': svdd_checkpoint,
        }
    }

    # Validate files exist
    if not Path(clap_checkpoint).exists():
        raise FileNotFoundError(
            f"CLAP checkpoint not found: {clap_checkpoint}")

    logging.info(f"Using {network_type} checkpoints for {model_type} model")
    logging.info(f"AE checkpoint: {ae_checkpoint}")
    logging.info(f"SVDD checkpoint: {svdd_checkpoint}")
    return paths


def extract_embedding(wav_path, clap_checkpoint, device):
    """Extract embedding from audio file."""
    try:
        logging.info(f"Extracting embedding from: {wav_path}")
        extractor = Wav2Embedding(
            model_ckpt_path=clap_checkpoint,
            audio_model='HTSAT-base',
            device=device
        )

        embedding = extractor.extract_embedding(wav_path)
        if embedding is None:
            raise RuntimeError("Failed to extract embedding")

        return embedding.squeeze()
    except Exception as e:
        logging.error(f"Embedding extraction failed: {e}")
        raise


def load_detector(model_config, svdd_config, ae_checkpoint,
                  svdd_checkpoint, device):
    """Load and initialize the anomaly detector."""
    try:
        logging.info("Loading anomaly detector...")
        detector = AnomalyDetector(
            configs=[model_config, svdd_config],
            checkpoint_paths=[ae_checkpoint, svdd_checkpoint],
            device=device
        )
        detector.load_models()
        logging.info("Models loaded successfully")
        return detector
    except Exception as e:
        logging.error(f"Failed to load detector: {e}")
        raise


def detect_loop(detector, embedding, threshold):
    """Perform loop detection."""
    try:
        results = detector.get_detected_loops([embedding], threshold)
        return results[0]
    except Exception as e:
        logging.error(f"Detection failed: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Detect if a WAV file is a loop using anomaly detection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Network Types:
  AE        - Standard AutoEncoder with regularization (recommended)
  AEwRES    - AutoEncoder with residual connections (for complex patterns)
  Baseline  - AutoEncoder without regularization (for comparison)
  DeepAE    - Deep 5-layer AutoEncoder (for complex datasets)
  CompactAE - Compact 2-layer AutoEncoder (for smaller datasets)

Examples:
  python loop_detector.py audio.wav --model bass --network AE \\
      --clap-checkpoint /path/to/clap.pt
  python loop_detector.py audio.wav --model guitar --network AEwRES \\
      --clap-checkpoint /path/to/clap.pt --threshold 0.5
  python loop_detector.py audio.wav --model bass --network DeepAE \\
      --clap-checkpoint /path/to/clap.pt --device cuda
  python loop_detector.py audio.wav --model guitar --dry-run \\
      --clap-checkpoint /path/to/clap.pt
        """
    )

    # Required arguments
    parser.add_argument('wav_path', type=str, help='Path to input WAV file')
    parser.add_argument('--model', type=str, choices=['bass', 'guitar'],
                        required=True,
                        help='Model type to use (bass or guitar)')
    parser.add_argument('--network', type=str,
                        choices=['AE', 'AEwRES', 'Baseline', 'DeepAE',
                                 'CompactAE'],
                        default='AEwRES',
                        help='Network architecture type (default: AEwRES)')

    # Optional configuration
    parser.add_argument('--config', type=str, default='single_model',
                        help='Configuration name (default: single_model)')
    parser.add_argument('--clap-checkpoint', type=str, required=True,
                        help='Path to CLAP checkpoint file')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom threshold for loop detection '
                        '(default: 0.5 for bass, 0.6 for guitar)')
    parser.add_argument('--device', type=str,
                        choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Device to use (default: auto)')

    # Utility options
    parser.add_argument('--log-level', type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Log level (default: INFO)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate setup without running detection')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    try:
        # Validate input file
        if not Path(args.wav_path).exists():
            raise FileNotFoundError(
                f"Audio file not found: {args.wav_path}")

        # Load config
        logging.info(f"Loading configuration: {args.config}")
        config = load_experiment_config(args.config)

        # Validate network type
        if args.network not in config.networks:
            available_networks = list(config.networks.keys())
            raise ValueError(
                f"Unsupported network type: {args.network}. "
                f"Available: {available_networks}")

        # Get model configuration
        model_config = config.networks[args.network]
        svdd_config = config.deepSVDD

        # Set default thresholds based on model type if not provided
        default_thresholds = {'bass': 0.5, 'guitar': 0.6}
        threshold = args.threshold
        if threshold is None:
            threshold = default_thresholds[args.model]
            logging.info(
                f"Using default threshold for {args.model}: {threshold}")
        else:
            logging.info(f"Using custom threshold: {threshold}")

        # Set num_features if needed
        if (not hasattr(model_config, 'num_features') or
                model_config.num_features is None):
            model_config.num_features = 1024

        # Initialize device
        if args.device == 'auto':
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        logging.info(f"Using device: {device}")

        # Get checkpoint paths
        paths = get_checkpoint_paths(args.network, args.model, 
                                   args.clap_checkpoint)

        if args.dry_run:
            print("Dry-run validation successful:")
            print(f"  Audio file: {args.wav_path} (exists)")
            print(f"  Config: {args.config} (loaded)")
            print(f"  Network: {args.network}")
            print(f"  Model: {args.model}")
            print(f"  Threshold: {threshold}")
            print(f"  Device: {device}")
            return 0

        # Extract embedding
        embedding = extract_embedding(args.wav_path, paths['clap'], device)

        # Load detector
        detector = load_detector(
            model_config, svdd_config,
            paths[args.model]['ae'], paths[args.model]['svdd'], device)

        # Run detection
        result = detect_loop(detector, embedding, threshold)

        # Display results
        print("\nResults:")
        print(f"  Input: {args.wav_path}")
        print(f"  Network: {args.network}")
        print(f"  Model: {args.model}")
        print(f"  Threshold: {threshold}")
        print(f"  Distance: {result['distance']:.6f}")
        is_loop = 'LOOP DETECTED' if result['is_loop'] else 'Not a loop'
        print(f"  Result: {is_loop}")

        return 0

    except Exception as e:
        logging.error(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
