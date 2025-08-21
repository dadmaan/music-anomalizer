"""
Simple single-model training script for DeepSVDD anomaly detection.

This script provides a user-friendly interface for training individual DeepSVDD models
using the existing configuration system. It maintains consistency with the project's
architecture while offering simplicity for practical applications.

Key features:
- Single model training with intuitive CLI interface
- Uses existing configuration system (single_model.yaml)
- Support for all network architectures (AE, AEwRES, Baseline, DeepAE, CompactAE)
- Consistent with train_models.py but simplified for single-model use
- Built-in dataset validation and preprocessing
- Comprehensive logging with emoji indicators
- WandB integration with sensible defaults

Usage:
    python train.py --dataset data.pkl --network AE
    python train.py --dataset data.pkl --network AEwRES --model-name my_model
    python train.py --dataset data.pkl --network DeepAE --epochs 500 --batch-size 64
    python train.py --dataset data.pkl --network CompactAE --wandb-project "MyProject"
    python train.py --help  # Show all available options
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch

# Add the project root to the Python path to enable module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from music_anomalizer.utils import load_pickle, PickleHandler, create_folder
from music_anomalizer.config import load_experiment_config
from music_anomalizer.models.deepSVDD import DeepSVDDTrainer


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration with emoji indicators and proper formatting.
    
    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        logging.Logger: Configured logger instance
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def initialize_device(device_preference: str = "auto") -> torch.device:
    """Initialize computation device with fallback support.
    
    Args:
        device_preference (str): Device preference ('auto', 'cpu', 'cuda')
        
    Returns:
        torch.device: Selected device
    """
    logger = logging.getLogger(__name__)
    
    if device_preference == "cpu":
        device = torch.device('cpu')
        logger.info("🖥️  Using CPU (forced)")
    elif device_preference == "cuda":
        if not torch.cuda.is_available():
            logger.warning("⚠️  CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            logger.info(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
    else:  # auto
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        if use_cuda:
            logger.info(f"🚀 Auto-detected CUDA device: {torch.cuda.get_device_name()}")
        else:
            logger.info("🖥️  Auto-detected CPU (CUDA not available)")
    
    return device


def validate_dataset(dataset_path: str) -> Optional[Any]:
    """Validate and load dataset with comprehensive checks.
    
    Args:
        dataset_path (str): Path to dataset pickle file
        
    Returns:
        Optional[Any]: Loaded dataset or None if validation fails
    """
    logger = logging.getLogger(__name__)
    
    # Convert to Path object for better handling
    dataset_file = Path(dataset_path)
    
    # Check if file exists
    if not dataset_file.exists():
        logger.error(f"❌ Dataset file not found: {dataset_path}")
        return None
    
    # Check file extension
    if dataset_file.suffix.lower() != '.pkl':
        logger.warning(f"⚠️  Dataset file doesn't have .pkl extension: {dataset_path}")
    
    # Check file size
    file_size = dataset_file.stat().st_size
    if file_size == 0:
        logger.error(f"❌ Dataset file is empty: {dataset_path}")
        return None
    
    # Load and validate dataset
    try:
        data = load_pickle(str(dataset_file))
        if data is None:
            logger.error(f"❌ Dataset loaded as None: {dataset_path}")
            return None
        
        if len(data) == 0:
            logger.error(f"❌ Dataset is empty: {dataset_path}")
            return None
        
        logger.info(f"✅ Dataset loaded successfully: {len(data)} samples ({file_size / (1024*1024):.1f} MB)")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset from {dataset_path}: {e}")
        return None


def train_single_model(
    dataset_path: str,
    network_type: str,
    model_name: Optional[str] = None,
    output_dir: str = "./models",
    # Training parameter overrides
    batch_size: Optional[int] = None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    # WandB parameters
    wandb_project: Optional[str] = None,
    wandb_log_model: bool = False,
    wandb_disabled: bool = False,
    enable_progress_bar: bool = False,
    # Device
    device: torch.device = None,
    # Logging
    logger: Optional[logging.Logger] = None
) -> Optional[Dict[str, Any]]:
    """Train a single DeepSVDD model using the configuration system.
    
    Args:
        dataset_path (str): Path to the dataset pickle file
        network_type (str): Type of network ('AE', 'AEwRES', 'Baseline', 'DeepAE', 'CompactAE')
        model_name (Optional[str]): Name for the trained model
        output_dir (str): Directory to save model checkpoints
        batch_size (Optional[int]): Override batch size from config
        max_epochs (Optional[int]): Override max epochs from config  
        patience (Optional[int]): Override patience from config
        wandb_project (Optional[str]): WandB project name
        wandb_log_model (bool): Whether to log model to WandB
        wandb_disabled (bool): Whether to disable WandB completely
        enable_progress_bar (bool): Whether to show training progress
        device (torch.device): Computing device
        logger (Optional[logging.Logger]): Logger instance
        
    Returns:
        Optional[Dict[str, Any]]: Training results or None if failed
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Generate model name if not provided
    if model_name is None:
        dataset_name = Path(dataset_path).stem
        model_name = f"{network_type}_{dataset_name}"
    
    logger.info(f"🚀 Starting training for model: {model_name}")
    logger.info(f"📊 Network type: {network_type}")
    logger.info(f"📁 Dataset: {dataset_path}")
    
    # Validate and load dataset
    data = validate_dataset(dataset_path)
    if data is None:
        return None
    
    # Load configuration
    try:
        config = load_experiment_config("single_model")
        logger.info("✅ Configuration loaded: single_model.yaml")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return None
    
    # Validate network type
    if network_type not in config.networks:
        available_networks = list(config.networks.keys())
        logger.error(f"❌ Unsupported network type: {network_type}")
        logger.error(f"   Available networks: {available_networks}")
        return None
    
    # Get network configuration
    ae_config = config.networks[network_type]
    logger.info(f"✅ Network configuration: {network_type}")
    logger.info(f"   Architecture: {ae_config.class_name}")
    logger.info(f"   Hidden dims: {ae_config.hidden_dims}")
    logger.info(f"   Dropout: {ae_config.dropout_rate}")
    logger.info(f"   Batch norm: {ae_config.use_batch_norm}")
    
    # Prepare trainer configuration with overrides
    trainer_config = config.trainer.model_copy()
    
    # Apply CLI overrides
    if batch_size is not None:
        trainer_config.batch_size = batch_size
        logger.info(f"🔧 Batch size override: {batch_size}")
    
    if max_epochs is not None:
        trainer_config.max_epochs = max_epochs
        logger.info(f"🔧 Max epochs override: {max_epochs}")
    
    if patience is not None:
        trainer_config.patience = patience
        logger.info(f"🔧 Patience override: {patience}")
    
    # Handle WandB configuration
    if wandb_disabled:
        trainer_config.wandb_project_name = None
        trainer_config.wandb_log_model = False
        logger.info("📊 WandB: Disabled")
    else:
        if wandb_project:
            trainer_config.wandb_project_name = wandb_project
            logger.info(f"📊 WandB project override: {wandb_project}")
        else:
            trainer_config.wandb_project_name = f"DeepSVDD-{model_name}"
            logger.info(f"📊 WandB project: DeepSVDD-{model_name}")
        
        trainer_config.wandb_log_model = wandb_log_model
        logger.info(f"📁 WandB model logging: {wandb_log_model}")
    
    # Progress bar setting
    trainer_config.enable_progress_bar = enable_progress_bar
    logger.info(f"📊 Progress bar: {enable_progress_bar}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path.absolute()}")
    
    try:
        # Initialize trainer with the same pattern as train_models.py
        trainer = DeepSVDDTrainer(
            [ae_config.model_dump(), config.deepSVDD.model_dump()],
            data,
            device,
            **trainer_config.model_dump()
        )
        logger.info("✅ DeepSVDD trainer initialized successfully")
        
        # Display training configuration
        logger.info("⚙️  Training configuration:")
        logger.info(f"   Batch size: {trainer_config.batch_size}")
        logger.info(f"   Max epochs: {trainer_config.max_epochs}")
        logger.info(f"   Patience: {trainer_config.patience}")
        logger.info(f"   AE learning rate: {ae_config.learning_rate}")
        logger.info(f"   SVDD learning rate: {config.deepSVDD.learning_rate}")
        
        # Run training
        logger.info("🔄 Starting training process...")
        trainer.run(model_name)
        logger.info("✅ Training completed successfully")
        
        # Get training results
        best_model_paths = trainer.get_best_model_path()
        trained_model = trainer.get_trained_dsvdd_model()
        center = trainer.get_center()
        
        # Save center vector
        center_path = output_path / f"{model_name}_center.pkl"
        ph = PickleHandler(str(center_path))
        ph.dump_data(center)
        logger.info(f"💾 Center vector saved: {center_path}")
        
        # Prepare results
        results = {
            'model_name': model_name,
            'network_type': network_type,
            'best_model_paths': best_model_paths,
            'trained_model': trained_model,
            'center': center,
            'center_path': str(center_path),
            'output_directory': str(output_path),
            'dataset_path': dataset_path,
            'dataset_size': len(data),
            'configuration': {
                'network_config': ae_config.model_dump(),
                'svdd_config': config.deepSVDD.model_dump(),
                'trainer_config': trainer_config.model_dump()
            }
        }
        
        logger.info("🎉 Training process completed successfully!")
        logger.info(f"📄 Model paths:")
        for key, path in best_model_paths.items():
            logger.info(f"   {key}: {path}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None


def main():
    """Main function with argument parsing and training orchestration."""
    parser = argparse.ArgumentParser(
        description="Train a single DeepSVDD model for anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Network Types:
  AE        - Standard AutoEncoder with regularization (recommended)
  AEwRES    - AutoEncoder with residual connections (for complex patterns)
  Baseline  - AutoEncoder without regularization (for comparison)
  DeepAE    - Deep 5-layer AutoEncoder (for complex datasets)
  CompactAE - Compact 2-layer AutoEncoder (for smaller datasets)

Examples:
  python train.py --dataset data.pkl --network AE
  python train.py --dataset bass_data.pkl --network AEwRES --model-name my_bass_model
  python train.py --dataset guitar_data.pkl --network DeepAE --epochs 500 --batch-size 64
  python train.py --dataset data.pkl --network CompactAE --wandb-project "MyProject"
  python train.py --dataset data.pkl --network AE --no-wandb --progress-bar
  python train.py --dataset data.pkl --network Baseline --output-dir ./my_models
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset pickle file"
    )
    parser.add_argument(
        "--network",
        type=str,
        choices=["AE", "AEwRES", "Baseline", "DeepAE", "CompactAE"],
        required=True,
        help="Network architecture type"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom name for the model (default: {network}_{dataset_name})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Output directory for model checkpoints (default: ./models)"
    )
    
    # Training parameter overrides
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from configuration"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override maximum training epochs from configuration"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience from configuration"
    )
    
    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computing device (default: auto)"
    )
    
    # WandB configuration
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name (default: DeepSVDD-{model_name})"
    )
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="Log model artifacts to WandB"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging completely"
    )
    
    # UI options
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show training progress bars"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Dry run option
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and dataset without training"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("🎯 DeepSVDD Single Model Training")
    logger.info("=" * 50)
    
    # Initialize device
    device = initialize_device(args.device)
    
    # Generate model name if not provided
    model_name = args.model_name
    if model_name is None:
        dataset_name = Path(args.dataset).stem
        model_name = f"{args.network}_{dataset_name}"
    
    # Display configuration
    logger.info(f"🏷️  Model name: {model_name}")
    logger.info(f"🏗️  Network type: {args.network}")
    logger.info(f"📊 Dataset: {args.dataset}")
    logger.info(f"💾 Output directory: {args.output_dir}")
    logger.info(f"🔧 Device: {device}")
    
    if args.batch_size:
        logger.info(f"⚙️  Batch size override: {args.batch_size}")
    if args.epochs:
        logger.info(f"🔄 Epochs override: {args.epochs}")
    if args.patience:
        logger.info(f"⏳ Patience override: {args.patience}")
    
    if args.no_wandb:
        logger.info("📊 WandB: Disabled")
    else:
        project = args.wandb_project or f"DeepSVDD-{model_name}"
        logger.info(f"📊 WandB project: {project}")
        logger.info(f"📁 WandB model logging: {args.wandb_log_model}")
    
    logger.info(f"📊 Progress bar: {args.progress_bar}")
    logger.info("=" * 50)
    
    # Handle dry run
    if args.dry_run:
        logger.info("🧪 Dry run mode - validating configuration and dataset")
        
        # Validate dataset
        data = validate_dataset(args.dataset)
        if data is None:
            logger.error("❌ Dataset validation failed")
            sys.exit(1)
        
        # Load and validate configuration
        try:
            config = load_experiment_config("single_model")
            if args.network not in config.networks:
                available_networks = list(config.networks.keys())
                logger.error(f"❌ Unsupported network type: {args.network}")
                logger.error(f"   Available networks: {available_networks}")
                sys.exit(1)
            
            ae_config = config.networks[args.network]
            logger.info("✅ Configuration validation successful")
            logger.info(f"   Network: {ae_config.class_name}")
            logger.info(f"   Hidden dims: {ae_config.hidden_dims}")
            logger.info(f"   Parameters: dropout={ae_config.dropout_rate}, batch_norm={ae_config.use_batch_norm}")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
        
        logger.info("🎉 Dry run completed successfully!")
        sys.exit(0)
    
    # Run training
    results = train_single_model(
        dataset_path=args.dataset,
        network_type=args.network,
        model_name=model_name,
        output_dir=args.output_dir,
        # Training parameter overrides
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        # WandB parameters
        wandb_project=args.wandb_project,
        wandb_log_model=args.wandb_log_model,
        wandb_disabled=args.no_wandb,
        enable_progress_bar=args.progress_bar,
        # Other
        device=device,
        logger=logger
    )
    
    if results is None:
        logger.error("❌ Training failed!")
        sys.exit(1)
    else:
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📁 Checkpoints saved in: {results['output_directory']}")
        logger.info(f"📄 Model paths: {list(results['best_model_paths'].keys())}")
        sys.exit(0)


if __name__ == "__main__":
    main()