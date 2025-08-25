"""
Simple single-model training script for DeepSVDD anomaly detection.

This script provides a user-friendly interface for training individual DeepSVDD models
using the existing configuration system. It maintains consistency with the project's
architecture while offering simplicity for practical applications.

Key features:
- Single model training with intuitive CLI interface
- Uses existing configuration system with config selection
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
    python train.py --dataset data.pkl --network AE --config exp1
    python train.py --dataset data.pkl --network AEwRES --config exp2_deeper
    python train.py --help  # Show all available options
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import torch
import shutil

# Add the project root to the Python path to enable module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from music_anomalizer.utils import load_pickle, PickleHandler, create_folder, setup_logging, initialize_device, validate_dataset
from music_anomalizer.config import load_experiment_config
from music_anomalizer.models.deepSVDD import DeepSVDDTrainer


def train_single_model(
    dataset_path: str,
    network_type: str,
    config_name: str = "single_model",
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
        config_name (str): Name of the configuration file to use (default: "single_model")
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
    logger.info(f"⚙️  Configuration: {config_name}")
    logger.info(f"📁 Dataset: {dataset_path}")
    
    # Validate and load dataset
    is_valid, error_msg, data = validate_dataset(dataset_path, load_data=True)
    if not is_valid:
        logger.error(f"❌ Dataset validation failed: {error_msg}")
        return None
    
    # Load configuration
    try:
        config = load_experiment_config(config_name)
        logger.info(f"✅ Configuration loaded: {config_name}.yaml")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration '{config_name}': {e}")
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
        trained_encoder = trainer.get_trained_encoder_model()
        trained_dsvdd = trainer.get_trained_dsvdd_model()
        center = trainer.get_center()
        
        # Copy checkpoint files to output directory and save trained models
        saved_model_paths = {}
        
        # Copy AE checkpoint
        if f"{model_name}-AE" in best_model_paths:
            ae_source = best_model_paths[f"{model_name}-AE"]
            ae_dest = output_path / f"{model_name}_AE.ckpt"
            shutil.copy2(ae_source, ae_dest)
            saved_model_paths["ae_checkpoint"] = str(ae_dest)
            logger.info(f"💾 AE checkpoint saved: {ae_dest}")
        
        # Copy DSVDD checkpoint
        if f"{model_name}-DSVDD" in best_model_paths:
            dsvdd_source = best_model_paths[f"{model_name}-DSVDD"]
            dsvdd_dest = output_path / f"{model_name}_DSVDD.ckpt"
            shutil.copy2(dsvdd_source, dsvdd_dest)
            saved_model_paths["dsvdd_checkpoint"] = str(dsvdd_dest)
            logger.info(f"💾 DSVDD checkpoint saved: {dsvdd_dest}")
        
        # Save trained encoder model as PyTorch state dict
        if trained_encoder is not None:
            encoder_path = output_path / f"{model_name}_encoder.pth"
            torch.save(trained_encoder.state_dict(), encoder_path)
            saved_model_paths["encoder_model"] = str(encoder_path)
            logger.info(f"💾 Encoder model saved: {encoder_path}")
        
        # Save trained DSVDD model as PyTorch state dict
        if trained_dsvdd is not None:
            dsvdd_path = output_path / f"{model_name}_dsvdd.pth"
            torch.save(trained_dsvdd.state_dict(), dsvdd_path)
            saved_model_paths["dsvdd_model"] = str(dsvdd_path)
            logger.info(f"💾 DSVDD model saved: {dsvdd_path}")
        
        # Save center vector (consistent with train_models.py naming)
        center_path = output_path / f"{model_name}_z_vector.pkl"
        ph = PickleHandler(str(center_path))
        ph.dump_data(center)
        saved_model_paths["center_vector"] = str(center_path)
        logger.info(f"💾 Center vector saved: {center_path}")
        
        # Prepare results
        results = {
            'model_name': model_name,
            'network_type': network_type,
            'best_model_paths': best_model_paths,
            'saved_model_paths': saved_model_paths,
            'trained_encoder': trained_encoder,
            'trained_dsvdd': trained_dsvdd,
            'center': center,
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
        logger.info(f"📁 Models saved to: {output_path.absolute()}")
        logger.info(f"📄 Saved files:")
        for key, path in saved_model_paths.items():
            filename = Path(path).name
            logger.info(f"   {key}: {filename}")
        
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
    parser.add_argument(
        "--config",
        type=str,
        default="single_model",
        help="Configuration file name (default: single_model)"
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
        is_valid, error_msg, data = validate_dataset(args.dataset, load_data=True)
        if not is_valid:
            logger.error(f"❌ Dataset validation failed: {error_msg}")
            sys.exit(1)
        
        # Load and validate configuration
        try:
            config = load_experiment_config(args.config)
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
        config_name=args.config,
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
        logger.info(f"📁 Models saved in: {results['output_directory']}")
        logger.info(f"📄 Saved files: {list(results['saved_model_paths'].keys())}")
        sys.exit(0)


if __name__ == "__main__":
    main()