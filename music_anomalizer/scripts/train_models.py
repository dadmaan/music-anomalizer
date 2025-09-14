"""
Script for training DeepSVDD models on different network configurations and datasets.

This script loads configurations from a YAML file, trains DeepSVDD models for each
network configuration and dataset combination, and organizes models using the
checkpoint registry system.

The script performs the following steps:
1. Load configuration from YAML file with validation
2. For each network configuration and dataset combination:
   - Load and validate the dataset
   - Initialize and run DeepSVDD trainer with proper error handling
   - Save the best model paths and trained models
3. Organize model files using structured directory layout
4. Register checkpoint paths for future use

Usage:
    python train_models.py                                    # Train with default config (exp1)
    python train_models.py --config exp2_deeper             # Train specific experiment
    python train_models.py --config exp1 --dry-run          # Validate configuration only
    python train_models.py --config exp1 --device cpu       # Force CPU training
    python train_models.py --config exp2_deeper --device cuda # GPU training
    python train_models.py --log-level DEBUG                # Verbose logging
    python train_models.py --wandb-project "MY_PROJECT"     # Override wandb project name
    python train_models.py --wandb-log-model                # Enable wandb model logging
    python train_models.py --wandb-disabled                 # Disable wandb logging
    python train_models.py --enable-progress-bar            # Show training progress bars
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
import torch

# Add the project root to the Python path to enable module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_anomalizer.utils import (
    write_to_json, create_folder, move_and_rename_files, PickleHandler,
    setup_logging, initialize_device, validate_dataset, load_pickle
)
from music_anomalizer.config import load_experiment_config, get_checkpoint_registry
from music_anomalizer.models.deepSVDD import DeepSVDDTrainer


def train_model_combination(
    network_name: str, 
    dataset_name: str, 
    ae_config: Any, 
    deepsvdd_config: Any, 
    trainer_config: Any,
    data: Any, 
    device: torch.device,
    wandb_project: Optional[str] = None,
    wandb_log_model: bool = False,
    wandb_disabled: bool = False,
    enable_progress_bar: bool = False
) -> Optional[Tuple[Dict[str, str], Any, Any]]:
    """Train a single model combination with comprehensive error handling.
    
    Args:
        network_name (str): Name of the network configuration
        dataset_name (str): Name of the dataset
        ae_config: AutoEncoder configuration
        deepsvdd_config: DeepSVDD configuration
        trainer_config: Trainer configuration
        data: Dataset to train on
        device: PyTorch device
        wandb_project (Optional[str]): Override wandb project name
        wandb_log_model (bool): Enable wandb model artifact logging
        wandb_disabled (bool): Disable wandb logging completely
        enable_progress_bar (bool): Enable training progress bar display
        
    Returns:
        Optional[Tuple]: (best_model_paths, trained_model, center) or None if failed
    """
    logger = logging.getLogger(__name__)
    model_id = f"{network_name}-{dataset_name}"
    
    logger.info(f"  Training model: {model_id}")
    
    try:
        # Prepare trainer configuration with wandb overrides
        trainer_params = trainer_config.model_dump()
        
        # Handle wandb configuration
        if wandb_disabled:
            # Disable wandb completely by setting project to None
            trainer_params['wandb_project_name'] = None
            trainer_params['wandb_log_model'] = False
        else:
            # Override wandb project name if provided
            if wandb_project:
                trainer_params['wandb_project_name'] = wandb_project
            
            # Override wandb model logging if specified
            trainer_params['wandb_log_model'] = wandb_log_model
        
        # Override progress bar setting
        trainer_params['enable_progress_bar'] = enable_progress_bar
        
        # Initialize trainer with error handling
        trainer = DeepSVDDTrainer(
            [ae_config.model_dump(), deepsvdd_config.model_dump()],
            data,
            device,
            **trainer_params
        )
        logger.info(f" Trainer initialized for {model_id}")
        
    except Exception as e:
        logger.error(f" Failed to initialize trainer for {model_id}: {e}")
        return None
    
    try:
        # Run training with progress logging
        logger.info(f" Starting training for {model_id}")
        trainer.run(model_id)
        logger.info(f" Training completed successfully for {model_id}")
        
        # Extract results
        best_model_paths = trainer.get_best_model_path()
        trained_model = trainer.get_trained_dsvdd_model()
        center = trainer.get_center()
        
        return best_model_paths, trained_model, center
        
    except Exception as e:
        logger.error(f" Training failed for {model_id}: {e}")
        return None


def organize_model_files(
    network_name: str,
    dataset_name: str, 
    best_model_paths: Dict[str, str],
    center: Any,
    target_dir: str
) -> Optional[Dict[str, str]]:
    """Organize trained model files into structured directory layout.
    
    Args:
        network_name (str): Network configuration name
        dataset_name (str): Dataset name
        best_model_paths (Dict[str, str]): Dictionary of model paths
        center: Center data to save
        target_dir (str): Target directory for organized files
        
    Returns:
        Optional[Dict[str, str]]: New file paths or None if failed
    """
    logger = logging.getLogger(__name__)
    base_name = f"{network_name}-{dataset_name}"
    
    try:
        # Extract model paths
        ae_ckpt_path = best_model_paths[base_name + "-AE"]
        svdd_ckpt_path = best_model_paths[base_name + "-DSVDD"]
        
        # Create destination directory
        destination_dir = os.path.join(target_dir, network_name)
        create_folder(destination_dir)
        logger.debug(f"Created directory: {destination_dir}")
        
        # Check if files are already in wandb directory structure
        if ae_ckpt_path.startswith("./wandb/checkpoints/"):
            # Files are already in wandb, just save center vector
            logger.info(f"✓ Model files already in wandb directory for {base_name}")
            ae_new_path = ae_ckpt_path
            svdd_new_path = svdd_ckpt_path
        else:
            # Move and rename files (legacy behavior for older checkpoints)
            ae_new_path, svdd_new_path = move_and_rename_files(
                [ae_ckpt_path, svdd_ckpt_path], 
                destination_dir
            )
            logger.info(f"✓ Model files organized for {base_name}")
        
        # Save center data as pickle file in the same directory as the model
        center_file_path = os.path.splitext(ae_new_path)[0] + '_z_vector.pkl'
        ph = PickleHandler(center_file_path)
        ph.dump_data(center)
        logger.debug(f"Saved center vector: {center_file_path}")
        
        return {
            base_name + "-AE": ae_new_path,
            base_name + "-DSVDD": svdd_new_path
        }
        
    except Exception as e:
        logger.error(f"Failed to organize model files for {base_name}: {e}")
        return None


def save_model_registry(
    organized_paths: Dict[str, Dict[str, str]], 
    config_name: str
) -> None:
    """Save model paths to JSON file for legacy compatibility.
    
    Args:
        organized_paths (Dict): Dictionary of organized model paths
        config_name (str): Configuration name for file naming
    """
    logger = logging.getLogger(__name__)
    
    try:
        output_path = f'./wandb/checkpoints/{config_name}_best_models_path.json'
        write_to_json(organized_paths, output_path)
        logger.info(f" Model registry saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save model registry: {e}")


def update_checkpoint_registry(config_name: str) -> None:
    """Update the checkpoint registry with discovered models.
    
    Args:
        config_name (str): Configuration name to update registry for
    """
    logger = logging.getLogger(__name__)
    
    try:
        checkpoint_registry = get_checkpoint_registry()
        discovered_checkpoints = checkpoint_registry.get_experiment_checkpoints(config_name)
        logger.info(f" Checkpoint registry updated with {len(discovered_checkpoints)} model combinations")
    except Exception as e:
        logger.warning(f"Could not update checkpoint registry: {e}")


def display_training_summary(
    successful_models: int, 
    total_models: int, 
    failed_models: list
) -> None:
    """Display comprehensive training summary.
    
    Args:
        successful_models (int): Number of successfully trained models
        total_models (int): Total number of model combinations attempted
        failed_models (list): List of failed model identifiers
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info(" TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f" Successful models: {successful_models}/{total_models}")
    logger.info(f" Failed models: {len(failed_models)}")
    
    if failed_models:
        logger.info("Failed model combinations:")
        for model_id in failed_models:
            logger.info(f"  - {model_id}")
    
    success_rate = (successful_models / total_models) * 100 if total_models > 0 else 0
    logger.info(f" Success rate: {success_rate:.1f}%")
    logger.info("=" * 60)


def main(
    config_name: str = "exp1", 
    device_override: str = "auto", 
    log_level: str = "INFO",
    wandb_project: Optional[str] = None,
    wandb_log_model: bool = False,
    wandb_disabled: bool = False,
    enable_progress_bar: bool = False
) -> bool:
    """Main function to train DeepSVDD models with comprehensive error handling and logging.
    
    Args:
        config_name (str): Name of the experiment configuration to use
        device_override (str): Device override ('auto', 'cpu', 'cuda')
        log_level (str): Logging level for output verbosity
        wandb_project (Optional[str]): Override wandb project name
        wandb_log_model (bool): Enable wandb model artifact logging
        wandb_disabled (bool): Disable wandb logging completely
        enable_progress_bar (bool): Enable training progress bar display
        
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    # Setup logging
    logger = setup_logging(log_level)
    logger.info(" Starting DeepSVDD model training")
    logger.info(f"Configuration: {config_name}")
    
    # Display wandb configuration
    if wandb_disabled:
        logger.info("WandB: Disabled")
    else:
        logger.info(f"WandB: Enabled")
        if wandb_project:
            logger.info(f"  Project override: {wandb_project}")
        logger.info(f"  Model logging: {'Enabled' if wandb_log_model else 'Disabled'}")
    logger.info(f"Progress bar: {'Enabled' if enable_progress_bar else 'Disabled'}")
    
    # Initialize device
    device = initialize_device(device_override)
    
    # Load and validate configurations
    try:
        configs = load_experiment_config(config_name)
        logger.info(f" Configuration loaded: {configs.config_name}")
        logger.info(f"  Networks: {list(configs.networks.keys())}")
        logger.info(f"  Datasets: {list(configs.dataset_paths.keys())}")
    except Exception as e:
        logger.error(f" Error loading configuration '{config_name}': {e}")
        return False
    
    # Initialize tracking variables
    trained_models = {}
    best_models_paths = {}
    centers = {}
    failed_models = []
    successful_models = 0
    
    # Calculate total combinations
    total_combinations = len(configs.networks) * len(configs.dataset_paths)
    logger.info(f" Training {total_combinations} model combinations")
    
    # Train models for each network configuration and dataset combination
    combination_count = 0
    for network_name, ae_config in configs.networks.items():
        for dataset_name, dataset_path in configs.dataset_paths.items():
            combination_count += 1
            model_id = f"{network_name}-{dataset_name}"
            
            logger.info(f" Processing combination {combination_count}/{total_combinations}: {model_id}")
            
            # Validate and load dataset
            is_valid, error_msg, data = validate_dataset(dataset_path, load_data=True)
            if not is_valid:
                logger.error(f"Dataset validation failed for {dataset_name}: {error_msg}")
                continue
            if data is None:
                failed_models.append(model_id)
                continue
            
            # Train model combination
            training_result = train_model_combination(
                network_name, dataset_name, ae_config, 
                configs.deepSVDD, configs.trainer, data, device,
                wandb_project, wandb_log_model, wandb_disabled, enable_progress_bar
            )
            
            if training_result is None:
                failed_models.append(model_id)
                continue
            
            # Store successful training results
            best_model_paths, trained_model, center = training_result
            best_models_paths[model_id] = best_model_paths
            trained_models[model_id] = trained_model
            centers[model_id] = center
            successful_models += 1
    
    # Check if any models were trained successfully
    if successful_models == 0:
        logger.error(" No models were trained successfully")
        display_training_summary(successful_models, total_combinations, failed_models)
        return False
    
    # Organize and save trained models
    logger.info(" Organizing model files...")
    subfolder = configs.config_name.upper()
    target_dir = f"./wandb/checkpoints/{subfolder}"
    
    organized_paths = {}
    for network_name, ae_config in configs.networks.items():
        for dataset_name, dataset_path in configs.dataset_paths.items():
            model_id = f"{network_name}-{dataset_name}"
            
            # Skip failed models
            if model_id not in best_models_paths:
                continue
            
            # Organize model files
            new_paths = organize_model_files(
                network_name, dataset_name,
                best_models_paths[model_id],
                centers[model_id],
                target_dir
            )
            
            if new_paths:
                organized_paths[model_id] = new_paths
    
    # Save model registry and update checkpoint registry
    if organized_paths:
        save_model_registry(organized_paths, configs.config_name)
        update_checkpoint_registry(configs.config_name)
    
    # Display final summary
    display_training_summary(successful_models, total_combinations, failed_models)
    
    logger.info(" Training process completed")
    return successful_models > 0


def validate_configuration(config_name: str) -> bool:
    """Validate configuration without training (dry-run mode).
    
    Args:
        config_name (str): Configuration name to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    logger = setup_logging("INFO")
    
    try:
        configs = load_experiment_config(config_name)
        logger.info(f"✓ Configuration '{config_name}' is valid")
        logger.info(f"  Networks: {list(configs.networks.keys())}")
        logger.info(f"  Datasets: {list(configs.dataset_paths.keys())}")
        logger.info(f"  Batch size: {configs.trainer.batch_size}")
        logger.info(f"  Max epochs: {configs.trainer.max_epochs}")
        
        # Validate dataset paths
        logger.info(" Validating dataset paths...")
        for dataset_name, dataset_path in configs.dataset_paths.items():
            if os.path.exists(dataset_path):
                file_size = os.path.getsize(dataset_path)
                logger.info(f"   {dataset_name}: {dataset_path} ({file_size / (1024*1024):.1f} MB)")
            else:
                logger.warning(f"    {dataset_name}: {dataset_path} (file not found)")
        
        return True
        
    except Exception as e:
        logger.error(f" Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DeepSVDD models for anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py                                    # Train with default config (exp1)
  python train_models.py --config exp2_deeper             # Train specific experiment
  python train_models.py --config exp1 --dry-run          # Validate configuration only
  python train_models.py --config exp1 --device cpu       # Force CPU training
  python train_models.py --config exp2_deeper --device cuda # GPU training
  python train_models.py --log-level DEBUG                # Verbose logging
  python train_models.py --wandb-project "MY_PROJECT"     # Override wandb project name
  python train_models.py --wandb-log-model                # Enable wandb model logging
  python train_models.py --wandb-disabled                 # Disable wandb logging
  python train_models.py --enable-progress-bar            # Show training progress bars
  python train_models.py --config exp1 --wandb-disabled --enable-progress-bar  # Train without wandb but with progress
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="exp1",
        help="Experiment configuration name (default: exp1)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cpu", "cuda"], 
        default="auto",
        help="Device to use for training (default: auto)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Validate configuration without training"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override wandb project name (default: use config value)"
    )
    parser.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="Enable wandb model artifact logging"
    )
    parser.add_argument(
        "--wandb-disabled",
        action="store_true", 
        help="Disable wandb logging completely"
    )
    parser.add_argument(
        "--enable-progress-bar",
        action="store_true",
        help="Enable training progress bar display"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        success = validate_configuration(args.config)
        sys.exit(0 if success else 1)
    else:
        success = main(
            args.config, 
            args.device, 
            args.log_level,
            args.wandb_project,
            args.wandb_log_model,
            args.wandb_disabled,
            args.enable_progress_bar
        )
        sys.exit(0 if success else 1)