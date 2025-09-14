#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for AutoEncoder Model on Audio Embeddings

This script performs systematic hyperparameter tuning for AutoEncoder models using 
Weights & Biases (wandb) sweeps. It explores different configurations to identify 
optimal parameters that minimize validation loss for anomaly detection tasks.

The script performs the following operations:
1. Loads and validates audio embedding datasets with comprehensive checks
2. Configures hyperparameter sweep with flexible parameter ranges
3. Orchestrates training of multiple model configurations using PyTorch Lightning
4. Logs performance metrics and model artifacts with wandb integration
5. Analyzes results with statistical summaries and visualizations
6. Provides comprehensive error handling and progress tracking

Usage:
    python hp_tuning_loop_detection.py                                   # Basic tuning with defaults
    python hp_tuning_loop_detection.py --data data/bass_embeddings.pkl   # Use different dataset
    python hp_tuning_loop_detection.py --device cuda --runs 50           # GPU with more runs
    python hp_tuning_loop_detection.py --method bayes --epochs 100       # Bayesian optimization
    python hp_tuning_loop_detection.py --analyze results.csv             # Analyze existing results
python hp_tuning_loop_detection.py --project MyProject --seed 42     # Custom project and seed
       
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, Tuple, List
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Set deterministic behavior for PyTorch Lightning
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from music_anomalizer.models.networks import AutoEncoder
from music_anomalizer.data.data_loader import DatasetSampler
from music_anomalizer.utils import (
    setup_logging, set_random_seeds, initialize_device, validate_dataset
)

# Conditional wandb import with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Hyperparameter sweeps will be disabled.")


def prepare_data_loaders(
    data: np.ndarray,
    train_split: float = 0.8,
    batch_size: int = 32,
    num_workers: int = 4,
    random_seed: int = 0
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare train and validation data loaders with robust error handling.
    
    Args:
        data (np.ndarray): Input dataset
        train_split (float): Fraction of data to use for training
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes
        random_seed (int): Random seed for data shuffling
        
    Returns:
        Tuple[DataLoader, DataLoader, int]: (train_loader, val_loader, num_features)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Shuffle data with seed for reproducibility
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(data))
        data_shuffled = data[shuffled_indices]
        
        # Split data
        num_data = data_shuffled.shape[0]
        num_train = int(num_data * train_split)
        
        train_data = data_shuffled[:num_train]
        val_data = data_shuffled[num_train:]
        
        logger.info(f" Data split - Train: {train_data.shape}, Validation: {val_data.shape}")
        
        # Create data loaders with error handling for workers
        try:
            train_params = {
                'batch_size': batch_size, 
                'shuffle': True, 
                'pin_memory': True, 
                'num_workers': num_workers
            }
            val_params = {
                'batch_size': batch_size, 
                'shuffle': False, 
                'pin_memory': True, 
                'num_workers': num_workers
            }
            
            train_loader = DataLoader(DatasetSampler(train_data), **train_params)
            val_loader = DataLoader(DatasetSampler(val_data), **val_params)
            
        except Exception as e:
            logger.warning(f"  Error with multiprocessing workers: {e}. Falling back to single-threaded.")
            train_params['num_workers'] = 0
            val_params['num_workers'] = 0
            train_loader = DataLoader(DatasetSampler(train_data), **train_params)
            val_loader = DataLoader(DatasetSampler(val_data), **val_params)
        
        num_features = data.shape[1]
        logger.info(f" Data loaders created successfully - Features: {num_features}")
        
        return train_loader, val_loader, num_features
        
    except Exception as e:
        logger.error(f" Error preparing data loaders: {e}")
        raise


def create_sweep_config(
    method: str = "random",
    metric_name: str = "val_loss",
    metric_goal: str = "minimize",
    learning_rate_range: Tuple[float, float] = (1e-6, 1e-2),
    dropout_values: List[float] = [0.1, 0.2, 0.3],
    hidden_dims_options: List[List[int]] = None,
    activation_functions: List[str] = ["ReLU", "ELU", "LeakyReLU"],
    use_batch_norm_options: List[bool] = [True, False]
) -> Dict[str, Any]:
    """Create comprehensive sweep configuration with flexible parameters.
    
    Args:
        method (str): Sweep method ('random', 'grid', 'bayes')
        metric_name (str): Metric to optimize
        metric_goal (str): Optimization goal ('minimize', 'maximize')
        learning_rate_range (Tuple[float, float]): Learning rate range
        dropout_values (List[float]): Dropout rate options
        hidden_dims_options (List[List[int]]): Hidden dimension configurations
        activation_functions (List[str]): Activation function options
        use_batch_norm_options (List[bool]): Batch normalization options
        
    Returns:
        Dict[str, Any]: Sweep configuration dictionary
    """
    if hidden_dims_options is None:
        hidden_dims_options = [
            [512, 256, 128],
            [256, 128, 64],
            [1024, 512, 256],
            [512, 256, 128, 64]
        ]
    
    sweep_config = {
        'method': method,
        'metric': {
            'name': metric_name,
            'goal': metric_goal   
        },
        'parameters': {
            'train_data_length': {
                'values': [0]  # Will be updated with actual train set length
            },
            'learning_rate': {
                'min': learning_rate_range[0],
                'max': learning_rate_range[1],
                'distribution': 'log_uniform'
            },
            'dropout_rate': {
                'values': dropout_values
            },
            'hidden_dims': {
                'values': hidden_dims_options
            },
            'activation_fn': {
                'values': activation_functions
            },
            'use_batch_norm': {
                'values': use_batch_norm_options
            }
        }
    }
    
    return sweep_config


def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_features: int,
    max_epochs: int = 200,
    project_name: str = "CLAP_DeepSVDD",
    enable_checkpointing: bool = True
) -> Dict[str, Any]:
    """Train AutoEncoder model with given configuration and comprehensive error handling.
    
    Args:
        config (Dict[str, Any]): Model configuration parameters
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_features (int): Number of input features
        max_epochs (int): Maximum training epochs
        project_name (str): WandB project name
        enable_checkpointing (bool): Whether to enable model checkpointing
        
    Returns:
        Dict[str, Any]: Training metrics and results
    """
    logger = logging.getLogger(__name__)
    
    if not WANDB_AVAILABLE:
        logger.error(" WandB not available. Cannot proceed with training.")
        return {}
    
    try:
        with wandb.init(config=config, reinit=True):
            wandb_config = wandb.config
            
            # Get activation function class
            try:
                activation_fn = getattr(nn, wandb_config.activation_fn)()
            except AttributeError:
                logger.warning(f"  Unknown activation function: {wandb_config.activation_fn}. Using ReLU.")
                activation_fn = nn.ReLU()
            
            # Initialize model with error handling
            try:
                model = AutoEncoder(
                    num_features=num_features,
                    hidden_dims=wandb_config.hidden_dims,
                    activation_fn=activation_fn,
                    dropout_rate=wandb_config.dropout_rate,
                    use_batch_norm=wandb_config.use_batch_norm,
                    learning_rate=wandb_config.learning_rate
                )
                logger.info(f" Model initialized: {wandb_config.hidden_dims}")
                
            except Exception as e:
                logger.error(f" Model initialization failed: {e}")
                return {"error": str(e)}
            
            # Initialize logger and callbacks
            callbacks = []
            
            if enable_checkpointing:
                try:
                    checkpoint_callback = ModelCheckpoint(
                        monitor='val_loss',
                        dirpath='./model/checkpoints',
                        filename=f'autoencoder-{wandb.run.name}',
                        save_top_k=1,
                        mode='min'
                    )
                    callbacks.append(checkpoint_callback)
                except Exception as e:
                    logger.warning(f"  Checkpointing setup failed: {e}")
            
            try:
                wandb_logger = WandbLogger(log_model="all", project=project_name)
            except Exception as e:
                logger.warning(f"  WandB logger setup failed: {e}")
                wandb_logger = None
            
            # Initialize trainer with robust configuration
            trainer_args = {
                'max_epochs': max_epochs,
                'deterministic': True,
                'default_root_dir': './model',
                'enable_progress_bar': False,  # Reduce noise in logs
                'enable_model_summary': False
            }
            
            if callbacks:
                trainer_args['callbacks'] = callbacks
            if wandb_logger:
                trainer_args['logger'] = wandb_logger
            
            try:
                trainer = pl.Trainer(**trainer_args)
            except Exception as e:
                logger.error(f" Trainer initialization failed: {e}")
                return {"error": str(e)}
            
            # Train the model with error handling
            try:
                logger.info(f"  Starting training with {len(train_loader)} train batches, {len(val_loader)} val batches")
                trainer.fit(model, train_loader, val_loader)
                logger.info(" Training completed successfully")
                
                # Collect metrics
                metrics = dict(trainer.logged_metrics)
                
                # Add best model path if checkpointing was enabled
                if enable_checkpointing and callbacks:
                    try:
                        metrics['best_model_path'] = callbacks[0].best_model_path
                    except Exception as e:
                        logger.warning(f"  Could not retrieve best model path: {e}")
                
                return metrics
                
            except Exception as e:
                logger.error(f" Training failed: {e}")
                return {"error": str(e)}
            
            finally:
                # Ensure wandb run is properly finished
                try:
                    wandb.finish()
                except:
                    pass
                
    except Exception as e:
        logger.error(f" Training setup failed: {e}")
        return {"error": str(e)}


def run_hyperparameter_sweep(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_features: int,
    sweep_config: Dict[str, Any],
    max_runs: int = 20,
    project_name: str = "CLAP_DeepSVDD",
    max_epochs: int = 200
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Execute hyperparameter sweep with comprehensive error handling and progress tracking.
    
    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_features (int): Number of input features
        sweep_config (Dict[str, Any]): Sweep configuration
        max_runs (int): Maximum number of sweep runs
        project_name (str): WandB project name
        max_epochs (int): Maximum epochs per run
        
    Returns:
        Tuple[Optional[str], List[Dict]]: (sweep_id, results_list)
    """
    logger = logging.getLogger(__name__)
    
    if not WANDB_AVAILABLE:
        logger.error(" WandB not available. Cannot run hyperparameter sweep.")
        return None, []
    
    results = []
    successful_runs = 0
    
    def train_with_logging(config=None):
        nonlocal successful_runs
        try:
            result = train_model(
                config, train_loader, val_loader, num_features,
                max_epochs, project_name
            )
            
            if "error" not in result:
                results.append(result)
                successful_runs += 1
                logger.info(f" Run {successful_runs} completed successfully")
            else:
                logger.error(f" Run failed: {result['error']}")
                
            return result
            
        except Exception as e:
            logger.error(f" Unexpected error in training run: {e}")
            return {"error": str(e)}
    
    try:
        # Update sweep config with actual train set length
        sweep_config['parameters']['train_data_length']['values'] = [len(train_loader)]
        
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        logger.info(f" Started hyperparameter sweep: {sweep_id}")
        
        # Run sweep with limited runs
        try:
            wandb.agent(sweep_id, train_with_logging, count=max_runs)
        except KeyboardInterrupt:
            logger.info("  Sweep interrupted by user")
        except Exception as e:
            logger.error(f" Sweep execution failed: {e}")
        
        logger.info(f" Sweep completed: {successful_runs}/{max_runs} runs successful")
        return sweep_id, results
        
    except Exception as e:
        logger.error(f" Sweep initialization failed: {e}")
        return None, results


def analyze_results(
    results: List[Dict[str, Any]], 
    csv_path: Optional[str] = None,
    show_plots: bool = True
) -> Optional[pd.DataFrame]:
    """Analyze hyperparameter tuning results with statistical summaries and visualizations.
    
    Args:
        results (List[Dict]): List of training results
        csv_path (Optional[str]): Path to CSV file to analyze (alternative to results)
        show_plots (bool): Whether to display plots
        
    Returns:
        Optional[pd.DataFrame]: Results DataFrame or None if analysis fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load results from CSV or use provided results
        if csv_path and os.path.exists(csv_path):
            logger.info(f" Loading results from {csv_path}")
            df = pd.read_csv(csv_path)
        elif results:
            logger.info(f" Analyzing {len(results)} training results")
            df = pd.DataFrame(results)
        else:
            logger.warning("  No results available for analysis")
            return None
        
        # Clean and prepare data
        df = df.dropna(axis=1)
        
        # Filter out error results
        if 'error' in df.columns:
            error_count = df['error'].notna().sum()
            if error_count > 0:
                logger.info(f"  Filtered out {error_count} failed runs")
                df = df[df['error'].isna()]
        
        if df.empty:
            logger.warning("  No successful results to analyze")
            return None
        
        # Sort by validation loss if available
        if 'val_loss' in df.columns:
            df = df.sort_values(by='val_loss').reset_index(drop=True)
            logger.info(" Results sorted by validation loss")
            
            # Display top results
            top_n = min(5, len(df))
            logger.info(f" Top {top_n} results:")
            for i in range(top_n):
                row = df.iloc[i]
                logger.info(f"  {i+1}. Val Loss: {row.get('val_loss', 'N/A'):.4f}")
        
        # Statistical summary
        logger.info(" Statistical summary:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary = df[numeric_cols].describe()
            logger.info(f"\n{summary}")
        
        # Visualizations
        if show_plots and 'val_loss' in df.columns:
            create_analysis_plots(df)
        
        return df
        
    except Exception as e:
        logger.error(f" Results analysis failed: {e}")
        return None


def create_analysis_plots(df: pd.DataFrame) -> None:
    """Create visualization plots for hyperparameter analysis.
    
    Args:
        df (pd.DataFrame): Results DataFrame
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Plot 1: Validation loss distribution
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Loss distribution histogram
        plt.subplot(2, 2, 1)
        plt.hist(df['val_loss'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Validation Loss')
        plt.ylabel('Frequency')
        plt.title('Distribution of Validation Loss')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Loss by activation function (if available)
        if 'activation_fn' in df.columns:
            plt.subplot(2, 2, 2)
            activation_loss = df.groupby('activation_fn')['val_loss'].mean().sort_values()
            activation_loss.plot(kind='bar', color='lightgreen', alpha=0.7)
            plt.xlabel('Activation Function')
            plt.ylabel('Average Validation Loss')
            plt.title('Average Loss by Activation Function')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Subplot 3: Loss by learning rate (if available)
        if 'learning_rate' in df.columns:
            plt.subplot(2, 2, 3)
            # Bin learning rates for better visualization
            df['lr_bin'] = pd.cut(df['learning_rate'], bins=5)
            lr_loss = df.groupby('lr_bin')['val_loss'].mean()
            lr_loss.plot(kind='bar', color='salmon', alpha=0.7)
            plt.xlabel('Learning Rate Range')
            plt.ylabel('Average Validation Loss')
            plt.title('Average Loss by Learning Rate')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Subplot 4: Loss by dropout rate (if available)
        if 'dropout_rate' in df.columns:
            plt.subplot(2, 2, 4)
            dropout_loss = df.groupby('dropout_rate')['val_loss'].mean().sort_values()
            dropout_loss.plot(kind='bar', color='gold', alpha=0.7)
            plt.xlabel('Dropout Rate')
            plt.ylabel('Average Validation Loss')
            plt.title('Average Loss by Dropout Rate')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info(" Analysis plots created successfully")
        
    except Exception as e:
        logger.error(f" Plot creation failed: {e}")


def main(
    data_path: str = "data/HTSAT-base_musicradar_guitar_embeddings.pkl",
    device_override: str = "auto",
    log_level: str = "INFO",
    random_seed: int = 0,
    train_split: float = 0.8,
    batch_size: int = 32,
    max_epochs: int = 200,
    max_sweep_runs: int = 20,
    project_name: str = "CLAP_DeepSVDD",
    sweep_method: str = "random",
    output_csv: Optional[str] = None,
    analyze_existing: Optional[str] = None,
    show_plots: bool = True
) -> bool:
    """Main function to orchestrate hyperparameter tuning pipeline.
    
    Args:
        data_path (str): Path to dataset pickle file
        device_override (str): Device preference ('auto', 'cpu', 'cuda')
        log_level (str): Logging level for output verbosity
        random_seed (int): Random seed for reproducibility
        train_split (float): Training data split ratio
        batch_size (int): Batch size for training
        max_epochs (int): Maximum epochs per training run
        max_sweep_runs (int): Maximum number of sweep runs
        project_name (str): WandB project name
        sweep_method (str): Sweep method ('random', 'grid', 'bayes')
        output_csv (Optional[str]): Output CSV file path for results
        analyze_existing (Optional[str]): Path to existing CSV file to analyze
        show_plots (bool): Whether to display analysis plots
        
    Returns:
        bool: True if process completed successfully, False otherwise
    """
    # Setup logging
    logger = setup_logging(log_level)
    logger.info("Starting Hyperparameter Tuning for AutoEncoder")
    logger.info(f"Dataset: {data_path}")
    logger.info(f"Device: {device_override}")
    logger.info(f"Max Runs: {max_sweep_runs}")
    
    try:
        # Set random seeds
        set_random_seeds(random_seed)
        
        # Check if we're only analyzing existing results
        if analyze_existing:
            logger.info(f" Analyzing existing results from {analyze_existing}")
            analyze_results([], analyze_existing, show_plots)
            return True
        
        # Check WandB availability
        if not WANDB_AVAILABLE:
            logger.error(" WandB is required for hyperparameter tuning but not available")
            return False
        
        # Validate and load dataset
        logger.info(" Loading and validating dataset...")
        is_valid, error_msg, data = validate_dataset(data_path, load_data=True)
        if not is_valid:
            logger.error(f"Dataset validation failed: {error_msg}")
            return None
        if data is None:
            return False
        
        # Prepare data loaders
        logger.info(" Preparing data loaders...")
        train_loader, val_loader, num_features = prepare_data_loaders(
            data, train_split, batch_size, num_workers=4, random_seed=random_seed
        )
        
        # Create sweep configuration
        logger.info("  Creating sweep configuration...")
        sweep_config = create_sweep_config(method=sweep_method)
        
        # Run hyperparameter sweep
        logger.info(f" Starting hyperparameter sweep with {max_sweep_runs} runs...")
        sweep_id, results = run_hyperparameter_sweep(
            train_loader, val_loader, num_features, sweep_config,
            max_sweep_runs, project_name, max_epochs
        )
        
        if sweep_id:
            logger.info(f" Sweep completed: {sweep_id}")
        else:
            logger.error(" Sweep failed to initialize")
            return False
        
        # Save results to CSV if requested
        if output_csv and results:
            try:
                results_df = pd.DataFrame(results)
                results_df.to_csv(output_csv, index=False)
                logger.info(f" Results saved to {output_csv}")
            except Exception as e:
                logger.error(f" Failed to save results: {e}")
        
        # Analyze results
        if results:
            logger.info(" Analyzing sweep results...")
            analyze_results(results, show_plots=show_plots)
        else:
            logger.warning("  No successful results to analyze")
        
        logger.info(" Hyperparameter tuning completed successfully")
        return True
        
    except Exception as e:
        logger.error(f" Hyperparameter tuning failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for AutoEncoder models using WandB sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hp_tuning_loop_detection.py                                    # Basic tuning with defaults
  python hp_tuning_loop_detection.py --data data/bass_embeddings.pkl   # Use different dataset
  python hp_tuning_loop_detection.py --device cuda --runs 50           # GPU with more runs
  python hp_tuning_loop_detection.py --method bayes --epochs 100       # Bayesian optimization
  python hp_tuning_loop_detection.py --analyze results.csv             # Analyze existing results
  python hp_tuning_loop_detection.py --project MyProject --seed 42     # Custom project and seed
        """
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/HTSAT-base_musicradar_guitar_embeddings.pkl",
        help="Path to dataset pickle file (default: guitar embeddings)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training (default: auto)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training data split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum epochs per run (default: 200)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Maximum number of sweep runs (default: 20)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="CLAP_DeepSVDD",
        help="WandB project name (default: CLAP_DeepSVDD)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "grid", "bayes"],
        default="random",
        help="Sweep method (default: random)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path for results"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        help="Analyze existing CSV results file"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation"
    )
    
    args = parser.parse_args()
    
    success = main(
        data_path=args.data,
        device_override=args.device,
        log_level=args.log_level,
        random_seed=args.seed,
        train_split=args.train_split,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        max_sweep_runs=args.runs,
        project_name=args.project,
        sweep_method=args.method,
        output_csv=args.output,
        analyze_existing=args.analyze,
        show_plots=not args.no_plots
    )
    
    sys.exit(0 if success else 1)