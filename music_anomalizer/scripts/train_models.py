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
"""

import os
import sys
import argparse
import torch
from torch import nn

# Add the project root to the Python path to enable module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_anomalizer.utils import write_to_json, load_json, create_folder, move_and_rename_files, PickleHandler
from music_anomalizer.config import load_experiment_config, get_checkpoint_registry
from music_anomalizer.models.deepSVDD import DeepSVDDTrainer
from music_anomalizer.utils import load_pickle


def main(config_name: str = "exp1", device_override: str = "auto"):
    """Main function to train DeepSVDD models.
    
    Args:
        config_name (str): Name of the experiment configuration to use
        device_override (str): Device override ('auto', 'cpu', 'cuda')
    """
    # Initialize device with override support
    if device_override == "cpu":
        device = torch.device('cpu')
    elif device_override == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:  # auto
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
    
    print(f'Using device: {device}')
    
    # Load and validate configurations
    try:
        configs = load_experiment_config(config_name)
        print(f"Loaded configuration: {configs.config_name}")
    except Exception as e:
        print(f"Error loading configuration '{config_name}': {e}")
        return
    trained_models = {}
    best_models_paths = {}
    centers = {}
    
    # Train models for each network configuration and dataset combination
    for name, AE_config in configs.networks.items():
        for dataset_name, path in configs.dataset_paths.items():
            print(f"Training model: {name} on dataset: {dataset_name}")
            
            # Load and validate dataset
            try:
                data = load_pickle(path)
                if data is None or len(data) == 0:
                    print(f"Warning: Empty or invalid dataset at {path}")
                    continue
                print(f"Dataset loaded: {len(data)} samples")
            except Exception as e:
                print(f"Error loading dataset {path}: {e}")
                continue
            
            # Initialize and run trainer with error handling
            try:
                trainer = DeepSVDDTrainer([AE_config.dict(), configs.deepSVDD.dict()], 
                                          data, 
                                          device, 
                                          **configs.trainer.dict())
            except Exception as e:
                print(f"Error initializing trainer for {name}-{dataset_name}: {e}")
                continue
            
            # Run training with error handling
            try:
                trainer.run(f"{name}-{dataset_name}")
                print(f"Training completed successfully for {name}-{dataset_name}")
            except Exception as e:
                print(f"Training failed for {name}-{dataset_name}: {e}")
                continue
            
            # Store best model paths, trained models, and centers
            best_models_paths[f"{name}-{dataset_name}"] = trainer.get_best_model_path()
            trained_models[f"{name}-{dataset_name}"] = trainer.get_trained_dsvdd_model()
            centers[f"{name}-{dataset_name}"] = trainer.get_center()
    
    # Organize and save trained models
    subfolder = configs.config_name.upper()
    target_dir = f"./checkpoints/loop_benchmark/{subfolder}"
    
    new_paths = {}
    for name, AE_config in configs.networks.items():
        for dataset_name, path in configs.dataset_paths.items():
            base_name = f"{name}-{dataset_name}"
            ae_ckpt_path = best_models_paths[base_name][base_name+"-AE"]
            svdd_ckpt_path = best_models_paths[base_name][base_name+"-DSVDD"]
            
            # Create destination directory
            destination_dir = os.path.join(target_dir, name)
            create_folder(destination_dir)
            
            # Move and rename files
            ae_new_path, svdd_new_path = move_and_rename_files([ae_ckpt_path, svdd_ckpt_path], destination_dir)
            
            # Save center data as pickle file
            file_path = os.path.splitext(ae_new_path)[0] + '_z_vector.pkl'
            ph = PickleHandler(file_path)
            ph.dump_data(centers[base_name])
            
            # Store new paths
            new_paths[base_name] = {base_name+"-AE": ae_new_path, 
                                    base_name+"-DSVDD": svdd_new_path}
    
    # Save best model paths to JSON file (legacy compatibility)
    write_to_json(new_paths, f'./checkpoints/{configs.config_name}_best_models_path.json')
    
    # Register checkpoints in the new system
    try:
        checkpoint_registry = get_checkpoint_registry()
        discovered_checkpoints = checkpoint_registry.get_experiment_checkpoints(configs.config_name)
        print(f"Checkpoint registry updated with {len(discovered_checkpoints)} model combinations")
    except Exception as e:
        print(f"Warning: Could not update checkpoint registry: {e}")
    
    print("Training completed and model paths saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepSVDD models for anomaly detection")
    parser.add_argument("--config", type=str, default="exp1", 
                       help="Experiment configuration name (default: exp1)")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device to use for training (default: auto)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate configuration without training")
    
    args = parser.parse_args()
    
    if args.dry_run:
        # Validate configuration only
        try:
            configs = load_experiment_config(args.config)
            print(f"✓ Configuration '{args.config}' is valid")
            print(f"  Networks: {list(configs.networks.keys())}")
            print(f"  Datasets: {list(configs.dataset_paths.keys())}")
            print(f"  Batch size: {configs.trainer.batch_size}")
            print(f"  Max epochs: {configs.trainer.max_epochs}")
        except Exception as e:
            print(f"✗ Configuration validation failed: {e}")
    else:
        main(args.config, args.device)
