"""
Script for training DeepSVDD models on different network configurations and datasets.

This script loads configurations from a JSON file, trains DeepSVDD models for each
network configuration and dataset combination, and saves the best models to a
designated directory structure.

The script performs the following steps:
1. Load configuration from JSON file
2. For each network configuration and dataset combination:
   - Load the dataset
   - Initialize and run DeepSVDD trainer
   - Save the best model paths and trained models
3. Move and rename model files to organized directory structure
4. Save the paths of the best models to a JSON file
"""

import os
import torch
from torch import nn

# Add the project root to the Python path to enable module imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils import write_to_json, load_json, create_folder, move_and_rename_files, PickleHandler
from modules.deepSVDD import DeepSVDDTrainer
from modules.utils import load_pickle


def main():
    """Main function to train DeepSVDD models."""
    # Initialize available devices
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using device:', device)
    
    # Load configurations
    configs = load_json("./configs/exp1_preliminary_benchmark.json")
    trained_models = {}
    best_models_paths = {}
    centers = {}
    
    # Train models for each network configuration and dataset combination
    for name, AE_config in configs['networks'].items():
        for dataset_name, path in configs['dataset_paths'].items():
            print(f"Training model: {name} on dataset: {dataset_name}")
            
            # Load dataset
            data = load_pickle(path)
            
            # Initialize and run trainer
            trainer = DeepSVDDTrainer([AE_config, configs["deepSVDD"]], 
                                      data, 
                                      device, 
                                      **configs["trainer"])
            
            trainer.run(f"{name}-{dataset_name}")
            
            # Store best model paths, trained models, and centers
            best_models_paths[f"{name}-{dataset_name}"] = trainer.get_best_model_path()
            trained_models[f"{name}-{dataset_name}"] = trainer.get_trained_dsvdd_model()
            centers[f"{name}-{dataset_name}"] = trainer.get_center()
    
    # Organize and save trained models
    subfolder = configs["config_name"].upper()
    target_dir = f"./checkpoints/loop_benchmark/{subfolder}"
    
    new_paths = {}
    for name, AE_config in configs["networks"].items():
        for dataset_name, path in configs["dataset_paths"].items():
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
    
    # Save best model paths to JSON file
    write_to_json(new_paths, f'./checkpoints/{configs["config_name"]}_best_models_path.json')
    print("Training completed and model paths saved.")


if __name__ == "__main__":
    main()
