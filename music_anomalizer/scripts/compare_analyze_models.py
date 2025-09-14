#!/usr/bin/env python3
"""
Script for comparing and analyzing trained models.

This script loads trained models, computes anomaly scores on training and validation sets,
visualizes latent spaces, and evaluates model performance across different datasets.
It uses Weights & Biases for logging results and plots.

Key functionalities:
- Loads trained models from checkpoint paths
- Computes anomaly scores for train and validation sets
- Visualizes latent space representations
- Determines anomaly thresholds using different methods
- Logs results to Weights & Biases
"""

import os
import sys
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import torch
import pytorch_lightning as pl
import wandb

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_anomalizer.data.data_loader import DataHandler
from music_anomalizer.utils import (
    load_json, load_pickle, cleanup, set_deterministic_behavior, 
    initialize_device, setup_logging
)
from music_anomalizer.config import load_experiment_config
from music_anomalizer.evaluation.statistics import determine_threshold_by_boxcox, determine_threshold_by_quantile, determine_threshold_by_std_dev
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.visualization.visualizer import LatentSpaceVisualizer


def load_configurations(config_name="exp2_deeper"):
    """Load configuration from YAML file with validation."""
    configs = load_experiment_config(config_name)
    return configs


def load_model_checkpoints(configs):
    """Load model checkpoint paths from JSON file."""
    ckpt_paths = load_json(f'./checkpoints/{configs["config_name"]}_best_models_path.json')
    return ckpt_paths


def process_train_set(configs, ckpt_paths, device, save_plots=True):
    """
    Process the training set for all models and datasets.
    
    Args:
        configs: Configuration dictionary
        ckpt_paths: Checkpoint paths dictionary
        device: Device to use for computation
        save_plots: Whether to save plots to disk and log to wandb
        
    Returns:
        tuple: (latents, scores, models, figures)
    """
    latents = {}
    scores = {}
    models = {}
    figures = {}
    
    GRID_LAYOUT = (2, 2)
    
    for name, AE_config in configs['networks'].items():
        for dataset_name, path in configs['dataset_paths'].items():
            base_name = f"{name}-{dataset_name}"
            ae_ckpt_path = ckpt_paths[base_name][base_name+"-AE"]
            svdd_ckpt_path = ckpt_paths[base_name][base_name+"-DSVDD"]

            data = load_pickle(path)
            dh = DataHandler(data)
            dh.load_data()

            AE_config['num_features'] = dh.get_num_features()
            AE_config['train_data_length'] = dh.get_num_data()

            config = [AE_config, configs['deepSVDD']]

            detector = AnomalyDetector(config, [ae_ckpt_path, svdd_ckpt_path], device)
            detector.load_models()
            models[f"{name}-{dataset_name}"] = detector

            output = detector.compute_anomaly_scores(dh.get_train_set())
            valid_scores = detector.compute_anomaly_scores(dh.get_val_set())['scores']
            anomaly_scores = [output['scores'], valid_scores]
            latents[f"{name}-{dataset_name}"] = output['embeddings']
            scores[f"{name}-{dataset_name}"] = output['scores']
            latent_dim = output['embeddings'].shape[-1]

            visualizer = LatentSpaceVisualizer(output['embeddings'], 
                                               latent_dim, 
                                               f"{name}-{dataset_name}", 
                                               anomaly_scores)

            subfolder = configs["config_name"].upper()
            save_dir = f"./plots/{subfolder}/train" if save_plots else None
            
            # Using quantile method for threshold determination
            thres = determine_threshold_by_quantile(anomaly_scores[0], 0.92)
            print(f"Threshold value for anomalies: {thres}")

            visualizer.visualize_all(grid_layout=GRID_LAYOUT, threshold=thres, save_dir=save_dir,
                                     title=f"{name} Model")

            figures[base_name] = visualizer.get_figure()

            cleanup()
            
    return latents, scores, models, figures


def log_figures_to_wandb(configs, figures):
    """
    Log figures to Weights & Biases.
    
    Args:
        configs: Configuration dictionary
        figures: Dictionary of figures to log
    """
    wandb.init(project=configs["trainer"]["wandb_project_name"])
    for name, fig in figures.items():
        wandb.log({f"{name} Plots": wandb.Image(fig)})
    wandb.finish()


def evaluate_models_on_validation_set(configs, models, save_plots=True):
    """
    Evaluate all models across validation sets.
    
    Args:
        configs: Configuration dictionary
        models: Dictionary of trained models
        save_plots: Whether to save plots to disk
    """
    GRID_LAYOUT = (2, 2)
    
    for dataset_name, path in configs["dataset_paths"].items():
        for name, detector in models.items():
            data = load_pickle(path)
            dh = DataHandler(data)
            dh.load_data()

            scr = detector.compute_anomaly_scores(dh.get_train_set())['scores']
            output = detector.compute_anomaly_scores(dh.get_val_set())

            latent_dim = output['embeddings'].shape[-1]
            visualizer = LatentSpaceVisualizer(output['embeddings'], latent_dim, f"{name}-{dataset_name}", anomaly_scores=[output['scores']])
            
            # Using Box-Cox method for threshold determination on validation set
            thres = determine_threshold_by_boxcox(scr)

            subfolder = configs["config_name"].upper()
            save_dir = f"./plots/{subfolder}/eval" if save_plots else None
            visualizer.visualize_all(grid_layout=GRID_LAYOUT, threshold=thres, save_dir=save_dir)

            cleanup()


def main():
    """Main function to run the model comparison and analysis."""
    # Set deterministic behavior for reproducibility
    set_deterministic_behavior()
    
    # Initialize device
    device = initialize_device()
    
    # Load configurations
    configs = load_configurations()
    
    # Load model checkpoints
    ckpt_paths = load_model_checkpoints(configs)
    
    # Process training set
    latents, scores, models, figures = process_train_set(configs, ckpt_paths, device)
    
    # Log figures to wandb
    if figures:
        log_figures_to_wandb(configs, figures)
    
    # Evaluate models on validation set
    evaluate_models_on_validation_set(configs, models)


if __name__ == "__main__":
    main()
