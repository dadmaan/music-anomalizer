#!/usr/bin/env python3
"""
Main Experiment Benchmark Script

This script runs the main experiment benchmark, evaluating different anomaly detection models
on music datasets. It includes:
1. Baseline models (Isolation Forest, PCA)
2. Deep SVDD models with visualization
3. Statistical analysis and plotting

The script loads configurations, processes datasets, runs evaluations, and generates plots
for analysis of anomaly detection performance.
"""

import os
import sys

# Set deterministic behavior for pytorch lightning
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_anomalizer.data.data_loader import *
from music_anomalizer.utils import load_json, write_to_json, load_pickle, cleanup, plot_score_distributions, box_plot_anomaly_scores
from music_anomalizer.config import load_experiment_config, get_checkpoint_registry
from music_anomalizer.evaluation.statistics import determine_threshold_by_quantile, divide_predictions_and_ttest, perform_pairwise_ttests_dynamic
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.visualization.visualizer import LatentSpaceVisualizer
from music_anomalizer.models.base_models import isolation_forest, pca_reconstruction_error

# Set matrix multiplication precision
torch.set_float32_matmul_precision('medium')  # set it to 'high' for higher precision


def set_random_seeds(seed=0):
    """Set random seeds for reproducibility."""
    random_seed = seed
    pl.seed_everything(random_seed)
    return random_seed


def initialize_device():
    """Initialize available devices (CUDA or CPU)."""
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using device:', device)
    return device, use_cuda


def load_configurations(config_name="exp2_deeper"):
    """Load experiment configurations from YAML file with validation."""
    configs = load_experiment_config(config_name)
    return configs


def evaluate_baseline_models(configs, random_seed):
    """
    Evaluate baseline models (Isolation Forest, PCA) on all datasets.
    
    Args:
        configs (dict): Configuration dictionary
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Results from baseline models
    """
    basemodels_results = {}
    
    for dataset_name, path in configs['dataset_paths'].items():
        print(f"\n--- Running Evaluation on {dataset_name} ---")
        data = load_pickle(path)
        train_set, valid_set = train_test_split(data, test_size=0.2, random_state=random_seed)
        np.random.shuffle(valid_set)  # Shuffle validation set

        print(f"Train embeddings shape: {train_set.shape}")
        print(f"Validation embeddings shape: {valid_set.shape}")

        # 2. Apply Baselines
        print("\n--- Isolation Forest ---")
        if_scores_train, if_scores_valid = isolation_forest(train_set, valid_set, random_state=42)
        if_thres = determine_threshold_by_quantile(if_scores_train, 0.95)  # calculate threshold value
        print("First 5 IF scores (validation):", if_scores_valid[:5])
        print("IF threshold:", if_thres)

        print("\n--- PCA Reconstruction Error ---")
        pca_scores_train, pca_scores_valid = pca_reconstruction_error(train_set, valid_set, 
                                                    variance_threshold=0.95, standardize=True)
        pca_thres = determine_threshold_by_quantile(pca_scores_train, 0.95)  # calculate threshold value
        print("First 5 PCA scores:", pca_scores_valid[:5])
        print("PCA threshold:", pca_thres)

        # store results for each dataset, separated by method
        if dataset_name not in basemodels_results:
            basemodels_results[dataset_name] = {}

        basemodels_results[dataset_name] = {
            'IF_scores_train': if_scores_train,
            'IF_scores_valid': if_scores_valid,
            'IF_thres': if_thres,
            'PCA_scores_train': pca_scores_train,
            'PCA_scores_valid': pca_scores_valid,
            'PCA_thres': pca_thres
        }

        print("\n--- Evaluation Finished ---")
        
    return basemodels_results


def evaluate_deep_svdd_models(configs, device, random_seed):
    """
    Evaluate Deep SVDD models on all datasets with visualization.
    
    Args:
        configs (dict): Configuration dictionary
        device (torch.device): Device to run computations on
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (dsvdd_results, figures)
    """
    GRID_LAYOUT = (1, 3)
    LOG_TO_WANDB = False  # save to disk and log to wandb

    dsvdd_results = {}
    figures = {}

    # Use checkpoint registry for automatic discovery
    checkpoint_registry = get_checkpoint_registry()
    ckpt_paths = checkpoint_registry.get_experiment_checkpoints(configs.config_name)

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

            # Get embeddings and anomaly scores
            train_dict = detector.compute_anomaly_scores(dh.get_train_set())
            valid_dict = detector.compute_anomaly_scores(dh.get_val_set())
            embeddings = [train_dict['embeddings'], valid_dict['embeddings']]
            anomaly_scores = [train_dict['scores'], valid_dict['scores']]
            latent_dim = embeddings[0].shape[-1]

            # Determine threshold for anomalies
            thres = determine_threshold_by_quantile(anomaly_scores[0], 0.95)
            print(f"Threshold value for anomalies (training examples): {thres}")

            print("----------------- Visualizing Training -----------------")
            visualizer = LatentSpaceVisualizer(embeddings[0], latent_dim, anomaly_scores)

            # Set save directory
            subfolder = configs["config_name"].upper()
            save_dir = f"./plots/{subfolder}/train" if LOG_TO_WANDB else None

            # Set the title
            title = f"{name} - {dataset_name.split('_')[-1].capitalize()} Dataset"

            # Plotting
            visualizer.visualize_all(grid_layout=GRID_LAYOUT, 
                                     threshold=thres, 
                                     save_dir=save_dir,
                                     title=title,
                                     width_per_col=4,
                                     height_per_row=4,
                                     plot_anomaly_distribution=False)
            figures[f"{base_name}_train"] = visualizer.get_figure()

            # delete the visualizer to free up memory
            del visualizer

            # Visualizing Validation
            print("----------------- Visualizing Validation -----------------")
            visualizer = LatentSpaceVisualizer(embeddings[1], latent_dim, [anomaly_scores[1]])

            # Set save directory
            subfolder = configs["config_name"].upper()
            save_dir = f"./plots/{subfolder}/valid" if LOG_TO_WANDB else None

            # Set the title
            title = f"{name} - {dataset_name.split('_')[-1].capitalize()} Dataset"

            # Plotting
            visualizer.visualize_all(grid_layout=GRID_LAYOUT, 
                                     threshold=thres, 
                                     save_dir=save_dir,
                                     title=title,
                                     width_per_col=4,
                                     height_per_row=4,
                                     plot_anomaly_distribution=False)
            figures[f"{base_name}_valid"] = visualizer.get_figure()

            # Store results
            if dataset_name not in dsvdd_results:
                dsvdd_results[dataset_name] = {}
            dsvdd_results[dataset_name][f"{name}_scores_train"] = anomaly_scores[0]
            dsvdd_results[dataset_name][f"{name}_scores_valid"] = anomaly_scores[1]
            dsvdd_results[dataset_name][f"{name}_thres"] = thres

            # Cleanup the memory
            cleanup()
            
    return dsvdd_results, figures


def organize_results(dsvdd_results, basemodels_results):
    """
    Organize results into train, validation, and threshold dictionaries.
    
    Args:
        dsvdd_results (dict): Results from Deep SVDD models
        basemodels_results (dict): Results from baseline models
        
    Returns:
        tuple: (train_results, valid_results, thresholds)
    """
    train_results = {}
    valid_results = {}
    thresholds = {}
    
    # Process DSVDD results
    for dataset_name, results in dsvdd_results.items():
        if dataset_name not in train_results:
            train_results[dataset_name] = {}
        if dataset_name not in valid_results:
            valid_results[dataset_name] = {}
        if dataset_name not in thresholds:
            thresholds[dataset_name] = {}

        for key, value in results.items():
            if "train" in key:
                train_results[dataset_name][key] = value
            elif "valid" in key:
                valid_results[dataset_name][key] = value
            elif "thres" in key:
                thresholds[dataset_name][key] = value
            else:
                print(f"Unknown key: {key} in {dataset_name}")

    # Process baseline results
    for dataset_name, results in basemodels_results.items():
        if dataset_name not in train_results:
            train_results[dataset_name] = {}
        if dataset_name not in valid_results:
            valid_results[dataset_name] = {}

        for key, value in results.items():
            if "train" in key:
                train_results[dataset_name][key] = value
            elif "valid" in key:
                valid_results[dataset_name][key] = value
            elif "thres" in key:
                thresholds[dataset_name][key] = value
            else:
                print(f"Unknown key: {key} in {dataset_name}")
                
    return train_results, valid_results, thresholds


def create_visualizations(train_results, valid_results, thresholds):
    """
    Create visualizations for the results.
    
    Args:
        train_results (dict): Training results
        valid_results (dict): Validation results
        thresholds (dict): Threshold values
    """
    dataset_names = list(train_results.keys())
    titles = ['Bass Dataset', 'Guitar Dataset']

    # Create box plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create vertically stacked subplots

    for dataset, title, ax in zip(dataset_names, titles, axes):
        box_plot_anomaly_scores(train_results, valid_results, dataset, ax, title=title)

    plt.tight_layout()
    plt.show()

    # Create score distribution plots
    for dataset_name in dataset_names:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # Adjusted layout for better visualization
        axes = axes.flatten()

        models = ['AE', 'AEwRES', 'IF', 'PCA']
        titles = [
            f"{'PCA Rec.' if model == 'PCA' else model} - {dataset_name.split('_')[-1].capitalize()} Dataset (Threshold: {thresholds[dataset_name][f'{model}_thres']:.3f})"
            for model in models
            ]

        for i, model in enumerate(models):
            train_key = f"{model}_scores_train"
            valid_key = f"{model}_scores_valid"
            thres_key = f"{model}_thres"

            plot_score_distributions(
                train_results[dataset_name][train_key],
                valid_results[dataset_name][valid_key],
                thresholds[dataset_name][thres_key],
                titles[i],
                axes[i]
            )

        plt.tight_layout()
        plt.show()


def perform_statistical_analysis(train_results, valid_results):
    """
    Perform statistical analysis on the results.
    
    Args:
        train_results (dict): Training results
        valid_results (dict): Validation results
    """
    dataset_name = 'HTSAT_base_musicradar_bass' 
    
    ttest_results = perform_pairwise_ttests_dynamic(train_results, dataset_name)
    print("T-test results (train):", ttest_results)
    
    ttest_results = perform_pairwise_ttests_dynamic(valid_results, dataset_name)
    print("T-test results (validation):", ttest_results)


def main():
    """Main function to run the experiment benchmark."""
    # Set up environment
    random_seed = set_random_seeds()
    device, use_cuda = initialize_device()
    
    # Load configurations
    configs = load_configurations()
    
    # Evaluate baseline models
    basemodels_results = evaluate_baseline_models(configs, random_seed)
    
    # Evaluate Deep SVDD models
    dsvdd_results, figures = evaluate_deep_svdd_models(configs, device, random_seed)
    
    # Organize results
    train_results, valid_results, thresholds = organize_results(dsvdd_results, basemodels_results)
    
    # Create visualizations
    create_visualizations(train_results, valid_results, thresholds)
    
    # Perform statistical analysis
    perform_statistical_analysis(train_results, valid_results)
    
    print("Experiment benchmark completed successfully.")


if __name__ == "__main__":
    main()
