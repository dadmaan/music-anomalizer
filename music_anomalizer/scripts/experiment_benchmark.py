#!/usr/bin/env python3
"""
Main Experiment Benchmark Script

This script runs comprehensive benchmark evaluations of different anomaly detection models
on music datasets, including baseline models and Deep SVDD models with visualization.

The script performs the following operations:
1. Loads and validates experiment configurations from YAML files
2. Evaluates baseline models (Isolation Forest, PCA Reconstruction Error)
3. Evaluates Deep SVDD models with latent space visualization
4. Performs statistical analysis and generates comparative plots
5. Provides comprehensive reporting and error handling


"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# Set deterministic behavior for pytorch lightning
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Add project root to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_anomalizer.data.data_loader import DataHandler
from music_anomalizer.utils import (
    load_pickle, cleanup, plot_score_distributions, box_plot_anomaly_scores,
    setup_logging, set_random_seeds, initialize_device, validate_dataset
)
from music_anomalizer.config import load_experiment_config, get_checkpoint_registry
from music_anomalizer.evaluation.statistics import (
    determine_threshold_by_quantile, 
    perform_pairwise_ttests_dynamic
)
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.visualization.visualizer import LatentSpaceVisualizer
from music_anomalizer.models.base_models import isolation_forest, pca_reconstruction_error

# Set matrix multiplication precision
torch.set_float32_matmul_precision('medium')


def validate_configuration(config_name: str) -> bool:
    """Validate experiment configuration and prerequisites.
    
    Args:
        config_name (str): Configuration name to validate
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load and validate configuration
        configs = load_experiment_config(config_name)
        logger.info(f" Configuration '{config_name}' loaded successfully")
        logger.info(f"  Networks: {list(configs.networks.keys())}")
        logger.info(f"  Datasets: {list(configs.dataset_paths.keys())}")
        
        # Validate dataset paths
        logger.info(" Validating dataset paths...")
        for dataset_name, dataset_path in configs.dataset_paths.items():
            if os.path.exists(dataset_path):
                file_size = os.path.getsize(dataset_path)
                logger.info(f"   {dataset_name}: {dataset_path} ({file_size / (1024*1024):.1f} MB)")
            else:
                logger.error(f"   {dataset_name}: {dataset_path} (file not found)")
                return False
        
        # Validate checkpoint registry
        try:
            checkpoint_registry = get_checkpoint_registry()
            ckpt_paths = checkpoint_registry.get_experiment_checkpoints(configs.config_name)
            logger.info(f" Checkpoint registry validated with {len(ckpt_paths)} model combinations")
        except Exception as e:
            logger.error(f" Checkpoint registry validation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f" Configuration validation failed: {e}")
        return False




def evaluate_baseline_models(
    configs: Any, 
    random_seed: int, 
    quantile_threshold: float = 0.95
) -> Dict[str, Dict[str, Any]]:
    """Evaluate baseline models (Isolation Forest, PCA) on all datasets with robust error handling.
    
    Args:
        configs: Configuration object with dataset paths
        random_seed (int): Random seed for reproducibility
        quantile_threshold (float): Quantile threshold for anomaly detection
        
    Returns:
        Dict[str, Dict[str, Any]]: Results from baseline models
    """
    logger = logging.getLogger(__name__)
    logger.info("  Starting baseline model evaluation")
    
    baseline_results = {}
    successful_evaluations = 0
    total_datasets = len(configs.dataset_paths)
    
    for dataset_idx, (dataset_name, dataset_path) in enumerate(configs.dataset_paths.items(), 1):
        logger.info(f" Processing dataset {dataset_idx}/{total_datasets}: {dataset_name}")
        
        # Validate and load dataset
        is_valid, error_msg, data = validate_dataset(dataset_path, load_data=True)
        if not is_valid:
            logger.error(f" Skipping dataset {dataset_name} due to validation failure: {error_msg}")
            continue
        
        try:
            # Split dataset
            train_set, valid_set = train_test_split(
                data, test_size=0.2, random_state=random_seed
            )
            np.random.shuffle(valid_set)
            
            logger.info(f" Train embeddings shape: {train_set.shape}")
            logger.info(f" Validation embeddings shape: {valid_set.shape}")
            
            dataset_results = {}
            
            # Evaluate Isolation Forest
            logger.info(" Evaluating Isolation Forest...")
            try:
                if_scores_train, if_scores_valid = isolation_forest(
                    train_set, valid_set, random_state=random_seed
                )
                if_threshold = determine_threshold_by_quantile(if_scores_train, quantile_threshold)
                
                dataset_results.update({
                    'IF_scores_train': if_scores_train,
                    'IF_scores_valid': if_scores_valid,
                    'IF_thres': if_threshold
                })
                
                logger.info(f" Isolation Forest completed - Threshold: {if_threshold:.4f}")
                logger.debug(f"First 5 IF validation scores: {if_scores_valid[:5]}")
                
            except Exception as e:
                logger.error(f" Isolation Forest evaluation failed for {dataset_name}: {e}")
            
            # Evaluate PCA Reconstruction Error
            logger.info(" Evaluating PCA Reconstruction Error...")
            try:
                pca_scores_train, pca_scores_valid = pca_reconstruction_error(
                    train_set, valid_set, variance_threshold=0.95, standardize=True
                )
                pca_threshold = determine_threshold_by_quantile(pca_scores_train, quantile_threshold)
                
                dataset_results.update({
                    'PCA_scores_train': pca_scores_train,
                    'PCA_scores_valid': pca_scores_valid,
                    'PCA_thres': pca_threshold
                })
                
                logger.info(f" PCA Reconstruction Error completed - Threshold: {pca_threshold:.4f}")
                logger.debug(f"First 5 PCA validation scores: {pca_scores_valid[:5]}")
                
            except Exception as e:
                logger.error(f" PCA evaluation failed for {dataset_name}: {e}")
            
            # Store results if any evaluation succeeded
            if dataset_results:
                baseline_results[dataset_name] = dataset_results
                successful_evaluations += 1
                logger.info(f" Baseline evaluation completed for {dataset_name}")
            else:
                logger.error(f" All baseline evaluations failed for {dataset_name}")
                
        except Exception as e:
            logger.error(f" Dataset processing failed for {dataset_name}: {e}")
    
    logger.info(f" Baseline evaluation summary: {successful_evaluations}/{total_datasets} datasets processed successfully")
    return baseline_results


def evaluate_deep_svdd_models(
    configs: Any, 
    device: torch.device, 
    random_seed: int,
    quantile_threshold: float = 0.95,
    enable_visualization: bool = True,
    save_plots: bool = False
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Evaluate Deep SVDD models on all datasets with comprehensive error handling and visualization.
    
    Args:
        configs: Configuration object
        device (torch.device): Device to run computations on
        random_seed (int): Random seed for reproducibility
        quantile_threshold (float): Quantile threshold for anomaly detection
        enable_visualization (bool): Whether to generate visualizations
        save_plots (bool): Whether to save plots to disk
        
    Returns:
        Tuple[Dict, Dict]: (deep_svdd_results, figures)
    """
    logger = logging.getLogger(__name__)
    logger.info(" Starting Deep SVDD model evaluation")
    
    GRID_LAYOUT = (1, 3)
    deep_svdd_results = {}
    figures = {}
    successful_evaluations = 0
    total_combinations = len(configs.networks) * len(configs.dataset_paths)
    
    try:
        # Get checkpoint registry
        checkpoint_registry = get_checkpoint_registry()
        ckpt_paths = checkpoint_registry.get_experiment_checkpoints(configs.config_name)
        logger.info(f" Retrieved checkpoints for {len(ckpt_paths)} model combinations")
        
    except Exception as e:
        logger.error(f" Failed to retrieve checkpoint paths: {e}")
        return {}, {}
    
    combination_count = 0
    for network_name, ae_config in configs.networks.items():
        for dataset_name, dataset_path in configs.dataset_paths.items():
            combination_count += 1
            base_name = f"{network_name}-{dataset_name}"
            
            logger.info(f" Processing combination {combination_count}/{total_combinations}: {base_name}")
            
            try:
                # Validate checkpoint paths
                if base_name not in ckpt_paths:
                    logger.error(f" Checkpoint paths not found for {base_name}")
                    continue
                
                ae_ckpt_path = ckpt_paths[base_name][base_name + "-AE"]
                svdd_ckpt_path = ckpt_paths[base_name][base_name + "-DSVDD"]
                
                # Validate checkpoint files exist
                if not os.path.exists(ae_ckpt_path):
                    logger.error(f" AE checkpoint not found: {ae_ckpt_path}")
                    continue
                if not os.path.exists(svdd_ckpt_path):
                    logger.error(f" DSVDD checkpoint not found: {svdd_ckpt_path}")
                    continue
                
                # Validate and load dataset
                is_valid, error_msg, data = validate_dataset(dataset_path, load_data=True)
                if not is_valid:
                    logger.error(f"Dataset validation failed for {dataset_name}: {error_msg}")
                    continue
                
                # Setup data handler
                dh = DataHandler(data)
                dh.load_data()
                
                # Configure model
                model_config = ae_config.model_copy()
                model_config.num_features = dh.get_num_features()
                model_config.train_data_length = dh.get_num_data()
                
                config = [model_config.model_dump(), configs.deepSVDD.model_dump()]
                
                # Initialize and load detector
                detector = AnomalyDetector(config, [ae_ckpt_path, svdd_ckpt_path], device)
                detector.load_models()
                logger.info(f" Models loaded successfully for {base_name}")
                
                # Compute anomaly scores
                train_dict = detector.compute_anomaly_scores(dh.get_train_set())
                valid_dict = detector.compute_anomaly_scores(dh.get_val_set())
                
                embeddings = [train_dict['embeddings'], valid_dict['embeddings']]
                anomaly_scores = [train_dict['scores'], valid_dict['scores']]
                latent_dim = embeddings[0].shape[-1]
                
                # Determine threshold
                threshold = determine_threshold_by_quantile(anomaly_scores[0], quantile_threshold)
                logger.info(f" Anomaly threshold for {base_name}: {threshold:.4f}")
                
                # Visualization
                if enable_visualization:
                    try:
                        # Training set visualization
                        logger.info(f" Creating training visualization for {base_name}")
                        visualizer = LatentSpaceVisualizer(embeddings[0], latent_dim, anomaly_scores)
                        
                        subfolder = configs.config_name.upper()
                        save_dir = f"./plots/{subfolder}/train" if save_plots else None
                        title = f"{network_name} - {dataset_name.split('_')[-1].capitalize()} Dataset"
                        
                        visualizer.visualize_all(
                            grid_layout=GRID_LAYOUT,
                            threshold=threshold,
                            save_dir=save_dir,
                            title=title,
                            width_per_col=4,
                            height_per_row=4,
                            plot_anomaly_distribution=False
                        )
                        figures[f"{base_name}_train"] = visualizer.get_figure()
                        del visualizer
                        
                        # Validation set visualization
                        logger.info(f" Creating validation visualization for {base_name}")
                        visualizer = LatentSpaceVisualizer(embeddings[1], latent_dim, [anomaly_scores[1]])
                        
                        save_dir = f"./plots/{subfolder}/valid" if save_plots else None
                        
                        visualizer.visualize_all(
                            grid_layout=GRID_LAYOUT,
                            threshold=threshold,
                            save_dir=save_dir,
                            title=title,
                            width_per_col=4,
                            height_per_row=4,
                            plot_anomaly_distribution=False
                        )
                        figures[f"{base_name}_valid"] = visualizer.get_figure()
                        del visualizer
                        
                    except Exception as e:
                        logger.warning(f"  Visualization failed for {base_name}: {e}")
                
                # Store results
                if dataset_name not in deep_svdd_results:
                    deep_svdd_results[dataset_name] = {}
                
                deep_svdd_results[dataset_name][f"{network_name}_scores_train"] = anomaly_scores[0]
                deep_svdd_results[dataset_name][f"{network_name}_scores_valid"] = anomaly_scores[1]
                deep_svdd_results[dataset_name][f"{network_name}_thres"] = threshold
                
                successful_evaluations += 1
                logger.info(f" Deep SVDD evaluation completed for {base_name}")
                
                # Cleanup memory
                cleanup()
                
            except Exception as e:
                logger.error(f" Deep SVDD evaluation failed for {base_name}: {e}")
    
    logger.info(f" Deep SVDD evaluation summary: {successful_evaluations}/{total_combinations} combinations processed successfully")
    return deep_svdd_results, figures


def organize_results(
    dsvdd_results: Dict[str, Dict[str, Any]], 
    baseline_results: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Organize results into train, validation, and threshold dictionaries with error handling.
    
    Args:
        dsvdd_results (Dict): Results from Deep SVDD models
        baseline_results (Dict): Results from baseline models
        
    Returns:
        Tuple[Dict, Dict, Dict]: (train_results, valid_results, thresholds)
    """
    logger = logging.getLogger(__name__)
    logger.info(" Organizing evaluation results")
    
    train_results = {}
    valid_results = {}
    thresholds = {}
    
    def process_results(results_dict: Dict[str, Dict[str, Any]], source_name: str) -> None:
        """Helper function to process results from a source."""
        for dataset_name, results in results_dict.items():
            # Initialize dataset entries
            for result_dict in [train_results, valid_results, thresholds]:
                if dataset_name not in result_dict:
                    result_dict[dataset_name] = {}
            
            for key, value in results.items():
                try:
                    if "train" in key:
                        train_results[dataset_name][key] = value
                    elif "valid" in key:
                        valid_results[dataset_name][key] = value
                    elif "thres" in key:
                        thresholds[dataset_name][key] = value
                    else:
                        logger.debug(f"  Unrecognized key pattern: {key} in {dataset_name} ({source_name})")
                        
                except Exception as e:
                    logger.warning(f"  Error processing key {key} for {dataset_name}: {e}")
    
    # Process Deep SVDD results
    if dsvdd_results:
        process_results(dsvdd_results, "Deep SVDD")
        logger.info(f" Processed Deep SVDD results for {len(dsvdd_results)} datasets")
    
    # Process baseline results
    if baseline_results:
        process_results(baseline_results, "Baseline")
        logger.info(f" Processed baseline results for {len(baseline_results)} datasets")
    
    # Summary statistics
    total_train_entries = sum(len(results) for results in train_results.values())
    total_valid_entries = sum(len(results) for results in valid_results.values())
    total_threshold_entries = sum(len(results) for results in thresholds.values())
    
    logger.info(f" Result organization summary:")
    logger.info(f"  Train results: {total_train_entries} entries across {len(train_results)} datasets")
    logger.info(f"  Validation results: {total_valid_entries} entries across {len(valid_results)} datasets")
    logger.info(f"  Thresholds: {total_threshold_entries} entries across {len(thresholds)} datasets")
    
    return train_results, valid_results, thresholds


def create_visualizations(
    train_results: Dict[str, Dict[str, Any]], 
    valid_results: Dict[str, Dict[str, Any]], 
    thresholds: Dict[str, Dict[str, Any]],
    show_plots: bool = True
) -> None:
    """Create comprehensive visualizations for the evaluation results with error handling.
    
    Args:
        train_results (Dict): Training results
        valid_results (Dict): Validation results  
        thresholds (Dict): Threshold values
        show_plots (bool): Whether to display plots
    """
    logger = logging.getLogger(__name__)
    logger.info(" Creating result visualizations")
    
    if not train_results or not valid_results or not thresholds:
        logger.warning("  Insufficient results for visualization")
        return
    
    try:
        dataset_names = list(train_results.keys())
        dataset_titles = [name.split('_')[-1].capitalize() + ' Dataset' for name in dataset_names]
        
        # Create box plots
        logger.info(" Creating box plot comparisons")
        try:
            fig, axes = plt.subplots(1, len(dataset_names), figsize=(5 * len(dataset_names), 4))
            if len(dataset_names) == 1:
                axes = [axes]
            
            for dataset, title, ax in zip(dataset_names, dataset_titles, axes):
                box_plot_anomaly_scores(train_results, valid_results, dataset, ax, title=title)
            
            plt.tight_layout()
            if show_plots:
                plt.show()
            plt.close()
            logger.info(" Box plots created successfully")
            
        except Exception as e:
            logger.error(f" Box plot creation failed: {e}")
        
        # Create score distribution plots
        logger.info(" Creating score distribution plots")
        for dataset_name in dataset_names:
            try:
                # Determine available models for this dataset
                available_models = []
                model_candidates = ['AE', 'AEwRES', 'Baseline', 'IF', 'PCA']
                
                for model in model_candidates:
                    train_key = f"{model}_scores_train"
                    valid_key = f"{model}_scores_valid" 
                    thres_key = f"{model}_thres"
                    
                    if (train_key in train_results.get(dataset_name, {}) and 
                        valid_key in valid_results.get(dataset_name, {}) and
                        thres_key in thresholds.get(dataset_name, {})):
                        available_models.append(model)
                
                if not available_models:
                    logger.warning(f"  No complete model results found for {dataset_name}")
                    continue
                
                # Create subplot layout
                n_models = len(available_models)
                cols = min(2, n_models)
                rows = (n_models + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                if n_models == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes if n_models > 1 else [axes]
                else:
                    axes = axes.flatten()
                
                for i, model in enumerate(available_models):
                    train_key = f"{model}_scores_train"
                    valid_key = f"{model}_scores_valid"
                    thres_key = f"{model}_thres"
                    
                    title = f"{'PCA Rec.' if model == 'PCA' else model} - {dataset_name.split('_')[-1].capitalize()} Dataset (Threshold: {thresholds[dataset_name][thres_key]:.3f})"
                    
                    plot_score_distributions(
                        train_results[dataset_name][train_key],
                        valid_results[dataset_name][valid_key],
                        thresholds[dataset_name][thres_key],
                        title,
                        axes[i] if n_models > 1 else axes[0]
                    )
                
                # Hide unused subplots
                for i in range(n_models, len(axes)):
                    if isinstance(axes, (list, np.ndarray)) and i < len(axes):
                        axes[i].set_visible(False)
                
                plt.tight_layout()
                if show_plots:
                    plt.show()
                plt.close()
                logger.info(f" Score distributions created for {dataset_name}")
                
            except Exception as e:
                logger.error(f" Score distribution plots failed for {dataset_name}: {e}")
        
        logger.info(" Visualization creation completed")
        
    except Exception as e:
        logger.error(f" Visualization creation failed: {e}")


def perform_statistical_analysis(
    train_results: Dict[str, Dict[str, Any]], 
    valid_results: Dict[str, Dict[str, Any]],
    target_dataset: Optional[str] = None
) -> None:
    """Perform statistical analysis on the evaluation results with error handling.
    
    Args:
        train_results (Dict): Training results
        valid_results (Dict): Validation results
        target_dataset (Optional[str]): Specific dataset to analyze, or None for first available
    """
    logger = logging.getLogger(__name__)
    logger.info(" Performing statistical analysis")
    
    if not train_results or not valid_results:
        logger.warning("  Insufficient results for statistical analysis")
        return
    
    # Select target dataset
    if target_dataset is None:
        target_dataset = list(train_results.keys())[0]
        logger.info(f" Using first available dataset for analysis: {target_dataset}")
    elif target_dataset not in train_results:
        logger.error(f" Target dataset '{target_dataset}' not found in results")
        return
    
    try:
        # Training set analysis
        logger.info(f" Performing pairwise t-tests on training results for {target_dataset}")
        train_ttest_results = perform_pairwise_ttests_dynamic(train_results, target_dataset)
        logger.info(f" Training set t-test results: {train_ttest_results}")
        
        # Validation set analysis
        logger.info(f" Performing pairwise t-tests on validation results for {target_dataset}")
        valid_ttest_results = perform_pairwise_ttests_dynamic(valid_results, target_dataset)
        logger.info(f" Validation set t-test results: {valid_ttest_results}")
        
        # Summary
        logger.info(" Statistical analysis completed successfully")
        
    except Exception as e:
        logger.error(f" Statistical analysis failed: {e}")


def display_benchmark_summary(
    baseline_results: Dict[str, Dict[str, Any]],
    dsvdd_results: Dict[str, Dict[str, Any]],
    total_datasets: int
) -> None:
    """Display comprehensive benchmark execution summary.
    
    Args:
        baseline_results (Dict): Results from baseline models
        dsvdd_results (Dict): Results from Deep SVDD models  
        total_datasets (int): Total number of datasets processed
    """
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info(" BENCHMARK EXECUTION SUMMARY")
    logger.info("=" * 70)
    
    # Baseline results summary
    baseline_datasets = len(baseline_results)
    baseline_success_rate = (baseline_datasets / total_datasets) * 100 if total_datasets > 0 else 0
    logger.info(f"  Baseline Models:")
    logger.info(f"    Successful datasets: {baseline_datasets}/{total_datasets}")
    logger.info(f"    Success rate: {baseline_success_rate:.1f}%")
    
    # Deep SVDD results summary
    dsvdd_datasets = len(dsvdd_results)
    dsvdd_success_rate = (dsvdd_datasets / total_datasets) * 100 if total_datasets > 0 else 0
    logger.info(f" Deep SVDD Models:")
    logger.info(f"    Successful datasets: {dsvdd_datasets}/{total_datasets}")
    logger.info(f"    Success rate: {dsvdd_success_rate:.1f}%")
    
    # Overall summary
    overall_success = baseline_datasets + dsvdd_datasets > 0
    logger.info(f" Overall Status: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if not overall_success:
        logger.error(" No models were evaluated successfully")
    
    logger.info("=" * 70)


def main(
    config_name: str = "exp2_deeper",
    device_override: str = "auto", 
    log_level: str = "INFO",
    quantile_threshold: float = 0.95,
    random_seed: int = 0,
    enable_visualization: bool = True,
    save_plots: bool = False,
    show_plots: bool = True,
    target_dataset: Optional[str] = None
) -> bool:
    """Main function to run the comprehensive experiment benchmark.
    
    Args:
        config_name (str): Name of the experiment configuration
        device_override (str): Device preference ('auto', 'cpu', 'cuda')
        log_level (str): Logging level for output verbosity
        quantile_threshold (float): Quantile threshold for anomaly detection
        random_seed (int): Random seed for reproducibility
        enable_visualization (bool): Whether to generate visualizations
        save_plots (bool): Whether to save plots to disk
        show_plots (bool): Whether to display plots
        target_dataset (Optional[str]): Specific dataset for statistical analysis
        
    Returns:
        bool: True if benchmark completed successfully, False otherwise
    """
    # Setup logging
    logger = setup_logging(log_level)
    logger.info(" Starting Music Anomaly Detection Benchmark")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Device: {device_override}")
    logger.info(f"Quantile Threshold: {quantile_threshold}")
    logger.info(f"Random Seed: {random_seed}")
    
    try:
        # Set random seeds for reproducibility
        set_random_seeds(random_seed)
        
        # Initialize device
        device = initialize_device(device_override)
        
        # Load and validate configurations
        logger.info(" Loading experiment configuration")
        configs = load_experiment_config(config_name)
        logger.info(f" Configuration loaded: {configs.config_name}")
        logger.info(f"  Networks: {list(configs.networks.keys())}")
        logger.info(f"  Datasets: {list(configs.dataset_paths.keys())}")
        
        total_datasets = len(configs.dataset_paths)
        
        # Evaluate baseline models
        baseline_results = evaluate_baseline_models(
            configs, random_seed, quantile_threshold
        )
        
        # Evaluate Deep SVDD models
        dsvdd_results, figures = evaluate_deep_svdd_models(
            configs, device, random_seed, quantile_threshold, 
            enable_visualization, save_plots
        )
        
        # Check if any evaluations succeeded
        if not baseline_results and not dsvdd_results:
            logger.error(" All model evaluations failed")
            display_benchmark_summary(baseline_results, dsvdd_results, total_datasets)
            return False
        
        # Organize results
        logger.info(" Organizing evaluation results")
        train_results, valid_results, thresholds = organize_results(
            dsvdd_results, baseline_results
        )
        
        # Create visualizations
        if train_results and valid_results and thresholds:
            create_visualizations(train_results, valid_results, thresholds, show_plots)
        else:
            logger.warning("  Skipping visualizations due to insufficient results")
        
        # Perform statistical analysis
        if train_results and valid_results:
            perform_statistical_analysis(train_results, valid_results, target_dataset)
        else:
            logger.warning("  Skipping statistical analysis due to insufficient results")
        
        # Display summary
        display_benchmark_summary(baseline_results, dsvdd_results, total_datasets)
        
        logger.info(" Experiment benchmark completed successfully")
        return True
        
    except Exception as e:
        logger.error(f" Benchmark execution failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive anomaly detection benchmark on music datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_exp_benchmark.py                                    # Run with default settings
  python main_exp_benchmark.py --config exp1                     # Use different experiment config
  python main_exp_benchmark.py --device cuda --seed 42           # GPU with specific seed
  python main_exp_benchmark.py --threshold 0.9 --no-viz          # Custom threshold, no plots
  python main_exp_benchmark.py --save-plots --log-level DEBUG    # Save plots with verbose logging
  python main_exp_benchmark.py --dry-run                         # Validate configuration only
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="exp2_deeper",
        help="Experiment configuration name (default: exp2_deeper)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cpu", "cuda"], 
        default="auto",
        help="Device to use for computations (default: auto)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Quantile threshold for anomaly detection (default: 0.95)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization generation"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to disk"
    )
    parser.add_argument(
        "--no-display",
        action="store_true", 
        help="Don't display plots (useful for headless environments)"
    )
    parser.add_argument(
        "--target-dataset",
        type=str,
        help="Specific dataset for statistical analysis"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running benchmark"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        success = validate_configuration(args.config)
        sys.exit(0 if success else 1)
    else:
        success = main(
            config_name=args.config,
            device_override=args.device,
            log_level=args.log_level,
            quantile_threshold=args.threshold,
            random_seed=args.seed,
            enable_visualization=not args.no_viz,
            save_plots=args.save_plots,
            show_plots=not args.no_display,
            target_dataset=args.target_dataset
        )
        sys.exit(0 if success else 1)