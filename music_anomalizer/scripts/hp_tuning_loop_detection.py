"""
Hyperparameter Tuning Script for AutoEncoder Model on Audio Embeddings

This script performs hyperparameter tuning for an AutoEncoder model using Weights & Biases (wandb) sweeps.
It systematically explores different configurations to identify optimal parameters that minimize validation loss.

The process involves:
1. Defining a sweep configuration with hyperparameters to tune
2. Initializing a wandb sweep to orchestrate training of multiple models
3. Training each model configuration using PyTorch Lightning
4. Logging performance metrics with wandb
5. Analyzing results and visualizing performance
"""

import os
# Set deterministic behavior for PyTorch Lightning
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"

import wandb
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Import custom modules
from music_anomalizer.models.networks import AutoEncoder
from music_anomalizer.data.data_loader import DatasetSampler  # Use existing DatasetSampler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed for reproducibility
random_seed = 0
pl.seed_everything(random_seed)

# Initialize available devices
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Using device:', device)


def load_and_prepare_data(data_path='data/HTSAT-base_musicradar_guitar_embeddings.pkl'):
    """
    Load and prepare data for training.
    
    Parameters:
        data_path (str): Path to the pickle file containing embeddings
        
    Returns:
        tuple: (train_set, val_set, num_features)
    """
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print('The number of data:', data.shape)
    
    # Shuffle and split
    random.shuffle(data)
    
    num_data = data.shape[0]
    num_train = int(num_data * 0.8)
    
    train_data = data[:num_train]
    val_data = data[num_train:]
    
    print('The number of train:', train_data.shape)
    print('The number of validation:', val_data.shape)
    
    # Create data loaders
    batch_size = 32
    num_features = data.shape[1]
    
    train_params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True, 'num_workers': 4}
    val_params = {'batch_size': batch_size, 'shuffle': False, 'pin_memory': True, 'num_workers': 4}
    
    train_set = DataLoader(DatasetSampler(train_data), **train_params)
    val_set = DataLoader(DatasetSampler(val_data), **val_params)
    
    return train_set, val_set, num_features


# Define sweep configuration for hyperparameter tuning
sweep_config = {
    'method': 'random',  # or 'grid' or 'bayes'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'   
    },
    'parameters': {
        'train_data_length': {
            'values': [0]  # Will be updated with actual train set length
        },
        'learning_rate': {
            'min': 1e-6,
            'max': 1e-2,
            'distribution': 'log_uniform'
        },
        'dropout_rate': {
            'values': [0.2]
        },
        'hidden_dims': {
            'values': [[512, 256, 128]]
        },
        'activation_fn': {
            'values': ['ELU']
        },
        'use_batch_norm': {
            'values': [True]
        }
    }
}


def train(config=None):
    """
    Train the AutoEncoder model with given configuration.
    
    Parameters:
        config: Configuration parameters for the model
        
    Returns:
        dict: Metrics from the training process
    """
    with wandb.init(config=config):
        config = wandb.config
        
        # Initialize model
        AE = AutoEncoder(
            num_features=num_features, 
            hidden_dims=config.hidden_dims, 
            activation_fn=getattr(nn, config.activation_fn)(), 
            dropout_rate=config.dropout_rate, 
            use_batch_norm=config.use_batch_norm,
            learning_rate=config.learning_rate
        )
        
        # Initialize logger
        wandb_ae_logger = WandbLogger(log_model="all", project="CLAP_DeepSVDD")
        
        # Initialize trainer
        trainer = pl.Trainer(
            num_nodes=1,
            max_epochs=200,
            deterministic=True,
            default_root_dir='./model',
            logger=wandb_ae_logger
        )
        
        # Train the model
        trainer.fit(AE, train_set, val_set)
        
        # Collect metrics
        metrics = trainer.logged_metrics
        # Note: checkpoint_callback is commented out in original code
        # metrics['best_model_path'] = checkpoint_callback.best_model_path
        
        wandb.finish()
        
        return metrics


def sweep_callback():
    """
    Execute the hyperparameter sweep.
    
    Returns:
        pd.DataFrame: Results from the sweep
    """
    results = []
    
    def train_with_logging(config=None):
        result = train(config)
        results.append(result)
        return result
    
    sweep_id = wandb.sweep(sweep_config, project="CLAP_DeepSVDD")
    wandb.agent(sweep_id, train_with_logging)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


def analyze_results(csv_path='csv/wandb_logs_guitar_116.csv'):
    """
    Analyze and visualize the results from hyperparameter tuning.
    
    Parameters:
        csv_path (str): Path to the CSV file containing results
    """
    # Load results
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1)
    df = df.sort_values(by='val_loss').reset_index(drop=True)
    
    print("Top 5 results:")
    print(df.head(5))
    
    print("\nStatistical summary:")
    print(df.describe())
    
    # Find specific run
    specific_run = df[df['Name'] == 'giddy-sweep-79']
    print("\nSpecific run 'giddy-sweep-79':")
    print(specific_run)
    
    # Visualization: Average validation loss per activation function
    avg_val_loss = df.groupby('activation_fn')['val_loss'].mean().reset_index()
    
    plt.figure(figsize=(6, 4))
    plt.bar(avg_val_loss['activation_fn'], avg_val_loss['val_loss'], color='skyblue')
    plt.xlabel('Activation Function')
    plt.ylabel('Average Validation Loss')
    plt.title('Average Validation Loss per Activation Function')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Visualization: Average validation loss per learning rate
    avg_val_loss = df.groupby('learning_rate')['val_loss'].mean().reset_index()
    
    plt.figure(figsize=(6, 4))
    plt.bar(avg_val_loss['learning_rate'], avg_val_loss['val_loss'], color='skyblue')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Validation Loss')
    plt.title('Average Validation Loss per Learning Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the hyperparameter tuning pipeline."""
    global train_set, val_set, num_features
    
    # Load and prepare data
    train_set, val_set, num_features = load_and_prepare_data()
    
    # Update sweep configuration with actual train set length
    sweep_config['parameters']['train_data_length']['values'] = [len(train_set)]
    
    # Run hyperparameter sweep
    print("Starting hyperparameter sweep...")
    results_df = sweep_callback()
    results_df.to_csv('wandb_sweep_results_guitar.csv')
    
    # Analyze results (if results file exists)
    try:
        analyze_results()
    except FileNotFoundError:
        print("Results file not found. Skipping analysis.")


if __name__ == "__main__":
    main()
