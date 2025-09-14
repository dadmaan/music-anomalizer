import os
import logging

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb

from music_anomalizer.models.networks import create_network, SVDD
from music_anomalizer.data.data_loader import DataHandler
from music_anomalizer.utils import get_z_vector, write_to_json, load_pickle, set_random_seeds

# set deterministic behavior for pl
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
# Set matrix multiplication precision
torch.set_float32_matmul_precision('medium') # set it to 'high' for higher precision
    
# init seed
set_random_seeds(1234)


class DeepSVDDTrainer:
    """
    A class to handle the training of a Deep SVDD model using PyTorch Lightning.

    Attributes:
        AE_config (dict): Configuration settings for the Autoencoder model.
        SVDD_config (dict): Configuration settings for the SVDD model.
        dataset (dict): A dictionary mapping dataset names to their respective paths.
        device (str): Device to run the model on ('cpu' or 'cuda').
        net (torch.nn.Module): The neural network model after pretraining.
        svdd (SVDD): The SVDD model instance.
        center (torch.Tensor): The center of the hypersphere for SVDD.
        best_model_paths (dict): A dictionary to store the paths of the best models saved during training.
        wandb_project_name (str): WandB project name.

    Methods:
        cleanup():
            Clears the CUDA cache if available.

        pretraining(name, dataset_name, train_set, val_set, max_epochs, patience):
            Pretrains an autoencoder model using the provided dataset.

        train_deepSVDD(name, dataset_name, train_set, val_set, max_epochs, patience):
            Trains the SVDD model using the pretrained weights from the autoencoder.

        setup_callbacks(name, dataset_name, model_type, patience):
            Sets up the necessary callbacks for training.

        setup_logger(name, dataset_name, model_type):
            Configures the WandbLogger for logging the training process.

        setup_trainer(max_epochs, callbacks, logger):
            Configures the PyTorch Lightning trainer.

        save_best_model_paths(save_path):
            Saves the paths of the best models to a JSON file.

        run(name, save_path='best_model_paths.json'):
            Runs the complete training pipeline for all datasets specified.
    """
    def __init__(self, configs, dataset, device, 
                 batch_size=32, max_epochs=1000, min_epochs=None, patience=10, 
                 wandb_project_name=None, wandb_log_model=False, 
                 enable_progress_bar=False, deterministic=False):
        self.AE_config, self.SVDD_config = configs
        self.dataset = dataset
        self.device = device
        
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience = patience
        
        self.net = None
        self.dsvdd = None
        self.center = None
        
        self.wandb_project_name = wandb_project_name
        self.wandb_log_model = wandb_log_model
        
        self.enable_progress_bar = enable_progress_bar
        self.deterministic = deterministic
        
        self.best_model_path = {}

    def cleanup(self):
        torch.cuda.empty_cache()

    def pretraining(self, name, train_set, val_set):
        model = create_network(self.AE_config)
        
        callbacks = self.setup_callbacks(name, 'AE', self.patience)
        wandb_logger = self.setup_logger(name, 'AE')
        trainer = self.setup_trainer(self.max_epochs, self.min_epochs, callbacks, wandb_logger)
        
        trainer.fit(model, train_set, val_set)
        wandb.finish()
 
        self.center = get_z_vector(model.to(self.device), train_set, self.device)
        self.net = model.encoder
        
        self.best_model_path[f"{name}-AE"] = callbacks[0].best_model_path

    def train_deepSVDD(self, name, train_set, val_set):
        self.net.train()
        self.dsvdd = SVDD(self.net, self.center.to(self.device), len(train_set), **self.SVDD_config)
        
        callbacks = self.setup_callbacks(name, 'DSVDD', self.patience)
        wandb_logger = self.setup_logger(name, 'DSVDD')
        trainer = self.setup_trainer(self.max_epochs, self.min_epochs, callbacks, wandb_logger)
        
        trainer.fit(self.dsvdd, train_set, val_set)
        wandb.finish()
        
        self.best_model_path[f"{name}-DSVDD"] = callbacks[0].best_model_path

    def setup_callbacks(self, name, model_type, patience):
        # Create wandb checkpoint directory
        checkpoint_dir = f"./wandb/checkpoints/{name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor='val_loss', 
            filename=f'{name}-{model_type}-{{epoch:02d}}-{{val_loss:.2f}}'
        )
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=patience, verbose=False, mode='min')
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')
        return [checkpoint_callback, early_stopping_callback, lr_monitor_callback]

    def setup_logger(self, name, model_type):
        if self.wandb_project_name is None:
            # Wandb is disabled, return None (PyTorch Lightning will use default logger)
            return None
        return WandbLogger(log_model=self.wandb_log_model, 
                           project=self.wandb_project_name if self.wandb_project_name else name, 
                           name=f"{name}-{model_type}")

    def setup_trainer(self, max_epochs, min_epochs, callbacks, logger):
        return pl.Trainer(
            num_nodes=1,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            deterministic=self.deterministic,
            default_root_dir='./wandb',
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=self.enable_progress_bar
        )

    def save_best_model_path(self, save_path):
        write_to_json(self.best_model_path, save_path)

    def get_best_model_path(self):
        return self.best_model_path
    
    def get_trained_dsvdd_model(self):
        return self.dsvdd
    
    def get_trained_encoder_model(self):
        return self.net
    
    def get_center(self):
        return self.center
        
    def run(self, name):
        logger = logging.getLogger()
            
        logger.info(f"Training the model with {name} dataset")
        
        dh = DataHandler(self.dataset, self.batch_size)
        dh.load_data()
        
        self.AE_config['num_features'] = dh.get_num_features()
        self.AE_config['train_data_length'] = dh.get_num_data()
        train_set = dh.get_train_set()
        val_set = dh.get_val_set()
        
        logger.info('Starting pretraining ...')
        self.pretraining(name, train_set, val_set)
        
        logger.info('Starting training ...')
        self.train_deepSVDD(name, train_set, val_set)
        
        self.cleanup()




