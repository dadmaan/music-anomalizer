import wandb
import torch
import torch.optim as optim
import pytorch_lightning as pl
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

from torch import nn

from music_anomalizer.models.layers import *

logger = logging.getLogger(__name__)


class BaseAutoEncoder(pl.LightningModule):
    """Base class for AutoEncoder models with shared functionality."""
    
    def __init__(self, num_features: int, train_data_length: int, hidden_dims: Optional[List[int]] = None, 
                 activation_fn: Optional[nn.Module] = None, dropout_rate: Optional[float] = None, 
                 use_batch_norm: bool = False, learning_rate: float = 1e-3, weight_decay: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        self.automatic_optimization = False

        # Validate inputs
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if train_data_length <= 0:
            raise ValueError(f"train_data_length must be positive, got {train_data_length}")
        if hidden_dims is None or not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
            raise ValueError(f"hidden_dims must be a non-empty list/tuple, got {hidden_dims}")

        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_data_length = train_data_length
        self.bias = bias

        try:
            # Initialize encoder and decoder
            self.encoder = Encoder(num_features, self.hidden_dims, activation_fn, 
                                   self.use_batch_norm, self.dropout_rate, self.bias)
            
            self.decoder = Decoder(self.hidden_dims, num_features, activation_fn, 
                                   self.use_batch_norm, self.dropout_rate, self.bias)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize encoder/decoder: {str(e)}")

        self.save_hyperparameters(ignore=['activation_fn'])

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1e3, eta_min=5e-6),
            'interval': 'step',
            'name': 'cosine_annealing_lr_log'
        }

        return [optimizer], [scheduler]

    def _training_step_common(self, train_batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        """Common training step logic for both AutoEncoder types."""
        try:
            opt = self.optimizers()
            sch = self.lr_schedulers()
            
            opt.zero_grad()
            
            x = train_batch
            if x is None or x.numel() == 0:
                logger.warning(f"Empty batch at step {batch_idx}")
                return None
            
            x_recon = self.forward(x)
            if x_recon is None:
                raise RuntimeError("Forward pass returned None")
            
            loss = nn.MSELoss()(x_recon, x)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss value: {loss}")
                return None
            
            self.log('train_loss', loss, prog_bar=True)
            
            self.manual_backward(loss)
            opt.step()
            
            if self.trainer.is_last_batch:
                sch.step()
            
            return loss
        except Exception as e:
            logger.error(f"{self.__class__.__name__} training step failed at batch {batch_idx}: {str(e)}")
            raise

    def _validation_step_common(self, val_batch: torch.Tensor, batch_idx: int) -> Optional[torch.Tensor]:
        """Common validation step logic for both AutoEncoder types."""
        try:
            x = val_batch
            if x is None or x.numel() == 0:
                logger.warning(f"Empty validation batch at step {batch_idx}")
                return None
            
            x_recon = self.forward(x)
            if x_recon is None:
                raise RuntimeError("Forward pass returned None")
            
            loss = nn.MSELoss()(x_recon, x)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid validation loss value: {loss}")
                return None
            
            self.log('val_loss', loss, prog_bar=True)
            return loss
        except Exception as e:
            logger.error(f"{self.__class__.__name__} validation step failed at batch {batch_idx}: {str(e)}")
            raise

    def on_train_start(self) -> None:
        wandb.watch(self, log='all', log_freq=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement forward method")


class AutoEncoder(BaseAutoEncoder):
    """Standard AutoEncoder implementation."""
    
    def __init__(self, num_features: int, train_data_length: int, hidden_dims: Optional[List[int]] = None, 
                 activation_fn: Optional[nn.Module] = None, dropout_rate: Optional[float] = None, 
                 use_batch_norm: bool = False, learning_rate: float = 1e-3, weight_decay: float = 1e-5, bias: bool = False) -> None:
        super().__init__(num_features, train_data_length, hidden_dims, activation_fn, 
                         dropout_rate, use_batch_norm, learning_rate, weight_decay, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        scheduler = {
        'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1e3, eta_min=5e-6),
        'interval': 'step',
        'name': 'cosine_annealing_lr_log'
        }

        # scheduler = {
        # 'scheduler': optim.lr_scheduler.OneCycleLR(
        #     optimizer, 
        #     max_lr=0.01, 
        #     total_steps=None,
        #     epochs=self.trainer.max_epochs, 
        #     steps_per_epoch=self.train_data_length, # 108 for guitar dataset
        #     pct_start=0.3, 
        #     anneal_strategy='cos', 
        #     final_div_factor=1e4,
        #     verbose=True
        # ),
        # 'interval': 'step',
        # 'name': 'one_cycle_lr_log'
        # }

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        x = train_batch
        x_recon = self.forward(x)
        loss = nn.MSELoss()(x_recon, x)
        self.log('train_loss', loss, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        x_recon = self.forward(x)
        loss = nn.MSELoss()(x_recon, x)
        # self.validation_step_outputs.append(loss)
        self.log('val_loss', loss, prog_bar=True)

        return loss
    
    # def on_validation_epoch_end(self):
    #     epoch_average_loss = torch.stack(self.validation_step_outputs).mean()
    #     self.log('validation_epoch_average_loss', epoch_average_loss)
    #     scheduler_plateau = self.lr_schedulers()
    #     scheduler_plateau.step(epoch_average_loss)
    #     self.validation_step_outputs.clear()  # free memory

    def on_train_start(self):
        wandb.watch(self, log='all', log_freq=100)


class AutoEncoderWithResidual(pl.LightningModule):
    def __init__(self, num_features, train_data_length, hidden_dims=None, activation_fn=None, 
                 dropout_rate=None, use_batch_norm=False, learning_rate=1e-3, weight_decay=1e-5, bias=False):
        super().__init__()
        self.automatic_optimization = False
        # self.validation_step_outputs = [] # for ReduceLROnPlateau scheduler

        self.num_features = num_features
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_data_length = train_data_length
        self.bias = bias

        # Initialize encoder and decoder
        self.encoder = Encoder(num_features, self.hidden_dims, activation_fn, 
                               self.use_batch_norm, self.dropout_rate, self.bias)
        
        self.decoder = Decoder(self.hidden_dims, num_features, activation_fn, 
                               self.use_batch_norm, self.dropout_rate, self.bias)

        self.save_hyperparameters(ignore=['activation_fn'])

    def forward(self, x):
        residuals = []
        for layer in self.encoder.layers[:-1]:
            x = layer(x)
            residuals.append(x)
        x = self.encoder.layers[-1](x)

        residuals.reverse()
        for layer, residual in zip(self.decoder.layers[:-1], residuals):
            x = layer(x)
            x = x + residual
        return self.decoder.layers[-1](x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        scheduler = {
        'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1e3, eta_min=5e-6),
        'interval': 'step',
        'name': 'cosine_annealing_lr_log'
        }

        # scheduler = {
        # 'scheduler': optim.lr_scheduler.OneCycleLR(
        #     optimizer, 
        #     max_lr=0.01, 
        #     total_steps=None,
        #     epochs=self.trainer.max_epochs, 
        #     steps_per_epoch=self.train_data_length, # 108 for guitar dataset
        #     pct_start=0.3, 
        #     anneal_strategy='cos', 
        #     final_div_factor=1e4,
        #     verbose=True
        # ),
        # 'interval': 'step',
        # 'name': 'one_cycle_lr_log'
        # }

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        x = train_batch
        x_recon = self.forward(x)
        loss = nn.MSELoss()(x_recon, x)
        self.log('train_loss', loss, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        x_recon = self.forward(x)
        loss = nn.MSELoss()(x_recon, x)
        # self.validation_step_outputs.append(loss)
        self.log('val_loss', loss, prog_bar=True)

        return loss
    
    # def on_validation_epoch_end(self):
    #     epoch_average_loss = torch.stack(self.validation_step_outputs).mean()
    #     self.log('validation_epoch_average_loss', epoch_average_loss)
    #     scheduler_plateau = self.lr_schedulers()
    #     scheduler_plateau.step(epoch_average_loss)
    #     self.validation_step_outputs.clear()  # free memory

    def on_train_start(self):
        wandb.watch(self, log='all', log_freq=100)



class SVDD(pl.LightningModule):
    def __init__(self, encoder, center, train_data_length, learning_rate=1e-3, weight_decay=1e-5):
        super().__init__()
        self.automatic_optimization = False

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.encoder = encoder
        self.center = center
        self.train_data_length = train_data_length

        self.save_hyperparameters(ignore=['encoder'])

    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1e3, eta_min=5e-6)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        opt.zero_grad()
        
        x = train_batch
        batch_size = x.shape[0]
        
        z = self.encoder(x)
        loss = nn.MSELoss()(z, self.center.repeat(batch_size, 1))
        self.log('train_loss', loss, prog_bar=True)
        
        self.manual_backward(loss)
        opt.step()
        
        if self.trainer.is_last_batch:
            sch.step()
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch
        batch_size = x.shape[0]
        z = self.encoder(x)
        loss = nn.MSELoss()(z, self.center.repeat(batch_size, 1))
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def on_train_start(self):
        wandb.watch(self, log='all', log_freq=100)

activation_functions = {
    "ELU": nn.ELU(),
    "ReLU": nn.ReLU(),
    "LeakyReLU": nn.LeakyReLU(negative_slope=0.1)
    # Add other activation functions as needed
}

# Secure class registry - replace eval() with explicit mapping
CLASS_REGISTRY = {
    'AutoEncoder': AutoEncoder,
    'AutoEncoderWithResidual': AutoEncoderWithResidual,
    'SVDD': SVDD,
    # Add aliases used in configuration files
    'AE': AutoEncoder,
    'AEwRES': AutoEncoderWithResidual,
}

def create_network(config: Dict[str, Any]) -> pl.LightningModule:
    """
    Create a neural network from configuration dictionary.
    
    Args:
        config: Configuration dictionary with required parameters
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If config is invalid or missing required parameters
        RuntimeError: If model instantiation fails
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    
    # Validate required config parameters
    required_params = [
        'class_name', 'num_features', 'train_data_length', 'hidden_dims',
        'activation_fn', 'dropout_rate', 'use_batch_norm', 'learning_rate',
        'weight_decay', 'bias'
    ]
    
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"Missing required config parameters: {missing_params}")
    
    class_name = config['class_name']
    if class_name not in CLASS_REGISTRY:
        raise ValueError(f"Unknown class: {class_name}. Available classes: {list(CLASS_REGISTRY.keys())}")
    
    # Validate activation function
    activation_fn_name = config['activation_fn']
    if activation_fn_name not in activation_functions:
        raise ValueError(f"Unknown activation function: {activation_fn_name}. Available: {list(activation_functions.keys())}")
    
    # Validate numeric parameters
    try:
        num_features = int(config['num_features'])
        train_data_length = int(config['train_data_length'])
        if num_features <= 0 or train_data_length <= 0:
            raise ValueError(f"num_features and train_data_length must be positive")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid numeric parameters: {str(e)}")
    
    # Validate hidden_dims
    hidden_dims = config['hidden_dims']
    if not isinstance(hidden_dims, (list, tuple)) or not all(isinstance(d, int) and d > 0 for d in hidden_dims):
        raise ValueError(f"hidden_dims must be a list/tuple of positive integers, got {hidden_dims}")
    
    try:
        model_class = CLASS_REGISTRY[class_name]
        logger.info(f"Creating {class_name} with {num_features} features")
        
        model = model_class(
            num_features=num_features,
            train_data_length=train_data_length,
            hidden_dims=list(hidden_dims),
            activation_fn=activation_functions[activation_fn_name],
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            bias=config['bias']
        )
        
        if model is None:
            raise RuntimeError(f"Model instantiation returned None")
        
        logger.info(f"Successfully created {class_name} model")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create network: {str(e)}")
        raise RuntimeError(f"Network creation failed: {str(e)}")

def load_AE_from_checkpoint(checkpoint_path: Union[str, Path], config: Dict[str, Any], device: str) -> pl.LightningModule:
    """
    Load an AutoEncoder model from checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary
        device: Target device ('cuda', 'cpu', etc.)
        
    Returns:
        Loaded model
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If config or device is invalid
        RuntimeError: If model loading fails
    """
    # Validate inputs
    if not isinstance(checkpoint_path, (str, Path)):
        raise ValueError(f"checkpoint_path must be string or Path, got {type(checkpoint_path)}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    
    if not isinstance(device, str):
        raise ValueError(f"Device must be a string, got {type(device)}")
    
    # Validate device format
    valid_devices = ['cpu'] + [f'cuda:{i}' for i in range(8)] + ['cuda']
    if device not in valid_devices and not device.startswith('cuda:'):
        logger.warning(f"Unusual device specification: {device}")
    
    # Validate required config parameters
    required_params = [
        'class_name', 'num_features', 'train_data_length', 'hidden_dims',
        'activation_fn', 'dropout_rate', 'use_batch_norm', 'learning_rate',
        'weight_decay', 'bias'
    ]
    
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"Missing required config parameters: {missing_params}")
    
    class_name = config['class_name']
    if class_name not in CLASS_REGISTRY:
        raise ValueError(f"Unknown class: {class_name}. Available classes: {list(CLASS_REGISTRY.keys())}")
    
    # Validate activation function
    activation_fn_name = config['activation_fn']
    if activation_fn_name not in activation_functions:
        raise ValueError(f"Unknown activation function: {activation_fn_name}. Available: {list(activation_functions.keys())}")
    
    try:
        AE_class = CLASS_REGISTRY[class_name]
        logger.info(f"Loading {class_name} from {checkpoint_path} to device {device}")
        
        # Attempt to load checkpoint
        model = AE_class.load_from_checkpoint(
            str(checkpoint_path),
            num_features=config['num_features'],
            train_data_length=config['train_data_length'],
            hidden_dims=config['hidden_dims'],
            activation_fn=activation_functions[activation_fn_name],
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            bias=config['bias'],
            map_location=device
        )
        
        if model is None:
            raise RuntimeError(f"Model loading returned None")
        
        # Move model to specified device
        try:
            model = model.to(device)
        except Exception as e:
            raise RuntimeError(f"Failed to move model to device {device}: {str(e)}")
        
        logger.info(f"Successfully loaded {class_name} model on device {device}")
        return model
        
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to load model from checkpoint: {str(e)}")
        raise RuntimeError(f"Checkpoint loading failed: {str(e)}")