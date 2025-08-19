import wandb
import torch
import torch.optim as optim
import pytorch_lightning as pl

from torch import nn

from music_anomalizer.models.layers import *
    

class AutoEncoder(pl.LightningModule):
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

def create_network(config):
    model = eval(config['class_name'])(
                num_features=config['num_features'],
                train_data_length=config['train_data_length'],
                hidden_dims=config['hidden_dims'],
                activation_fn=activation_functions[config['activation_fn']],
                dropout_rate=config['dropout_rate'],
                use_batch_norm=config['use_batch_norm'],
                learning_rate=config['learning_rate'],
                weight_decay=config['weight_decay'],
                bias=config['bias']
    )
    return model

def load_AE_from_checkpoint(checkpoint_path, config, device):
    AE_class = globals()[config['class_name']]
    model = AE_class.load_from_checkpoint(checkpoint_path,
                                        num_features=config['num_features'],
                                        train_data_length=config['train_data_length'],
                                        hidden_dims=config['hidden_dims'],
                                        activation_fn=activation_functions[config['activation_fn']],
                                        dropout_rate=config['dropout_rate'],
                                        use_batch_norm=config['use_batch_norm'],
                                        learning_rate=config['learning_rate'],
                                        weight_decay=config['weight_decay'],
                                        bias=config['bias'],
                                        device=device)
    return model