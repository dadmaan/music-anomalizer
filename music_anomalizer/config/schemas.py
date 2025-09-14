"""Configuration schemas for validation and type safety."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class NetworkConfig(BaseModel):
    """Configuration for neural network architecture."""
    class_name: str = Field(..., description="Network class name (e.g., AutoEncoder)")
    num_features: Optional[int] = Field(None, description="Number of input features")
    train_data_length: Optional[int] = Field(None, description="Training data length")
    hidden_dims: List[int] = Field(..., description="Hidden layer dimensions")
    activation_fn: str = Field("ELU", description="Activation function")
    dropout_rate: Optional[float] = Field(None, description="Dropout rate (0.0-1.0)")
    use_batch_norm: bool = Field(False, description="Use batch normalization")
    learning_rate: float = Field(1e-5, description="Learning rate")
    weight_decay: float = Field(1e-5, description="Weight decay for regularization")
    bias: bool = Field(False, description="Use bias in layers")

    @validator('dropout_rate')
    def validate_dropout_rate(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Dropout rate must be between 0 and 1')
        return v

    @validator('learning_rate', 'weight_decay')
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError('Learning rate and weight decay must be positive')
        return v


class DeepSVDDConfig(BaseModel):
    """Configuration for Deep SVDD training."""
    learning_rate: float = Field(1e-5, description="Learning rate for Deep SVDD")
    weight_decay: float = Field(1e-5, description="Weight decay for Deep SVDD")

    @validator('learning_rate', 'weight_decay')
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError('Learning rate and weight decay must be positive')
        return v


class TrainerConfig(BaseModel):
    """Configuration for model training."""
    wandb_project_name: str = Field(..., description="Weights & Biases project name")
    batch_size: int = Field(32, description="Training batch size")
    max_epochs: int = Field(1000, description="Maximum training epochs")
    min_epochs: Optional[int] = Field(None, description="Minimum training epochs")
    patience: int = Field(10, description="Early stopping patience")
    wandb_log_model: bool = Field(False, description="Log model to W&B")
    enable_progress_bar: bool = Field(False, description="Show training progress bar")
    deterministic: bool = Field(False, description="Use deterministic training")

    @validator('batch_size', 'max_epochs', 'patience')
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError('Batch size, max epochs, and patience must be positive')
        return v


class AudioPreprocessingConfig(BaseModel):
    """Configuration for audio preprocessing."""
    sample_rate: int = Field(32000, description="Audio sample rate (Hz)")
    target_audio_length: int = Field(10, description="Target audio length (seconds)")
    mono: bool = Field(True, description="Convert to mono audio")
    only_pad: bool = Field(False, description="Only pad audio, don't repeat")

    @validator('sample_rate', 'target_audio_length')
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError('Sample rate and target length must be positive')
        return v


class ThresholdConfig(BaseModel):
    """Pre-computed anomaly detection thresholds."""
    pass  # Dynamic fields for different model-dataset combinations


class CheckpointPaths(BaseModel):
    """Model checkpoint paths for inference."""
    ae_checkpoint: Optional[str] = Field(None, description="AutoEncoder checkpoint path")
    svdd_checkpoint: Optional[str] = Field(None, description="Deep SVDD checkpoint path")


class CheckpointConfig(BaseModel):
    """Checkpoint configuration with experiment discovery and manual paths."""
    experiment_name: Optional[str] = Field(None, description="Experiment name for automatic checkpoint discovery")
    manual_paths: Optional[Dict[str, str]] = Field(None, description="Manual checkpoint path specifications")
    
    class Config:
        extra = "allow"  # Allow additional dynamic checkpoint fields


class ExperimentConfig(BaseModel):
    """Main experiment configuration."""
    config_name: str = Field(..., description="Experiment configuration name")
    networks: Dict[str, NetworkConfig] = Field(..., description="Network configurations")
    deepSVDD: DeepSVDDConfig = Field(..., description="Deep SVDD configuration")
    trainer: TrainerConfig = Field(..., description="Training configuration")
    dataset_paths: Dict[str, str] = Field(..., description="Dataset file paths")
    audio_preprocessing: Optional[AudioPreprocessingConfig] = Field(None, description="Audio preprocessing config")
    threshold: Optional[Dict[str, float]] = Field(None, description="Pre-computed thresholds")
    checkpoints: Optional[CheckpointConfig] = Field(None, description="Model checkpoint configuration")
    
    class Config:
        extra = "allow"  # Allow additional fields like threshold values