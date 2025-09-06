# Music Anomalizer Technical Reference

This document provides detailed technical information about the algorithms, models, and implementation details of the Music Anomalizer system.

## Table of Contents
1. [Model Architectures](#model-architectures)
2. [Training Process](#training-process)
3. [Anomaly Detection](#anomaly-detection)
4. [Data Processing](#data-processing)
5. [Configuration Reference](#configuration-reference)

## Model Architectures

### AutoEncoder Networks

Music Anomalizer uses AutoEncoder networks for representation learning. Two main variants are implemented:

#### Standard AutoEncoder (AE)

A standard feedforward neural network with an encoder-decoder structure:

```python
class AutoEncoder(BaseAutoEncoder):
    def __init__(self, num_features, hidden_dims, activation_fn="ELU", 
                 dropout_rate=0.2, use_batch_norm=True, learning_rate=1e-5, 
                 weight_decay=1e-5, bias=False):
        super().__init__()
        # Encoder with progressive dimension reduction
        self.encoder = Encoder(num_features, hidden_dims, activation_fn, 
                              bias, dropout_rate, use_batch_norm)
        # Decoder with symmetric expansion
        self.decoder = Decoder(hidden_dims, num_features, activation_fn, 
                              bias, dropout_rate, use_batch_norm)
```

#### AutoEncoder with Residual Connections (AEwRES)

An enhanced version with skip connections to improve gradient flow:

```python
class AutoEncoderWithResidual(BaseAutoEncoder):
    def __init__(self, num_features, hidden_dims, activation_fn="ELU", 
                 dropout_rate=0.2, use_batch_norm=True, learning_rate=1e-5, 
                 weight_decay=1e-5, bias=False):
        super().__init__()
        # Encoder with residual connections
        self.encoder = ResidualEncoder(num_features, hidden_dims, activation_fn, 
                                      bias, dropout_rate, use_batch_norm)
        # Decoder with residual connections
        self.decoder = ResidualDecoder(hidden_dims, num_features, activation_fn, 
                                      bias, dropout_rate, use_batch_norm)
```

### Deep Support Vector Data Description (Deep SVDD)

Deep SVDD is used for anomaly detection. It learns a hypersphere in the latent space:

```python
class DeepSVDD(pl.LightningModule):
    def __init__(self, ae_model, learning_rate=1e-5, weight_decay=1e-5):
        super().__init__()
        self.ae_model = ae_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # Hypersphere center (learned during training)
        self.register_buffer('center', torch.zeros(ae_model.latent_dim))
```

The loss function minimizes the distance of normal data points to the hypersphere center:

```python
def svdd_loss(self, z):
    """Compute Deep SVDD loss."""
    # Distance to center
    dist = torch.sum((z - self.center) ** 2, dim=1)
    # Minimize distances for normal data
    return torch.mean(dist)
```

## Training Process

### AutoEncoder Pretraining

AutoEncoders are pretrained to learn meaningful representations:

1. **Data Preparation**: Audio files are converted to embeddings using CLAP
2. **Model Initialization**: Network architecture is configured based on hyperparameters
3. **Training Loop**: 
   - Forward pass through encoder and decoder
   - Reconstruction loss computation (MSE)
   - Backpropagation and parameter updates

```python
def training_step(self, batch, batch_idx):
    x, _ = batch  # Input features, labels (ignored for AE)
    # Forward pass
    x_reconstructed = self(x)
    # Reconstruction loss
    loss = F.mse_loss(x_reconstructed, x)
    self.log('train_loss', loss)
    return loss
```

### Deep SVDD Training

After AutoEncoder pretraining, Deep SVDD is trained:

1. **Center Initialization**: Compute initial center as mean of encoded normal data
2. **Training Loop**:
   - Encode data using pretrained AutoEncoder
   - Compute distance to hypersphere center
   - Update model parameters to minimize distances

```python
def training_step(self, batch, batch_idx):
    x, _ = batch
    # Encode using pretrained AE
    with torch.no_grad():
        z = self.ae_model.encoder(x)
    # Compute SVDD loss
    loss = self.svdd_loss(z)
    self.log('train_loss', loss)
    return loss
```

### Network Variants

Five network architectures are available:

1. **AE**: Standard AutoEncoder with regularization
2. **AEwRES**: AutoEncoder with residual connections
3. **Baseline**: AutoEncoder without regularization
4. **DeepAE**: Deep 5-layer AutoEncoder
5. **CompactAE**: Compact 2-layer AutoEncoder

## Anomaly Detection

### Score Computation

Anomaly scores are computed based on the distance in latent space:

```python
def compute_anomaly_score(self, embedding):
    """Compute anomaly score for a single embedding."""
    # Encode using AutoEncoder
    with torch.no_grad():
        z = self.ae_model.encoder(embedding)
    # Compute distance to SVDD center
    distance = torch.sum((z - self.center) ** 2, dim=1)
    return distance.item()
```

### Threshold-Based Classification

Precomputed thresholds determine classification:

```python
def is_anomaly(self, score, threshold):
    """Classify based on threshold."""
    return score > threshold
```

### Batch Processing Optimization

For efficiency, embeddings are processed in batches:

```python
def compute_anomaly_scores(self, dataset, batch_size=None):
    """Compute anomaly scores with optimized batching."""
    if batch_size is None:
        batch_size = self._determine_optimal_batch_size(dataset)
    
    scores = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batch_scores = self._process_batch(batch)
        scores.extend(batch_scores)
    return scores
```

## Data Processing

### Audio Preprocessing

Audio files are preprocessed to a consistent format:

1. **Resampling**: Convert to 32kHz sample rate
2. **Mono Conversion**: Convert to mono channel
3. **Length Adjustment**: Pad or truncate to 10 seconds

```python
def preprocess_audio(self, audio_file):
    """Preprocess audio file to consistent format."""
    # Load audio
    audio, sr = librosa.load(audio_file, sr=None)
    # Resample to target rate
    if sr != self.target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
    # Convert to mono
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)
    # Adjust length
    audio = self._adjust_length(audio)
    return audio
```

### Embedding Extraction

Audio embeddings are extracted using the CLAP model:

1. **Model Loading**: Load pretrained CLAP model
2. **Feature Extraction**: Convert audio to embeddings
3. **Normalization**: Normalize embeddings for consistency

```python
def extract_embedding(self, audio):
    """Extract CLAP embedding from audio."""
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio).float()
    # Extract embedding
    with torch.no_grad():
        embedding = self.clap_model.get_audio_embedding_from_data(
            audio_tensor, use_tensor=True
        )
    return embedding.cpu().numpy()
```

### Data Validation

Comprehensive validation ensures data quality:

```python
def validate_dataset(self, dataset_path):
    """Validate dataset file and contents."""
    # File existence check
    if not os.path.exists(dataset_path):
        return False, f"Dataset file not found: {dataset_path}"
    
    # File format check
    try:
        data = load_pickle(dataset_path)
    except Exception as e:
        return False, f"Failed to load dataset: {e}"
    
    # Content validation
    if not isinstance(data, list):
        return False, "Dataset should be a list of embeddings"
    
    if len(data) == 0:
        return False, "Dataset is empty"
    
    return True, "Dataset is valid"
```

## Configuration Reference

### Experiment Configuration

YAML-based configuration files define experiment parameters:

```yaml
# Base configuration (inherited by all experiments)
config_name: exp2_deeper

# Network architectures
networks:
  AE:
    class_name: AutoEncoder
    hidden_dims: [512, 256, 128, 64, 32]
    dropout_rate: 0.2
    use_batch_norm: true
  
  AEwRES:
    class_name: AutoEncoderWithResidual
    hidden_dims: [512, 256, 128, 64, 32]
    dropout_rate: 0.2
    use_batch_norm: true

# Dataset paths
dataset_paths:
  HTSAT_base_musicradar_bass: "data/pickle/embedding/musicradar/HTSAT-base_musicradar_bass_embeddings.pkl"
  HTSAT_base_musicradar_guitar: "data/pickle/embedding/musicradar/HTSAT-base_musicradar_guitar_embeddings.pkl"

# Precomputed thresholds
threshold:
  AEwRES_bass: 0.004654786549508572
  AEwRES_guitar: 0.03549588881432997
  AE_bass: 0.005436893552541733
  AE_guitar: 0.0010853302665054798

# Checkpoint management
checkpoints:
  experiment_name: "EXP2_DEEPER"
  manual_paths:
    AEwRES_bass_ae: "./checkpoints/loop_benchmark/EXP2_DEEPER/AEwRES/AEwRES-HTSAT_base_musicradar_bass-AE-epoch=203-val_loss=0.01.ckpt"
    # ... additional paths
```

### Training Configuration

Training parameters are defined in the configuration:

```yaml
# Training settings
trainer:
  batch_size: 32
  max_epochs: 1000
  patience: 10
  wandb_project_name: "LOOP-DSVDD-EXP2-DEEPER"
  wandb_log_model: false
  enable_progress_bar: false
```

### Network Configuration Schema

Pydantic schemas ensure configuration validity:

```python
class NetworkConfig(BaseModel):
    class_name: str
    hidden_dims: List[int]
    dropout_rate: Optional[float] = None
    use_batch_norm: bool = False
    activation_fn: str = "ELU"
    learning_rate: float = 1e-5
    weight_decay: float = 1e-5
    bias: bool = False

class ExperimentConfig(BaseModel):
    config_name: str
    base_config: Optional[str] = None
    networks: Dict[str, NetworkConfig]
    dataset_paths: Dict[str, str]
    threshold: Dict[str, float]
    checkpoints: CheckpointConfig
    trainer: TrainerConfig
    deepSVDD: DeepSVDDConfig
```

This technical reference provides detailed information about the implementation of Music Anomalizer. For usage instructions, please refer to the User Guide, and for development information, see the Developer Guide.
