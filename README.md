# Music Anomalizer 🎵

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-orange.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code implementation and supplementary materials for the paper **"Learning Normal Patterns in Musical Samples: Unsupervised Anomaly Detection for Variable-Length Audio Inputs"**.

This project implements a **Deep Support Vector Data Description (Deep SVDD)** framework for identification of musical patterns that deviate from normal patterns within training data.

## 🚀 Key Features

- **🎯 Unsupervised Anomaly Detection**: Deep SVDD-based models for musical audio loop analysis
- **🧠 Multiple Neural Architectures**: AutoEncoders with residual connections, compact architectures, and baseline models
- **🎵 Audio Processing Pipeline**: Complete WAV preprocessing with HTSAT embeddings and feature extraction
- **📊 Interactive Web Interface**: Streamlit-based application for real-time audio analysis
- **⚡ Training**: PyTorch Lightning integration with WandB logging
- **🔧 Flexible Configuration**: YAML-based experiment configuration system

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Dataset Requirements](#dataset-requirements)
- [Training](#training)
- [Web Interface](#web-interface)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- FFmpeg (for audio processing)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-username/music-anomalizer.git
cd music-anomalizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .

```

### Docker Installation

```bash
# Build the Docker image
docker-compose build

# Run the application
docker-compose up
```

## 🚀 Quick Start

### 1. Preprocess Audio Data

```bash
# Process WAV files with metadata
python music_anomalizer/scripts/preprocess_wav_loops.py \
    --wav_dir data/audio_files \
    --metadata_path data/metadata.json \
    --output_path processed_data.pkl \
    --sample_rate 32000 \
    --target_length 10
```

### 2. Train a Model

```bash
# Train using the simple interface
python train.py \
    --dataset processed_data.pkl \
    --network AE \
    --model-name my_anomaly_detector \
    --epochs 500 \
    --batch-size 32
```

### 3. Compute Anomaly Scores

```bash
# Analyze new audio files
python music_anomalizer/scripts/compute_anomaly_scores.py \
    --model_path models/my_anomaly_detector \
    --test_data test_data.pkl \
    --output_path results.json
```

### 4. Launch Web Interface

```bash
# Start the Streamlit app
streamlit run app/pages/Home.py
```

## 📚 Documentation

Find relevant documentations in `docs/` directory.

### Audio Processing Pipeline

```
WAV Files → Feature Extraction → CLAP Embeddings → Deep SVDD → Anomaly Scores
```

1. **Audio Preprocessing**: Normalize, resample, and segment audio
2. **Feature Extraction**: HTSAT embeddings
3. **Deep SVDD Training**: Learn normal pattern representations
4. **Anomaly Detection**: Compute distances from learned center

## 📖 Usage

### Training Custom Models

```python
from music_anomalizer.models.deepSVDD import DeepSVDDTrainer
from music_anomalizer.config import load_experiment_config

# Load configuration
config = load_experiment_config("configs/exp1.yaml")

# Initialize trainer
trainer = DeepSVDDTrainer(
    AE_config=config['AE_config'],
    SVDD_config=config['SVDD_config'],
    dataset={"train": "data/train.pkl"},
    device="cuda"
)

# Train the model
trainer.pretraining("my_model", "train", train_data, val_data, 
                   max_epochs=500, patience=10)
trainer.train_deepSVDD("my_model", "train", train_data, val_data, 
                      max_epochs=1000, patience=20)
```

### Computing Anomaly Scores

```python
from music_anomalizer.models.anomaly_detector import AnomalyDetector

# Load trained model
detector = AnomalyDetector()
detector.load_model("path/to/model.ckpt")

# Compute scores for new data
scores = detector.compute_anomaly_scores(test_dataset)
anomalies = detector.predict_anomalies(test_dataset, threshold=0.8)
```

### Batch Processing

```python
# Process multiple audio files
results = {}
for audio_file in audio_files:
    features = extract_features(audio_file)
    score = detector.compute_anomaly_scores([features])
    results[audio_file] = score[0]
```

## 🗃️ Dataset Requirements

### Input Data Format

The system expects preprocessed data in pickle format with the following structure:

```python
{
    'embeddings': np.ndarray,  # Shape: (n_samples, feature_dim)
    'labels': List[str],       # Optional: ground truth labels
    'metadata': Dict           # Additional sample information
}
```

### Audio Requirements

- **Format**: WAV files (16-bit or 24-bit)
- **Sample Rate**: 32 kHz (configurable)
- **Duration**: Variable length (automatically processed to target length)
- **Channels**: Mono (stereo files converted automatically)

### Metadata Format

```json
{
    "sample_id_1": {
        "file_path": "audio/drum_loop_1.wav",
        "bpm": 120,
        "key": "C",
        "genre": ["electronic", "dance"],
        "tags": ["drum", "loop", "4/4"]
    },
    "sample_id_2": {
        "file_path": "audio/bass_line_2.wav",
        "bpm": 128,
        "key": "Am",
        "genre": ["house"],
        "tags": ["bass", "synth", "loop"]
    }
}
```

## 🎯 Training

### Single Model Training

Use the simplified `train.py` script for individual model training:

```bash
# Basic training
python train.py --dataset data.pkl --network AE

# Advanced configuration
python train.py \
    --dataset data.pkl \
    --network AEwRES \
    --model-name advanced_detector \
    --epochs 1000 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --wandb-project "MusicAnomalyDetection"
```

### Experiment Management

For complex experiments, use the configuration system:

```bash
# Run predefined experiments
python music_anomalizer/scripts/train_models.py --config configs/exp1.yaml
python music_anomalizer/scripts/train_models.py --config configs/exp2_deeper.yaml
```

### Available Network Types

| Network | Description | Use Case |
|---------|-------------|----------|
| `AE` | Standard AutoEncoder | Baseline experiments |
| `AEwRES` | AutoEncoder with Residual | Complex patterns |
| `Baseline` | Simple baseline model | Comparison studies |

### Hyperparameter Tuning

```bash
# Run hyperparameter optimization
python music_anomalizer/scripts/hp_tuning_loop_detection.py \
    --dataset data.pkl \
    --trials 50 \
    --config configs/hp_config.yaml
```

## 🌐 Web Interface

The project includes a Streamlit-based web application for basic interactive analysis:

### Features
- **🎵 Audio Upload**: Drag-and-drop interface for audio files
- **📊 Real-time Analysis**: Instant anomaly score computation
- **📈 Visualization**: Interactive plots and spectrograms

### Launching the App

```bash
streamlit run app/pages/Home.py
```

Navigate to `http://localhost:8501` to access the interface.

### Available Pages
- **Home**: Overview and quick analysis
- **Overview**: Model architecture and methodology
- **Upload & Analyze**: Detailed audio file analysis
- **Visualization**: Advanced plotting and comparison tools

## ⚙️ Configuration

The system uses YAML configuration files for experiment management:

### Base Configuration (`configs/base.yaml`)

```yaml
# Deep SVDD parameters
deepSVDD:
  learning_rate: 0.00001
  weight_decay: 0.00001

# Training configuration
trainer:
  batch_size: 32
  max_epochs: 1000
  patience: 10

# Network defaults
network_defaults:
  activation_fn: "ELU"
  learning_rate: 0.00001
  weight_decay: 0.00001
  bias: false

# Audio preprocessing
audio_preprocessing:
  sample_rate: 32000
  target_audio_length: 10
  mono: true
```

### Custom Experiments

Create custom configuration files by extending the base configuration:

```yaml
# configs/my_experiment.yaml
base: "base.yaml"

# Override specific parameters
trainer:
  batch_size: 64
  max_epochs: 2000

# Add experiment-specific networks
networks:
  - name: "CustomAE"
    type: "AE"
    hidden_dims: [1024, 512, 256, 128]
```

## 📁 Project Structure

```
music-anomalizer/
├── 📁 music_anomalizer/          # Main package
│   ├── 📁 models/                # Neural network models
│   │   ├── networks.py           # AutoEncoder architectures
│   │   ├── deepSVDD.py          # Deep SVDD trainer
│   │   ├── anomaly_detector.py   # High-level detection interface
│   │   └── layers.py             # Custom neural network layers
│   ├── 📁 preprocessing/         # Data preprocessing
│   │   ├── extract_embed.py      # Feature extraction
│   │   └── wav2embed.py          # Audio-to-embedding conversion
│   ├── 📁 data/                  # Data loading utilities
│   ├── 📁 config/               # Configuration management
│   ├── 📁 scripts/              # Training and evaluation scripts
│   └── 📁 visualization/        # Plotting and analysis tools
├── 📁 app/                      # Streamlit web interface
│   └── 📁 pages/                # Web app pages
├── 📁 configs/                  # YAML configuration files
├── 📄 train.py                  # Simple training script
├── 📄 prepare_data.py           # Data preparation utilities
└── 📄 README.md                 # This file
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dadman2025learningnormalpatternsmusical,
    title={Learning Normal Patterns in Musical Loops}, 
    author={Shayan Dadman and Bernt Arild Bremdal and Børre Bang and Rune Dalmo},
    year={2025},
    eprint={2505.23784},
    archivePrefix={arXiv},
    primaryClass={cs.SD},
    url={https://arxiv.org/abs/2505.23784},
    note={Code available at: https://github.com/your-username/music-anomalizer}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**🎵 Happy anomaly detecting! 🎵**

