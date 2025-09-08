# Music Anomalizer User Guide

Welcome to the Music Anomalizer User Guide! This guide will help you get started with using the Music Anomalizer system for detecting anomalies in musical audio loops using deep learning techniques.

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Web Interface Guide](#web-interface-guide)
4. [Command-Line Tools](#command-line-tools)
5. [Configuration System](#configuration-system)
6. [Troubleshooting](#troubleshooting)

## Introduction

Music Anomalizer is a sophisticated system that uses deep learning models to identify whether audio loops are similar to training data patterns. It employs several advanced techniques:

- **HTSAT-base**: Audio feature extraction from CLAP model
- **AutoEncoder with Residual Connections (AEwRES)**: Representation learning
- **Deep Support Vector Data Description (Deep SVDD)**: Anomaly detection

The system can work with different types of musical instruments, currently supporting bass and guitar loops.

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support)
- Audio files in WAV format for analysis

### Running the Web Interface

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/music-anomalizer.git
   cd music-anomalizer
   ```

2. Start the Streamlit web application:
   ```bash
   docker-compose up app
   ```

3. Access the application at http://localhost:8501

### Running with GPU Support

For faster processing, use GPU acceleration:
```bash
docker-compose up app
```

Make sure you have NVIDIA Docker runtime installed for GPU support.

## Web Interface Guide

The Music Anomalizer web interface consists of two main pages:

### 1. Overview Page

This page allows you to:
- View examples from the training dataset
- Listen to the most "normal" and most "anomalous" loops
- Understand what the model considers typical patterns

Features:
- **Top 3 'Normal' Training Loops**: Shows the loops most similar to training data
- **Lowest 3 'Normal' Training Loops**: Shows the loops least similar to training data
- **Configuration Selection**: Choose between different experiment configurations and network types

### 2. Upload & Analyze Page

This page allows you to:
- Upload your own audio files for analysis
- Compare your audio against the trained models
- Visualize results in the latent space

Features:
- **Audio Upload**: Upload WAV files for analysis
- **Anomaly Detection**: Determine if your audio is similar to training data
- **Latent Space Visualization**: 2D/3D visualization of training data and your analyzed file
- **Interactive Exploration**: Click on data points to listen to corresponding audio files

## Command-Line Tools

Music Anomalizer provides several command-line tools for users and batch processing:

### Data Preparation

Prepare your audio data for training:
```bash
python prepare_data.py --audio-dir data/bass_loops
```

### Training Models

Train a single model:
```bash
python train.py --dataset data.pkl --network AE
```

Train multiple models with experiment configurations:
```bash
python music_anomalizer/scripts/train_models.py --config exp2_deeper
```

### Anomaly Score Computation

Compute anomaly scores for training datasets:
```bash
python music_anomalizer/scripts/compute_anomaly_scores.py --config exp2_deeper --model-type bass
```

### Loop Detection

Analyze individual audio files:
```bash
python loop_detector.py audio.wav --model bass
```

### Embedding Extraction

Extract audio embeddings from datasets:
```bash
python music_anomalizer/scripts/embedding_extraction_wav.py --dataset data/dataset/guitar --output data/embeddings
```

## Configuration System

Music Anomalizer uses a flexible YAML-based configuration system:

### Experiment Configurations

Available configurations:
- `exp1`: Initial experiment configuration
- `exp2_deeper`: Deep networks with 5-layer architectures
- `single_model`: Single model configuration

### Network Types

Available network architectures:
- `AE`: Standard AutoEncoder with regularization (recommended)
- `AEwRES`: AutoEncoder with residual connections (for complex patterns)
- `Baseline`: AutoEncoder without regularization (for comparison)
- `DeepAE`: Deep 5-layer AutoEncoder (for complex datasets)
- `CompactAE`: Compact 2-layer AutoEncoder (for smaller datasets)

Note that `DeepAE` and `CompactAE` are experimental.



## Troubleshooting

### Common Issues

1. **Docker Container Fails to Start**
   - Ensure Docker and Docker Compose are properly installed
   - Check that NVIDIA Docker runtime is installed for GPU support
   - Verify sufficient system resources (RAM, disk space)

2. **Model Loading Errors**
   - Ensure checkpoint files exist in the correct locations
   - Check that configuration files reference valid paths
   - Verify that dataset files are properly formatted

3. **Audio Processing Issues**
   - Ensure audio files are in WAV format
   - Check that audio files have sufficient length (minimum 10 seconds recommended)
   - Verify that audio files contain actual audio data (not silent)

### Getting Help

If you encounter issues not covered in this guide:
1. Check the existing documentation files in the `docs/` directory
2. Review the logs from Docker containers for error messages
3. Open an issue on the GitHub repository with detailed information about the problem

