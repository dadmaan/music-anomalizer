import pytest
import numpy as np
import torch
from pathlib import Path


@pytest.fixture
def synthetic_audio_features():
    """Generate synthetic audio feature embeddings for testing."""
    np.random.seed(42)
    
    # Normal features - tighter distribution
    normal_features = np.random.normal(0, 0.5, (50, 64)).astype(np.float32)
    
    # Anomalous features - wider distribution  
    anomalous_features = np.random.normal(0, 2.0, (10, 64)).astype(np.float32)
    
    return {
        'normal': normal_features,
        'anomalous': anomalous_features,
        'mixed': np.vstack([normal_features, anomalous_features])
    }


@pytest.fixture
def mock_ae_config():
    """Basic AutoEncoder configuration for testing."""
    return {
        'class_name': 'AutoEncoder',
        'num_features': 64,
        'train_data_length': 100,
        'hidden_dims': [32, 16, 8],
        'activation_fn': 'ReLU',
        'dropout_rate': 0.1,
        'use_batch_norm': False,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'bias': False
    }


@pytest.fixture  
def mock_svdd_config():
    """Basic SVDD configuration for testing."""
    return {
        'learning_rate': 1e-3,
        'weight_decay': 1e-5
    }


@pytest.fixture
def mock_center_vector():
    """Mock center vector for SVDD testing."""
    return torch.randn(8, dtype=torch.float32)