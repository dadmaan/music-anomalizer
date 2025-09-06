import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from ..models.networks import create_network, AutoEncoder
from ..models.layers import LinearLayer, Encoder, Decoder  
from ..models.base_models import isolation_forest, pca_reconstruction_error
from ..models.anomaly_detector import AnomalyDetector


class TestNetworkCreation:
    """Test basic network creation functionality."""
    
    def test_create_network_basic(self, mock_ae_config):
        """Test creating AutoEncoder with valid config."""
        model = create_network(mock_ae_config)
        
        assert isinstance(model, AutoEncoder)
        assert model.num_features == 64
        assert model.hidden_dims == [32, 16, 8]
    
    def test_create_network_invalid_config(self):
        """Test network creation with invalid config."""
        with pytest.raises(ValueError, match="Config must be a dictionary"):
            create_network("invalid")
    
    def test_network_forward_pass(self, mock_ae_config):
        """Test basic forward pass through network."""
        model = create_network(mock_ae_config)
        x = torch.randn(5, 64)
        
        output = model(x)
        
        assert output.shape == (5, 64)
        assert not torch.isnan(output).any()


class TestLayers:
    """Test individual layer components."""
    
    def test_linear_layer_forward(self):
        """Test LinearLayer forward pass."""
        layer = LinearLayer(64, 32, torch.nn.ReLU(), bias=False)
        x = torch.randn(10, 64)
        
        output = layer(x)
        
        assert output.shape == (10, 32)
        assert (output >= 0).all()  # ReLU activation
    
    def test_encoder_decoder_consistency(self):
        """Test Encoder-Decoder dimension consistency."""
        hidden_dims = [32, 16, 8]
        encoder = Encoder(64, hidden_dims, torch.nn.ReLU(), False, None, False)
        decoder = Decoder(hidden_dims, 64, torch.nn.ReLU(), False, None, False)
        
        x = torch.randn(5, 64)
        encoded = encoder(x)
        decoded = decoder(encoded)
        
        assert encoded.shape == (5, 8)  # Last hidden dim
        assert decoded.shape == (5, 64)  # Original input dim


class TestBaseModels:
    """Test baseline anomaly detection models."""
    
    def test_isolation_forest_basic(self, synthetic_audio_features):
        """Test Isolation Forest with synthetic data."""
        normal_data = synthetic_audio_features['normal']
        mixed_data = synthetic_audio_features['mixed']
        
        train_scores, eval_scores = isolation_forest(normal_data, mixed_data, random_state=42)
        
        assert len(train_scores) == 50
        assert len(eval_scores) == 60  # 50 normal + 10 anomalous
        assert isinstance(train_scores, np.ndarray)
        assert isinstance(eval_scores, np.ndarray)
    
    def test_pca_reconstruction_basic(self, synthetic_audio_features):
        """Test PCA reconstruction with synthetic data."""
        normal_data = synthetic_audio_features['normal']
        mixed_data = synthetic_audio_features['mixed']
        
        train_scores, eval_scores = pca_reconstruction_error(
            normal_data, mixed_data, n_components=10, standardize=True
        )
        
        assert len(train_scores) == 50
        assert len(eval_scores) == 60
        assert np.all(train_scores >= 0)  # Reconstruction errors are non-negative
        assert np.all(eval_scores >= 0)


class TestAnomalyDetector:
    """Test AnomalyDetector main functionality."""
    
    def test_anomaly_detector_init(self, mock_ae_config, mock_svdd_config):
        """Test AnomalyDetector initialization."""
        configs = (mock_ae_config, mock_svdd_config)
        checkpoint_paths = ['/fake/ae.ckpt', '/fake/svdd.ckpt']
        
        detector = AnomalyDetector(configs, checkpoint_paths, 'cpu')
        
        assert detector.device == 'cpu'
        assert len(detector.checkpoint_paths) == 2
        assert detector.ae_model is None  # Not loaded yet
        assert detector.svdd_model is None
    
    def test_models_not_loaded_error(self, mock_ae_config, mock_svdd_config):
        """Test error when trying to use detector without loading models."""
        configs = (mock_ae_config, mock_svdd_config)
        detector = AnomalyDetector(configs, ['/fake/ae.ckpt', '/fake/svdd.ckpt'], 'cpu')
        
        dummy_data = [np.random.randn(64).astype(np.float32) for _ in range(5)]
        
        with pytest.raises(RuntimeError, match="Models not loaded"):
            detector.compute_anomaly_scores(dummy_data)
    
    @patch('music_anomalizer.models.anomaly_detector.load_AE_from_checkpoint')
    @patch('music_anomalizer.models.anomaly_detector.SVDD')
    @patch('music_anomalizer.models.anomaly_detector.load_pickle')
    @patch('pathlib.Path.exists')
    def test_load_models_success(self, mock_exists, mock_load_pickle, mock_svdd, 
                                mock_load_ae, mock_ae_config, mock_svdd_config, mock_center_vector):
        """Test successful model loading with mocks."""
        # Setup mocks
        mock_exists.return_value = True
        mock_ae_model = Mock()
        mock_ae_model.encoder = Mock()
        mock_load_ae.return_value = mock_ae_model
        mock_load_pickle.return_value = mock_center_vector
        mock_svdd_model = Mock()
        mock_svdd.load_from_checkpoint.return_value = mock_svdd_model
        
        configs = (mock_ae_config, mock_svdd_config)
        detector = AnomalyDetector(configs, ['/fake/ae.ckpt', '/fake/svdd.ckpt'], 'cpu')
        
        # This should not raise an exception
        detector.load_models()
        
        assert detector.ae_model is not None
        assert detector.svdd_model is not None  
        assert detector.z_vector is not None
    
    def test_get_extreme_anomalies(self, mock_ae_config, mock_svdd_config):
        """Test extreme anomaly detection functionality."""
        detector = AnomalyDetector((mock_ae_config, mock_svdd_config), ['/fake1', '/fake2'], 'cpu')
        
        # Mock anomaly scores - some high, some low
        scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5]
        
        extremes = detector.get_extreme_anomalies(scores, n_samples=3)
        
        assert 'most_anomalous' in extremes
        assert 'least_anomalous' in extremes
        assert len(extremes['most_anomalous']) == 3
        assert len(extremes['least_anomalous']) == 3
        
        # Verify most anomalous have higher scores than least anomalous
        most_scores = [score for _, score in extremes['most_anomalous']]
        least_scores = [score for _, score in extremes['least_anomalous']]
        assert min(most_scores) > max(least_scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])