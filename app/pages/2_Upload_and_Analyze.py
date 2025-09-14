import streamlit as st
import os
import torch
import pandas as pd
import io
import numpy as np
import tempfile
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Import music_anomalizer core modules
from music_anomalizer.utils import (
    load_pickle, get_anomaly_scores_manager, setup_logging, 
    initialize_device, validate_file_path
)
from music_anomalizer.config.loader import load_experiment_config
from music_anomalizer.config.checkpoint_manager import get_checkpoint_registry
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.preprocessing.wav2embed import Wav2Embedding
from music_anomalizer.visualization.visualizer import LatentSpaceVisualizer

# Setup logging
logger = setup_logging("INFO")

# Paths (relative to this file - go up two levels: pages -> app -> root)
BASE_DIR = Path(__file__).parent.parent.parent
# Available experiment configurations
AVAILABLE_CONFIGS = ['exp1', 'exp2_deeper', 'single_model']
DEFAULT_CONFIG = 'exp2_deeper'

# Initialize checkpoint registry
checkpoint_registry = get_checkpoint_registry()

def get_config_path(config_name=DEFAULT_CONFIG):
    """Get config path for selected experiment."""
    return BASE_DIR / 'configs' / f'{config_name}.yaml'

CLAP_CKPT = BASE_DIR / 'checkpoints' / 'laion_clap' / 'music_audioset_epoch_15_esc_90.14.pt'

def get_available_networks(config_name):
    """Get available network types from config."""
    try:
        config = load_experiment_config(config_name, str(BASE_DIR / 'configs'))
        return list(config.networks.keys())
    except Exception as e:
        logger.error(f"Failed to load network types: {e}")
        return ['AEwRES']  # fallback

@st.cache_data
def get_model_choices(config_name=DEFAULT_CONFIG, network_key='AEwRES'):
    """Load model configuration and checkpoint paths using proper config management."""
    try:
        # Load experiment configuration
        config_path = str(get_config_path(config_name))
        config = load_experiment_config(config_name, str(BASE_DIR / 'configs'))
        
        # Get checkpoints from registry or manual paths
        experiment_name = config.config_name.upper()
        
        def get_checkpoint_paths(model_type: str, net_key: str = None):
            """Get checkpoint paths for a model type using checkpoint registry."""
            if net_key is None:
                net_key = network_key
            try:
                # Try to get checkpoints from registry first
                checkpoints = checkpoint_registry.get_model_checkpoints(
                    experiment_name, net_key, f'HTSAT_base_musicradar_{model_type}'
                )
                return {
                    'ae_ckpt': checkpoints.get(f'{net_key}-HTSAT_base_musicradar_{model_type}-AE'),
                    'svdd_ckpt': checkpoints.get(f'{net_key}-HTSAT_base_musicradar_{model_type}-DSVDD'),
                }
            except (KeyError, FileNotFoundError) as e:
                logger.warning(f"Checkpoint registry failed for {model_type}: {e}")
                # Fallback to manual paths from config
                manual_paths = config.checkpoints.manual_paths if config.checkpoints else {}
                
                def resolve_path(path_key, default_path):
                    path = manual_paths.get(path_key, default_path)
                    if isinstance(path, str) and path.startswith('./'):
                        path = path[2:]
                    return str(BASE_DIR / path)
                
                return {
                    'ae_ckpt': resolve_path(f'{net_key}_{model_type}_ae', 
                        f'checkpoints/loop_benchmark/{experiment_name}/{net_key}/{net_key}-HTSAT_base_musicradar_{model_type}-AE-epoch=149-val_loss=0.01.ckpt'),
                    'svdd_ckpt': resolve_path(f'{net_key}_{model_type}_svdd',
                        f'checkpoints/loop_benchmark/{experiment_name}/{net_key}/{net_key}-HTSAT_base_musicradar_{model_type}-DSVDD-epoch=132-val_loss=0.00.ckpt'),
                }
        
        model_choices = {}
        for model_type in ['bass', 'guitar']:
            checkpoints = get_checkpoint_paths(model_type, network_key)
            model_choices[model_type] = {
                **checkpoints,
                'threshold_key': f'{network_key}_{model_type}',
                'model_key': network_key,
            }
        
        return model_choices
    
    except Exception as e:
        logger.error(f"Failed to load model choices: {e}")
        st.error(f"Error loading model configuration: {e}")
        return {}

@st.cache_data
def load_anomaly_scores(model_type, config_name=DEFAULT_CONFIG, network_key='AEwRES'):
    """Load anomaly scores for a given model type, computing them if necessary."""
    manager = get_anomaly_scores_manager()
    
    def progress_callback(message, progress):
        # Use st.info to show progress messages in Streamlit
        if hasattr(st, '_current_progress_bar'):
            st._current_progress_bar.progress(progress, text=message)
        else:
            # Fallback for when no progress bar is available
            if progress < 1.0:
                st.info(f"⏳ {message}")
            else:
                st.success(f"✅ {message}")
    
    # Try to load scores, auto-computing if missing
    scores, error = manager.load_scores(
        model_type=model_type,
        config_name=config_name,
        network_key=network_key,
        auto_compute=True,
        progress_callback=progress_callback
    )
    
    if error:
        st.error(f"❌ Error loading anomaly scores: {error}")
        return []
    
    return scores

@st.cache_data
def load_training_data(model_type, config_name=DEFAULT_CONFIG, network_key='AEwRES'):
    """Load training embeddings and anomaly scores for a given model type."""
    try:
        config = load_experiment_config(config_name, str(BASE_DIR / 'configs'))
        
        # Load embeddings using proper config management
        dataset_path = config.dataset_paths.get(f'HTSAT_base_musicradar_{model_type}')
        if not dataset_path:
            raise ValueError(f"Dataset path not found for {model_type}")
        
        # Handle relative paths
        if dataset_path.startswith('./'):
            dataset_path = dataset_path[2:]
        full_dataset_path = BASE_DIR / dataset_path
        
        # Validate dataset file exists
        is_valid, error_msg = validate_file_path(str(full_dataset_path), "dataset")
        if not is_valid:
            raise FileNotFoundError(error_msg)
        
        embeddings = load_pickle(str(full_dataset_path))
        if embeddings is None:
            raise ValueError(f"Failed to load embeddings from {full_dataset_path}")
        
        # Load anomaly scores using the manager
        scores = load_anomaly_scores(model_type, config_name, network_key)
        
        logger.info(f"Loaded {len(embeddings)} embeddings and {len(scores)} scores for {model_type}")
        return embeddings, scores
        
    except Exception as e:
        logger.error(f"Failed to load training data for {model_type}: {e}")
        st.error(f"Error loading training data: {e}")
        return None, None

class StreamlitVisualizationService:
    """Service for creating Streamlit-compatible visualizations using core visualization modules."""
    
    @staticmethod
    def create_interactive_pca_plot(embeddings, scores, analyzed_embedding=None, 
                                  analyzed_score=None, plot_type="3D", point_size=1, 
                                  opacity=0.5, color_by="anomaly_score", threshold=None):
        """Create interactive PCA plot using music_anomalizer visualization with Plotly output."""
        try:
            from sklearn.decomposition import PCA
            
            # Normalize the embeddings
            scaler = MinMaxScaler(feature_range=(-1,1))
            embeddings_normalized = scaler.fit_transform(embeddings)
            
            # Perform PCA to reduce to 2D or 3D
            n_components = 3 if plot_type == "3D" else 2
            pca = PCA(n_components=n_components)
            pca.fit(embeddings_normalized)
            embeddings_pca = pca.transform(embeddings_normalized)
            
            # Normalize the analyzed embedding if provided
            analyzed_embedding_normalized = None
            if analyzed_embedding is not None:
                analyzed_embedding_normalized = scaler.transform(analyzed_embedding.reshape(1, -1))
            
            # Create DataFrame for plotting
            plot_data = {
                'x': embeddings_pca[:, 0],
                'y': embeddings_pca[:, 1],
                'anomaly_score': [s['anomaly_score'] for s in scores],
                'file_path': [s['file_path'] for s in scores],
                'index': range(len(scores))
            }
            
            if plot_type == "3D":
                plot_data['z'] = embeddings_pca[:, 2]
            
            df = pd.DataFrame(plot_data)
            
            # Determine coloring scheme
            if color_by == "threshold" and threshold is not None:
                df['color'] = ['blue' if score['anomaly_score'] <= threshold else 'red' for score in scores]
                color_column = 'color'
                color_title = 'Threshold Classification'
                color_kwargs = {
                    'color_discrete_map': {'blue': 'blue', 'red': 'red'}
                }
            else:
                color_column = 'anomaly_score'
                color_title = 'Anomaly Score'
                color_kwargs = {
                    'color_continuous_scale': 'Viridis'
                }
            
            # Create base plot
            common_kwargs = {
                'data_frame': df,
                'color': color_column,
                'title': f'{plot_type} PCA of Training Data ({len(embeddings)} samples)',
                'labels': {color_column: color_title},
                'hover_data': {
                    'anomaly_score': ':.6f', 
                    'file_path': True,
                    'index': False
                },
                'opacity': opacity,
                **color_kwargs
            }
            
            if plot_type == "3D":
                fig = px.scatter_3d(x='x', y='y', z='z', **common_kwargs)
            else:
                fig = px.scatter(x='x', y='y', **common_kwargs)
            
            # Add analyzed file point if provided
            if analyzed_embedding is not None and analyzed_score is not None:
                analyzed_pca = pca.transform(analyzed_embedding_normalized)
                
                if plot_type == "3D":
                    fig.add_trace(go.Scatter3d(
                        x=[analyzed_pca[0, 0]],
                        y=[analyzed_pca[0, 1]],
                        z=[analyzed_pca[0, 2]],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='diamond'),
                        name='Analyzed File',
                        hovertemplate=f'Analyzed File<br>Anomaly Score: {analyzed_score:.6f}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=[analyzed_pca[0, 0]],
                        y=[analyzed_pca[0, 1]],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='diamond'),
                        name='Analyzed File',
                        hovertemplate=f'Analyzed File<br>Anomaly Score: {analyzed_score:.6f}<extra></extra>'
                    ))
                
                # Add annotation
                fig.update_layout(
                    annotations=[
                        dict(
                            x=0.5, y=1.05,
                            xref="paper", yref="paper",
                            text=f"Analyzed File Anomaly Score: {analyzed_score:.6f}",
                            showarrow=False,
                            font=dict(size=14, color="red")
                        )
                    ]
                )
            
            # Update layout with PCA variance information
            variance_labels = {
                'x': f"PC1={pca.explained_variance_ratio_[0]*100:.2f}%",
                'y': f"PC2={pca.explained_variance_ratio_[1]*100:.2f}%"
            }
            
            if plot_type == "3D":
                variance_labels['z'] = f"PC3={pca.explained_variance_ratio_[2]*100:.2f}%"
                fig.update_layout(scene=dict(
                    xaxis_title=variance_labels['x'],
                    yaxis_title=variance_labels['y'],
                    zaxis_title=variance_labels['z']
                ))
            else:
                fig.update_layout(
                    xaxis_title=variance_labels['x'],
                    yaxis_title=variance_labels['y']
                )
            
            if color_by == "anomaly_score":
                fig.update_layout(coloraxis_colorbar=dict(title="Anomaly Score"))
            
            logger.info(f"Created {plot_type} PCA plot with {len(embeddings)} points")
            return fig, pca, df
            
        except Exception as e:
            logger.error(f"Failed to create PCA plot: {e}")
            raise

# Create global instance
@st.cache_resource
def get_visualization_service():
    """Get cached visualization service instance."""
    return StreamlitVisualizationService()

def create_pca_plot(embeddings, scores, analyzed_embedding=None, analyzed_score=None, 
                   plot_type="3D", point_size=1, opacity=0.5, color_by="anomaly_score", threshold=None):
    """Wrapper function for backward compatibility."""
    service = get_visualization_service()
    return service.create_interactive_pca_plot(
        embeddings, scores, analyzed_embedding, analyzed_score, 
        plot_type, point_size, opacity, color_by, threshold
    )

def get_audio_player(file_path):
    """Create an audio player for the given file path with proper error handling."""
    try:
        # Construct full path
        full_path = BASE_DIR / file_path
        
        # Validate file exists and is accessible
        is_valid, error_msg = validate_file_path(str(full_path), "audio file")
        if not is_valid:
            logger.warning(f"Audio file validation failed: {error_msg}")
            return None, error_msg
        
        # Read audio file
        with open(full_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        logger.debug(f"Loaded audio file: {file_path} ({len(audio_bytes)} bytes)")
        return audio_bytes, None
        
    except Exception as e:
        error_msg = f"Error loading audio: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


class LoopDetectionService:
    """Service class for loop detection using proper music_anomalizer integration."""
    
    def __init__(self, config_name=DEFAULT_CONFIG, network_key='AEwRES'):
        self.config_name = config_name
        self.network_key = network_key
        self.config = None
        self.device = initialize_device()
        self.detector = None
        self.extractor = None
        
    def initialize(self, model_type: str):
        """Initialize the service with proper configuration management."""
        try:
            # Load experiment configuration
            self.config = load_experiment_config(self.config_name, str(BASE_DIR / 'configs'))
            
            # Get model choices
            model_choices = get_model_choices(self.config_name, self.network_key)
            if model_type not in model_choices:
                raise ValueError(f"Model type '{model_type}' not available")
            
            model_choice = model_choices[model_type]
            
            # Validate checkpoint files exist
            ae_ckpt = model_choice['ae_ckpt']
            svdd_ckpt = model_choice['svdd_ckpt']
            
            for ckpt_path, ckpt_type in [(ae_ckpt, 'AE'), (svdd_ckpt, 'SVDD')]:
                is_valid, error_msg = validate_file_path(ckpt_path, f"{ckpt_type} checkpoint")
                if not is_valid:
                    raise FileNotFoundError(error_msg)
            
            # Validate CLAP checkpoint
            is_valid, error_msg = validate_file_path(str(CLAP_CKPT), "CLAP checkpoint")
            if not is_valid:
                raise FileNotFoundError(error_msg)
            
            # Get network configurations using proper config management
            network_config = self.config.networks[model_choice['model_key']]
            svdd_config = self.config.deepSVDD
            
            # Convert to dict and ensure all required parameters
            model_config = network_config.dict()
            model_config['class_name'] = model_choice['model_key']
            
            # Set defaults if missing
            if model_config.get('num_features') is None:
                model_config['num_features'] = 1024
            if model_config.get('train_data_length') is None:
                model_config['train_data_length'] = 1000
            
            # Initialize embedding extractor
            self.extractor = Wav2Embedding(
                model_ckpt_path=str(CLAP_CKPT),
                audio_model='HTSAT-base',
                device=str(self.device)
            )
            
            # Initialize anomaly detector
            self.detector = AnomalyDetector(
                configs=[model_config, svdd_config.dict()],
                checkpoint_paths=[ae_ckpt, svdd_ckpt],
                device=str(self.device)
            )
            self.detector.load_models()
            
            logger.info(f"Loop detection service initialized for {model_type} model with {self.network_key} network")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize loop detection service: {e}")
            raise
    
    def detect_loop(self, wav_path: str, model_type: str, threshold: float = None):
        """Detect if audio file is a loop using initialized models."""
        try:
            # Initialize if not already done
            if self.detector is None or self.extractor is None:
                self.initialize(model_type)
            
            # Validate audio file
            is_valid, error_msg = validate_file_path(wav_path, "audio file")
            if not is_valid:
                raise FileNotFoundError(error_msg)
            
            # Get threshold from config if not provided
            if threshold is None:
                model_choices = get_model_choices(self.config_name, self.network_key)
                threshold_key = model_choices[model_type]['threshold_key']
                threshold = self.config.threshold.get(threshold_key)
                if threshold is None:
                    raise ValueError(f"Threshold not found for {threshold_key}")
            
            # Extract embedding
            logger.info(f"Extracting embedding from: {wav_path}")
            embedding = self.extractor.extract_embedding(wav_path)
            if embedding is None:
                raise RuntimeError("Failed to extract embedding from audio file")
            
            embedding_squeezed = embedding.squeeze()
            
            # Detect loop
            logger.info(f"Running loop detection with threshold: {threshold}")
            result = self.detector.get_detected_loops([embedding_squeezed], threshold)
            
            if not result or 0 not in result:
                raise RuntimeError("Loop detection failed to return results")
            
            detection_result = result[0]
            is_loop = detection_result['is_loop']
            distance = detection_result['distance']
            
            logger.info(f"Loop detection complete: is_loop={is_loop}, distance={distance:.6f}")
            return is_loop, distance, threshold, embedding_squeezed
            
        except Exception as e:
            logger.error(f"Loop detection failed: {e}")
            raise

# Global service instance
@st.cache_resource
def get_loop_detection_service(config_name=DEFAULT_CONFIG, network_key='AEwRES'):
    """Get cached loop detection service instance."""
    return LoopDetectionService(config_name, network_key)

def detect_loop(wav_path, model_type, threshold=None, config_name=DEFAULT_CONFIG, network_key='AEwRES'):
    """Wrapper function for backward compatibility."""
    service = get_loop_detection_service(config_name, network_key)
    return service.detect_loop(wav_path, model_type, threshold)

def main():
    st.title("📤 Upload and Analyze Your Audio")
    st.write("Upload your WAV file to compare it against the training data and identify if it's similar.")

    # Initialize session state variables
    if 'selected_audio_path' not in st.session_state:
        st.session_state.selected_audio_path = None
    if 'selected_audio_info' not in st.session_state:
        st.session_state.selected_audio_info = None
    if 'fig' not in st.session_state:
        st.session_state.fig = None
    if 'plot_dataframe' not in st.session_state:
        st.session_state.plot_dataframe = None
    if 'last_model' not in st.session_state:
        st.session_state.last_model = None

    # Configuration and model selection
    col1, col2, col3 = st.columns(3)
    with col1:
        config_name = st.selectbox(
            '⚙️ Experiment Config', 
            options=AVAILABLE_CONFIGS,
            index=AVAILABLE_CONFIGS.index(DEFAULT_CONFIG),
            help="Choose the experiment configuration to use"
        )
        
        # Add "Set New Config" button to reset states when changing configs
        if st.button("Set Config", help="Reset all states to use this configuration"):
            # Clear all session state variables related to analysis
            keys_to_clear = [
                'selected_audio_path', 'selected_audio_info', 'fig', 'plot_dataframe',
                'last_model', 'last_config', 'last_network'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear cached data to force re-computation with new config
            st.cache_data.clear()
            
            st.success("✅ States reset for new configuration!")
            st.rerun()  # Refresh the page
    with col2:
        # Get available networks for the selected config
        available_networks = get_available_networks(config_name)
        default_network = 'AEwRES' if 'AEwRES' in available_networks else available_networks[0] if available_networks else 'AEwRES'
        
        network_type = st.selectbox(
            '🧠 Network Type', 
            options=available_networks,
            index=available_networks.index(default_network) if default_network in available_networks else 0,
            help="Choose the network architecture to use"
        )
    with col3:
        model = st.selectbox(
            '🎸 Model type', 
            options=['bass', 'guitar'],
            help="Choose the model trained on bass or guitar dataset"
        )
    
    # Sidebar with additional controls
    with st.sidebar:
        st.header("🔧 Advanced Options")
        
        # Show current anomaly scores status
        manager = get_anomaly_scores_manager()
        scores_info = manager.get_scores_info(model, config_name, network_type)
        
        if scores_info['exists'] and scores_info['valid']:
            st.success(f"✅ Anomaly scores loaded ({scores_info['num_scores']} samples)")
            if scores_info['last_modified']:
                st.caption(f"Last updated: {scores_info['last_modified'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.warning("⚠️ Anomaly scores need computation")
        
        # Force recompute button
        if st.button("🔄 Recompute Anomaly Scores", 
                    help="Force recomputation of anomaly scores (this may take several minutes)"):
            with st.spinner("Recomputing anomaly scores..."):
                progress_bar = st.progress(0.0, text="Initializing recomputation...")
                st._current_progress_bar = progress_bar
                
                try:
                    success, error = manager.compute_missing_scores(
                        model_type=model,
                        config_name=config_name,
                        network_key=network_type,
                        force_recompute=True
                    )
                    
                    if success:
                        st.success("✅ Anomaly scores recomputed successfully!")
                        st.rerun()  # Refresh the page to show new scores
                    else:
                        st.error(f"❌ Failed to recompute scores: {error}")
                        
                finally:
                    st._current_progress_bar = None

    # Store config selection in session state
    if 'last_config' not in st.session_state:
        st.session_state.last_config = None
    if 'last_network' not in st.session_state:
        st.session_state.last_network = None
        
    # If the model, config, or network changes, clear the previous visualization and selection
    if (st.session_state.last_model != model or 
        st.session_state.last_config != config_name or 
        st.session_state.last_network != network_type):
        st.session_state.fig = None
        st.session_state.selected_audio_path = None
        st.session_state.selected_audio_info = None
        st.session_state.last_model = model
        st.session_state.last_config = config_name
        st.session_state.last_network = network_type

    uploaded_file = st.file_uploader('Choose a WAV file', type=['wav', 'mp3'],
                                     help="Upload a WAV file to analyze")

    # Get default threshold from config using proper config management
    try:
        config = load_experiment_config(config_name, str(BASE_DIR / 'configs'))
        model_choices = get_model_choices(config_name)
        threshold_key = model_choices[model]['threshold_key']
        default_threshold = config.threshold.get(threshold_key, 0.0) if config.threshold else 0.0
    except Exception as e:
        logger.error(f"Failed to load config for threshold: {e}")
        st.error(f"Configuration error: {e}")
        default_threshold = 0.0
    
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.number_input(
            f'Custom Threshold (default: {default_threshold:.6f})', 
            value=default_threshold, 
            format="%f",
            help=f"Lower values = stricter detection. Config default: {default_threshold:.6f}"
        )
    with col2:
        use_default_threshold = st.checkbox(
            'Use default threshold from config', 
            value=True,
            help=f"Use the optimized threshold ({default_threshold:.6f}) from {threshold_key}"
        )

    if uploaded_file is not None:
        file_content = uploaded_file.read()

        st.markdown("---")
        st.subheader("Uploaded Audio Preview")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**📁 File Name:** {uploaded_file.name}")
            st.write(f"**📊 File Size:** {uploaded_file.size / 1024:.1f} KB")
        with col2:
            audio_bytes = io.BytesIO(file_content)
            st.audio(audio_bytes, format='audio/wav')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            tmp_wav.write(file_content)
            tmp_wav_path = tmp_wav.name

        st.markdown("---")
        st.subheader("🔍 Run Analysis")
        st.write("Click the button below to analyze your audio file against the training data.")

        if st.button('Run Analysis'):
            th = None if use_default_threshold else threshold
            
            try:
                with st.spinner('Running analysis...'):
                    is_loop, distance, used_threshold, embedding = detect_loop(tmp_wav_path, model, th, config_name, network_type)

                st.markdown("---")
                st.subheader("📊 Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**📁 Input File:** {uploaded_file.name}")
                    st.write(f"**🎸 Model:** {model.capitalize()}")
                    st.write(f"**🧠 Network:** {network_type}")
                with col2:
                    st.write(f"**🎯 Threshold:** {used_threshold:.6f}")
                    if distance is not None:
                        st.write(f"**📏 Distance:** {distance:.6f}")
                    else:
                        st.write("**📏 Distance:** N/A")

                if is_loop is None:
                    st.error('❌ Failed to extract embedding from input WAV file.')
                    st.info("💡 **Tip:** Make sure your audio file is a valid WAV format and contains audio data.")
                elif is_loop:
                    st.success('✅ **SIMILARITY DETECTED**')
                    if distance is not None:
                        st.info(f"Your audio file is similar to the original data!")
                else:
                    st.info('❌ **Not a good match**')
                    if distance is not None:
                        st.info(f"Your audio file is not similar to the original data!")
                        
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                st.error(f"❌ Analysis failed: {str(e)}")
                st.info("Please check your audio file and try again.")

        st.markdown("---")
        st.subheader("🔬 Latent Space Analysis")
        st.write("Visualization of the training data and your analyzed file in the model's latent space.")
        st.info("💡 **Interactive Feature:** Choose 2D plot type and cick on any datapoint in the plot to listen to the corresponding audio file!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            plot_type = st.radio("Plot Type", ["2D", "3D"], index=1)
        with col2:
            point_size = st.slider("Point Size", min_value=1, max_value=10, value=1)
        with col3:
            opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        with col4:
            color_by = st.radio("Color By", ["anomaly_score", "threshold"], index=0)

        if st.button('Run Visualization'):
            th = None if use_default_threshold else threshold
            
            try:
                with st.spinner('Preparing the Visualization...'):
                    is_loop, distance, used_threshold, embedding = detect_loop(tmp_wav_path, model, th, config_name, network_type)
                    
                    if embedding is not None:
                        # Check if training data (especially anomaly scores) need to be computed
                        manager = get_anomaly_scores_manager()
                        scores_exist, error = manager.check_scores_exist(model, config_name, network_type)
                        
                        if not scores_exist:
                            st.warning(f"⚠️ Anomaly scores for {model} model ({config_name}) not found. Computing them now...")
                            with st.spinner("Computing anomaly scores for visualization, this may take a few minutes..."):
                                progress_bar = st.progress(0.0, text="Initializing...")
                                st._current_progress_bar = progress_bar
                        
                        with st.spinner(f'Generating {plot_type} visualization...'):
                            embeddings, scores = load_training_data(model, config_name, network_type)
                            
                            if embeddings is not None and scores is not None:
                                fig, _, df = create_pca_plot(
                                    embeddings, scores, embedding, distance,
                                    plot_type, point_size, opacity, color_by, used_threshold
                                )
                                # Store the figure and dataframe in the session state
                                st.session_state.fig = fig
                                st.session_state.plot_dataframe = df
                                logger.info(f"Successfully generated {plot_type} visualization")
                            else:
                                st.error("Failed to load training data for visualization")
                                
                        if hasattr(st, '_current_progress_bar'):
                            st._current_progress_bar = None  # Ensure cleanup
                    else:
                        st.error("Could not generate visualization because embedding extraction failed.")
                        
            except Exception as e:
                logger.error(f"Visualization failed: {e}")
                st.error(f"❌ Visualization failed: {str(e)}")
                st.info("Please check your audio file and configuration, then try again.")
                if hasattr(st, '_current_progress_bar'):
                    st._current_progress_bar = None

        # Display the plot if it exists in the session state
        if st.session_state.fig:
            event = st.plotly_chart(
                st.session_state.fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key=f"pca_plot_{model}_{plot_type}_{color_by}"
            )

            # Handle point selection events with error handling
            try:
                if event.selection and event.selection['points']:
                    selected_index = event.selection['points'][0]['point_index']
                    df = st.session_state.plot_dataframe
                    
                    if df is not None and 0 <= selected_index < len(df):
                        selected_file_path = df.iloc[selected_index]['file_path']
                        selected_anomaly_score = df.iloc[selected_index]['anomaly_score']

                        # Store selected audio info in session state
                        st.session_state.selected_audio_path = selected_file_path
                        st.session_state.selected_audio_info = {
                            'file_path': selected_file_path,
                            'anomaly_score': selected_anomaly_score,
                            'index': selected_index
                        }
                        logger.debug(f"Selected audio sample: {selected_file_path}")
                    else:
                        logger.warning(f"Invalid selection index: {selected_index}")
            except Exception as e:
                logger.error(f"Error handling point selection: {e}")
            
            if color_by == "threshold":
                st.info("💡 **Interpretation:** "
                        "Each point represents a training sample. "
                        "Blue points are below the threshold (normal), red points are above the threshold (anomalous). "
                        "The red diamond shows where your file is positioned in the latent space. ")
            else:
                st.info("💡 **Interpretation:** "
                        "Each point represents a training sample. "
                        "Color indicates anomaly score (darker = more anomalous). "
                        "The red diamond shows where your file is positioned in the latent space. ")
            
            if plot_type == "3D":
                st.warning("⚠️ You can click and listen to the data points interactively by choosng 2D plot type. "
                        "This functionality is not available yet for 3D plot type due to Streamlit limitations.")

        # Display audio player for selected point if it exists in session state
        if st.session_state.selected_audio_path and st.session_state.selected_audio_info:
            st.markdown("---")
            st.subheader("🎵 Selected Audio Sample")
            selected_info = st.session_state.selected_audio_info
            st.write(f"**📁 File Path:** {selected_info['file_path']}")
            st.write(f"**📊 Anomaly Score:** {selected_info['anomaly_score']:.6f}")
            
            try:
                audio_bytes, error = get_audio_player(selected_info['file_path'])
                if audio_bytes and not error:
                    st.audio(audio_bytes, format='audio/wav')
                elif error:
                    st.error(f"Error loading audio: {error}")
            except Exception as e:
                logger.error(f"Failed to load selected audio: {e}")
                st.error(f"Failed to load audio file: {str(e)}")
            
            if st.button("Clear Selection"):
                try:
                    st.session_state.selected_audio_path = None
                    st.session_state.selected_audio_info = None
                    logger.debug("Cleared audio selection")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error clearing selection: {e}")

if __name__ == '__main__':
    main()
