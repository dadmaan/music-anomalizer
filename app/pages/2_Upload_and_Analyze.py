import streamlit as st
import os
import torch
import pickle
import pandas as pd
import io
import numpy as np
from music_anomalizer.utils import load_pickle
from music_anomalizer.config.loader import load_yaml_config
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.preprocessing.wav2embed import Wav2Embedding
import tempfile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Paths (relative to this file - go up two levels: pages -> app -> root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Available experiment configurations
AVAILABLE_CONFIGS = ['exp1', 'exp2_deeper', 'single_model']
DEFAULT_CONFIG = 'exp2_deeper'

def get_config_path(config_name=DEFAULT_CONFIG):
    """Get config path for selected experiment."""
    return os.path.join(BASE_DIR, 'configs', f'{config_name}.yaml')
CLAP_CKPT = os.path.join(BASE_DIR, 'checkpoints', 'laion_clap', 'music_audioset_epoch_15_esc_90.14.pt')

@st.cache_data
def get_model_choices(config_name=DEFAULT_CONFIG):
    """Load model configuration from config file."""
    config_path = get_config_path(config_name)
    config = load_yaml_config(config_path)
    
    # Get checkpoint paths from config or use defaults
    manual_paths = config.get('checkpoints', {}).get('manual_paths', {})
    
    def resolve_checkpoint_path(path_key, default_path):
        """Resolve checkpoint path, handling relative paths from config."""
        path = manual_paths.get(path_key, default_path)
        if path.startswith('./'):
            path = path[2:]  # Remove ./ prefix
        return os.path.join(BASE_DIR, path)
    
    return {
        'bass': {
            'ae_ckpt': resolve_checkpoint_path('AEwRES_bass_ae', 
                'checkpoints/loop_benchmark/EXP2_DEEPER/AEwRES/AEwRES-HTSAT_base_musicradar_bass-AE-epoch=149-val_loss=0.01.ckpt'),
            'svdd_ckpt': resolve_checkpoint_path('AEwRES_bass_svdd',
                'checkpoints/loop_benchmark/EXP2_DEEPER/AEwRES/AEwRES-HTSAT_base_musicradar_bass-DSVDD-epoch=132-val_loss=0.00.ckpt'),
            'threshold_key': 'AEwRES_bass',
            'model_key': 'AEwRES',
        },
        'guitar': {
            'ae_ckpt': resolve_checkpoint_path('AEwRES_guitar_ae',
                'checkpoints/loop_benchmark/EXP2_DEEPER/AEwRES/AEwRES-HTSAT_base_musicradar_guitar-AE-epoch=153-val_loss=0.01.ckpt'),
            'svdd_ckpt': resolve_checkpoint_path('AEwRES_guitar_svdd',
                'checkpoints/loop_benchmark/EXP2_DEEPER/AEwRES/AEwRES-HTSAT_base_musicradar_guitar-DSVDD-epoch=34-val_loss=0.01.ckpt'),
            'threshold_key': 'AEwRES_guitar',
            'model_key': 'AEwRES',
        }
    }

@st.cache_data
def load_anomaly_scores(model_type):
    """Load anomaly scores for a given model type."""
    file_path = os.path.join(BASE_DIR, 'output', f'anomaly_scores_{model_type}.pkl')
    with open(file_path, 'rb') as f:
        scores = pickle.load(f)
    return scores

@st.cache_data
def load_training_data(model_type, config_name=DEFAULT_CONFIG):
    """Load training embeddings and anomaly scores for a given model type."""
    config_path = get_config_path(config_name)
    config = load_yaml_config(config_path)
    
    # Load embeddings - handle relative paths from config
    dataset_path = config['dataset_paths'][f'HTSAT_base_musicradar_{model_type}']
    if dataset_path.startswith('./'):
        dataset_path = dataset_path[2:]  # Remove ./ prefix
    full_dataset_path = os.path.join(BASE_DIR, dataset_path)
    embeddings = load_pickle(full_dataset_path)
    
    # Load anomaly scores
    scores = load_anomaly_scores(model_type)
    
    return embeddings, scores

def create_pca_plot(embeddings, scores, analyzed_embedding=None, analyzed_score=None, plot_type="3D", point_size=1, opacity=0.5, color_by="anomaly_score", threshold=None):
    """Create a 2D or 3D PCA plot of training data colored by anomaly scores or threshold, with optional analyzed file."""
    # Normalize the embeddings
    scaler = MinMaxScaler(feature_range=(-1,1))
    embeddings_normalized = scaler.fit_transform(embeddings)
    
    # Perform PCA to reduce to 2D or 3D
    n_components = 3 if plot_type == "3D" else 2
    pca = PCA(n_components=n_components)
    pca.fit(embeddings_normalized)
    embeddings_pca = pca.transform(embeddings_normalized)
    
    # Normalize the analyzed embedding if provided
    if analyzed_embedding is not None:
        analyzed_embedding_normalized = scaler.transform(analyzed_embedding.reshape(1, -1))
    
    # Create DataFrame for plotting
    if plot_type == "3D":
        df = pd.DataFrame({
            'x': embeddings_pca[:, 0],
            'y': embeddings_pca[:, 1],
            'z': embeddings_pca[:, 2],
            'anomaly_score': [s['anomaly_score'] for s in scores],
            'file_path': [s['file_path'] for s in scores],
            'index': range(len(scores))  # Add index for click handling
        })
    else:
        df = pd.DataFrame({
            'x': embeddings_pca[:, 0],
            'y': embeddings_pca[:, 1],
            'anomaly_score': [s['anomaly_score'] for s in scores],
            'file_path': [s['file_path'] for s in scores],
            'index': range(len(scores))  # Add index for click handling
        })
    
    # Create scatter plot based on color_by option
    if color_by == "threshold" and threshold is not None:
        # Color by threshold: blue for below threshold, red for above
        df['color'] = ['blue' if score['anomaly_score'] <= threshold else 'red' for score in scores]
        color_column = 'color'
        color_title = 'Threshold Classification'
        color_continuous_scale = None
        color_discrete_map = {'blue': 'blue', 'red': 'red'}
    else:
        # Color by anomaly score (default)
        color_column = 'anomaly_score'
        color_title = 'Anomaly Score'
        color_continuous_scale = 'Viridis'
        color_discrete_map = None
    
    if plot_type == "3D":
        if color_by == "threshold" and threshold is not None:
            fig = px.scatter_3d(df, x='x', y='y', z='z', 
                                color=color_column,
                                color_discrete_map=color_discrete_map,
                                title=f'3D PCA of Training Data ({len(embeddings)} samples)',
                                labels={color_column: color_title},
                                hover_data={
                                    'anomaly_score': ':.6f', 
                                    'file_path': True,
                                    'index': False  # Hide index in hover
                                },
                                size_max=point_size,
                                opacity=opacity)
        else:
            fig = px.scatter_3d(df, x='x', y='y', z='z', 
                                color=color_column,
                                color_continuous_scale=color_continuous_scale,
                                title=f'3D PCA of Training Data ({len(embeddings)} samples)',
                                labels={color_column: color_title},
                                hover_data={
                                    'anomaly_score': ':.6f', 
                                    'file_path': True,
                                    'index': False  # Hide index in hover
                                },
                                size_max=point_size,
                                opacity=opacity)
    else:
        if color_by == "threshold" and threshold is not None:
            fig = px.scatter(df, x='x', y='y', 
                             color=color_column,
                             color_discrete_map=color_discrete_map,
                             title=f'2D PCA of Training Data ({len(embeddings)} samples)',
                             labels={color_column: color_title},
                             hover_data={
                                 'anomaly_score': ':.6f', 
                                 'file_path': True,
                                 'index': False  # Hide index in hover
                             },
                             size_max=point_size,
                             opacity=opacity)
        else:
            fig = px.scatter(df, x='x', y='y', 
                             color=color_column,
                             color_continuous_scale=color_continuous_scale,
                             title=f'2D PCA of Training Data ({len(embeddings)} samples)',
                             labels={color_column: color_title},
                             hover_data={
                                 'anomaly_score': ':.6f', 
                                 'file_path': True,
                                 'index': False  # Hide index in hover
                             },
                             size_max=point_size,
                             opacity=opacity)
    
    # Add analyzed file if provided
    if analyzed_embedding is not None and analyzed_score is not None:
        # Transform the analyzed embedding using the same PCA
        analyzed_pca = pca.transform(analyzed_embedding_normalized)
        
        # Add the analyzed file as a distinct point
        if plot_type == "3D":
            fig.add_trace(go.Scatter3d(
                x=[analyzed_pca[0, 0]],
                y=[analyzed_pca[0, 1]],
                z=[analyzed_pca[0, 2]],
                mode='markers',
                marker=dict(
                    size=7,
                    color='red',
                    symbol='diamond'
                ),
                name='Analyzed File',
                hovertemplate=f'Analyzed File<br>Anomaly Score: {analyzed_score:.6f}<extra></extra>'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[analyzed_pca[0, 0]],
                y=[analyzed_pca[0, 1]],
                mode='markers',
                marker=dict(
                    size=7,
                    color='red',
                    symbol='diamond'
                ),
                name='Analyzed File',
                hovertemplate=f'Analyzed File<br>Anomaly Score: {analyzed_score:.6f}<extra></extra>'
            ))
        
        # Add annotation
        fig.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text=f"Analyzed File Anomaly Score: {analyzed_score:.6f}",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
            ]
        )
    
    # Update layout
    if plot_type == "3D":
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1={pca.explained_variance_ratio_[0]*100:.2f}%",
                yaxis_title=f"PC2={pca.explained_variance_ratio_[1]*100:.2f}%",
                zaxis_title=f"PC3={pca.explained_variance_ratio_[2]*100:.2f}%"
            )
        )
        if color_by == "anomaly_score":
            fig.update_layout(coloraxis_colorbar=dict(title="Anomaly Score"))
    else:
        fig.update_layout(
            xaxis_title=f"PC1={pca.explained_variance_ratio_[0]*100:.2f}%",
            yaxis_title=f"PC2={pca.explained_variance_ratio_[1]*100:.2f}%"
        )
        if color_by == "anomaly_score":
            fig.update_layout(coloraxis_colorbar=dict(title="Anomaly Score"))
    
    return fig, pca, df

def get_audio_player(file_path):
    """Create an audio player for the given file path."""
    try:
        # Construct full path
        full_path = os.path.join(BASE_DIR, file_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            return None, f"Audio file not found: {file_path}"
        
        # Read audio file
        with open(full_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        return audio_bytes, None
    except Exception as e:
        return None, f"Error loading audio: {str(e)}"


def detect_loop(wav_path, model, threshold=None, config_name=DEFAULT_CONFIG):
    config_path = get_config_path(config_name)
    config = load_yaml_config(config_path)
    model_choices = get_model_choices()
    model_choice = model_choices[model]
    model_config = config['networks'][model_choice['model_key']]
    svdd_config = config['deepSVDD']
    if threshold is None:
        threshold = config['threshold'][model_choice['threshold_key']]
    if model_config.get('num_features', None) is None:
        model_config['num_features'] = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = Wav2Embedding(
        model_ckpt_path=CLAP_CKPT,
        audio_model='HTSAT-base',
        device=device
    )
    embedding = extractor.extract_embedding(wav_path)
    if embedding is None:
        return None, None, None, None
    embedding_squeezed = embedding.squeeze()
    embedding_list = [embedding_squeezed]
    detector = AnomalyDetector(
        configs=[model_config, svdd_config],
        checkpoint_paths=[model_choice['ae_ckpt'], model_choice['svdd_ckpt']],
        device=device
    )
    detector.load_models()
    result = detector.get_detected_loops(embedding_list, threshold)
    distance = result[0]['distance']
    is_loop = result[0]['is_loop']

    return is_loop, distance, threshold, embedding_squeezed

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
    col1, col2 = st.columns(2)
    with col1:
        config_name = st.selectbox(
            '⚙️ Experiment Config', 
            options=AVAILABLE_CONFIGS,
            index=AVAILABLE_CONFIGS.index(DEFAULT_CONFIG),
            help="Choose the experiment configuration to use"
        )
    with col2:
        model = st.selectbox(
            '🎸 Model type', 
            options=['bass', 'guitar'],
            help="Choose the model trained on bass or guitar dataset"
        )

    # Store config selection in session state
    if 'last_config' not in st.session_state:
        st.session_state.last_config = None
        
    # If the model or config changes, clear the previous visualization and selection
    if st.session_state.last_model != model or st.session_state.last_config != config_name:
        st.session_state.fig = None
        st.session_state.selected_audio_path = None
        st.session_state.selected_audio_info = None
        st.session_state.last_model = model
        st.session_state.last_config = config_name

    uploaded_file = st.file_uploader('Choose a WAV file', type=['wav', 'mp3'],
                                     help="Upload a WAV file to analyze")

    # Get default threshold from config
    config_path = get_config_path(config_name)
    config = load_yaml_config(config_path)
    model_choices = get_model_choices(config_name)
    threshold_key = model_choices[model]['threshold_key']
    default_threshold = config.get('threshold', {}).get(threshold_key, 0.0)
    
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
            with st.spinner('Running analysis...'):
                is_loop, distance, used_threshold, embedding = detect_loop(tmp_wav_path, model, th, config_name)

            st.markdown("---")
            st.subheader("📊 Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**📁 Input File:** {uploaded_file.name}")
                st.write(f"**🎸 Model:** {model.capitalize()}")
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

        st.markdown("---")
        st.subheader("🔬 Latent Space Analysis")
        st.write("Visualization of the training data and your analyzed file in the model's latent space.")
        st.info("💡 **Interactive Feature:** Click on any datapoint in the plot to listen to the corresponding audio file!")

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
            with st.spinner('Preparing the Visualization...'):
                is_loop, distance, used_threshold, embedding = detect_loop(tmp_wav_path, model, th, config_name)
                if embedding is not None:
                    with st.spinner(f'Generating {plot_type} visualization...'):
                        try:
                            embeddings, scores = load_training_data(model, config_name)
                            fig, pca, df = create_pca_plot(
                                embeddings, scores, embedding, distance,
                                plot_type, point_size, opacity, color_by, used_threshold
                            )
                            # Store the figure and dataframe in the session state
                            st.session_state.fig = fig
                            st.session_state.plot_dataframe = df
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                else:
                    st.error("Could not generate visualization because embedding extraction failed.")

        # Display the plot if it exists in the session state
        if st.session_state.fig:
            event = st.plotly_chart(
                st.session_state.fig,
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key=f"pca_plot_{model}_{plot_type}_{color_by}"
            )

            # Handle point selection events
            if event.selection and event.selection['points']:
                selected_index = event.selection['points'][0]['point_index']
                df = st.session_state.plot_dataframe
                if 0 <= selected_index < len(df):
                    selected_file_path = df.iloc[selected_index]['file_path']
                    selected_anomaly_score = df.iloc[selected_index]['anomaly_score']

                    # Store selected audio info in session state
                    st.session_state.selected_audio_path = selected_file_path
                    st.session_state.selected_audio_info = {
                        'file_path': selected_file_path,
                        'anomaly_score': selected_anomaly_score,
                        'index': selected_index
                    }
            
            if color_by == "threshold":
                st.info("💡 **Interpretation:** "
                        "Each point represents a training sample. "
                        "Blue points are below the threshold (normal), red points are above the threshold (anomalous). "
                        "The red diamond shows where your file is positioned in the latent space. "
                        "**Click on any blue or red point to listen to that audio file!**")
            else:
                st.info("💡 **Interpretation:** "
                        "Each point represents a training sample. "
                        "Color indicates anomaly score (darker = more anomalous). "
                        "The red diamond shows where your file is positioned in the latent space. "
                        "**Click on any point to listen to that audio file!**")

        # Display audio player for selected point if it exists in session state
        if st.session_state.selected_audio_path and st.session_state.selected_audio_info:
            st.markdown("---")
            st.subheader("🎵 Selected Audio Sample")
            selected_info = st.session_state.selected_audio_info
            st.write(f"**📁 File Path:** {selected_info['file_path']}")
            st.write(f"**📊 Anomaly Score:** {selected_info['anomaly_score']:.6f}")
            
            audio_bytes, error = get_audio_player(selected_info['file_path'])
            if audio_bytes and not error:
                st.audio(audio_bytes, format='audio/wav')
            elif error:
                st.error(f"Error loading audio: {error}")
            
            if st.button("Clear Selection"):
                st.session_state.selected_audio_path = None
                st.session_state.selected_audio_info = None
                st.rerun()

if __name__ == '__main__':
    main()

