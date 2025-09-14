import streamlit as st
import os
from music_anomalizer.utils import get_anomaly_scores_manager
from music_anomalizer.config.loader import load_experiment_config

# Paths (relative to this file - go up two levels: pages -> app -> root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Available experiment configurations
AVAILABLE_CONFIGS = ['exp1', 'exp2_deeper', 'single_model']
DEFAULT_CONFIG = 'exp2_deeper'

def get_config_path(config_name=DEFAULT_CONFIG):
    """Get config path for selected experiment."""
    return os.path.join(BASE_DIR, 'configs', f'{config_name}.yaml')

def get_available_networks(config_name):
    """Get available network types from config."""
    try:
        config = load_experiment_config(config_name, os.path.join(BASE_DIR, 'configs'))
        return list(config.networks.keys())
    except Exception:
        return ['AEwRES']  # fallback

@st.cache_data
def load_anomaly_scores(model_type, config_name=DEFAULT_CONFIG, network_key='AEwRES'):
    """Load anomaly scores for a given model type, computing them if necessary."""
    manager = get_anomaly_scores_manager()
    
    def progress_callback(message, progress):
        # Use st.info to show progress messages in Streamlit
        if hasattr(st, '_current_progress_bar') and st._current_progress_bar is not None:
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

def get_audio_file_path(relative_path):
    """Convert relative path from anomaly scores to absolute path."""
    return os.path.join(BASE_DIR, relative_path)

def display_top_loops(model_type, config_name=DEFAULT_CONFIG, network_type='AEwRES'):
    """Display the top 3 loops with lowest anomaly scores."""
    st.subheader(f"🎵 Top 3 'Normal' Training Loops ({model_type.capitalize()})")
    st.subheader(f"   Config: {config_name} - Network: {network_type}")
    st.write("These are the most typical examples from the training dataset "
             "(lowest anomaly scores). Listen to understand what the model "
             "considers 'normal' examples.")
    
    # Check if scores need to be computed
    manager = get_anomaly_scores_manager()
    scores_exist, error = manager.check_scores_exist(model_type, config_name, network_type)
    
    if not scores_exist:
        st.warning(f"⚠️ Anomaly scores for {model_type} model ({config_name}) not found. Computing them now...")
        with st.spinner("Computing anomaly scores, this may take a few minutes..."):
            # Create a progress bar
            progress_bar = st.progress(0.0, text="Initializing...")
            st._current_progress_bar = progress_bar
            
            try:
                scores = load_anomaly_scores(model_type, config_name, network_type)
                st._current_progress_bar = None  # Clean up
            finally:
                st._current_progress_bar = None  # Ensure cleanup
    else:
        scores = load_anomaly_scores(model_type, config_name, network_type)
    
    try:
        top_n = scores[:3]
        
        # Display summary info
        total_loops = len(scores)
        st.write(f"The training dataset contain {total_loops} {model_type} loops.")
        st.info(f"Showing top 3 with lowest anomaly scores.")
        
        for i, entry in enumerate(top_n):
            with st.expander(f"#{i+1}: {os.path.basename(entry['file_path'])} - Score: {entry['anomaly_score']:.6f}"):
                audio_path = get_audio_file_path(entry['file_path'])
                
                if os.path.exists(audio_path):
                    # Display audio player
                    with open(audio_path, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/wav')
                    
                    # Display file info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**🎯 Anomaly Score:** {entry['anomaly_score']:.6f}")
                        st.write(f"**🏆 Rank:** {i+1} of {total_loops}")
                    with col2:
                        st.write(f"**📁 File:** {os.path.basename(entry['file_path'])}")
                        st.write(f"**📂 Path:** {entry['file_path']}")
                        # st.write(f"**📊 Percentile:** {((i+1)/total_loops)*100:.1f}%")
                        
                    # Add explanation
                    # if i == 0:
                    #     st.success("🥇 **Best match to training data** - This loop is most similar to what the model learned as 'normal'")
                    # elif i < 3:
                    #     st.info("🥈 **Very typical loop** - This loop closely matches the training data patterns")
                    # else:
                    #     st.info("🥉 **Typical loop** - This loop represents normal patterns in the training data")
                else:
                    st.error(f"❌ Audio file not found: {audio_path}")
                    
    except Exception as e:
        st.error(f"❌ Error loading anomaly scores for {model_type}: {str(e)}")

def display_last_loops(model_type, config_name=DEFAULT_CONFIG, network_type='AEwRES'):
    """Display the top 3 loops with lowest anomaly scores."""
    st.subheader(f"🎵 Lowest 3 'Normal' Training Loops ({model_type.capitalize()})")
    st.subheader(f"   Config: {config_name} - Network: {network_type}")
    st.write("These are the least typical examples from the training dataset "
             "(highest anomaly scores). Listen to understand what the model "
             "considers 'anomaly' examples.")
    
    # Check if scores need to be computed (should already be computed from display_top_loops)
    manager = get_anomaly_scores_manager()
    scores_exist, error = manager.check_scores_exist(model_type, config_name, network_type)
    
    if not scores_exist:
        st.warning(f"⚠️ Anomaly scores for {model_type} model ({config_name}) not found. Computing them now...")
        with st.spinner("Computing anomaly scores, this may take a few minutes..."):
            # Create a progress bar
            progress_bar = st.progress(0.0, text="Initializing...")
            st._current_progress_bar = progress_bar
            
            try:
                scores = load_anomaly_scores(model_type, config_name, network_type)
                st._current_progress_bar = None  # Clean up
            finally:
                st._current_progress_bar = None  # Ensure cleanup
    else:
        scores = load_anomaly_scores(model_type, config_name, network_type)
    
    try:
        last_n = scores[-3:]
        
        # Display summary info
        total_loops = len(scores)
        st.info(f"Showing last 3 with highest anomaly scores.")
        
        for i, entry in enumerate(last_n):
            with st.expander(f"#{i+1}: {os.path.basename(entry['file_path'])} - Score: {entry['anomaly_score']:.6f}"):
                audio_path = get_audio_file_path(entry['file_path'])
                
                if os.path.exists(audio_path):
                    # Display audio player
                    with open(audio_path, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/wav')
                    
                    # Display file info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**🎯 Anomaly Score:** {entry['anomaly_score']:.6f}")
                        st.write(f"**🏆 Rank:** {total_loops-i} of {total_loops}")
                    with col2:
                        st.write(f"**📁 File:** {os.path.basename(entry['file_path'])}")
                        st.write(f"**📂 Path:** {entry['file_path']}")
                        # st.write(f"**📊 Percentile:** {((total_loops-i)/total_loops)*100:.1f}%")
                        
                    # Add explanation
                    # if i == 0:
                    #     st.success("🥇 **Best match to training data** - This loop is most similar to what the model learned as 'normal'")
                    # elif i < 3:
                    #     st.info("🥈 **Very typical loop** - This loop closely matches the training data patterns")
                    # else:
                    #     st.info("🥉 **Typical loop** - This loop represents normal patterns in the training data")
                else:
                    st.error(f"❌ Audio file not found: {audio_path}")
                    
    except Exception as e:
        st.error(f"❌ Error loading anomaly scores for {model_type}: {str(e)}")

def main():
    st.title('Loop Detector')
    st.write('Identifies the similarity of a WAV file to the original (training) data '
             'using anomaly detection based on deep learning models. The model is '
             'trained on a dataset of MusicRadar loops.')
    
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
        if st.button("🔄 Set New Config", help="Reset all states to use the new configuration"):
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
        model = st.selectbox('🎸 Model type', options=['bass', 'guitar'], 
                            help="Choose the model trained on bass or guitar dataset")
    
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
    
    # Display top and lowest 3 training examples
    display_top_loops(model, config_name, network_type)
    display_last_loops(model, config_name, network_type)
    

if __name__ == '__main__':
    main()
