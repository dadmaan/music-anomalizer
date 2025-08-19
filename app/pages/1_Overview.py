import streamlit as st
import os
import pickle
from modules.utils import load_json, load_pickle

# Paths (relative to this file)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'exp2_deeper.json')

@st.cache_data
def load_anomaly_scores(model_type):
    """Load anomaly scores for a given model type."""
    file_path = os.path.join(BASE_DIR, f'anomaly_scores_{model_type}.pkl')
    with open(file_path, 'rb') as f:
        scores = pickle.load(f)
    return scores

def get_audio_file_path(relative_path):
    """Convert relative path from anomaly scores to absolute path."""
    return os.path.join(BASE_DIR, relative_path)

def display_top_loops(model_type):
    """Display the top 3 loops with lowest anomaly scores."""
    st.subheader(f"🎵 Top 3 'Normal' Training Loops ({model_type.capitalize()})")
    st.write("These are the most typical examples from the training dataset (lowest anomaly scores). Listen to understand what the model considers 'normal' examples.")
    
    try:
        scores = load_anomaly_scores(model_type)
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

def display_last_loops(model_type):
    """Display the top 3 loops with lowest anomaly scores."""
    st.subheader(f"🎵 Lowest 3 'Normal' Training Loops ({model_type.capitalize()})")
    st.write("These are the least typical examples from the training dataset (highest anomaly scores). Listen to understand what the model considers 'anomaly' examples.")
    
    try:
        scores = load_anomaly_scores(model_type)
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
    st.write('Identifies the similarity of a WAV file to the original (training) data using anomaly detection based on deep learning models. The model is trained on a dataset of MusicRadar loops.')
    
    # Model selection at the top
    model = st.selectbox('🎸 Model type', options=['bass', 'guitar'], 
                        help="Choose the model trained on bass or guitar dataset")
    
    # Display top and lowest 3 training examples
    display_top_loops(model)
    display_last_loops(model)
    

if __name__ == '__main__':
    main()
