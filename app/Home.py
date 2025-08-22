import streamlit as st

def main():
    st.set_page_config(
        page_title="Music Anomaly Detector",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Music Anomaly Detector")
    st.markdown("---")
    
    st.write("""
    Welcome to the Music Anomaly Detector! This application uses deep learning models 
    to identify whether audio loops are similar to training data patterns.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Overview")
        st.write("""
        View and listen to examples from the training dataset to understand 
        what the model considers 'normal' and 'anomalous' patterns.
        """)
        if st.button("Go to Overview", use_container_width=True):
            st.switch_page("pages/1_Overview.py")
    
    with col2:
        st.subheader("📤 Upload & Analyze")
        st.write("""
        Upload your own audio files to analyze them against the trained models 
        and visualize results in the latent space.
        """)
        if st.button("Go to Upload & Analyze", use_container_width=True):
            st.switch_page("pages/2_Upload_and_Analyze.py")
    
    st.markdown("---")
    
    st.subheader("📋 How It Works")
    
    with st.expander("🧠 Model Architecture"):
        st.write("""
        The system uses:
        - **HTSAT-base**: Audio feature extraction from CLAP model
        - **AEwRES**: Autoencoder with residual connections for representation learning
        - **Deep SVDD**: Support Vector Data Description for anomaly detection
        """)
    
    with st.expander("📊 Training Data"):
        st.write("""
        Models are trained on MusicRadar datasets:
        - **Bass loops**: Various bass guitar patterns and styles
        - **Guitar loops**: Different guitar playing techniques and genres
        
        **Available Experiments:**
        - **exp1**: Initial experiment configuration
        - **exp2_deeper**: Deep networks with 5-layer architectures
        - **single_model**: Single model configuration
        """)
    
    with st.expander("🎯 Anomaly Detection"):
        st.write("""
        The system computes anomaly scores to determine similarity:
        - **Lower scores**: More similar to training data (normal)
        - **Higher scores**: Less similar to training data (anomalous)
        - **Threshold**: Configurable cutoff for classification
        """)

if __name__ == "__main__":
    main()