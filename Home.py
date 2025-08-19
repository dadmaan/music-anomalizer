import streamlit as st

st.set_page_config(
    page_title="Loop Detector",
    page_icon="🎵",
    layout="wide"
)

st.title("Loop Detector App")
st.write("Welcome to the Loop Detector app! Use the sidebar to navigate between pages.")
st.write("This app identifies the similarity of a WAV file to the original (training) data using anomaly detection based on deep learning models. The model is trained on a dataset of MusicRadar loops.")
