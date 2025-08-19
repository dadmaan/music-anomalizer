"""
Audio Embedding Extraction Script

This script extracts embeddings from audio files using the CLAP model.
It processes WAV files and saves the embeddings as pickle files.

The script uses concurrent processing for efficient handling of large datasets.
"""

import os
import pickle
import torch
import glob
import json
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

from modules.utils import PickleHandler
import laion_clap
from modules.clap_audio_tagging import *
from modules.extract_embed import *

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
EXTRACT_FEATURES = True
AUDIO_MODEL = 'HTSAT-base'  # enable_fusion=False

# Initialize model
clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel=AUDIO_MODEL)
clap_model.load_ckpt('checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt')  # for HTSAT-base; their best model trained on music audioset


def extract_embedding(model, audio_path, device, extract_features=True):
    """
    Extracts the embedding from the model for a single audio file.
    We pass the audio data directly to the audio_branch and bypass the projection layer at the end.

    Parameters:
        model: The CLAP model
        audio_path (str): Path to the audio file
        device: Device to run the model on
        extract_features (bool): Whether to extract features or not
        
    Returns:
        np.array: Audio embedding or None if error occurs
    """
    try:
        if extract_features:
            audio_data = load_audio(audio_path, expand_dim=True)
            audio_input = get_features(audio_data, clap_model)
            audio_tensor = prepare_input_dict(audio_input, device)
        else:
            audio_data = load_audio(audio_path, expand_dim=False)
            audio_tensor = prepare_audio_as_tensor(audio_data, device)
            
        # Get model predictions
        model.eval()
        with torch.no_grad():
            output_dict = model(audio_tensor, device)
        
        audio_embed = output_dict['embedding'].detach().cpu().numpy()
        
        # Optimize memory usage
        del audio_tensor
        torch.cuda.empty_cache()
        
        return audio_embed
        
    except Exception as e:
        logging.error(f"Error processing file {audio_path}: {str(e)}")
        return None


def worker(path, model, device, extract_features):
    """
    Worker function for processing a single audio file.
    
    Parameters:
        path (str): Path to the audio file
        model: The CLAP model
        device: Device to run the model on
        extract_features (bool): Whether to extract features or not
        
    Returns:
        tuple: (path, embedding) or (path, None) if error occurs
    """
    return path, extract_embedding(model, path, device, extract_features)


def run_process(audio_paths, model, device, output_file, extract_features=True, max_workers=8):
    """
    Process all audio files in a directory to extract embeddings using Python concurrent,
    and save the results and indices to pickle files.

    Args:
        audio_paths (list): List of audio paths.
        model (torch.nn.Module): The model used to extract embeddings.
        device (torch.device): Device to run the model on.
        output_file (str): Path to the output pickle file to save embeddings.
        extract_features (bool): Flag to decide whether to extract features or not.
        max_workers (int): Maximum number of workers. Set None for automatic estimation.
    """
    index_file = output_file.replace('.pkl', '_index.pkl')
    
    pickle_handler = PickleHandler(output_file)
    index_handler = PickleHandler(index_file)
    
    processed_data = []
    index_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for path in tqdm(audio_paths):
            futures.append(executor.submit(worker, path, model, device, extract_features))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(audio_paths)):
            path, emb = future.result()
            if emb is not None:
                processed_data.append(emb.flatten())
                index_data.append((audio_paths.index(path), path))
    
    if processed_data:
        pickle_handler.dump_data(np.array(processed_data))
        print(f"Data saved to {output_file}")
        index_handler.dump_data(index_data)
        print(f"Index data saved to {index_file}")
    else:
        print("No valid data to save.")


def main():
    """Main function to run the embedding extraction pipeline."""
    # Audio Samples - currently configured for bass guitar samples
    audio_paths = glob.glob("data/MusicRadar/SamplePack/musicradar-bass-guitar-samples"+"/**/*.wav", recursive=True)
    output_file = f'data/pickle/embedding/musicradar/{AUDIO_MODEL}_musicradar_{len(audio_paths)}_nff_bass_embeddings.pkl'
    
    print(f"Found {len(audio_paths)} audio files.")
    
    # Get audio branch from model
    audio_branch = get_audio_branch(clap_model)
    
    # Run the process
    run_process(audio_paths, audio_branch, device, output_file, EXTRACT_FEATURES, max_workers=None)


if __name__ == "__main__":
    main()
