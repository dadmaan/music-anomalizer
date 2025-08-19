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

from music_anomalizer.utils import PickleHandler
import laion_clap
# from modules.clap_audio_tagging import *  # TODO: This module doesn't exist in new structure
from music_anomalizer.preprocessing.extract_embed import *

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


def process_dataset(dataset_path, output_dir, model, device, config):
    """Process a complete dataset and extract embeddings."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Find all audio files
    audio_paths = list(dataset_path.glob("**/*.wav"))
    if not audio_paths:
        logging.warning(f"No audio files found in {dataset_path}")
        return
    
    logging.info(f"Found {len(audio_paths)} audio files in {dataset_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    dataset_name = dataset_path.name
    model_name = config['audio_model']
    output_file = output_dir / f"{model_name}_{dataset_name}_{len(audio_paths)}_embeddings.pkl"
    
    # Get audio branch from model
    audio_branch = get_audio_branch(model)
    
    # Run the process
    run_process(audio_paths, audio_branch, device, str(output_file), 
               config['extract_features'], config['max_workers'])


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Extract embeddings from audio files using CLAP model")
    parser.add_argument("--dataset", type=str, default="data/dataset/bass",
                       help="Path to dataset directory (default: data/dataset/bass)")
    parser.add_argument("--output", type=str, default="data/prep",
                       help="Output directory for embeddings (default: data/prep)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                       help="Device to use for processing (default: auto)")
    parser.add_argument("--model", type=str, default="HTSAT-base",
                       help="CLAP audio model variant (default: HTSAT-base)")
    parser.add_argument("--checkpoint", type=str, 
                       default="checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt",
                       help="Path to CLAP model checkpoint")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of worker threads (default: 8)")
    parser.add_argument("--no-features", action="store_true",
                       help="Skip feature extraction preprocessing")
    
    args = parser.parse_args()
    
    # Initialize device
    device = initialize_device(args.device)
    
    # Prepare configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'audio_model': args.model,
        'checkpoint_path': args.checkpoint,
        'max_workers': args.workers,
        'extract_features': not args.no_features
    })
    
    try:
        # Initialize CLAP model
        clap_model = initialize_clap_model(config['checkpoint_path'], 
                                         config['audio_model'], device)
        
        # Process dataset
        process_dataset(args.dataset, args.output, clap_model, device, config)
        
        logging.info("Embedding extraction completed successfully")
        
    except Exception as e:
        logging.error(f"Embedding extraction failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
