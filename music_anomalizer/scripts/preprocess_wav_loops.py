"""
Audio Preprocessing Script for WAV Loops

This script preprocesses audio files by:
1. Loading audio with specified sample rate and mono conversion
2. Adjusting audio length to a target duration
3. Saving processed data with labels as a pickle file

The script uses multiprocessing for efficient processing of large datasets.
"""

import os
import librosa
import yaml
import json
import numpy as np
import pickle
import multiprocessing
from music_anomalizer.utils import construct_file_path
from music_anomalizer.config import load_audio_preprocessing_config
from sklearn.preprocessing import MultiLabelBinarizer


def encode_audio_tags(tags):
    """
    Encode audio tags using MultiLabelBinarizer.
    
    Parameters:
        tags (list): List of audio tags to encode.
        
    Returns:
        np.array: Encoded tags as binary matrix.
    """
    mlb = MultiLabelBinarizer()
    encoder = mlb.fit(tags)
    et = encoder.transform(tags)
    print(f"Number of classes: {len(mlb.classes_)}, Classes: {mlb.classes_}")
    return et


def process_audio_length(wav, sec_wav, target_length, sr, only_pad):
    """
    Adjusts the length of the audio time series to match the target length.

    Parameters:
        wav (np.array): The audio time series.
        sec_wav (float): The original length of the audio in seconds.
        target_length (int): The desired length of the audio in seconds. 
        sr (int): The sampling rate of the audio.
        only_pad (bool): Only pad shorter audio files with zero.
    
    Returns:
        np.array: The processed audio time series of exactly target_length seconds.
    """
    current_length_samples = int(sec_wav * sr)
    target_length_samples = target_length * sr
    
    if not only_pad and current_length_samples < target_length_samples:
        # Calculate how many full repeats are needed
        repeat_count = target_length_samples // current_length_samples
        # Repeat the audio
        extended_wav = np.tile(wav, repeat_count)
        
        # If still not enough samples, pad the remaining
        if len(extended_wav) < target_length_samples:
            pad_length = target_length_samples - len(extended_wav)
            extended_wav = np.pad(extended_wav, (0, pad_length), 'constant')
    elif only_pad:
        pad_length = target_length_samples - len(wav)
        extended_wav = np.pad(wav, (0, pad_length), 'constant')
    else:
        # If the audio is longer than the target, just truncate it
        extended_wav = wav[:target_length_samples]
    
    return extended_wav


def process_audio_file(args):
    """
    Function to load and process a single audio file.
    
    Parameters:
        args (tuple): Tuple containing (path, label, config)
        
    Returns:
        tuple: (processed_audio, label) or None if error occurs
    """
    try:
        path, label, config = args
        
        sample_rate = config['sample_rate']
        target_length = config['target_audio_length']
        mono = config['mono']
        only_pad = config['only_pad']
        
        # Load audio file
        audio, _ = librosa.load(path, sr=sample_rate, mono=mono)
        sec_wav = librosa.get_duration(y=audio, sr=sample_rate)
        audio = process_audio_length(audio, sec_wav, target_length, sample_rate, only_pad)
        
        # Double check audio is exactly target_length seconds long
        if len(audio) > sample_rate * target_length:
            audio = audio[:int(sample_rate * target_length)]
        
        return (audio, label)
    
    except Exception as e:
        print(f"Error processing file {path} with {sec_wav}secs length: {e}")
        return None


def run_process(meta_info, config, output_dir='data', num_workers=4):
    """
    Preprocess audio files using multiprocessing and save them with labels as a pickle file.
    
    Parameters:
        meta_info (dict): Audio files metadata.
        config (dict): Configuration parameters.
        output_dir (str): Directory to save the pickle file.
        num_workers (int): Number of worker processes to use.
    """
    config_details = "\n".join(f"{key}: {value}" for key, value in config.items())
    print("Preprocessing files with the following configuration:\n", config_details)
    
    audio_paths = []
    audio_tags = []
    
    for info in meta_info.values():
        audio_paths.append(info['file_path'])
        audio_tags.append([info['audio_tags']])
    
    labels = encode_audio_tags(audio_tags)
    
    # Prepare arguments for multiprocessing
    args = [(path, label, config) for path, label in zip(audio_paths, labels)]
    
    if num_workers is None or num_workers == -1:
        num_workers = os.cpu_count() or 1
        
    with multiprocessing.Pool(num_workers) as pool:
        processed_data = pool.map(process_audio_file, args)
    
    processed_data = [item for item in processed_data if item is not None]
    
    if processed_data:
        output_path = construct_file_path(config, output_dir)
        output_file = os.path.join(output_path)
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        print(f"Data saved to {output_file}")
    else:
        print("No valid data to save.")


def main():
    """Main function to run the preprocessing pipeline."""
    # Load configuration
    with open('configs/audio_preprocessing_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('output/metadata/LP_meta_info_v2_processed_balanced_i.json', 'r') as f:
        meta_info = json.load(f)
    
    run_process(meta_info, config, num_workers=-1)


if __name__ == "__main__":
    main()
