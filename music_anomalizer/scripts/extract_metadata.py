"""
Script to extract metadata from audio files including BPM, duration, and keywords.

This script processes WAV audio files to extract metadata such as:
- Beats Per Minute (BPM) either from file path patterns or using a tempo detection service
- Audio duration
- Descriptive keywords from file paths

The script uses multiprocessing for efficient processing of multiple files and 
saves the extracted metadata to a JSON file.

Usage:
    python script/extract_metadata.py
"""

import os
import glob
import json
import re
import hashlib
import multiprocessing
import requests
import librosa
import numpy as np

import nltk
from nltk.corpus import words
from music_anomalizer.utils import generate_md5_hash

# Download required NLTK data
nltk.download('wordnet', quiet=True)




def load_existing_metadata(file_path):
    """Load existing metadata if file exists, otherwise return empty dictionary."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def update_metadata_file(metadata, file_path):
    """Update the metadata JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Error saving the metadata: {e}")


def extract_bpm_from_path(file_path):
    """Extract BPM from file path using defined patterns."""
    # Patterns to match bpm in directory or file name
    patterns = [
        r'(\d+)\s?bpm',  # Matches '95bpm' or '95 bpm'
        r'(\d+)-?bpm',   # Matches '95-bpm'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, file_path.lower())
        if match:
            bpm = int(match.group(1))
            return bpm
    
    return None


def extract_keywords(file_path):
    """Extract and process keywords from file path."""
    try:
        # Normalize the file path by replacing underscores and hyphens with spaces
        normalized_path = re.sub(r'\s+-\s+', ' ', file_path)
        normalized_path = normalized_path.replace('_', ' ')

        # Regex pattern to match unwanted characters
        chars_to_remove = r'[&(){}[\]!@#$%^*+=|\\:;"\'<>,.?/~`]'  

        # Split the path into components using the OS-specific separator
        path_components = normalized_path.split(os.sep)

        # Keywords to remove
        keywords_to_remove = {'data', 'LoopPack', 'MusicRadar', 'SamplePack', 'and', 'bpm', 'BPM'}

        # Musical notes that should be followed by 'major' or 'minor'
        major_musical_notes = {'C', 'D', 'E', 'F', 'G', 'A', 'B'}
        minor_musical_notes = {note.lower() for note in major_musical_notes}

        # Initialize a list to store processed keywords
        processed_keywords = []

        # Extract and refine keywords from each component
        for component in path_components:
            # Remove the file extension from the last component (assumed to be the filename)
            if component == path_components[-1]:
                component, _ = os.path.splitext(component)
                component = component.replace('-', ' ')

            # Split component into words based on spaces and filter out empty strings
            words = [word.strip() for word in component.split() if word.strip()]

            # Clean and refine each word
            for word in words:
                # Append 'major' or 'minor' to musical notes
                if word in major_musical_notes:
                    word += '-major'
                elif word in minor_musical_notes:
                    word += '-minor'

                # Skip unwanted keywords or those too short or purely numeric
                if word in keywords_to_remove or len(word) < 3 or word.isdigit():
                    continue

                # Remove trailing digits from keywords
                word = re.sub(r'(\D+)\d*$', r'\1', word)

                word = re.sub(chars_to_remove, '', word)  # Remove specific unwanted characters

                # Add to processed keywords if not already present
                if word not in processed_keywords:
                    processed_keywords.append(word)

            # First pass: handle camelCase transformation
            processed_keywords = [re.sub('([a-z0-9])([A-Z])', r'\1-\2', word).lower() for word in processed_keywords]

        # Ensure 'loop' is always included in the keywords
        processed_keywords.append('loop')
    
        return processed_keywords
    except Exception as e:
        print(f"Error processing keywords from file {file_path}: {e}")
        return []


def manage_bpm_keywords(keywords, bpm):
    """Manage BPM keywords in the keywords list."""
    try:
        specific_bpm_keyword = f"{bpm}bpm"
        
        # pattern for XXXbpm
        bpm_pattern = re.compile(r'\d+bpm')
        
        # search for any existing BPM value
        existing_bpm = None
        for keyword in keywords:
            if bpm_pattern.match(keyword):
                existing_bpm = keyword
                break
        
        # check if the existing BPM matches the specific BPM
        if existing_bpm:
            if existing_bpm == specific_bpm_keyword:
                return keywords # no changes
            else:
                keywords.remove(existing_bpm)
                keywords.append(specific_bpm_keyword)
        else:
            keywords.append(specific_bpm_keyword)
        
        return keywords
    except Exception as e:
        print(f"Error managing keywords: {e}")
        return []


def get_keywords_from_path(file_path, target_bpm):
    """Get keywords from file path and manage BPM keywords."""
    keywords = extract_keywords(file_path)
    keywords = manage_bpm_keywords(keywords, target_bpm)
    return list(keywords)


def check_tempo_value(bpm):
    """Check for a reasonable BPM value."""
    try:
        if bpm:
            return bpm if bpm >= int(50) and bpm <= int(220) else int(120)
        else:
            print("Cannot get the correct BPM. Set to default; 120.")
            return int(120)
    except Exception as e:
        print(f"Error checking tempo value: {e}")


def convert_ndarray_to_list(obj):
    """
    Recursively converts numpy ndarrays to lists in the given object.
    
    :param obj: The object to convert.
    :return: The converted object with numpy arrays replaced by lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(element) for element in obj]
    else:
        return obj


def get_tempo_from_detector(audio_file_path):
    """Method to communicate with tempo detection service."""
    url = 'http://tempo_detector:5000/estimate_tempo'
    headers = {'Content-Type': 'application/json'}
    payload = {'file_path': audio_file_path,
               'sample_rate': 22050,
               'clip_length': 1}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # This will raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Request failed for file path {audio_file_path}: {e}")
        return None, None  # Return a tuple of None values if the request fails

    try:
        data = response.json()
        tempo = data.get('tempo')
        confidence = data.get('confidence')
        return tempo, confidence
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
        return None, None


def process_file(file_path, metadata_file):
    """Process a single audio file to extract metadata."""
    metadata = load_existing_metadata(metadata_file)
    doc_idx = generate_md5_hash(file_path)

    if doc_idx in metadata:
        return

    try:
        wav, sr = librosa.load(file_path, sr=None)
            
        duration = librosa.get_duration(y=wav, sr=sr)
        
        tempo_bpm = extract_bpm_from_path(file_path)
        if tempo_bpm is None:
            tempo_bpm, _ = get_tempo_from_detector(file_path)
        tempo_bpm = check_tempo_value(tempo_bpm)
        
        audio_tags = get_keywords_from_path(file_path, tempo_bpm)
        
        metadata[doc_idx] = {
            'file_path': file_path,
            'sample_rate': sr,
            'bpm': tempo_bpm,
            'sec_wav': duration,
            'audio_tags': audio_tags
        }
        
        return metadata
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def process_audio_files(root_folder, metadata_file='meta_info.json'):
    """Process all audio files in the specified folder."""
    try:
        if not os.path.exists(metadata_file):
            update_metadata_file({}, metadata_file)
            print("Created an empty metadata file.")

        file_paths = glob.glob(os.path.join(root_folder, "**", "*.wav"), recursive=True)
        print(f"Found {len(file_paths)} WAV files to process.")          

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(process_file, [(file_path, metadata_file) for file_path in file_paths])
            
        results = [result for result in results if result is not None]
        update_metadata_file(results, metadata_file)
        return results
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """Main function to run the metadata extraction process."""
    # Example usage - modify these paths as needed
    root_folder = './data/MusicRadar/selection/guitar'
    metadata_file = 'MR_guitar_meta_info.json'
    
    print(f"Processing audio files in {root_folder}")
    results = process_audio_files(root_folder, metadata_file)
    print(f"Metadata extraction completed. Results saved to {metadata_file}")


if __name__ == "__main__":
    main()
