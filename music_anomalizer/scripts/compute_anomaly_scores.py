import json
import os
import sys
import torch

# Add the pann_aff directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pann_aff'))

from music_anomalizer.utils import load_json, load_pickle
from music_anomalizer.config import load_experiment_config
from music_anomalizer.data.data_loader import DataHandler
from music_anomalizer.models.anomaly_detector import AnomalyDetector
from music_anomalizer.preprocessing.wav2embed import Wav2Embedding
import pickle
from tqdm import tqdm

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = 'exp2_deeper'  # Use YAML config name instead of path
# Checkpoints
AEWRES_BASS_AE = os.path.join(BASE_DIR, 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_bass-AE-epoch=149-val_loss=0.01.ckpt')
AEWRES_BASS_SVDD = os.path.join(BASE_DIR, 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_bass-DSVDD-epoch=132-val_loss=0.00.ckpt')
AEWRES_GUITAR_AE = os.path.join(BASE_DIR, 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_guitar-AE-epoch=153-val_loss=0.01.ckpt')
AEWRES_GUITAR_SVDD = os.path.join(BASE_DIR, 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_guitar-DSVDD-epoch=34-val_loss=0.01.ckpt')
# Datasets
BASS_DATASET = os.path.join(BASE_DIR, 'data', 'pickle','embedding','musicradar','HTSAT-base_musicradar_bass_embeddings.pkl')
BASS_DATASET_INDEX = os.path.join(BASE_DIR, 'data','pickle','embedding','musicradar','HTSAT-base_musicradar_bass_embeddings_index.pkl')
GUITAR_DATASET = os.path.join(BASE_DIR, 'data','pickle','embedding','musicradar','HTSAT-base_musicradar_guitar_embeddings.pkl')
GUITAR_DATASET_INDEX = os.path.join(BASE_DIR, 'data','pickle','embedding','musicradar','HTSAT-base_musicradar_guitar_embeddings_index.pkl')

MODEL_CHOICES = {
    'bass': {
        'ae_ckpt': AEWRES_BASS_AE,
        'svdd_ckpt': AEWRES_BASS_SVDD,
        'model_key': 'AEwRES',
        'dataset': BASS_DATASET,
        'dataset_index': BASS_DATASET_INDEX
    },
    'guitar': {
        'ae_ckpt': AEWRES_GUITAR_AE,
        'svdd_ckpt': AEWRES_GUITAR_SVDD,
        'model_key': 'AEwRES',
        'dataset': GUITAR_DATASET,
        'dataset_index': GUITAR_DATASET_INDEX
    }
}

def compute_anomaly_scores(model_type):
    """Compute anomaly scores for all files in the training dataset for a given model type."""
    print(f"Computing anomaly scores for {model_type} model...")
    
    # Load config and model choice
    config = load_experiment_config(CONFIG_NAME)
    model_choice = MODEL_CHOICES[model_type]
    model_config = config['networks'][model_choice['model_key']]
    svdd_config = config['deepSVDD']
    
    # Load dataset
    print("Loading dataset...")
    data = load_pickle(model_choice['dataset'])
    data_index = load_pickle(model_choice['dataset_index'])
    print('Shape of data :', data.shape)
    print('Number of indices :', len(data_index))

    if model_config.get('num_features', None) is None:
        model_config['num_features'] = data.shape[-1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    
    detector = AnomalyDetector(
        configs=[model_config, svdd_config],
        checkpoint_paths=[model_choice['ae_ckpt'], model_choice['svdd_ckpt']],
        device=device
    )
    detector.load_models()
    
    
    results = []
    
    # Process each file
    print("Computing anomaly scores for all files...")
    for idx in tqdm(range(len(data)), desc="Processing files"):
        try:
            # Get anomaly score (distance from center) for current file
            file_data = data[idx:idx+1]  # Get single file data
            loop_data = detector.get_loop_score(file_data)
            _ , dist = loop_data[0]
        
            results.append({
                'file_id': idx,
                'file_path': data_index[idx][1],
                'anomaly_score': dist,
            })
            
        except Exception as e:
            print(f"Error processing file at index {idx}: {e}")
            # Add a default entry for failed files
            results.append({
                'file_id': idx,
                'file_path': data_index[idx][1] if idx < len(data_index) else f"unknown_{idx}",
                'anomaly_score': float('inf'),
            })
    
    # Sort by anomaly score (lowest first = most typical loops)
    results.sort(key=lambda x: x['anomaly_score'])
    
    # Save results
    output_file = os.path.join(BASE_DIR, f'anomaly_scores_{model_type}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved {len(results)} results to {output_file}")
    print(f"Top 5 loops (lowest anomaly scores):")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {os.path.basename(result['file_path'])} - Score: {result['anomaly_score']:.6f}")
    
    return results

def main():
    """Compute anomaly scores for both bass and guitar models."""
    for model_type in ['bass', 'guitar']:
        compute_anomaly_scores(model_type)
        print(f"\nCompleted {model_type} model\n")

if __name__ == '__main__':
    main()
