import argparse
import os
import torch
from modules.utils import load_json
from modules.anomaly_detector import AnomalyDetector
from modules.wav2embed import Wav2Embedding

# Paths
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'configs', 'exp2_deeper.json')
CLAP_CKPT = os.path.join(os.path.dirname(__file__), 'checkpoints', 'laion_clap', 'music_audioset_epoch_15_esc_90.14.pt')
AEWRES_BASS_AE = os.path.join(os.path.dirname(__file__), 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_bass-AE-epoch=149-val_loss=0.01.ckpt')
AEWRES_BASS_SVDD = os.path.join(os.path.dirname(__file__), 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_bass-DSVDD-epoch=132-val_loss=0.00.ckpt')
AEWRES_GUITAR_AE = os.path.join(os.path.dirname(__file__), 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_guitar-AE-epoch=153-val_loss=0.01.ckpt')
AEWRES_GUITAR_SVDD = os.path.join(os.path.dirname(__file__), 'checkpoints', 'loop_benchmark', 'EXP2_DEEPER', 'AEwRES', 'AEwRES-HTSAT_base_musicradar_guitar-DSVDD-epoch=34-val_loss=0.01.ckpt')

MODEL_CHOICES = {
    'bass': {
        'ae_ckpt': AEWRES_BASS_AE,
        'svdd_ckpt': AEWRES_BASS_SVDD,
        'threshold_key': 'AEwRES_bass',
        'model_key': 'AEwRES',
    },
    'guitar': {
        'ae_ckpt': AEWRES_GUITAR_AE,
        'svdd_ckpt': AEWRES_GUITAR_SVDD,
        'threshold_key': 'AEwRES_guitar',
        'model_key': 'AEwRES',
    }
}

def main():
    parser = argparse.ArgumentParser(description='Detect if a WAV file is a loop using anomaly detection.')
    parser.add_argument('wav_path', type=str, help='Path to input WAV file')
    parser.add_argument('--model', type=str, choices=['bass', 'guitar'], required=True, help='Model type to use (bass or guitar)')
    parser.add_argument('--threshold', type=float, default=None, help='Threshold for loop detection (default: from config)')
    args = parser.parse_args()

    # Load config
    config = load_json(CONFIG_PATH)
    model_choice = MODEL_CHOICES[args.model]
    model_config = config['networks'][model_choice['model_key']]
    svdd_config = config['deepSVDD']
    threshold = args.threshold
    if threshold is None:
        threshold = config['threshold'][model_choice['threshold_key']]

    # Set num_features if needed (not used for inference, but required by model)
    # If you want to infer num_features from a dataset, you can add that logic here.
    if model_config.get('num_features', None) is None:
        model_config['num_features'] = 1024  # Default, or infer from training data if available

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load embedding extractor
    extractor = Wav2Embedding(
        model_ckpt_path=CLAP_CKPT,
        audio_model='HTSAT-base',
        device=device
    )
    embedding = extractor.extract_embedding(args.wav_path)
    if embedding is None:
        print('Failed to extract embedding from input WAV file.')
        return

    # Prepare embedding for detector (expects a batch/list)
    embedding = [embedding.squeeze()]  # Squeeze in case shape is (1, N)

    # Load anomaly detector
    detector = AnomalyDetector(
        configs=[model_config, svdd_config],
        checkpoint_paths=[model_choice['ae_ckpt'], model_choice['svdd_ckpt']],
        device=device
    )
    detector.load_models()

    # Run detection
    results = detector.get_detected_loops(embedding, threshold)
    is_loop = results[0]['is_loop']
    print(f'Input: {args.wav_path}')
    print(f'Model: {args.model}')
    print(f'Threshold: {threshold}')
    print(f'Distance from center: { results[0]['distance']:.6f}')
    if is_loop:
        print('Result: LOOP DETECTED')
    else:
        print('Result: Not a loop')

if __name__ == '__main__':
    main()
