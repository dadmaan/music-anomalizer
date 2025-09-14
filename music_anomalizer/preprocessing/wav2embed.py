import laion_clap
import torch
import logging
import torch
from music_anomalizer.preprocessing.extract_embed import (
    get_features, prepare_input_dict, prepare_audio_as_tensor, 
    get_audio_branch, load_audio
)

class Wav2Embedding:
    """
    Class to perform preprocessing and embedding extraction for input audio files using CLAP models.
    """
    def __init__(self, 
                 model_ckpt_path='checkpoints/laion_clap/music_audioset_epoch_15_esc_90.14.pt',
                 audio_model='HTSAT-base',
                 device=None,
                 enable_fusion=False):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_model = audio_model
        self.enable_fusion = enable_fusion
        self.clap_model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model)
        self.clap_model.load_ckpt(model_ckpt_path) # using the full loaded model for feature engineering part; line 31
        self.audio_branch = get_audio_branch(self.clap_model) # using only the audio component of the model to extract embeddings; line 40

    def extract_embedding(self, audio_path, extract_features=True):
        """
        Extracts the embedding from the model for a single audio file.
        """
        try:
            if extract_features:
                audio_data = load_audio(audio_path, expand_dim=True)
                audio_input = get_features(audio_data, self.clap_model)
                audio_tensor = prepare_input_dict(audio_input, self.device)
            else:
                audio_data = load_audio(audio_path, expand_dim=False)
                audio_tensor = prepare_audio_as_tensor(audio_data, self.device)

            self.audio_branch.eval()

            with torch.no_grad():
                output_dict = self.audio_branch(audio_tensor, self.device)
            audio_embed = output_dict['embedding'].detach().cpu().numpy()

            del audio_tensor
            torch.cuda.empty_cache()

            return audio_embed

        except Exception as e:
            logging.error(f"Error processing file {audio_path}: {str(e)}")
            return None

    def extract_embeddings_batch(self, audio_paths, extract_features=True, max_workers=8):
        """
        Extract embeddings for a batch of audio files.
        Returns a dict: {audio_path: embedding or None}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(self.extract_embedding, path, extract_features): path for path in audio_paths}
            for future in tqdm(as_completed(future_to_path), total=len(audio_paths)):
                path = future_to_path[future]
                try:
                    emb = future.result()
                    results[path] = emb
                except Exception as e:
                    results[path] = None
        return results