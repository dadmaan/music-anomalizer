import os
import numpy as np
import torch
from music_anomalizer.models.networks import load_AE_from_checkpoint, SVDD
from music_anomalizer.utils import load_pickle

class AnomalyDetector:
    """
    A class for detecting anomalies using Autoencoder (AE) and Support Vector Data Description (SVDD) models.

    Attributes:
        ae_config (dict): Configuration parameters for the Autoencoder model.
        svdd_config (dict): Configuration parameters for the SVDD model.
        checkpoint_paths (dict): Dictionary containing paths to the checkpoints for AE and SVDD models.
        device (str): The device (e.g., 'cuda' or 'cpu') on which the models will be loaded and computations performed.
        ae_model (torch.nn.Module): The loaded Autoencoder model.
        svdd_model (torch.nn.Module): The loaded SVDD model.
        z_vector (torch.Tensor): The center vector for the SVDD model, loaded from a pickle file.

    Methods:
        __init__(configs, checkpoint_paths, device): Initializes the AnomalyDetector with configurations, paths, and device.
        load_models(): Loads the AE and SVDD models from their respective checkpoints and sets them to evaluation mode.
        compute_anomaly_scores(eval_set): Computes the anomaly scores for the given evaluation dataset.
    """
    def __init__(self, configs, checkpoint_paths, device):
        """
        Initializes the AnomalyDetector with necessary configurations, checkpoint paths, and computation device.

        Args:
            configs (tuple): A tuple containing two dictionaries for AE and SVDD configurations respectively.
            checkpoint_paths (dict): A dictionary with keys 'AE' and 'SVDD' pointing to their respective checkpoint paths.
            device (str): The device (e.g., 'cuda' or 'cpu') to load the models on and perform computations.
        """
        self.ae_config, self.svdd_config = configs
        self.checkpoint_paths = checkpoint_paths
        self.device = device
        self.ae_model = None
        self.svdd_model = None
        self.z_vector = None

    def load_models(self):
        """
        Loads the Autoencoder and SVDD models from their checkpoints, initializes the center vector for SVDD,
        and sets both models to evaluation mode.
        """
        ae_ckpt = self.checkpoint_paths[0]
        z_path = os.path.splitext(ae_ckpt)[0] +'_z_vector.pkl'
        svdd_ckpt = self.checkpoint_paths[1]

        # Load AE model
        self.ae_model = load_AE_from_checkpoint(ae_ckpt, self.ae_config, self.device)
        self.z_vector = load_pickle(z_path)
        self.ae_model.eval()

        # Load SVDD model
        self.svdd_model = SVDD.load_from_checkpoint(
            svdd_ckpt, encoder=self.ae_model.encoder, center=self.z_vector.to(self.device),
            learning_rate=self.svdd_config['learning_rate'], weight_decay=self.svdd_config['weight_decay']
        )
        self.svdd_model.eval()

    def compute_anomaly_scores(self, dataset):
        """
        Computes the anomaly scores for the provided evaluation dataset using the loaded AE and SVDD models.

        Args:
            dataset (iterable): An iterable of data points (e.g., list or DataLoader) to evaluate.

        Returns:
            dict: A dictionary containing 'embeddings' and 'scores'. 'embeddings' is a numpy array of embeddings
                  extracted by the AE model, and 'scores' is a list of anomaly scores computed by the SVDD model.
        """
        embeddings = []
        scores = []
        with torch.no_grad():
            for data in dataset:
                data = torch.tensor(data).float().unsqueeze(0).to(self.device) if isinstance(data, np.ndarray) else data.to(self.device)
                embedding = self.ae_model.encoder(data)
                embeddings.append(embedding)

                pred_z = self.svdd_model(data)
                for z in pred_z:
                    score = z - self.z_vector
                    score = score.square().mean().item()
                    scores.append(score)

        embeddings = torch.cat(embeddings, dim=0).detach().cpu().numpy()

        return {'embeddings': embeddings, 'scores': scores}
    
    # def get_detected_loops(self, inputs, threshold):
    #     """ Identify and return loops whose anomaly score falls below a specified threshold. """
    #     loop_data = []
    #     distances = []

    #     with torch.no_grad():
    #         for idx, data in enumerate(inputs):
    #             d = torch.tensor(data).float().unsqueeze(0).to(self.device)
    #             z = self.svdd_model(d)
    #             dist = z - self.z_vector
    #             dist = dist.square().mean()
    #             distances.append((idx, dist.item()))
                
    #             if dist < threshold:
    #                 loop_data.append((idx, dist.item()))

    #     print(f"Number of {len(loop_data)} loops detected from {len(inputs)} loops.")

    #     return loop_data, distances

    def get_detected_loops(self, inputs, threshold: float):
        """
        Returns one dictionary:
            {
                idx_0: {"distance": d0, "is_loop": bool},
                idx_1: {"distance": d1, "is_loop": bool},
                ...
            }
        You can later filter on `is_loop` to obtain only the loops.
        """
        results = {}
        with torch.no_grad():
            for idx, data in enumerate(inputs):
                d   = torch.tensor(data).float().unsqueeze(0).to(self.device)
                z   = self.svdd_model(d)
                dist = ((z - self.z_vector) ** 2).mean().item()
                results[idx] = {
                    "distance": dist,
                    "is_loop":  dist < threshold,
                    "threshold": threshold
                }
        n_loops = sum(info["is_loop"] for info in results.values())
        print(f"{n_loops} loops detected out of {len(inputs)} inputs.")
        return results

    def get_loop_score(self, inputs):
        """ Compute and return anomaly scores for each loop in the input sequence. """
        loop_data = []

        with torch.no_grad():
            for idx, data in enumerate(inputs):
                d = torch.tensor(data).float().unsqueeze(0).to(self.device)
                z = self.svdd_model(d)
                dist = z - self.z_vector
                dist = dist.square().mean()
                loop_data.append((idx, dist.item()))

        return loop_data
    
    def get_extreme_anomalies(self, anomaly_scores, n_samples=30):
        """
        Sorts a list of anomaly scores and returns the entries with the highest and lowest scores.

        Parameters:
            anomaly_scores (list of float): A list containing anomaly scores for each input.
            n_samples (int): Number of samples to retrieve from each extremes.

        Returns:
            dict: A dictionary containing two lists:
                - 'most_anomalous': The n entries with the highest anomaly scores.
                - 'least_anomalous': The n entries with the lowest anomaly scores.
        """
        indexed_scores = list(enumerate(anomaly_scores))
        indexed_scores.sort(key=lambda x: x[1])

        least_anomalous = indexed_scores[:n_samples]
        most_anomalous = indexed_scores[-n_samples:]

        return {
            'least_anomalous': least_anomalous,
            'most_anomalous': most_anomalous
        }