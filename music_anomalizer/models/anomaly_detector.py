import os
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Iterable, Optional, Any
from music_anomalizer.models.networks import load_AE_from_checkpoint, SVDD
from music_anomalizer.utils import load_pickle

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    A class for detecting anomalies using Autoencoder (AE) and Support Vector Data Description (SVDD) models.

    Attributes:
        ae_config (Dict[str, Any]): Configuration parameters for the Autoencoder model.
        svdd_config (Dict[str, Any]): Configuration parameters for the SVDD model.
        checkpoint_paths (List[str]): List containing paths to the checkpoints for AE and SVDD models.
        device (str): The device (e.g., 'cuda' or 'cpu') on which the models will be loaded and computations performed.
        ae_model (Optional[torch.nn.Module]): The loaded Autoencoder model.
        svdd_model (Optional[torch.nn.Module]): The loaded SVDD model.
        z_vector (Optional[torch.Tensor]): The center vector for the SVDD model, loaded from a pickle file.

    Methods:
        __init__(configs, checkpoint_paths, device): Initializes the AnomalyDetector with configurations, paths, and device.
        load_models(): Loads the AE and SVDD models from their respective checkpoints and sets them to evaluation mode.
        compute_anomaly_scores(eval_set): Computes the anomaly scores for the given evaluation dataset.
    """
    def __init__(self, configs: Tuple[Dict[str, Any], Dict[str, Any]], checkpoint_paths: List[str], device: str) -> None:
        """
        Initializes the AnomalyDetector with necessary configurations, checkpoint paths, and computation device.

        Args:
            configs: A tuple containing two dictionaries for AE and SVDD configurations respectively.
            checkpoint_paths: A list with AE and SVDD checkpoint paths.
            device: The device (e.g., 'cuda' or 'cpu') to load the models on and perform computations.
        """
        self.ae_config: Dict[str, Any]
        self.svdd_config: Dict[str, Any]
        self.ae_config, self.svdd_config = configs
        self.checkpoint_paths: List[str] = checkpoint_paths
        self.device: str = device
        self.ae_model: Optional[torch.nn.Module] = None
        self.svdd_model: Optional[torch.nn.Module] = None
        self.z_vector: Optional[torch.Tensor] = None

    def load_models(self) -> None:
        """
        Loads the Autoencoder and SVDD models from their checkpoints, initializes the center vector for SVDD,
        and sets both models to evaluation mode.
        
        Raises:
            FileNotFoundError: If checkpoint files don't exist
            RuntimeError: If model loading fails
            ValueError: If device is incompatible
        """
        try:
            # Validate checkpoint paths
            if len(self.checkpoint_paths) < 2:
                raise ValueError(f"Expected 2 checkpoint paths, got {len(self.checkpoint_paths)}")
            
            ae_ckpt = self.checkpoint_paths[0]
            svdd_ckpt = self.checkpoint_paths[1]
            z_path = os.path.splitext(ae_ckpt)[0] + '_z_vector.pkl'
            
            # Check file existence
            for path, name in [(ae_ckpt, "AE checkpoint"), (svdd_ckpt, "SVDD checkpoint"), (z_path, "Z vector file")]:
                if not Path(path).exists():
                    raise FileNotFoundError(f"{name} not found at: {path}")
            
            logger.info(f"Loading models on device: {self.device}")
            
            # Load AE model with error handling
            try:
                self.ae_model = load_AE_from_checkpoint(ae_ckpt, self.ae_config, self.device)
                if self.ae_model is None:
                    raise RuntimeError("AE model loading returned None")
                self.ae_model.eval()
                logger.info("AE model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load AE model from {ae_ckpt}: {str(e)}")
            
            # Load z vector with error handling
            try:
                self.z_vector = load_pickle(z_path)
                if self.z_vector is None:
                    raise RuntimeError("Z vector loading returned None")
                # Validate z_vector is a tensor
                if not isinstance(self.z_vector, torch.Tensor):
                    raise ValueError(f"Expected z_vector to be torch.Tensor, got {type(self.z_vector)}")
                logger.info("Z vector loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load z vector from {z_path}: {str(e)}")
            
            # Load SVDD model with error handling
            try:
                self.svdd_model = SVDD.load_from_checkpoint(
                    svdd_ckpt, 
                    encoder=self.ae_model.encoder, 
                    center=self.z_vector.to(self.device),
                    learning_rate=self.svdd_config['learning_rate'], 
                    weight_decay=self.svdd_config['weight_decay']
                )
                if self.svdd_model is None:
                    raise RuntimeError("SVDD model loading returned None")
                self.svdd_model.eval()
                logger.info("SVDD model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load SVDD model from {svdd_ckpt}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            # Clean up partially loaded models
            self.ae_model = None
            self.svdd_model = None
            self.z_vector = None
            raise

    def compute_anomaly_scores(self, dataset: Iterable[Union[np.ndarray, torch.Tensor]], batch_size: Optional[int] = None) -> Dict[str, Union[np.ndarray, List[float]]]:
        """
        Computes the anomaly scores for the provided evaluation dataset using the loaded AE and SVDD models.
        Uses batch processing for improved performance and memory efficiency.

        Args:
            dataset: An iterable of data points (e.g., list or DataLoader) to evaluate.
            batch_size: Batch size for processing. If None, automatically determines optimal size.

        Returns:
            A dictionary containing 'embeddings' and 'scores'. 'embeddings' is a numpy array of embeddings
            extracted by the AE model, and 'scores' is a list of anomaly scores computed by the SVDD model.
                  
        Raises:
            RuntimeError: If models are not loaded
            ValueError: If input data format is invalid
            torch.cuda.OutOfMemoryError: If GPU memory is insufficient
        """
        # Validate models are loaded
        if self.ae_model is None or self.svdd_model is None or self.z_vector is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if not hasattr(dataset, '__iter__'):
            raise ValueError("Dataset must be iterable")
        
        # Convert dataset to list for efficient batching
        dataset_list = list(dataset)
        if not dataset_list:
            raise ValueError("Dataset is empty")
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._determine_optimal_batch_size(dataset_list)
        
        logger.info(f"Processing {len(dataset_list)} samples with batch size {batch_size}")
        
        all_embeddings: List[torch.Tensor] = []
        all_scores: List[float] = []
        
        try:
            with torch.no_grad():
                # Process in batches
                for batch_start in range(0, len(dataset_list), batch_size):
                    batch_end = min(batch_start + batch_size, len(dataset_list))
                    batch_data = dataset_list[batch_start:batch_end]
                    
                    try:
                        # Process batch
                        batch_embeddings, batch_scores = self._process_batch(batch_data, batch_start)
                        all_embeddings.extend(batch_embeddings)
                        all_scores.extend(batch_scores)
                        
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"OOM with batch size {batch_size}, falling back to single-item processing")
                        # Fallback to single item processing for this batch
                        for i, data in enumerate(batch_data):
                            try:
                                single_embeddings, single_scores = self._process_batch([data], batch_start + i)
                                all_embeddings.extend(single_embeddings)
                                all_scores.extend(single_scores)
                            except Exception as e:
                                logger.error(f"Error processing item at index {batch_start + i}: {str(e)}")
                                raise RuntimeError(f"Failed to process item at index {batch_start + i}: {str(e)}")
                    
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
                        raise RuntimeError(f"Failed to process batch {batch_start}-{batch_end}: {str(e)}")
            
            # Validate we have embeddings to concatenate
            if not all_embeddings:
                raise ValueError("No valid embeddings computed from dataset")
            
            try:
                # Concatenate all embeddings efficiently
                embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Failed to concatenate embeddings: {str(e)}")

        except Exception as e:
            logger.error(f"Anomaly score computation failed: {str(e)}")
            raise

        logger.info(f"Computed anomaly scores for {len(all_scores)} samples")
        return {'embeddings': embeddings, 'scores': all_scores}
    
    def _determine_optimal_batch_size(self, dataset: List[Union[np.ndarray, torch.Tensor]]) -> int:
        """
        Automatically determines optimal batch size based on available memory and data characteristics.
        
        Args:
            dataset: The dataset to be processed
            
        Returns:
            Optimal batch size for processing
        """
        # Start with a reasonable default
        base_batch_size = 32
        
        # Check available GPU memory if using CUDA
        device_obj = torch.device(self.device) if isinstance(self.device, str) else self.device
        if device_obj.type == 'cuda':
            try:
                # Get GPU memory info
                total_memory = torch.cuda.get_device_properties(device_obj).total_memory
                allocated_memory = torch.cuda.memory_allocated(device_obj)
                available_memory = total_memory - allocated_memory
                
                # Conservative estimate: use 30% of available memory for batching
                memory_factor = available_memory / (1024**3)  # Convert to GB
                
                if memory_factor > 8:  # >8GB available
                    base_batch_size = 64
                elif memory_factor > 4:  # >4GB available
                    base_batch_size = 32
                elif memory_factor > 2:  # >2GB available
                    base_batch_size = 16
                else:  # Limited memory
                    base_batch_size = 8
                    
            except Exception:
                # Fallback if memory detection fails
                base_batch_size = 16
        else:
            # CPU processing - more conservative
            base_batch_size = 16
        
        # Adjust based on dataset size
        dataset_size = len(dataset)
        if dataset_size < base_batch_size:
            return dataset_size
        elif dataset_size < 100:
            return min(base_batch_size // 2, dataset_size)
        
        logger.debug(f"Determined optimal batch size: {base_batch_size}")
        return base_batch_size
    
    def _process_batch(self, batch_data: List[Union[np.ndarray, torch.Tensor]], batch_start_idx: int) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Process a batch of data efficiently with vectorized operations.
        
        Args:
            batch_data: List of data items to process as a batch
            batch_start_idx: Starting index for this batch (for error reporting)
            
        Returns:
            Tuple of (embeddings_list, scores_list) for the batch
        """
        # Prepare batch tensors
        batch_tensors = []
        valid_indices = []
        
        for i, data in enumerate(batch_data):
            try:
                # Convert and validate input data
                if isinstance(data, np.ndarray):
                    if data.size == 0:
                        logger.warning(f"Empty data at index {batch_start_idx + i}, skipping")
                        continue
                    tensor = torch.tensor(data).float()
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)  # Add batch dimension
                elif isinstance(data, torch.Tensor):
                    tensor = data.float()
                    if tensor.dim() == 1:
                        tensor = tensor.unsqueeze(0)  # Add batch dimension
                else:
                    raise ValueError(f"Unsupported data type: {type(data)} at index {batch_start_idx + i}")
                
                # Validate tensor shape
                if tensor.dim() < 2:
                    raise ValueError(f"Input tensor must have at least 2 dimensions, got {tensor.dim()} at index {batch_start_idx + i}")
                
                batch_tensors.append(tensor)
                valid_indices.append(i)
                
            except Exception as e:
                logger.error(f"Error preparing data at index {batch_start_idx + i}: {str(e)}")
                raise RuntimeError(f"Failed to prepare data at index {batch_start_idx + i}: {str(e)}")
        
        if not batch_tensors:
            return [], []
        
        try:
            # Stack tensors into a single batch - handle variable sizes
            if len(set(t.shape[1:] for t in batch_tensors)) == 1:
                # All tensors have same shape - can stack efficiently
                batch_tensor = torch.stack(batch_tensors, dim=0)
                if batch_tensor.dim() > 2:
                    # Flatten to (batch_size, features) if needed
                    batch_tensor = batch_tensor.view(batch_tensor.size(0), -1)
            else:
                # Handle variable shapes by processing individually but still in batch context
                return self._process_variable_batch(batch_tensors, batch_start_idx, valid_indices)
            
            # Move to device once for entire batch
            batch_tensor = batch_tensor.to(self.device)
            
            # Compute embeddings for entire batch
            batch_embeddings = self.ae_model.encoder(batch_tensor)
            if batch_embeddings is None or batch_embeddings.numel() == 0:
                raise RuntimeError(f"AE encoder returned invalid embeddings for batch starting at {batch_start_idx}")
            
            # Compute SVDD predictions for entire batch
            batch_pred_z = self.svdd_model(batch_tensor)
            if batch_pred_z is None:
                raise RuntimeError(f"SVDD model returned None for batch starting at {batch_start_idx}")
            
            # Compute anomaly scores efficiently
            batch_scores = []
            for i, z in enumerate(batch_pred_z):
                if z.shape != self.z_vector.shape:
                    raise ValueError(f"Shape mismatch: z={z.shape}, z_vector={self.z_vector.shape} at batch index {i}")
                score = ((z - self.z_vector) ** 2).mean().item()
                batch_scores.append(score)
            
            # Split embeddings back to list for consistency
            embedding_list = [batch_embeddings[i:i+1] for i in range(batch_embeddings.size(0))]
            
            return embedding_list, batch_scores
            
        except Exception as e:
            logger.error(f"Error in batch processing starting at {batch_start_idx}: {str(e)}")
            raise RuntimeError(f"Failed to process batch starting at {batch_start_idx}: {str(e)}")
    
    def _process_variable_batch(self, batch_tensors: List[torch.Tensor], batch_start_idx: int, valid_indices: List[int]) -> Tuple[List[torch.Tensor], List[float]]:
        """
        Handle batch processing when tensors have different shapes.
        Still more efficient than original single-item processing due to device optimization.
        """
        embeddings_list = []
        scores_list = []
        
        # Process tensors on device without repeated transfers
        device_tensors = [t.to(self.device) for t in batch_tensors]
        
        for i, tensor in enumerate(device_tensors):
            try:
                # Compute embedding
                embedding = self.ae_model.encoder(tensor)
                if embedding is None or embedding.numel() == 0:
                    raise RuntimeError(f"AE encoder returned invalid embedding at batch index {i}")
                embeddings_list.append(embedding)
                
                # Compute anomaly score
                pred_z = self.svdd_model(tensor)
                if pred_z is None:
                    raise RuntimeError(f"SVDD model returned None at batch index {i}")
                
                for z in pred_z:
                    if z.shape != self.z_vector.shape:
                        raise ValueError(f"Shape mismatch: z={z.shape}, z_vector={self.z_vector.shape} at batch index {i}")
                    score = ((z - self.z_vector) ** 2).mean().item()
                    scores_list.append(score)
                    
            except Exception as e:
                logger.error(f"Error processing variable batch item {i} (global index {batch_start_idx + valid_indices[i]}): {str(e)}")
                raise RuntimeError(f"Failed to process item at batch index {i}: {str(e)}")
        
        return embeddings_list, scores_list

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

    def get_detected_loops(self, inputs: Iterable[np.ndarray], threshold: float) -> Dict[int, Dict[str, Union[float, bool, str]]]:
        """
        Returns one dictionary:
            {
                idx_0: {"distance": d0, "is_loop": bool},
                idx_1: {"distance": d1, "is_loop": bool},
                ...
            }
        You can later filter on `is_loop` to obtain only the loops.
        
        Args:
            inputs: Iterable of numpy arrays to process
            threshold: Distance threshold for loop detection
        
        Returns:
            Dictionary mapping indices to detection results
        
        Raises:
            RuntimeError: If models are not loaded
            ValueError: If threshold is invalid or inputs are malformed
        """
        # Validate models and inputs
        if self.svdd_model is None or self.z_vector is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if not hasattr(inputs, '__iter__') or not hasattr(inputs, '__len__'):
            raise ValueError("Inputs must be an iterable with length")
        
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError(f"Threshold must be a non-negative number, got {threshold}")
        
        results: Dict[int, Dict[str, Union[float, bool, str]]] = {}
        try:
            with torch.no_grad():
                for idx, data in enumerate(inputs):
                    try:
                        # Validate and convert input data
                        if isinstance(data, np.ndarray):
                            if data.size == 0:
                                logger.warning(f"Empty data at index {idx}, setting high distance")
                                results[idx] = {
                                    "distance": float('inf'),
                                    "is_loop": False,
                                    "threshold": threshold
                                }
                                continue
                            d = torch.tensor(data).float().unsqueeze(0).to(self.device)
                        else:
                            raise ValueError(f"Expected numpy array, got {type(data)} at index {idx}")
                        
                        # Compute distance
                        z = self.svdd_model(d)
                        if z is None:
                            raise RuntimeError(f"SVDD model returned None at index {idx}")
                        
                        dist = ((z - self.z_vector) ** 2).mean().item()
                        if not isinstance(dist, (int, float)) or dist < 0:
                            raise ValueError(f"Invalid distance computed: {dist} at index {idx}")
                        
                        results[idx] = {
                            "distance": dist,
                            "is_loop": dist < threshold,
                            "threshold": threshold
                        }
                        
                    except Exception as e:
                        logger.error(f"Error processing input at index {idx}: {str(e)}")
                        # Set fallback values for failed computation
                        results[idx] = {
                            "distance": float('inf'),
                            "is_loop": False,
                            "threshold": threshold,
                            "error": str(e)
                        }
            
            n_loops = sum(info["is_loop"] for info in results.values())
            logger.info(f"{n_loops} loops detected out of {len(inputs)} inputs.")
            print(f"{n_loops} loops detected out of {len(inputs)} inputs.")
            
        except Exception as e:
            logger.error(f"Loop detection failed: {str(e)}")
            raise RuntimeError(f"Failed to detect loops: {str(e)}")
        
        return results

    def get_loop_score(self, inputs: Iterable[np.ndarray]) -> List[Tuple[int, float]]:
        """ 
        Compute and return anomaly scores for each loop in the input sequence.
        
        Args:
            inputs: Iterable of input data arrays
            
        Returns:
            List of tuples (index, score) for each input
            
        Raises:
            RuntimeError: If models are not loaded
            ValueError: If inputs are malformed
        """
        # Validate models are loaded
        if self.svdd_model is None or self.z_vector is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if not hasattr(inputs, '__iter__'):
            raise ValueError("Inputs must be iterable")
        
        loop_data: List[Tuple[int, float]] = []
        
        try:
            with torch.no_grad():
                for idx, data in enumerate(inputs):
                    try:
                        # Validate and convert input data
                        if isinstance(data, np.ndarray):
                            if data.size == 0:
                                logger.warning(f"Empty data at index {idx}, skipping")
                                continue
                            d = torch.tensor(data).float().unsqueeze(0).to(self.device)
                        else:
                            raise ValueError(f"Expected numpy array, got {type(data)} at index {idx}")
                        
                        # Compute anomaly score
                        z = self.svdd_model(d)
                        if z is None:
                            raise RuntimeError(f"SVDD model returned None at index {idx}")
                        
                        dist = z - self.z_vector
                        dist = dist.square().mean()
                        score = dist.item()
                        
                        if not isinstance(score, (int, float)):
                            raise ValueError(f"Invalid score computed: {score} at index {idx}")
                        
                        loop_data.append((idx, score))
                        
                    except Exception as e:
                        logger.error(f"Error computing score at index {idx}: {str(e)}")
                        # Continue processing other inputs
                        continue
                        
        except Exception as e:
            logger.error(f"Loop scoring failed: {str(e)}")
            raise RuntimeError(f"Failed to compute loop scores: {str(e)}")
        
        logger.info(f"Computed scores for {len(loop_data)} loops")
        return loop_data
    
    def get_extreme_anomalies(self, anomaly_scores: List[float], n_samples: int = 30) -> Dict[str, List[Tuple[int, float]]]:
        """
        Sorts a list of anomaly scores and returns the entries with the highest and lowest scores.

        Args:
            anomaly_scores: A list containing anomaly scores for each input.
            n_samples: Number of samples to retrieve from each extremes.

        Returns:
            A dictionary containing two lists:
            - 'most_anomalous': The n entries with the highest anomaly scores.
            - 'least_anomalous': The n entries with the lowest anomaly scores.
        """
        indexed_scores: List[Tuple[int, float]] = list(enumerate(anomaly_scores))
        indexed_scores.sort(key=lambda x: x[1])

        least_anomalous = indexed_scores[:n_samples]
        most_anomalous = indexed_scores[-n_samples:]

        return {
            'least_anomalous': least_anomalous,
            'most_anomalous': most_anomalous
        }