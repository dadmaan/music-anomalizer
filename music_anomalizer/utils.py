"""
Reference from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/pytorch_utils.py
"""
import torch
import datetime
import os
import json
import pandas as pd
import shutil
import numpy as np
import pickle
import logging
import random
import matplotlib.pyplot as plt
from music_anomalizer.models.networks import load_AE_from_checkpoint
from pathlib import Path
import gc
import seaborn as sns
import hashlib
import argparse
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List

# Import anomaly scores manager
from music_anomalizer.anomaly_scores_manager import AnomalyScoresManager, get_anomaly_scores_manager

# MARK: General Utils
def cleanup():
    torch.cuda.empty_cache()  # Clear CUDA cache
    gc.collect()  # Force garbage collection
    
def create_folder(fd):
    """Legacy wrapper for validate_directory_path - use validate_directory_path directly."""
    is_valid, error_msg = validate_directory_path(fd, create_if_missing=True)
    if not is_valid:
        print(f"An unexpected error occurred: {error_msg}")
    else:
        if not Path(fd).exists():
            print(f"Created new directory at {fd}.")
        else:
            print("Path already exists.") 

def load_json(file_path, as_dataframe=False):
    """
    Loads a JSON file from the specified path.

    :param file_path: Path to the JSON file.
    :param as_dataframe: If True, returns flattened data as pandas DataFrame.
    :return: Parsed JSON data as a Python dictionary or DataFrame.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        if as_dataframe:
            # Flatten the data structure for DataFrame
            flat_data = []
            for item in data:
                for key, value in item.items():
                    value['hash'] = key  # Optionally add the hash as a column
                    flat_data.append(value)
            
            return pd.DataFrame(flat_data)
        
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from file {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
          
def write_to_json(data, json_file_path):
    try:
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file)
            
    except Exception as e:
        print(f"Failed to update JSON file: {e}")



def load_pickle(file_path):
    """
    Loads a pickle file from the specified file path.

    Args:
    file_path (str): The path to the pickle file.

    Returns:
    object: The object loaded from the pickle file.
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Failed to load pickle file: {e}")
        return None
    
def move_and_rename_files(file_paths, destination_dir, new_names=False):
    """
    Moves files from their current locations to a new directory and optionally renames them.

    :param file_paths: List of strings, where each string is a path to a file to be moved.
    :param destination_dir: String, the directory to which the files should be moved.
    :param new_names: Set new names for the files by index.
    """
    try:
        new_paths = []
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        if new_names and len(new_names) != len(file_paths):
            raise ValueError("The list of new names must be the same length as the list of file paths.")
        
        for index, file_path in enumerate(file_paths):
            if new_names:
                file_name = os.path.basename(f"loop_{index}.mid")
            else:
                file_name = os.path.split(file_path)[-1]
            
            new_file_path = os.path.join(destination_dir, file_name)
            new_paths.append(new_file_path)
            
            shutil.move(file_path, new_file_path)
            print(f"Moved and renamed {file_path} to {new_file_path}")
            
        return new_paths
            
    except FileNotFoundError:
        print(f"The file(s) at {file_paths} does not exist.")
    except PermissionError:
        print(f"Permission denied. You do not have the necessary permissions to move the file.")
    except Exception as e:
        print(f"An error occurred: {e}")



class StatisticsContainer(object):
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations.
        "Ref.: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/utils/utilities.py"
        """
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'train': [], 'valid': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'train': [], 'valid': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict


class PickleHandler:
    def __init__(self, file_path):
        """
        Initializes the PickleHandler with a specific file path.

        Args:
            file_path (str): The path to the pickle file where data will be stored.
        """
        self.file_path = file_path
        self.ensure_directory_exists(file_path)
        self.ensure_file_exists()

    def ensure_directory_exists(self, file_path):
        """
        Ensures that the directory for the file path exists.

        Args:
            file_path (str): The path to the file for which the directory needs to be checked.
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")

    def ensure_file_exists(self):
        """
        Ensures that the pickle file exists. If it does not, it creates an empty file.
        """
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'wb') as f:
                pass  # Create an empty pickle file
            print(f"File {self.file_path} created as it did not exist.")

    def dump_data(self, data):
        """
        Dumps data to the pickle file.
        """
        with open(self.file_path, 'wb') as f:
            pickle.dump(data, f)
        
    def append_data(self, data):
        """
        Appends data to the pickle file.

        Args:
            data (any): The data to append to the file.
        """
        with open(self.file_path, 'ab') as f:
            pickle.dump(data, f)
        
    def read_data(self):
        """
        Reads and returns all data from the pickle file as a list.
        """
        data = []
        with open(self.file_path, 'rb') as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
        return data
    
# MARK: CLAP Utils
def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)
    Returns:
      out: (batch_size, ...)
    """
    out = (
            x.transpose(0, -1) * mixup_lambda
            + torch.flip(x, dims=[0]).transpose(0, -1) * (1 - mixup_lambda)
    ).transpose(0, -1)
    return out


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def construct_file_path(config, output_dir, prefix="LP_preprocessed_data", extension=".pkl"):
    """
    Constructs a file path for saving or loading preprocessed data based on the given configuration.

    Args:
    - config (dict): Configuration dictionary containing sample_rate, target_audio_length, mono, and only_pad.
    - output_dir (str): The directory where the file will be saved or loaded from.
    - prefix (str): The prefix for the filename to indicate the type of data or processing.

    Returns:
    - str: The full path to the file.
    """
    # Extract configuration details
    sample_rate = config['sample_rate']
    target_length = config['target_audio_length']
    channel_type = 'm' if config['mono'] else 's'
    pad_type = 'np' if config['only_pad'] else 'p'

    # Construct the filename using the configuration
    filename = f"{prefix}_{sample_rate}_{target_length}_{channel_type}_{pad_type}{extension}"

    # Combine the output directory and filename to form the full path
    full_path = os.path.join(output_dir, filename)
    
    return full_path


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        "Ref.: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/utils/utilities.py"
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)
    
    

# MARK: AE/SVDD
def get_z_vector(model, train_set, device):
    train_z = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, x_train in enumerate(train_set):
            z = model.encoder(x_train.to(device))
            train_z.append(z)
            
    z_vector = torch.vstack(train_z).mean(0)
    print('center shape :', z_vector.shape)
    return z_vector

def save_z_vector(config, checkpoint_path, train_set, device):
    ae_model = load_AE_from_checkpoint(checkpoint_path, config, device)
    z_vector = get_z_vector(ae_model, train_set, device)

    file_path = os.path.splitext(checkpoint_path)[0] + '_z_vector.pkl'
    ph = PickleHandler(file_path)
    ph.dump_data(z_vector)

    print(f"AE z-vector saved at '{file_path}'")

def get_dist_from_svdd(data_set, model, z_vector, device):
    z_set = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, x in enumerate(data_set):
            z = model(x.to(device))
            z_set.append(z)
            
    z_set = torch.vstack(z_set)
    
    # compute distance
    dist = z_set - z_vector.unsqueeze(0)
    dist = dist.square().mean(1)
    dist = dist.cpu().detach().numpy()
    
    return dist


def get_test_data_dist_from_svdd(test_set, model, z_vector, device):
    test_dist = []
    with torch.no_grad():
        for i , data in enumerate(test_set):
            input_data = torch.tensor(data).float().unsqueeze(0).to(device)
            z = model(input_data)
            
            # compute distance
            dist = z - z_vector
            dist = dist.square().mean()
            test_dist.append(dist.item())

    return test_dist

def get_detected_loops(test_set, model, z_vector, threshold, device):
    loop_data = []
    
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_set):
            input_midi = torch.tensor(data).float().unsqueeze(0).to(device)
            z = model(input_midi)
            # compute distance
            dist = z - z_vector
            dist = dist.square().mean()
    
            if dist < threshold:
                loop_data.append(idx)
    
    print(f"Number of {len(loop_data)} loops detected from {len(test_set)} loops.")
    return loop_data
    

def plot_svdd_output_distribution(svdd_outputs, bins=30, title='SVDD Output Distribution'):
    """
    Plots the distribution of SVDD outputs using a histogram.

    Parameters:
    - svdd_outputs (list or np.array): The outputs from the SVDD model.
    - bins (int): Number of bins for the histogram.
    - title (str): Title of the plot.
    """
    # Convert list to numpy array if necessary
    if isinstance(svdd_outputs, list):
        svdd_outputs = np.array(svdd_outputs)
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(svdd_outputs, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    
    # Adding aesthetics
    plt.title(title)
    plt.xlabel('SVDD Output Values')
    plt.ylabel('Frequency')
    
    # Show the plot
    plt.show()
    
def plot_score_distributions(scores_train, scores_valid, threshold, title, ax, kde=False, fontsize=10):
    """
    Helper function to plot score distributions with a threshold line.

    Args:
        scores_train (array-like): Training scores.
        scores_valid (array-like): Validation scores.
        threshold (float): Threshold value for anomaly detection.
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes): Axis to plot on.
    """
    sns.histplot(scores_train, kde=kde, element='step', stat='density', label="Train", color="orange", alpha=0.6, ax=ax)
    sns.histplot(scores_valid, kde=kde, element='step', stat='density', label="Validation", color="blue", alpha=0.6, ax=ax)
    ax.axvline(threshold, color='red', linestyle='--', label="Threshold")
    ax.set_yscale('log')
    ax.set_xlabel('Anomaly Score', fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.legend()
    
# MARK: Common Script Utilities
def setup_logging(log_level: str = "INFO"):
    """Configure logging with timestamps and proper formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('music_anomalizer')
    return logger


def initialize_device(device_override: Optional[str] = None) -> torch.device:
    """Initialize compute device with validation and fallback.
    
    Args:
        device_override: Override device selection ('auto', 'cpu', 'cuda')
    
    Returns:
        Validated torch device
    """
    logger = logging.getLogger('music_anomalizer')
    
    if device_override == 'cpu':
        device = torch.device('cpu')
        logger.info("Using CPU (forced)")
    elif device_override == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA requested but not available, falling back to CPU")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Auto-selected CUDA: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Auto-selected CPU (CUDA not available)")
    
    return device


def set_deterministic_behavior():
    """Set deterministic behavior for reproducibility."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_float32_matmul_precision('medium')
    random_seed = 0
    pl.seed_everything(random_seed)


def set_random_seeds(seed: int = 0) -> int:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
        
    Returns:
        int: The seed value used
    """
    logger = logging.getLogger('music_anomalizer')
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed} for reproducibility")
    return seed

def validate_dataset(dataset_path: str, dataset_index_path: str = None, load_data: bool = False) -> Tuple[bool, str, Any]:
    """Validate dataset files exist and optionally load data.
    
    Args:
        dataset_path: Path to dataset pickle file
        dataset_index_path: Optional path to dataset index pickle file
        load_data: If True, loads and validates the dataset content
    
    Returns:
        Tuple of (is_valid, error_message, loaded_data)
    """
    logger = logging.getLogger('music_anomalizer')
    
    # Convert to Path objects for consistency
    dataset_path_obj = Path(dataset_path) if isinstance(dataset_path, str) else dataset_path
    
    if not dataset_path_obj.exists():
        return False, f"Dataset file not found: {dataset_path}", None
    
    if dataset_index_path:
        index_path_obj = Path(dataset_index_path) if isinstance(dataset_index_path, str) else dataset_index_path
        if not index_path_obj.exists():
            return False, f"Dataset index file not found: {dataset_index_path}", None
    
    try:
        # Check file sizes
        dataset_size = dataset_path_obj.stat().st_size / (1024 * 1024)  # MB
        logger.debug(f"Dataset file size: {dataset_size:.1f} MB")
        
        if dataset_size < 0.1:  # Less than 0.1 MB is suspicious
            return False, f"Dataset file seems too small: {dataset_size:.1f} MB", None
        
        if dataset_size == 0:
            return False, f"Dataset file is empty: {dataset_path}", None
            
        # Optionally load and validate data
        if load_data:
            data = load_pickle(str(dataset_path_obj))
            if data is None:
                return False, f"Dataset loaded as None: {dataset_path}", None
            
            if hasattr(data, '__len__') and len(data) == 0:
                return False, f"Dataset is empty: {dataset_path}", None
                
            logger.info(f"Dataset loaded: {len(data) if hasattr(data, '__len__') else 'unknown'} samples ({dataset_size:.1f} MB)")
            return True, "", data
            
        return True, "", None
        
    except Exception as e:
        return False, f"Error validating dataset files: {e}", None
        



def generate_md5_hash(text: str, length: int = 32) -> str:
    """Generate a shorter hash using MD5.
    
    Args:
        text (str): Text to hash
        length (int): Length of hash to return
        
    Returns:
        str: MD5 hash truncated to specified length
    """
    hash_object = hashlib.md5(text.encode())
    return hash_object.hexdigest()[:length]


def create_standard_argument_parser(
    description: str,
    epilog: str = None,
    add_common_args: bool = True
) -> argparse.ArgumentParser:
    """Create a standard argument parser with common options.
    
    Args:
        description (str): Description for the parser
        epilog (str): Epilog text for examples
        add_common_args (bool): Whether to add common arguments
        
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    
    if add_common_args:
        parser.add_argument(
            '--device',
            choices=['auto', 'cpu', 'cuda'],
            default='auto',
            help='Device to use for processing (default: auto)'
        )
        
        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
            default='INFO',
            help='Logging level (default: INFO)'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Validate configuration without processing'
        )
    
    return parser


def display_execution_summary(stats: Dict[str, Any]) -> None:
    """Display comprehensive execution summary.
    
    Args:
        stats: Processing statistics dictionary
    """
    logger = logging.getLogger('music_anomalizer')
    
    logger.info("\n" + "="*60)
    logger.info(" EXECUTION SUMMARY")
    logger.info("="*60)
    
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            if 'rate' in key.lower() or 'percentage' in key.lower():
                logger.info(f" {key.replace('_', ' ').title()}: {value:.1f}%")
            else:
                logger.info(f" {key.replace('_', ' ').title()}: {value}")
        else:
            logger.info(f" {key.replace('_', ' ').title()}: {value}")
    
    logger.info("="*60)


def box_plot_anomaly_scores(data_train, data_valid, dataset_name, ax, title=None):
    """
    Plots the anomaly scores for training and validation examples in a single boxplot with a logarithmic scale.

    Args:
        data_train (dict): Dictionary containing training results.
        data_valid (dict): Dictionary containing validation results.
        dataset_name (str): Name of the dataset to plot.
        ax (matplotlib.axes.Axes): The axes to plot on.
        title (str, optional): Title of the plot. Defaults to None.
    """
    # Custom properties for the boxplot
    boxprops = dict(linestyle='-', linewidth=1, color='darkblue')
    medianprops = dict(linestyle='-', linewidth=1, color='firebrick')
    whiskerprops = dict(linestyle='--', linewidth=1, color='gray')
    capprops = dict(linestyle='-', linewidth=1, color='gray')
    flierprops = dict(marker='o', markerfacecolor='green', markersize=2, linestyle='none')

    # Extract scores and model names
    scores_train = list(data_train[dataset_name].values())
    scores_valid = list(data_valid[dataset_name].values())
    scores = scores_train + scores_valid
    model_names = [key.split("_")[0] for key in data_train[dataset_name].keys()]
    labels = [f"{'PCA Rec.' if name == 'PCA' else name} (Train)" for name in model_names] + [f"{'PCA Rec.' if name == 'PCA' else name} (Valid)" for name in model_names]

    # Plotting
    ax.boxplot(scores, vert=False, showfliers=True, 
               boxprops=boxprops, medianprops=medianprops,
               whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops)
    ax.set_xscale('log')
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Anomaly Score (log scale)', fontsize=10)
    ax.set_title(title or 'Comparison of Anomaly Scores', fontsize=10)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


# MARK: Validation and Error Handling Utilities
def validate_file_path(file_path: str, file_type: str = "file") -> Tuple[bool, str]:
    """Validate that a file path exists and is accessible.
    
    Args:
        file_path (str): Path to validate
        file_type (str): Type of file for error messages
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not file_path:
        return False, f"{file_type.capitalize()} path is empty"
    
    path_obj = Path(file_path)
    if not path_obj.exists():
        return False, f"{file_type.capitalize()} not found: {file_path}"
    
    if not path_obj.is_file():
        return False, f"{file_type.capitalize()} path is not a file: {file_path}"
    
    # Check file size for basic validation
    file_size = path_obj.stat().st_size
    if file_size == 0:
        return False, f"{file_type.capitalize()} is empty: {file_path}"
    
    return True, ""


def validate_directory_path(dir_path: str, create_if_missing: bool = False) -> Tuple[bool, str]:
    """Validate that a directory path exists and is accessible.
    
    Args:
        dir_path (str): Directory path to validate
        create_if_missing (bool): Create directory if it doesn't exist
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    if not dir_path:
        return False, "Directory path is empty"
    
    path_obj = Path(dir_path)
    
    if not path_obj.exists():
        if create_if_missing:
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                return True, ""
            except Exception as e:
                return False, f"Could not create directory {dir_path}: {e}"
        else:
            return False, f"Directory not found: {dir_path}"
    
    if not path_obj.is_dir():
        return False, f"Path is not a directory: {dir_path}"
    
    return True, ""


def safe_file_operation(operation_func, *args, **kwargs):
    """Safely execute a file operation with error handling.
    
    Args:
        operation_func: Function to execute
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Result of operation or None if failed
    """
    logger = logging.getLogger('music_anomalizer')
    
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        return None

