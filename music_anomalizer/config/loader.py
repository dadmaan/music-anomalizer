"""Configuration loading utilities with YAML inheritance support."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from .schemas import ExperimentConfig, AudioPreprocessingConfig


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_yaml_config(config_path: str, configs_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration with inheritance support."""
    if configs_dir is None:
        configs_dir = os.path.dirname(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if 'base_config' in config:
        base_config_path = os.path.join(configs_dir, config['base_config'])
        base_config = load_yaml_config(base_config_path, configs_dir)
        # Remove base_config key from current config
        config.pop('base_config')
        # Merge configurations
        config = merge_configs(base_config, config)
    
    return config


def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate experiment configuration."""
    config_dict = load_yaml_config(config_path)
    return ExperimentConfig(**config_dict)


def load_experiment_config(config_name: str, configs_dir: str = "configs") -> ExperimentConfig:
    """Load experiment configuration by name."""
    config_path = os.path.join(configs_dir, f"{config_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return load_config(config_path)


def load_audio_preprocessing_config(config_path: str = "configs/audio_preprocessing.yaml") -> AudioPreprocessingConfig:
    """Load audio preprocessing configuration."""
    config_dict = load_yaml_config(config_path)
    return AudioPreprocessingConfig(**config_dict)