"""
Test script for verifying the train_models.py functionality.

This script tests the imports and basic functionality without actually training models.
"""

import os
import sys
import torch

# Add the project root to the Python path to enable module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_anomalizer.utils import load_json, load_pickle
from music_anomalizer.config import load_experiment_config


def test_imports():
    """Test that all required modules can be imported."""
    try:
        from music_anomalizer.models.deepSVDD import DeepSVDDTrainer
        print("✓ DeepSVDDTrainer imported successfully")
    except Exception as e:
        print(f"✗ Failed to import DeepSVDDTrainer: {e}")
        return False
    
    try:
        from music_anomalizer.utils import write_to_json, create_folder, move_and_rename_files, PickleHandler
        print("✓ Utility functions imported successfully")
    except Exception as e:
        print(f"✗ Failed to import utility functions: {e}")
        return False
        
    return True


def test_config_loading():
    """Test that configuration can be loaded."""
    try:
        configs = load_experiment_config("exp1")
        print("✓ Configuration loaded successfully")
        print(f"  - Config name: {configs.config_name}")
        print(f"  - Networks: {list(configs.networks.keys())}")
        print(f"  - Datasets: {list(configs.dataset_paths.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def test_data_loading():
    """Test that dataset files can be loaded."""
    try:
        configs = load_experiment_config("exp1")
        for dataset_name, path in configs.dataset_paths.items():
            if os.path.exists(path):
                data = load_pickle(path)
                print(f"✓ Dataset '{dataset_name}' loaded successfully")
                print(f"  - Data type: {type(data)}")
                if hasattr(data, 'shape'):
                    print(f"  - Data shape: {data.shape}")
            else:
                print(f"⚠ Dataset file '{path}' not found")
        return True
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False


def main():
    """Main test function."""
    print("Testing train_models.py functionality...\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Loading Test", test_config_loading),
        ("Data Loading Test", test_data_loading)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The script should work correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
