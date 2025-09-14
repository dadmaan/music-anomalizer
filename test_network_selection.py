#!/usr/bin/env python3
"""
Test script to verify network selection functionality in compute_anomaly_scores.py
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from music_anomalizer.scripts.compute_anomaly_scores import compute_anomaly_scores
from music_anomalizer.config.loader import load_experiment_config


def test_network_selection():
    """Test that compute_anomaly_scores works with different network selections."""
    
    print("Testing network selection functionality...")
    
    # Test with exp2_deeper config which has both AE and AEwRES networks
    config_name = 'exp2_deeper'
    config = load_experiment_config(config_name, str(project_root / 'configs'))
    
    print(f"Available networks in {config_name}: {list(config.networks.keys())}")
    
    # Test AE network
    print("\n1. Testing AE network...")
    try:
        results_bass_ae = compute_anomaly_scores(
            model_type='bass',
            config_name=config_name,
            network_key='AE'
        )
        print(f"   ✓ AE bass scores computed: {len(results_bass_ae)} files")
        
        results_guitar_ae = compute_anomaly_scores(
            model_type='guitar',
            config_name=config_name,
            network_key='AE'
        )
        print(f"   ✓ AE guitar scores computed: {len(results_guitar_ae)} files")
        
    except Exception as e:
        print(f"   ✗ AE network test failed: {e}")
    
    # Test AEwRES network
    print("\n2. Testing AEwRES network...")
    try:
        results_bass_aewres = compute_anomaly_scores(
            model_type='bass',
            config_name=config_name,
            network_key='AEwRES'
        )
        print(f"   ✓ AEwRES bass scores computed: {len(results_bass_aewres)} files")
        
        results_guitar_aewres = compute_anomaly_scores(
            model_type='guitar',
            config_name=config_name,
            network_key='AEwRES'
        )
        print(f"   ✓ AEwRES guitar scores computed: {len(results_guitar_aewres)} "
              f"files")
        
    except Exception as e:
        print(f"   ✗ AEwRES network test failed: {e}")
    
    print("\nNetwork selection test completed!")


if __name__ == '__main__':
    test_network_selection()
