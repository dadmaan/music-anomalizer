# Streamlit Refactoring Documentation

## Problem Analysis

The current `music_anomalizer/scripts/compute_anomaly_scores.py` script has a hardcoded approach to model selection that doesn't align with the flexible network selection implemented in the Streamlit pages (`app/pages/1_Overview.py` and `app/pages/2_Upload_and_Analyze.py`).

### Current Issues:

1. **Hardcoded Model Choices**: The script uses `get_default_model_choices()` which always returns 'AEwRES' as the model_key, ignoring other networks defined in config files.

2. **No Network Selection Support**: The script doesn't accept a network_key parameter, making it impossible to compute scores for different network architectures defined in configs.

3. **Inconsistent with Streamlit Pages**: The Streamlit pages properly allow users to select from available networks in config files, but the underlying computation script doesn't support this flexibility.

### Streamlit Pages Implementation (Correct Approach):

The Streamlit pages correctly implement:
- Dynamic network selection using `get_available_networks(config_name)` 
- Config-based model choices using `get_model_choices(config_name, network_key)`
- Proper checkpoint path resolution based on selected network

## Solution Plan

### 1. Modify `compute_anomaly_scores.py` to Support Network Selection

**Key Changes Needed:**
- Add `network_key` parameter to `compute_anomaly_scores()` function
- Replace hardcoded `get_default_model_choices()` with config-based logic
- Update argument parsing to include network selection
- Modify checkpoint path logic to use selected network

### 2. Implementation Details

#### Function Signature Update:
```python
def compute_anomaly_scores(model_type: str, config_name: str = DEFAULT_CONFIG,
                         network_key: str = 'AEwRES',  # New parameter
                         device: torch.device = None, 
                         output_path: Optional[Path] = None) -> List[Dict]:
```

#### Config-Based Model Choices:
Replace the hardcoded `get_default_model_choices()` with logic that:
- Loads the experiment config
- Gets available networks from `config.networks.keys()`
- Uses the selected `network_key` to determine model configuration

#### Argument Parser Enhancement:
Add network selection argument:
```python
parser.add_argument(
    '--network',
    help='Network type to use (default: AEwRES)'
)
```

### 3. Backward Compatibility

The changes should maintain backward compatibility by:
- Defaulting to 'AEwRES' when no network is specified
- Supporting existing command-line usage patterns
- Maintaining the same output file naming conventions

### 4. Integration with AnomalyScoresManager

The `AnomalyScoresManager` in `music_anomalizer/anomaly_scores_manager.py` will need updates to:
- Pass the network_key parameter to `compute_anomaly_scores()`
- Support network-specific score file naming (if needed)

## Expected Benefits

1. **Consistency**: Script behavior will match Streamlit page functionality
2. **Flexibility**: Users can compute scores for any network defined in config files
3. **Maintainability**: Reduced code duplication and better alignment with config-based approach
4. **User Experience**: More intuitive workflow that matches the UI experience

## Implementation Steps

1. Update `compute_anomaly_scores.py` function signatures and logic
2. Modify argument parsing to support network selection
3. Update `AnomalyScoresManager` to pass network information
4. Test with various config files and network types
5. Verify compatibility with existing Streamlit page integration
