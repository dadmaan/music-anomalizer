# Script Conversion Reference

This document provides a reference for the conversion of Jupyter notebooks to Python scripts, detailing the changes made and the purpose of each script.

## Converted Scripts

### 1. preprocess_wav_loops.py
**Source:** `1_preprocess_wav_loops.ipynb`

**Purpose:** 
- Loads and preprocesses audio files with configurable parameters
- Adjusts audio length to a target duration
- Uses multiprocessing for efficient processing
- Saves processed data with labels as a pickle file

**Key Functions:**
- `encode_audio_tags()`: Encodes audio tags using MultiLabelBinarizer
- `process_audio_length()`: Adjusts audio to target length by repeating, padding, or truncating
- `process_audio_file()`: Processes a single audio file
- `run_process()`: Main processing function that handles multiprocessing

**Enhanced Features (2025-08):**
- **Configuration Integration**: Added support for audio preprocessing configuration loading
- **Improved Error Handling**: Better validation and error messages

**Usage:**
```bash
python music_anomalizer/scripts/preprocess_wav_loops.py
```

### 2. embedding_extraction_wav.py
**Source:** `4_embedding_extraction_wav.ipynb`

**Purpose:**
- Extracts embeddings from audio files using the CLAP model
- Processes WAV files using concurrent processing with configurable parameters
- Saves embeddings and index data as pickle files with flexible output naming

**Key Functions:**
- `extract_embedding()`: Extracts embedding from a single audio file
- `worker()`: Worker function for concurrent processing
- `run_process()`: Main processing function that handles concurrent execution
- `initialize_device()`: Device initialization with preference selection
- `initialize_clap_model()`: CLAP model loading with validation
- `process_dataset()`: Complete dataset processing pipeline

**Enhanced Features (2025-08):**
- **Command-line Interface**: Configurable dataset paths and processing options
- **Device Selection**: Explicit device control with fallback logic
- **Robust Error Handling**: Model loading validation and graceful failure handling
- **Flexible I/O**: Configurable input/output directories
- **Better Logging**: Structured logging with timestamps
- **Path Validation**: Checks for dataset and checkpoint existence

**Usage:**
```bash
# Basic usage with defaults
python music_anomalizer/scripts/embedding_extraction_wav.py

# Process specific dataset
python music_anomalizer/scripts/embedding_extraction_wav.py --dataset data/dataset/guitar --output data/embeddings

# Control processing parameters
python music_anomalizer/scripts/embedding_extraction_wav.py --device cuda --workers 16 --model HTSAT-base

# Use custom checkpoint
python music_anomalizer/scripts/embedding_extraction_wav.py --checkpoint path/to/clap_model.pt
```

### 3. hp_tuning_loop_detection.py
**Source:** `6_hp_tuning_loop_detecton.ipynb`

**Purpose:**
- Performs systematic hyperparameter tuning for AutoEncoder models using Weights & Biases (wandb) sweeps
- Explores different configurations to identify optimal parameters that minimize validation loss
- Orchestrates training of multiple model configurations using PyTorch Lightning
- Analyzes results with statistical summaries and visualizations
- Provides comprehensive error handling and progress tracking

**Key Functions:**
- `setup_logging()`: Configurable logging with timestamps and formatting
- `set_random_seeds()`: Sets random seeds for reproducibility across all libraries  
- `initialize_device()`: Device initialization with fallback support
- `validate_dataset()`: Validates and loads datasets with comprehensive checks
- `prepare_data_loaders()`: Prepares train/validation data loaders with error handling
- `create_sweep_config()`: Creates comprehensive sweep configuration with flexible parameters
- `train_model()`: Trains AutoEncoder model with comprehensive error handling
- `run_hyperparameter_sweep()`: Executes sweep with progress tracking and recovery
- `analyze_results()`: Analyzes results with statistical summaries and visualizations
- `create_analysis_plots()`: Creates visualization plots for hyperparameter analysis
- `main()`: Main orchestration function with comprehensive parameter control

**Enhanced Features (2025-08-20):**
- **Command-line Interface**: Comprehensive CLI with configurable datasets and sweep parameters
- **Robust Error Handling**: Continues sweep despite individual run failures with comprehensive recovery
- **Advanced Logging**: Structured logging with progress tracking and emoji status indicators  
- **Dataset Validation**: Pre-execution validation with integrity checks and sample counting
- **Flexible Configuration**: Customizable sweep parameters, methods, and hyperparameter ranges
- **Statistical Analysis**: Comprehensive analysis with top results, summaries, and visualizations
- **WandB Integration**: Conditional import handling and graceful degradation without WandB
- **Memory Management**: Proper cleanup and resource management throughout sweep execution
- **Modular Architecture**: Well-separated functions for maintainability and testing
- **Results Management**: Save/load results to CSV with analysis-only mode

**Usage:**
```bash
# Basic hyperparameter tuning with defaults
python music_anomalizer/scripts/hp_tuning_loop_detection.py

# Use different dataset
python music_anomalizer/scripts/hp_tuning_loop_detection.py --data data/bass_embeddings.pkl

# GPU training with more sweep runs
python music_anomalizer/scripts/hp_tuning_loop_detection.py --device cuda --runs 50

# Bayesian optimization with custom epochs
python music_anomalizer/scripts/hp_tuning_loop_detection.py --method bayes --epochs 100

# Analyze existing results without running new sweep
python music_anomalizer/scripts/hp_tuning_loop_detection.py --analyze results.csv

# Custom WandB project with specific seed
python music_anomalizer/scripts/hp_tuning_loop_detection.py --project MyProject --seed 42

# Save results and disable plots (headless)  
python music_anomalizer/scripts/hp_tuning_loop_detection.py --output sweep_results.csv --no-plots

# Debug mode with verbose logging
python music_anomalizer/scripts/hp_tuning_loop_detection.py --log-level DEBUG --runs 5
```

### 4. train_models.py
**Source:** `7_train_models.ipynb`

**Purpose:**
- Trains DeepSVDD models on different network configurations and datasets
- Loads configurations from YAML files with validation
- Organizes trained models in a structured directory with checkpoint registry
- Provides command-line interface for flexible training execution

**Key Functions:**
- `main(config_name, device_override)`: Main training orchestration with error handling
- Command-line argument parsing with validation options

**Enhanced Features (2025-08):**
- **Command-line Interface**: Configurable experiments via CLI arguments
- **Device Selection**: Explicit device control (auto/cpu/cuda) with fallback
- **Configuration Validation**: Pydantic schema validation with detailed error messages
- **Robust Error Handling**: Continues training other models if one fails
- **Dataset Validation**: Checks dataset existence and sample count
- **Checkpoint Registry Integration**: Registers trained models for automatic discovery
- **Dry-run Mode**: Validates configuration without expensive training

**Usage:**
```bash
# Basic training with default config
python music_anomalizer/scripts/train_models.py

# Train specific experiment
python music_anomalizer/scripts/train_models.py --config exp2_deeper

# Validate configuration only
python music_anomalizer/scripts/train_models.py --config exp1 --dry-run

# Force CPU training
python music_anomalizer/scripts/train_models.py --config exp1 --device cpu

# GPU training with validation
python music_anomalizer/scripts/train_models.py --config exp2_deeper --device cuda

# Verbose logging for debugging
python music_anomalizer/scripts/train_models.py --log-level DEBUG

# Complete refactor with enhanced features (2025-08-20)
python music_anomalizer/scripts/train_models.py --config exp1 --device auto --log-level INFO
```

**Error Handling Improvements:**
- Dataset loading failures don't stop entire training process
- Configuration validation before expensive operations
- Device availability checking with graceful fallback
- Clear error messages for troubleshooting

### 5. compare_analyze_models.py
**Source:** `8_compare_analyze_models.ipynb`

**Purpose:**
- Compares and analyzes trained models across different datasets
- Computes anomaly scores for training and validation sets
- Visualizes latent space representations of the data
- Determines anomaly thresholds using statistical methods
- Logs results to Weights & Biases for experiment tracking

**Key Functions:**
- `process_train_set()`: Processes training set for all models and datasets
- `evaluate_models_on_validation_set()`: Evaluates models on validation sets
- `log_figures_to_wandb()`: Logs visualization figures to Weights & Biases
- `main()`: Main function that orchestrates the comparison and analysis process

**Usage:**
```bash
python script/compare_analyze_models.py
```

### 6. main_exp_benchmark.py
**Source:** `11_main_exp_benchmark.ipynb`

**Purpose:**
- Runs comprehensive benchmark evaluations of different anomaly detection models on music datasets
- Evaluates baseline models (Isolation Forest, PCA Reconstruction Error)
- Evaluates Deep SVDD models with latent space visualization
- Performs statistical analysis and generates comparative plots
- Provides comprehensive reporting and error handling

**Key Functions:**
- `setup_logging()`: Configurable logging with timestamps and formatting
- `set_random_seeds()`: Sets random seeds for reproducibility across all libraries
- `initialize_device()`: Device initialization with fallback support
- `validate_configuration()`: Validates experiment configuration and prerequisites
- `validate_dataset()`: Validates and loads datasets with comprehensive checks
- `evaluate_baseline_models()`: Evaluates baseline models with robust error handling
- `evaluate_deep_svdd_models()`: Evaluates Deep SVDD models with visualization and error handling
- `organize_results()`: Organizes results with error handling and summary statistics
- `create_visualizations()`: Creates comprehensive visualizations with adaptive layouts
- `perform_statistical_analysis()`: Performs statistical analysis with configurable target datasets
- `display_benchmark_summary()`: Displays comprehensive execution summary
- `main()`: Main orchestration function with comprehensive parameter control

**Enhanced Features (2025-08-20):**
- **Command-line Interface**: Comprehensive CLI with configurable parameters and examples
- **Robust Error Handling**: Continues evaluation despite individual model/dataset failures
- **Advanced Logging**: Structured logging with progress tracking and emoji status indicators
- **Configuration Validation**: Pre-execution validation of configs, datasets, and checkpoints
- **Flexible Visualization**: Adaptive plot layouts with optional saving and display control
- **Statistical Analysis**: Configurable target datasets for pairwise significance testing
- **Dry-run Mode**: Configuration validation without expensive model evaluation
- **Device Management**: Intelligent device selection with detailed feedback
- **Memory Management**: Proper cleanup and memory management throughout evaluation
- **Modular Architecture**: Well-separated functions for maintainability and testing

**Usage:**
```bash
# Basic benchmark with default settings
python music_anomalizer/scripts/main_exp_benchmark.py

# Use different experiment configuration
python music_anomalizer/scripts/main_exp_benchmark.py --config exp1

# GPU evaluation with specific seed
python music_anomalizer/scripts/main_exp_benchmark.py --device cuda --seed 42

# Custom threshold without visualizations  
python music_anomalizer/scripts/main_exp_benchmark.py --threshold 0.9 --no-viz

# Save plots with verbose logging
python music_anomalizer/scripts/main_exp_benchmark.py --save-plots --log-level DEBUG

# Validate configuration only
python music_anomalizer/scripts/main_exp_benchmark.py --dry-run

# Headless environment (no plot display)
python music_anomalizer/scripts/main_exp_benchmark.py --no-display --save-plots

# Target specific dataset for statistical analysis
python music_anomalizer/scripts/main_exp_benchmark.py --target-dataset HTSAT_base_musicradar_bass
```

### 7. loop_evaluation.py
**Source:** `9_test.ipynb`

**Purpose:**
- Evaluates detected loops using statistical analysis and visualization
- Loads and processes MIDI loop data for evaluation
- Performs statistical tests on loop scores
- Generates visualizations of loop score distributions

**Key Functions:**
- `load_test_data()`: Loads test data for loop evaluation
- `evaluate_loops()`: Evaluates loops using a trained SVDD model
- `perform_statistical_test()`: Performs statistical tests on evaluation distances
- `plot_loop_score_distribution()`: Plots the distribution of loop scores
- `main()`: Main function that demonstrates the loop evaluation functionality

**Usage:**
```bash
python script/loop_evaluation.py
```

### 8. extract_metadata.py
**Source:** `extract_meta_data.ipynb`

**Purpose:**
- Extracts metadata from audio files including BPM, duration, and keywords
- Communicates with a tempo detection service via HTTP requests
- Processes audio files using multiprocessing for efficiency
- Saves extracted metadata to JSON files

**Key Functions:**
- `generate_md5_hash()`: Generates a hash for file identification
- `load_existing_metadata()`: Loads existing metadata from JSON file
- `update_metadata_file()`: Updates the metadata JSON file
- `extract_bpm_from_path()`: Extracts BPM from file path patterns
- `extract_keywords()`: Extracts and processes keywords from file paths
- `manage_bpm_keywords()`: Manages BPM keywords in the keywords list
- `get_tempo_from_detector()`: Communicates with tempo detection service
- `process_file()`: Processes a single audio file to extract metadata
- `process_audio_files()`: Processes all audio files in a specified folder
- `main()`: Main function that orchestrates the metadata extraction process

**Usage:**
```bash
python script/extract_metadata.py
```

## Changes Made During Conversion

### General Improvements:
1. **Added proper module documentation:** Each script now has a comprehensive docstring explaining its purpose and functionality.
2. **Structured code with functions:** Code is organized into logical functions with clear purposes and documentation.
3. **Added main() function and if-name-main pattern:** Each script can be run directly or imported as a module.
4. **Improved error handling:** Added appropriate try/except blocks where needed.
5. **Enhanced code comments:** Added detailed comments explaining complex operations.

### Specific Changes by Script:

#### preprocess_wav_loops.py:
- Converted notebook cells into logical functions
- Added comprehensive docstrings for all functions
- Added a main() function to orchestrate the preprocessing pipeline
- Improved error messages with more context

#### embedding_extraction_wav.py:
- Restructured code into logical functions with clear responsibilities
- Added detailed documentation for all functions
- Added a main() function to run the embedding extraction pipeline
- Improved variable naming for clarity

#### hp_tuning_loop_detection.py:
- Separated data loading, training, and analysis into distinct functions
- Added comprehensive documentation for all functions
- Added a main() function to orchestrate the hyperparameter tuning process
- Made result analysis more robust with error handling

#### compare_analyze_models.py:
- Converted notebook cells into logical functions with clear responsibilities
- Added comprehensive documentation for all functions
- Added a main() function to orchestrate the model comparison and analysis process
- Improved code structure with proper separation of concerns
- Added proper error handling and logging

#### main_exp_benchmark.py:
- Converted notebook cells into logical functions with clear responsibilities
- Added comprehensive documentation for all functions
- Added a main() function to orchestrate the experiment benchmark process
- Improved code structure with proper separation of concerns for different evaluation phases
- Organized code into distinct functions for baseline evaluation, Deep SVDD evaluation, result organization, visualization, and statistical analysis
- Added proper error handling and logging

#### loop_evaluation.py:
- Converted notebook cells into logical functions with clear responsibilities
- Added comprehensive documentation for all functions
- Added a main() function to demonstrate the loop evaluation functionality
- Organized code into distinct functions for data loading, loop evaluation, statistical testing, and visualization
- Added proper error handling and logging

## Configuration System Migration (2025-08)

### Migration from JSON to YAML Configuration System

**Purpose:** Improved configuration management with inheritance, validation, and better readability.

**Key Improvements:**
1. **YAML Format**: Enables comments for better documentation of configuration choices
2. **Configuration Inheritance**: Base configuration file reduces duplication across experiments
3. **Schema Validation**: Pydantic models provide type safety and validation
4. **Modular Structure**: Organized configuration loading utilities

**New Configuration Structure:**
```
configs/
├── base.yaml              # Base configuration with common defaults
├── exp1.yaml             # Experiment 1 configuration (inherits from base)
├── exp2_deeper.yaml      # Experiment 2 deeper networks (inherits from base)
└── audio_preprocessing.yaml  # Audio preprocessing configuration
```

**Schema Components:**
- `NetworkConfig`: Neural network architecture validation
- `DeepSVDDConfig`: Deep SVDD training parameters  
- `TrainerConfig`: Training loop configuration
- `AudioPreprocessingConfig`: Audio processing parameters
- `ExperimentConfig`: Main experiment configuration container

**Configuration Loading:**
```python
from music_anomalizer.config import load_experiment_config

# Load with validation
config = load_experiment_config("exp2_deeper")
```

**Migration Details:**
- Converted `configs/deeper_network.json` → `configs/exp2_deeper.yaml`
- Added comprehensive comments explaining parameter choices
- Implemented inheritance to reduce duplication
- Added validation to catch configuration errors early
- Updated all scripts to use new YAML config system:
  - `main_exp_benchmark.py` - Updated config loading and checkpoint discovery
  - `compare_analyze_models.py` - Migrated to YAML config with validation
  - `compute_anomaly_scores.py` - Updated config loading approach
  - `loop_detector.py` - Migrated to new config system
  - `test_train_models.py` - Updated for YAML configuration
  - `train_models.py` - Major improvements with CLI and error handling

**Validation Features:**
- Type checking for all configuration parameters
- Range validation (e.g., dropout rate 0-1, positive learning rates)
- Required field validation
- Custom validators for domain-specific constraints

### Checkpoint Management System (2025-08)

**Purpose:** Robust checkpoint management with automatic discovery and validation.

**Key Components:**
- `CheckpointRegistry`: Automatic checkpoint discovery and validation
- `CheckpointConfig`: Configuration for checkpoint naming and organization
- Integration with experiment configurations

**Features:**
- **Automatic Discovery**: Scans checkpoint directories to find best models
- **Validation Loss Selection**: Automatically selects checkpoints with lowest validation loss
- **Path Validation**: Ensures checkpoint files exist before use
- **Flexible Override**: Supports manual path specification when needed

**Checkpoint Directory Structure:**
```
checkpoints/
└── loop_benchmark/
    └── EXP2_DEEPER/
        ├── AE/
        │   ├── *-AE-epoch=*.ckpt
        │   └── *-DSVDD-epoch=*.ckpt
        └── AEwRES/
            ├── *-AE-epoch=*.ckpt
            └── *-DSVDD-epoch=*.ckpt
```

**Usage:**
```python
from music_anomalizer.config import get_checkpoint_registry

registry = get_checkpoint_registry()
checkpoints = registry.get_experiment_checkpoints("EXP2_DEEPER")
```

## Script Enhancement Session (2025-08)

### Comprehensive Script Improvements

**Purpose:** Modernize all scripts with robust CLI interfaces, error handling, and configuration integration.

**Scripts Enhanced:**

#### train_models.py (Major Overhaul - Enhanced 2025-08-20)
- **Command-line Interface**: Added configurable experiments via `--config`, `--device`, `--dry-run`, `--log-level`
- **Device Management**: Explicit device selection with automatic fallback and detailed device info logging
- **Configuration Validation**: Pydantic schema validation with detailed error reporting and dataset path verification
- **Robust Error Handling**: Continues training if individual models fail, with comprehensive error tracking and recovery
- **Dataset Validation**: Advanced checks for dataset existence, file size, sample count, and data integrity
- **Checkpoint Registry**: Integration with automatic checkpoint discovery system and legacy JSON compatibility
- **Comprehensive Logging**: Structured logging with timestamps, progress indicators, and emoji-based status messages
- **Modular Architecture**: Refactored into focused functions for better maintainability and testing
- **Training Progress**: Real-time progress tracking with combination counters and success/failure statistics
- **Enhanced CLI**: Rich help text with usage examples and comprehensive argument validation
- **Return Codes**: Proper exit codes for integration with CI/CD pipelines and automation scripts
- **Pydantic V2 Compatibility**: Updated from deprecated `.dict()` to `.model_dump()` method calls

#### embedding_extraction_wav.py (Significant Enhancement)
- **CLI Interface**: Added `--dataset`, `--output`, `--device`, `--workers`, `--model` arguments
- **Device Management**: Configurable device selection with validation
- **Model Loading**: Robust CLAP model initialization with checkpoint validation
- **Flexible Processing**: Configurable dataset paths and output directories
- **Better Logging**: Structured logging with timestamps and proper error reporting
- **Path Validation**: Dataset and checkpoint existence verification

#### hp_tuning_loop_detection.py (Module Fixes)
- **Import Resolution**: Fixed missing module imports using new package structure
- **Dependency Updates**: Uses `AutoEncoder` and `DatasetSampler` from proper modules

#### preprocess_wav_loops.py (Configuration Integration)
- **Config Loading**: Added audio preprocessing configuration support
- **Better Structure**: Improved import organization

#### test_train_models.py (Pydantic Migration)
- **Type-safe Access**: Updated to use Pydantic attributes instead of dictionary access
- **Configuration Testing**: Enhanced validation of YAML configuration loading

#### All Config-using Scripts (Systematic Update)
- **YAML Migration**: All scripts now use `load_experiment_config()` instead of `load_json()`
- **Validation**: Configuration validation through Pydantic schemas
- **Error Handling**: Proper error messages for configuration issues
- **Consistency**: Unified configuration loading patterns across all scripts

**New Features Added:**
- **Automatic Checkpoint Discovery**: Scripts can find best models automatically
- **Device Selection**: All scripts support explicit device control
- **Configuration Validation**: All configs validated before expensive operations
- **Dry-run Modes**: Validate configurations without running expensive operations
- **Better Error Recovery**: Scripts continue processing when individual operations fail

**Benefits Achieved:**
1. **User Experience**: Clear CLI interfaces with helpful arguments
2. **Reliability**: Comprehensive error handling and validation
3. **Flexibility**: Configurable parameters for different use cases
4. **Maintainability**: Consistent patterns across all scripts
5. **Performance**: Better device management and resource handling

## Latest Refactoring Session (2025-08-20)

### train_models.py Complete Overhaul

**Purpose:** Major structural refactoring following reference documentation conventions with enhanced observability and reliability.

**Key Architectural Changes:**
1. **Modular Function Design**: Broke down monolithic main() function into focused, testable components:
   - `setup_logging()`: Configurable logging with timestamps and formatting
   - `initialize_device()`: Robust device initialization with detailed feedback
   - `validate_dataset()`: Comprehensive dataset validation with integrity checks
   - `train_model_combination()`: Single model training with isolated error handling
   - `organize_model_files()`: File organization with proper error recovery
   - `display_training_summary()`: Rich training summary with statistics

2. **Enhanced Error Handling & Recovery**:
   - Individual model failures don't abort entire training process
   - Comprehensive error tracking with detailed logging
   - Graceful degradation when datasets or models fail
   - Return codes for CI/CD integration

3. **Advanced Logging & Observability**:
   - Structured logging with timestamps and severity levels
   - Progress indicators with emoji-based status messages
   - Real-time training progress with combination counters
   - Success/failure statistics and detailed summaries
   - Configurable log levels (DEBUG, INFO, WARNING, ERROR)

4. **Configuration & Validation Improvements**:
   - Enhanced dry-run mode with dataset path verification
   - File size and integrity checks for datasets
   - Advanced device detection with fallback mechanisms
   - Pydantic v2 compatibility (model_dump() vs dict())

5. **CLI Enhancement**:
   - Rich help text with usage examples
   - Comprehensive argument validation
   - Support for --log-level parameter
   - Proper exit codes for automation

**Implementation Benefits:**
- **Maintainability**: Clear separation of concerns with focused functions
- **Testability**: Each function can be tested independently
- **Reliability**: Continues processing despite individual failures
- **Observability**: Comprehensive logging and progress tracking
- **User Experience**: Clear feedback and helpful error messages
- **Integration**: Proper exit codes for CI/CD pipelines

**Technical Improvements:**
- Fixed import issues that prevented script execution
- Removed unused imports (datetime, Path, nn, load_json)
- Updated deprecated Pydantic .dict() calls to .model_dump()
- Enhanced type hints for better code documentation
- Improved error message clarity and actionability

### main_exp_benchmark.py Complete Overhaul (2025-08-20)

**Purpose:** Major structural refactoring following reference documentation conventions with comprehensive benchmark capabilities.

**Key Architectural Changes:**
1. **Modular Function Design**: Broke down monolithic functions into focused, testable components:
   - `setup_logging()`: Configurable logging with timestamps and proper formatting
   - `initialize_device()`: Robust device initialization with detailed feedback and fallback
   - `validate_configuration()`: Pre-execution validation of configs, datasets, and checkpoints
   - `validate_dataset()`: Comprehensive dataset validation with integrity and size checks
   - `evaluate_baseline_models()`: Baseline model evaluation with robust error handling
   - `evaluate_deep_svdd_models()`: Deep SVDD evaluation with visualization and memory management
   - `organize_results()`: Result organization with error handling and summary statistics
   - `create_visualizations()`: Adaptive visualization with flexible layouts and error recovery
   - `perform_statistical_analysis()`: Statistical testing with configurable target datasets
   - `display_benchmark_summary()`: Comprehensive execution summary with success rates

2. **Enhanced Error Handling & Recovery**:
   - Individual model/dataset failures don't abort entire benchmark
   - Comprehensive error tracking with detailed logging and recovery mechanisms
   - Graceful degradation when visualizations or analyses fail
   - Robust checkpoint and dataset validation before expensive operations
   - Return codes for CI/CD integration

3. **Advanced Logging & Observability**:
   - Structured logging with timestamps and severity levels
   - Progress indicators with emoji-based status messages and counters
   - Real-time evaluation progress tracking across models and datasets
   - Success/failure statistics and detailed execution summaries
   - Configurable log levels with debug information for troubleshooting

4. **Configuration & Validation Improvements**:
   - Enhanced dry-run mode with comprehensive prerequisite validation
   - Configuration, dataset, and checkpoint validation before execution
   - Advanced device detection with detailed hardware information
   - Pydantic model attribute access (replacing dictionary access patterns)
   - File size and integrity checks for all datasets

5. **CLI Enhancement & Flexibility**:
   - Comprehensive CLI with 10+ configurable parameters
   - Rich help text with detailed usage examples
   - Support for headless environments (--no-display)
   - Configurable visualization options (--save-plots, --no-viz)
   - Statistical analysis target selection (--target-dataset)
   - Flexible threshold and seed configuration
   - Proper exit codes for automation and CI/CD

6. **Visualization & Analysis Improvements**:
   - Adaptive plot layouts based on available results
   - Flexible model detection and dynamic subplot creation
   - Optional plot saving and display control for different environments
   - Robust statistical analysis with configurable target datasets
   - Memory management with proper cleanup after visualizations

**Implementation Benefits:**
- **Maintainability**: Clear separation of concerns with focused, testable functions
- **Reliability**: Continues evaluation despite individual component failures
- **Flexibility**: Highly configurable for different experimental scenarios and environments
- **Observability**: Comprehensive logging and progress tracking throughout execution
- **User Experience**: Clear feedback, helpful error messages, and flexible usage options
- **Integration**: Proper exit codes and headless operation for automation

**Technical Improvements:**
- Fixed configuration access patterns to use Pydantic model attributes
- Enhanced type hints and documentation for all functions
- Improved memory management with explicit cleanup operations  
- Robust error handling patterns with contextual error messages
- Eliminated hardcoded dataset names and made analysis more flexible
- Added comprehensive parameter validation and error checking

### hp_tuning_loop_detection.py Complete Overhaul (2025-08-20)

**Purpose:** Major structural refactoring following reference documentation conventions with comprehensive hyperparameter tuning capabilities.

**Key Architectural Changes:**
1. **Modular Function Design**: Broke down monolithic code into focused, testable components:
   - `setup_logging()`: Configurable logging with timestamps and proper formatting
   - `set_random_seeds()`: Comprehensive random seed management across all libraries
   - `initialize_device()`: Robust device initialization with detailed feedback and fallback
   - `validate_dataset()`: Comprehensive dataset validation with integrity and size checks
   - `prepare_data_loaders()`: Robust data loader creation with multiprocessing fallback
   - `create_sweep_config()`: Flexible sweep configuration with customizable parameters
   - `train_model()`: Individual model training with comprehensive error handling
   - `run_hyperparameter_sweep()`: Sweep orchestration with progress tracking and recovery
   - `analyze_results()`: Statistical analysis with comprehensive summaries and visualizations
   - `create_analysis_plots()`: Multi-panel visualization generation with error recovery

2. **Enhanced Error Handling & Recovery**:
   - Individual sweep run failures don't abort entire sweep
   - Comprehensive error tracking with detailed logging and graceful recovery  
   - Conditional WandB import with graceful degradation when unavailable
   - Robust multiprocessing fallback for data loading
   - Memory management and proper cleanup after each training run
   - Return codes for CI/CD integration

3. **Advanced Logging & Observability**:
   - Structured logging with timestamps and severity levels
   - Progress indicators with emoji-based status messages and run counters
   - Real-time sweep progress tracking across hyperparameter configurations
   - Success/failure statistics and detailed execution summaries
   - Configurable log levels with debug information for troubleshooting
   - Top results reporting with validation loss comparisons

4. **Configuration & Validation Improvements**:
   - Enhanced dataset validation with file size, integrity, and format checks
   - Flexible sweep configuration with customizable parameter ranges
   - Advanced device detection with detailed hardware information
   - Configurable hyperparameter search spaces and optimization methods
   - Results persistence with CSV save/load functionality

5. **CLI Enhancement & Flexibility**:
   - Comprehensive CLI with 13+ configurable parameters
   - Rich help text with detailed usage examples for different scenarios
   - Support for different sweep methods (random, grid, Bayesian)
   - Configurable dataset paths, batch sizes, and training parameters
   - Analysis-only mode for existing results without running new sweeps
   - Headless operation with plot generation control
   - Proper exit codes for automation and CI/CD

6. **Statistical Analysis & Visualization**:
   - Comprehensive statistical summaries with descriptive statistics
   - Multi-panel visualizations showing loss distributions and hyperparameter effects
   - Top results ranking and comparison functionality
   - Error filtering and data cleaning for robust analysis
   - Adaptive plotting based on available hyperparameter columns

**Implementation Benefits:**
- **Maintainability**: Clear separation of concerns with focused, testable functions
- **Reliability**: Continues hyperparameter search despite individual run failures
- **Flexibility**: Highly configurable for different datasets, models, and search strategies
- **Observability**: Comprehensive logging and progress tracking throughout sweep execution
- **User Experience**: Clear feedback, helpful error messages, and flexible usage options
- **Integration**: Proper exit codes and headless operation for automation
- **Scientific Rigor**: Reproducible experiments with proper seed management and validation

**Technical Improvements:**
- Conditional dependency imports with graceful degradation
- Enhanced type hints and comprehensive documentation for all functions
- Improved memory management with explicit cleanup operations
- Robust error handling patterns with contextual error messages
- Eliminated hardcoded paths and made all parameters configurable
- Added comprehensive parameter validation and error checking
- Proper WandB run management with cleanup and error recovery

## Notes:
- All scripts maintain the same functionality as the original notebooks
- Configuration files now use YAML format with inheritance and validation
- Scripts include comprehensive CLI interfaces for flexible usage
- All scripts support proper error handling and validation
- Dependencies include pydantic and PyYAML for configuration management
- Scripts are designed to be run from the project root directory
- Latest refactoring follows modular design patterns for better maintainability
