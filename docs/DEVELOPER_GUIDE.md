# Music Anomalizer Developer Guide

This guide is intended for developers who want to contribute to the Music Anomalizer project or understand its internal architecture.

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Code Structure](#code-structure)
3. [Development Setup](#development-setup)
4. [Testing](#testing)
5. [Contributing](#contributing)

## Project Architecture

Music Anomalizer follows a modular architecture with clearly separated components:

### Core Components

1. **Data Processing Pipeline**
   - Audio preprocessing and feature extraction
   - Embedding generation using CLAP models
   - Data loading and batching

2. **Model Architecture**
   - AutoEncoder networks (standard and with residual connections)
   - Deep SVDD anomaly detection
   - Model training and evaluation

3. **Anomaly Detection Engine**
   - Score computation for training data
   - Loop detection for new audio files
   - Threshold-based classification

4. **User Interface**
   - Streamlit web application
   - Interactive visualizations
   - Configuration management

### Data Flow

```
Audio Files → Preprocessing → Embeddings → AutoEncoder → Deep SVDD → Anomaly Scores
                                      ↘                    ↗
                                   New Audio Files → Loop Detection
```

## Code Structure

The project follows a Python package structure with the following key directories:

```
music_anomalizer/
├── config/              # Configuration management
├── data/                # Data loading utilities
├── evaluation/          # Model evaluation tools
├── models/              # Neural network models
├── preprocessing/       # Audio preprocessing
├── scripts/             # Command-line tools
├── tests/               # Unit and integration tests
├── visualization/       # Data visualization
├── utils.py             # Utility functions
└── anomaly_scores_manager.py  # Anomaly score management
```

### Key Modules

#### `music_anomalizer/models/`

Contains the core neural network implementations:

- `networks.py`: AutoEncoder and related architectures
- `anomaly_detector.py`: Deep SVDD implementation and anomaly detection logic
- `deepSVDD.py`: Deep SVDD training implementation
- `layers.py`: Custom neural network layers
- `losses.py`: Specialized loss functions
- `base_models.py`: Baseline models (Isolation Forest, PCA)

#### `music_anomalizer/config/`

Handles configuration management:

- `loader.py`: Configuration loading and validation
- `schemas.py`: Pydantic schemas for configuration validation
- `checkpoint_manager.py`: Model checkpoint discovery and management

#### `music_anomalizer/scripts/`

Command-line tools for various operations:

- `compute_anomaly_scores.py`: Compute anomaly scores for training data
- `embedding_extraction_wav.py`: Extract audio embeddings
- `train_models.py`: Train Deep SVDD models
- `loop_detector.py`: Detect loops in audio files
- `prepare_data.py`: Prepare data for training

#### `music_anomalizer/preprocessing/`

Audio preprocessing utilities:

- `wav2embed.py`: Convert WAV files to embeddings
- `extract_embed.py`: Embedding extraction pipeline

## Development Setup

### Prerequisites

1. Python 3.8+
2. Docker and Docker Compose
3. NVIDIA Docker runtime (for GPU support)

### Development Environment

For development work, use the development Docker setup:

```bash
# Start development environment with JupyterLab
docker-compose -f docker-compose.dev.yml up jupyter-dev

# Or start Streamlit with auto-reload
docker-compose -f docker-compose.dev.yml up streamlit-dev
```

### Local Development Setup

If you prefer to develop locally:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks (optional):
   ```bash
   pre-commit install
   ```

## Testing

The project includes a test suite to ensure code quality and prevent regressions.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=music_anomalizer

# Run specific test file
pytest music_anomalizer/tests/test_models_smoke.py
```

### Test Structure

Tests are organized in the `music_anomalizer/tests/` directory:

- `conftest.py`: Test fixtures and configuration
- `test_models_smoke.py`: Smoke tests for core models

### Writing Tests

When adding new functionality, include appropriate tests:

1. Use pytest fixtures for common setup
2. Test both normal and edge cases
3. Mock external dependencies when possible
4. Follow existing test patterns

Example test structure:
```python
def test_functionality(self, mock_config):
    # Arrange
    # Set up test data and mocks
    
    # Act
    # Call the function being tested
    
    # Assert
    # Verify expected behavior
```

## Contributing

We welcome contributions to Music Anomalizer! Here's how you can help:

### Reporting Issues

1. Check existing issues before creating a new one
2. Include detailed information about the problem
3. Provide steps to reproduce the issue
4. Include system information and error messages

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style

Follow these guidelines for code consistency:

1. Use Black for code formatting:
   ```bash
   black music_anomalizer/
   ```

2. Use isort for import organization:
   ```bash
   isort music_anomalizer/
   ```

3. Check for linting issues:
   ```bash
   flake8 music_anomalizer/
   ```

4. Follow PEP 8 guidelines for Python code
5. Use type hints for function signatures
6. Write docstrings for public functions and classes

### Documentation

When adding new features:

1. Update relevant documentation files
2. Add docstrings to new functions and classes
3. Include usage examples when appropriate
4. Update README.md if necessary

### Pull Request Process

1. Ensure your code follows the style guidelines
2. Add tests for new functionality
3. Update documentation as needed
4. Describe your changes in the pull request
5. Reference any related issues
6. Request review from maintainers

This developer guide provides an overview of the Music Anomalizer codebase. For more detailed information about specific components, refer to the technical documentation and source code comments.
