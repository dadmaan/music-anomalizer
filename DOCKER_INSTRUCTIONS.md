# Docker Setup Instructions

This project includes a docker-compose configuration that allows you to run the Music Anomalizer application in different environments.

## Services

1. **app** - Main Streamlit web application with GPU support
2. **jupyter** - Jupyter notebook environment with GPU support

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- NVIDIA Docker runtime (for GPU support)

## Usage

### Run the Streamlit Web Application (GPU)

```bash
docker-compose up app
```

Access the application at http://localhost:8501


### Run Jupyter Notebook

```bash
docker-compose up jupyter
```

Access Jupyter at http://localhost:8081

### Run All Services

```bash
docker-compose up
```

This will start all services. You can then access:
- Streamlit app at http://localhost:8501
- Jupyter notebook at http://localhost:8081

## Stopping Services

To stop the services, press `Ctrl+C` in the terminal where docker-compose is running, or run:

```bash
docker-compose down
```

## Package Structure

The project now uses a proper Python package structure with `music_anomalizer` as the main package. The Docker setup automatically installs the package in development mode with all dependencies.

## Development

### Installing the Package

The Docker container automatically installs the `music_anomalizer` package in development mode using:

```bash
pip install -e ".[dev]"
```

This includes all dependencies specified in `pyproject.toml`.

### Using the Package

Inside the container, you can import modules from the package:

```python
from music_anomalizer.models import AnomalyDetector
from music_anomalizer.preprocessing import Wav2Embedding
from music_anomalizer.utils import load_json
```

### Running Scripts

You can run the provided scripts directly:

```bash
python -m music_anomalizer.scripts.main_exp_benchmark
```

## Notes

- The application code is mounted as a volume, so changes to the code will be reflected in the containers
- GPU support requires NVIDIA Docker runtime to be installed on your system
- The package is installed in development mode, so local changes are immediately available
- Configuration files are available in the `configs/` directory
