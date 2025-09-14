# Docker Setup Instructions

This project includes multiple Docker configurations that allow you to run the Music Anomalizer application in different environments.

## Available Docker Configurations

### Production Setup (`Dockerfile` + `docker-compose.yml`)
1. **app** - Main Streamlit web application with GPU support
2. **jupyter** - Jupyter notebook environment with GPU support

### Development Setup (`Dockerfile.dev` + `docker-compose.dev.yml`)
1. **jupyter-dev** - JupyterLab environment optimized for development with live code updates
2. **streamlit-dev** - Streamlit app with auto-reload for development
3. **dev-shell** - Interactive bash shell for debugging and development

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- NVIDIA Docker runtime (for GPU support)

## Usage

### Production Environment

#### Run the Streamlit Web Application (GPU)

```bash
docker-compose up app
```

Access the application at http://localhost:8501

#### Run Jupyter Notebook

```bash
docker-compose up jupyter
```

Access Jupyter at http://localhost:8081

#### Run All Services

```bash
docker-compose up
```

This will start all services. You can then access:
- Streamlit app at http://localhost:8501
- Jupyter notebook at http://localhost:8081

### Development Environment

For development work where you need immediate reflection of code changes, use the development Docker configuration:

#### Run JupyterLab for Development

```bash
docker-compose -f docker-compose.dev.yml up jupyter-dev
```

Access JupyterLab at http://localhost:8081 (with enhanced development features)

#### Run Streamlit with Auto-Reload

```bash
docker-compose -f docker-compose.dev.yml up streamlit-dev
```

Access Streamlit at http://localhost:8501 (automatically reloads on code changes)

#### Interactive Development Shell

```bash
docker-compose -f docker-compose.dev.yml run dev-shell
```

Opens an interactive bash shell inside the container for debugging and development

#### Run All Development Services

```bash
docker-compose -f docker-compose.dev.yml up
```

This starts all development services with live code updates enabled.

## Stopping Services

To stop the services, press `Ctrl+C` in the terminal where docker-compose is running, or run:

### For Production Environment
```bash
docker-compose down
```

### For Development Environment
```bash
docker-compose -f docker-compose.dev.yml down
```

## Package Structure

The project now uses a proper Python package structure with `music_anomalizer` as the main package. The Docker setup automatically installs the package in development mode with all dependencies.

## Development

### Installing the Package

Both Docker setups automatically install the `music_anomalizer` package in development mode using:

```bash
pip install -e ".[dev]"
```

This includes all dependencies specified in `pyproject.toml`.

The development setup reinstalls the package on every container start to ensure the latest changes are available.

### Using the Package

Inside any container, you can import modules from the package:

```python
from music_anomalizer.models import AnomalyDetector
from music_anomalizer.preprocessing import Wav2Embedding
from music_anomalizer.utils import load_json
```

### Running Scripts

You can run the provided scripts directly in any container:

```bash
python -m music_anomalizer.scripts.main_exp_benchmark
```

### Development Workflow

1. Start the development environment:
   ```bash
   docker-compose -f docker-compose.dev.yml up jupyter-dev
   ```

2. Access JupyterLab at http://localhost:8081

3. Make changes to your local code - they will be immediately available in the container

4. The package is automatically reinstalled on container startup, ensuring all imports work correctly


