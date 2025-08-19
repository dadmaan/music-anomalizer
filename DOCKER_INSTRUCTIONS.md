# Docker Setup Instructions

This project includes a docker-compose configuration that allows you to run the Music Anomalizer application in different environments.

## Services

1. **app** - Main Streamlit web application with GPU support
2. **jupyter** - Jupyter notebook environment with GPU support
3. **app-cpu** - Streamlit web application for CPU-only systems

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

### Run the Streamlit Web Application (CPU-only)

```bash
docker-compose up app-cpu
```

Access the application at http://localhost:8502

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

## Notes

- The application code is mounted as a volume, so changes to the code will be reflected in the containers
- GPU support requires NVIDIA Docker runtime to be installed on your system
- Checkpoints, data, and output directories are mounted as volumes for persistence
