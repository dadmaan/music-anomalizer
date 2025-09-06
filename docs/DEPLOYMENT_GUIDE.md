# Music Anomalizer Deployment Guide

This guide explains how to deploy Music Anomalizer in different environments, from local development to production systems.

## Table of Contents
1. [Docker Setup](#docker-setup)
2. [Production Deployment](#production-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Performance Optimization](#performance-optimization)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Docker Setup

Music Anomalizer provides two Docker configurations: one for production and one for development.

### Production Setup

The production setup uses `Dockerfile` and `docker-compose.yml`:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 fluidsynth curl git tmux\
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python package
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY music_anomalizer/ ./music_anomalizer/
COPY configs/ ./configs/
RUN pip install --no-cache-dir -e ".[dev]"

# Copy remaining files
COPY . .

# Expose port for Jupyter
EXPOSE 8081

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8081", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
```

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - .:/usr/src/app
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["streamlit", "run", "app/pages/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

  jupyter:
    build: .
    ports:
      - "8081:8081"
    volumes:
      - .:/usr/src/app
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8081", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
```

To run the production setup:
```bash
# Start Streamlit web application
docker-compose up app

# Start Jupyter notebook
docker-compose up jupyter

# Start all services
docker-compose up
```

### Development Setup

The development setup uses `Dockerfile.dev` and `docker-compose.dev.yml` with live code updates:

```dockerfile
# Dockerfile.dev
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /usr/src/app

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    libsndfile1 fluidsynth curl git tmux \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision torchaudio \
    librosa numpy pandas matplotlib seaborn \
    scikit-learn scipy ipykernel notebook streamlit \
    plotly tqdm pydantic

# Install development tools
RUN pip install --no-cache-dir jupyter jupyterlab ipywidgets

# Create startup script for development mode
RUN echo '#!/bin/bash\n\
echo "Installing music_anomalizer package in development mode..."\n\
pip install -e ".[dev]" --quiet\n\
echo "Package installed successfully!"\n\
exec "$@"' > /usr/local/bin/dev-entrypoint.sh && \
    chmod +x /usr/local/bin/dev-entrypoint.sh

EXPOSE 8081 8501

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/dev-entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8081", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'"]
```

```yaml
# docker-compose.dev.yml
services:
  jupyter-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8081:8081"
    volumes:
      - .:/usr/src/app
      - jupyter-dev-cache:/root/.cache/pip
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONPATH=/usr/src/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["jupyter", "lab", "--ip=0.0.0.0", "--port=8081", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'"]

  streamlit-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8501:8501"
    volumes:
      - .:/usr/src/app
      - streamlit-dev-cache:/root/.cache/pip
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONPATH=/usr/src/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["streamlit", "run", "app/pages/Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=true"]

  dev-shell:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/usr/src/app
      - shell-dev-cache:/root/.cache/pip
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONPATH=/usr/src/app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["bash"]
    stdin_open: true
    tty: true

volumes:
  jupyter-dev-cache:
  streamlit-dev-cache:
  shell-dev-cache:
```

To run the development setup:
```bash
# Start JupyterLab for development
docker-compose -f docker-compose.dev.yml up jupyter-dev

# Start Streamlit with auto-reload
docker-compose -f docker-compose.dev.yml up streamlit-dev

# Start interactive development shell
docker-compose -f docker-compose.dev.yml run dev-shell

# Start all development services
docker-compose -f docker-compose.dev.yml up
```

## Production Deployment

### System Requirements

For production deployment, ensure your system meets these requirements:

- **CPU**: Multi-core processor (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3070/RTX 4070 or better)
- **Storage**: 50GB free space for models and data
- **OS**: Linux (Ubuntu 20.04+ recommended) or Windows 10/11 with WSL2

### Deployment Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/music-anomalizer.git
   cd music-anomalizer
   ```

2. **Prepare data and models**:
   - Place training data in appropriate directories
   - Download or train required models
   - Update configuration files with correct paths

3. **Configure environment**:
   ```bash
   # Create .env file with environment variables
   echo "NVIDIA_VISIBLE_DEVICES=all" > .env
   ```

4. **Start services**:
   ```bash
   docker-compose up -d
   ```

5. **Verify deployment**:
   - Access Streamlit UI at http://localhost:8501
   - Access Jupyter at http://localhost:8081
   - Check container logs for errors

### Configuration for Production

Update configuration files for production use:

```yaml
# configs/exp2_deeper.yaml (production settings)
trainer:
  batch_size: 64  # Larger batch size for better GPU utilization
  max_epochs: 1000
  patience: 20  # More patience for production training
  wandb_project_name: "LOOP-DSVDD-PROD"
  wandb_log_model: true  # Enable model logging in production
  enable_progress_bar: false  # Disable for cleaner logs
```

## Cloud Deployment

### AWS Deployment

To deploy on AWS EC2 with GPU support:

1. **Launch GPU-enabled instance**:
   - Use p3.2xlarge or better instance type
   - Select Ubuntu 20.04 AMI
   - Attach sufficient storage (50GB+)

2. **Install NVIDIA drivers**:
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install NVIDIA drivers
   sudo apt install ubuntu-drivers-common -y
   sudo ubuntu-drivers autoinstall
   
   # Reboot
   sudo reboot
   ```

3. **Install Docker and NVIDIA Container Toolkit**:
   ```bash
   # Install Docker
   sudo apt install docker.io -y
   
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt update
   sudo apt install nvidia-docker2 -y
   sudo systemctl restart docker
   ```

4. **Deploy Music Anomalizer**:
   ```bash
   git clone https://github.com/your-username/music-anomalizer.git
   cd music-anomalizer
   docker-compose up -d
   ```

### Google Cloud Deployment

To deploy on Google Cloud Platform with GPU support:

1. **Create Compute Engine instance**:
   - Select GPU-enabled machine type (n1-standard-8 with 1x T4 GPU)
   - Use Ubuntu 20.04 image
   - Enable GPU access

2. **Install dependencies**:
   ```bash
   # Install Docker
   sudo apt update
   sudo apt install docker.io docker-compose -y
   
   # Install NVIDIA drivers and container toolkit
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt update
   sudo apt install nvidia-driver-470 nvidia-docker2 -y
   sudo systemctl restart docker
   ```

3. **Deploy application**:
   ```bash
   git clone https://github.com/your-username/music-anomalizer.git
   cd music-anomalizer
   docker-compose up -d
   ```

### Azure Deployment

To deploy on Microsoft Azure with GPU support:

1. **Create Virtual Machine**:
   - Select GPU-enabled VM size (Standard_NC6 or better)
   - Use Ubuntu 20.04 image
   - Configure networking to allow ports 8501 and 8081

2. **Install dependencies**:
   ```bash
   # Install Docker
   sudo apt update
   sudo apt install docker.io docker-compose -y
   
   # Install NVIDIA drivers and container toolkit
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt update
   sudo apt install nvidia-driver-470 nvidia-docker2 -y
   sudo systemctl restart docker
   ```

3. **Deploy application**:
   ```bash
   git clone https://github.com/your-username/music-anomalizer.git
   cd music-anomalizer
   docker-compose up -d
   ```

## Performance Optimization

### GPU Utilization

To maximize GPU utilization:

1. **Batch Size Optimization**:
   ```yaml
   # configs/exp2_deeper.yaml
   trainer:
     batch_size: 128  # Adjust based on GPU memory
   ```

2. **Mixed Precision Training**:
   ```python
   # In training configuration
   trainer = pl.Trainer(
       precision=16,  # Use 16-bit floating point
       # ... other parameters
   )
   ```

3. **Memory Management**:
   ```python
   # In anomaly_detector.py
   def _determine_optimal_batch_size(self, dataset):
       """Determine optimal batch size based on available GPU memory."""
       if self.device == 'cpu':
           return 32
       
       # Get available GPU memory
       if torch.cuda.is_available():
           gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
           # Use 30% of available memory for safety
           available_memory = gpu_memory * 0.3
           # Estimate memory per sample (this is approximate)
           memory_per_sample = 1024 * 1024  # 1MB per sample estimate
           max_batch_size = int(available_memory / memory_per_sample)
           # Clamp to reasonable range
           return max(8, min(64, max_batch_size))
       return 32
   ```

### Caching Strategies

Implement caching to reduce recomputation:

```python
# In anomaly_scores_manager.py
@st.cache_data
def load_anomaly_scores(model_type, config_name=DEFAULT_CONFIG, network_key='AEwRES'):
    """Load anomaly scores for a given model type, computing them if necessary."""
    manager = get_anomaly_scores_manager()
    
    # Try to load scores, auto-computing if missing
    scores, error = manager.load_scores(
        model_type=model_type,
        config_name=config_name,
        network_key=network_key,
        auto_compute=True
    )
    
    if error:
        st.error(f"❌ Error loading anomaly scores: {error}")
        return []
    
    return scores
```

### Data Loading Optimization

Optimize data loading for better performance:

```python
# In data_loader.py
class OptimizedDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=4, pin_memory=True):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True  # Keep workers alive between epochs
        )
```

## Monitoring and Maintenance

### Health Checks

Implement health checks for monitoring:

```python
# In app/Home.py
def health_check():
    """Perform basic health checks."""
    checks = {
        'docker_running': check_docker_status(),
        'gpu_available': check_gpu_availability(),
        'models_loaded': check_model_availability(),
        'data_accessible': check_data_accessibility()
    }
    return checks

def check_docker_status():
    """Check if Docker is running."""
    try:
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False
```

### Logging and Monitoring

Configure structured logging for better monitoring:

```python
# In utils.py
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging with timestamps and proper formatting."""
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('music_anomalizer')
    return logger
```

### Backup and Recovery

Implement backup strategies for models and data:

```bash
#!/bin/bash
# backup.sh - Backup script for Music Anomalizer

# Configuration
BACKUP_DIR="/backups/music-anomalizer"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p ${BACKUP_DIR}/${DATE}

# Backup models
cp -r checkpoints/ ${BACKUP_DIR}/${DATE}/checkpoints/

# Backup configurations
cp -r configs/ ${BACKUP_DIR}/${DATE}/configs/

# Backup computed scores
cp -r output/ ${BACKUP_DIR}/${DATE}/output/

# Create backup archive
tar -czf ${BACKUP_DIR}/music-anomalizer-${DATE}.tar.gz -C ${BACKUP_DIR} ${DATE}

# Remove uncompressed backup
rm -rf ${BACKUP_DIR}/${DATE}

# Keep only last 7 backups
find ${BACKUP_DIR} -name "music-anomalizer-*.tar.gz" -mtime +7 -delete

echo "Backup completed: ${BACKUP_DIR}/music-anomalizer-${DATE}.tar.gz"
```

### Update Management

For safe updates:

1. **Create update script**:
   ```bash
   #!/bin/bash
   # update.sh - Update script for Music Anomalizer
   
   # Stop services
   docker-compose down
   
   # Backup current version
   ./backup.sh
   
   # Pull latest changes
   git pull origin main
   
   # Rebuild containers
   docker-compose build
   
   # Start services
   docker-compose up -d
   
   echo "Update completed"
   ```

2. **Rollback procedure**:
   ```bash
   # rollback.sh - Rollback to previous version
   # Stop services
   docker-compose down
   
   # Restore from backup
   # (Implementation depends on backup strategy)
   
   # Start services
   docker-compose up -d
   
   echo "Rollback completed"
   ```

This deployment guide provides comprehensive information for deploying Music Anomalizer in various environments. For usage instructions, please refer to the User Guide, and for development information, see the Developer Guide.
