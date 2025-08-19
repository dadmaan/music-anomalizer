# Use an official PyTorch image with CUDA support
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime 
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime


# Set the working directory in the container
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 fluidsynth curl git tmux\
    && rm -rf /var/lib/apt/lists/*

# Copy package configuration files first for better Docker layer caching
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Copy the package source code
COPY music_anomalizer/ ./music_anomalizer/
COPY configs/ ./configs/

# Install the package in development mode with all dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy additional project files
COPY . .

# To run jupyter in remote development scenario with VSCode
# from https://stackoverflow.com/questions/63998873/vscode-how-to-run-a-jupyter-notebook-in-a-docker-container-over-a-remote-serve
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini


# Expose the port Jupyter will listen on
EXPOSE 8081

# Use Tini as the container's entry point
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8081", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
