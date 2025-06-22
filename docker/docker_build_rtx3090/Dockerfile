# Dockerfile optimized for RTX 3090 (Compute 8.6)
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
      build-essential \
      git \
      python3.10 \
      python3.10-dev \
      python3.10-venv \
      python3-pip \
      wget \
      libffi-dev \
      libssl-dev \
      libxml2-dev \
      libxslt1-dev \
      zlib1g-dev \
      libjpeg-dev \
      libpng-dev \
      libopenblas-dev \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Set working directory
WORKDIR /app

# Install PyTorch for CUDA 11.8 (RTX 3090 compatible)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
RUN pip install \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    datasets==2.16.1 \
    trl==0.7.10 \
    peft==0.7.1 \
    scipy \
    sentencepiece \
    protobuf \
    xformers==0.0.23 \
    einops \
    ninja

# Install Flash Attention 2 for RTX 3090
# Note: This might need compilation, so we install from source
RUN pip install packaging && \
    pip install flash-attn==2.3.6 --no-build-isolation

# Install Unsloth
RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Optional: Install DeepSpeed for better memory efficiency
RUN pip install deepspeed==0.12.6

# Set environment variables for RTX 3090
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_VISIBLE_DEVICES="0,1,2"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"

# Create directories
RUN mkdir -p /app/models /app/data /app/output /app/scripts

# Copy training scripts
COPY unsloth_Accelerate-Docker.py /app/scripts/
COPY unsloth_Accelerate.py /app/scripts/

# Make scripts executable
RUN chmod +x /app/scripts/*.py

# Default command
CMD ["/bin/bash"]
