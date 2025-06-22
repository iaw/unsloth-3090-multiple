#!/bin/bash

# Docker launch script optimized for RTX 3090 GPUs (24GB VRAM)

# For training with 3 RTX 3090 GPUs
docker run --rm -it \
--gpus all \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-e NCCL_P2P_DISABLE=1 \
-e NCCL_DEBUG=INFO \
-e TORCH_DISTRIBUTED_DEBUG=DETAIL \
-e CUDA_LAUNCH_BLOCKING=1 \
-e TORCH_USE_CUDA_DSA=1 \
-e CUDA_VISIBLE_DEVICES=0,1,2,3 \
-e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True" \
-e TORCH_CUDA_ARCH_LIST="8.6" \
-v "/home/username/unsloth-3090-multiple":/app/unsloth_run \
-v "/home/username/ssd/data/hub/Hermes-3-Llama-3.1-8B":/app/models/Hermes-3-Llama-3.1-8B \
-v "/home/username/ssd/dadta/hub/alpaca":/app/data/unsloth_datasets \
-v "/home/username/unsloth_output":/app/output \
-w /app/unsloth_run \
unsloth-rtx3090:latest \
/bin/bash -c "accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 unsloth_Accelerate-Docker.py --output_dir /app/output"

# Alternative: Single GPU training (if multi-GPU OOMs)
# docker run --rm -it \
# --gpus '"device=0"' \
# --ipc=host \
# --ulimit memlock=-1 \
# --ulimit stack=67108864 \
# -e CUDA_VISIBLE_DEVICES=0 \
# -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True" \
# -v "$HOME/Desktop/Model Launchers/unsloth":/app/unsloth_run \
# -v "/media/user/Models/Text/Qwen3-32B":/app/models/Qwen3-32B \
# -v "/media/user/combined-alpaca-prompts/Final":/app/data/unsloth_datasets \
# -v "$HOME/Desktop/Docker-Unsloth-Output":/app/output \
# -w /app/unsloth_run \
# unslothai/unsloth:latest \
# python unsloth_Accelerate-Docker.py --output_dir /app/output

# For interactive debugging
# docker run --rm -it \
# --gpus all \
# --ipc=host \
# --ulimit memlock=-1 \
# --ulimit stack=67108864 \
# -e NCCL_P2P_DISABLE=1 \
# -e CUDA_VISIBLE_DEVICES=0,1,2 \
# -e PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True" \
# -v "$HOME/Desktop/Model Launchers/unsloth":/app/unsloth_run \
# -v "/media/user/Models/Text/Qwen3-32B":/app/models/Qwen3-32B \
# -v "/media/user/combined-alpaca-prompts/Final":/app/data/unsloth_datasets \
# -v "$HOME/Desktop/Docker-Unsloth-Output":/app/output \
# -w /app/unsloth_run \
# unslothai/unsloth:latest \
# /bin/bash
