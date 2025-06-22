#!/bin/bash

# Build script for RTX 3090 Docker image

# Set image name and tag
IMAGE_NAME="unsloth-rtx3090"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building Docker image for RTX 3090 GPUs..."
echo "Image name: ${FULL_IMAGE_NAME}"

# Create a temporary build directory
BUILD_DIR="docker_build_rtx3090"
mkdir -p ${BUILD_DIR}

# Copy necessary files to build directory
echo "Copying files to build directory..."
cp rtx3090_dockerfile ${BUILD_DIR}/Dockerfile
cp rtx3090_requirements.txt ${BUILD_DIR}/requirements.txt
cp unsloth_Accelerate-Docker.py ${BUILD_DIR}/
cp unsloth_Accelerate.py ${BUILD_DIR}/

# Optional: Copy your patched unsloth source if you have it
# cp -r unsloth/ ${BUILD_DIR}/
# cp -r unsloth_zoo/ ${BUILD_DIR}/

# Navigate to build directory
cd ${BUILD_DIR}

# Build the Docker image
echo "Starting Docker build..."
docker build \
  --build-arg CUDA_VERSION=11.8.0 \
  --build-arg PYTHON_VERSION=3.10 \
  -t ${FULL_IMAGE_NAME} \
  -f Dockerfile \
  .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully: ${FULL_IMAGE_NAME}"
    echo ""
    echo "To test the image, run:"
    echo "docker run --rm -it --gpus all ${FULL_IMAGE_NAME} python /app/scripts/test_rtx3090_setup.py"
    echo ""
    echo "To use for training, run your launch script:"
    echo "./rtx3090_docker_launch.sh"
else
    echo "❌ Docker build failed!"
    exit 1
fi

# Clean up build directory (optional)
cd ..
# rm -rf ${BUILD_DIR}  # Uncomment to auto-cleanup

# List the created image
echo ""
echo "Docker images:"
docker images | grep ${IMAGE_NAME}
