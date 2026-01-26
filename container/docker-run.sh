#!/bin/bash
# Simple Docker run script for CUDA Samples benchmarking
# This script builds and runs the container without docker-compose

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the parent directory (cuda-samples root)
CUDA_SAMPLES_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="cuda-samples:latest"

# Check for --rebuild flag
REBUILD=false
if [[ "$1" == "--rebuild" ]]; then
    REBUILD=true
    shift
fi

# Get current user's UID and GID
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(whoami)

# Build the image if it doesn't exist or if rebuild is requested
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]] || [[ "$REBUILD" == true ]]; then
    echo "Building Docker image from $SCRIPT_DIR/Dockerfile..."
    echo "Using UID=$USER_ID, GID=$GROUP_ID, USERNAME=$USERNAME"
    docker build -t $IMAGE_NAME \
        --build-arg USER_ID=$USER_ID \
        --build-arg GROUP_ID=$GROUP_ID \
        --build-arg USERNAME=$USERNAME \
        -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"
else
    echo "Using existing Docker image (use --rebuild to force rebuild)"
fi

# Create benchmark results directory
mkdir -p "$CUDA_SAMPLES_DIR/benchmark_results"
mkdir -p "$CUDA_SAMPLES_DIR/build"

# Run the container with source mounted
echo "Starting CUDA Samples container..."
echo "Mounting: $CUDA_SAMPLES_DIR -> /cuda-samples"
docker run --rm -it --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$CUDA_SAMPLES_DIR:/cuda-samples" \
    -w /cuda-samples \
    $IMAGE_NAME

echo "Container exited."
