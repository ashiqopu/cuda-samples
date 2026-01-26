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
    
    # Try to build with retries for rate limit errors
    MAX_RETRIES=3
    RETRY_DELAY=10
    for i in $(seq 1 $MAX_RETRIES); do
        echo "Build attempt $i of $MAX_RETRIES..."
        if docker build -t $IMAGE_NAME \
            --build-arg USER_ID=$USER_ID \
            --build-arg GROUP_ID=$GROUP_ID \
            --build-arg USERNAME=$USERNAME \
            -f "$SCRIPT_DIR/Dockerfile" "$SCRIPT_DIR"; then
            echo "Build successful!"
            break
        else
            if [ $i -lt $MAX_RETRIES ]; then
                echo "Build failed. Waiting ${RETRY_DELAY} seconds before retry..."
                sleep $RETRY_DELAY
                RETRY_DELAY=$((RETRY_DELAY * 2))  # Exponential backoff
            else
                echo "Build failed after $MAX_RETRIES attempts."
                echo ""
                echo "This is likely due to Docker Hub rate limiting."
                echo "Possible solutions:"
                echo "  1. Wait a few minutes and try again"
                echo "  2. Log in to Docker Hub: docker login"
                echo "  3. Use a mirror or different registry"
                exit 1
            fi
        fi
    done
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
