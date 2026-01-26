# CUDA Samples Docker Benchmarking Guide

This Docker setup allows you to benchmark your NVIDIA GPU using CUDA samples without making any changes to your host system.

## Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Docker Runtime** (nvidia-docker2) installed
3. **Docker** installed

### Installing NVIDIA Docker Runtime

On Ubuntu/Debian:
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker
```

### Verify NVIDIA Docker Installation

```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Option 1: Using the Provided Script (Recommended)

1. **Run the script from the cuda-samples root directory:**
   ```bash
   ./container/docker-run.sh
   ```
   
   This script will:
   - Automatically detect your user ID and group ID
   - Build the image with a non-root user matching your host credentials
   - Mount the entire cuda-samples directory
   - Start an interactive container

2. **Inside the container, build the samples:**
   ```bash
   mkdir -p build && cd build
   cmake ..
   make -j$(nproc)
   ```

3. **Run benchmarks:**
   ```bash
   cd /cuda-samples
   ./run_benchmarks.sh
   ```

4. **Exit the container:**
   ```bash
   exit
   ```

Results will be saved in `./benchmark_results/` on your host system, owned by your user (no sudo needed!).

### Option 2: Using Docker Directly

1. **Build the Docker image with your user credentials:**
   ```bash
   docker build -t cuda-samples:latest \
     --build-arg USER_ID=$(id -u) \
     --build-arg GROUP_ID=$(id -g) \
     --build-arg USERNAME=$(whoami) \
     -f ./container/Dockerfile ./container
   ```

2. **Run the container:**
   ```bash
   docker run --rm -it --gpus all \
     -v "$(pwd):/cuda-samples" \
     -w /cuda-samples \
     cuda-samples:latest
   ```

3. **Build and run benchmarks as shown in Option 1**

## Running Individual Samples

After starting the container and building the samples, you can run individual samples:

```bash
# From inside the container
cd /cuda-samples/build

# Check GPU information
./Samples/1_Utilities/deviceQuery/deviceQuery

# Test memory bandwidth
./Samples/1_Utilities/bandwidthTest/bandwidthTest

# Matrix multiplication benchmark
./Samples/0_Introduction/matrixMul/matrixMul

# N-body simulation
./Samples/5_Domain_Specific/nbody/nbody

# Peer-to-peer bandwidth test (multi-GPU)
./Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest
```

**Note:** All files created in the build directory will be owned by your user on the host, making it easy to clean up or modify without sudo.

## Running All Tests

To run the comprehensive test suite:

```bash
cd /cuda-samples
python3 run_tests.py --output ./test_output --dir ./build/Samples --config test_args.json
```

## Benchmark Results

Results are automatically saved to:
- `./benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.txt` - Timestamped results
- `./benchmark_results/latest.txt` - Symlink to the most recent results

## Customizing Benchmarks

Edit `run_benchmarks.sh` to add or remove specific benchmarks. The script includes:

- **deviceQuery** - GPU hardware information
- **bandwidthTest** - Memory bandwidth measurements
- **matrixMul** - Matrix multiplication performance
- **nbody** - N-body simulation
- **reduction** - Parallel reduction
- **transpose** - Matrix transpose
- **BlackScholes** - Financial modeling

## Sample Output

```
======================================
CUDA Samples Benchmark Results
Timestamp: Wed Dec 17 10:30:45 UTC 2025
======================================

=== GPU Information ===
NVIDIA GeForce RTX 4090, 545.29.06, 24564 MiB, 8.9

=== Running: deviceQuery ===
CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)
...
```

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Docker runtime is properly installed:
```bash
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
```

### Permission Denied (Docker)

Add your user to the docker group:
```bash
sudo usermod -aG docker $USER
```
Then log out and log back in.

### Architecture Not Supported Error

If you see `nvcc fatal: Unsupported gpu architecture 'compute_XX'`, your GPU architecture might not be supported by the CUDA version. The samples require CUDA 13.0 for full compatibility.

### File Permission Issues

The container runs as a non-root user matching your host UID/GID, so all files created inside the container will be owned by you on the host. If you still encounter permission issues:

```bash
# Clean and rebuild
rm -rf build
./container/docker-run.sh --rebuild
```

### Build Failures

Some samples may fail to build if:
- GPU architecture is not supported
- Optional dependencies are missing (OpenGL, Vulkan)
- The build continues anyway, and core benchmarking samples should still work

## Using a Different CUDA Version

Edit the first line of `container/Dockerfile` to use a different CUDA version:
```dockerfile
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
```

**Note:** The cuda-samples are designed for CUDA 13.0. Using older versions may result in build errors due to unsupported GPU architectures.

Available versions: https://hub.docker.com/r/nvidia/cuda/tags

## Performance Tips

1. Close unnecessary applications on the host
2. Run benchmarks multiple times for consistency
3. Monitor GPU temperature: `nvidia-smi -l 1`
4. Compare results with different CUDA versions

## Rebuilding the Container

To rebuild after changing the Dockerfile or to clean up:

```bash
# Rebuild the image
./container/docker-run.sh --rebuild

# Or manually
docker rmi cuda-samples:latest
./container/docker-run.sh
```

## Cleaning Up

Remove the Docker image:
```bash
docker rmi cuda-samples:latest
```

Remove build artifacts (no sudo needed!):
```bash
rm -rf build/ benchmark_results/
```

## Key Features

- ✅ **Non-root user**: Container runs as your user (matching UID/GID) - no permission issues!
- ✅ **Live mounting**: Source code mounted from host - edit on host, build in container
- ✅ **Persistent builds**: Build directory persists on host between container runs
- ✅ **Easy cleanup**: All files owned by you - no sudo needed
- ✅ **CUDA 13.0**: Full support for latest GPU architectures

## Workflow Summary

```bash
# 1. Start container (first time will build the image)
cd /path/to/cuda-samples
./container/docker-run.sh

# 2. Inside container: build samples
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 3. Run benchmarks
cd /cuda-samples
./run_benchmarks.sh

# 4. Exit container
exit

# 5. Check results on host (owned by you, no sudo needed!)
cat benchmark_results/latest.txt

# 6. Optional: Rebuild container after changes
./container/docker-run.sh --rebuild
```

## Additional Resources

- [CUDA Samples Documentation](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Container README](./README.md)
