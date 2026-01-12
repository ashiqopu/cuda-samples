# CUDA Samples Docker Container

This directory contains Docker setup for running CUDA samples in an isolated environment.

## Quick Start

```bash
# From the cuda-samples root directory, run:
./container/docker-run.sh
```

This will:
1. Build the Docker image (if not already built)
2. Mount the cuda-samples directory into the container
3. Start an interactive bash session

## Building Samples Inside Container

Once inside the container:

```bash
# Build all samples
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run benchmarks
cd /cuda-samples
./run_benchmarks.sh
```

## Features

- **Live Source Mounting**: Your cuda-samples directory is mounted into the container, so any changes you make on the host are immediately available inside the container
- **Persistent Build**: The `build/` directory persists on your host, so you don't need to rebuild every time
- **Result Export**: Benchmark results are saved to `benchmark_results/` on your host

## Directory Structure

```
container/
├── Dockerfile              # Container image definition
├── docker-run.sh          # Script to build and run container
├── docker-compose.yml     # Docker Compose configuration
├── DOCKER_BENCHMARK_GUIDE.md  # Detailed documentation
└── README.md              # This file
```

## Workflow

1. Make changes to source files on your host
2. Build inside the container
3. Run benchmarks
4. Results appear in `benchmark_results/` on your host

## Commands

```bash
# Build image and start container
./container/docker-run.sh

# Rebuild image (after Dockerfile changes)
./container/docker-run.sh --rebuild

# Inside container: Build samples
mkdir -p build && cd build && cmake .. && make -j$(nproc)

# Inside container: Run benchmarks
./run_benchmarks.sh

# Inside container: Run individual sample
./build/Samples/1_Utilities/deviceQuery/deviceQuery
```
