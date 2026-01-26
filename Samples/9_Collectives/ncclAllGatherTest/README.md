# ncclAllGatherTest

## Description

This sample demonstrates the NCCL AllGather collective operation across multiple GPUs. AllGather collects data from all participating GPUs and distributes the complete gathered data to all GPUs.

## Key Concepts

- NCCL Collective Operations
- Multi-GPU Communication
- AllGather Pattern
- Performance Benchmarking

## Supported SM Architectures

SM 7.5, 8.0, 8.6, 8.7, 8.9, 9.0, 10.0, 11.0, 12.0

## Supported Operating Systems

Linux

## Prerequisites

- CUDA Toolkit 11.0 or later
- NCCL 2.0 or later
- At least 2 CUDA-capable GPUs

## Build Instructions

```bash
cd /path/to/cuda-samples/build
cmake ..
make ncclAllGatherTest
```

## Usage

Basic usage:
```bash
./ncclAllGatherTest
```

With specific number of GPUs:
```bash
./ncclAllGatherTest --num_gpus=4
```

Custom configuration:
```bash
./ncclAllGatherTest --num_gpus=8 --size=128 --iterations=1000
```

## Command Line Options

- `--help`: Display help information
- `--num_gpus=N`: Number of GPUs to use (default: all available)
- `--size=N`: Send buffer size in MB per GPU (default: 32, recv will be N * num_gpus)
- `--iterations=N`: Number of benchmark iterations (default: 100)
- `--warmup=N`: Number of warmup iterations (default: 10)

## How It Works

1. Initializes NCCL communicators for all participating GPUs
2. Allocates send buffers (size N) and receive buffers (size N * num_gpus) on each GPU
3. Each GPU initializes its send buffer with rank-specific values (rank+1)
4. Performs warmup iterations to stabilize GPU clocks
5. Runs benchmark iterations and measures performance
6. Verifies correctness by checking that each GPU received data from all other GPUs
7. Reports performance metrics including bandwidth and latency

## Expected Output

```
NCCL AllGather Test Starting...

Running AllGather test with 4 GPUs
Buffer size per GPU (send): 32 MB
Buffer size per GPU (recv): 128 MB
Iterations: 100 (warmup: 10)

Starting warmup...
Starting benchmark...

Performance Results:
-----------------------------------
Number of GPUs:         4
Send buffer per GPU:    32.00 MB
Recv buffer per GPU:    128.00 MB
Average time:           3.2145 ms
Bus bandwidth:          119.87 GB/s
Algorithm bandwidth:    29.97 GB/s
-----------------------------------

Verifying results...
Rank 0: PASSED
Rank 1: PASSED
Rank 2: PASSED
Rank 3: PASSED

Test PASSED
```

## Performance Notes

- Each GPU sends N bytes and receives N * (num_gpus - 1) bytes from other GPUs
- Bus bandwidth represents total data movement across all GPUs
- Algorithm bandwidth represents the effective per-GPU bandwidth
- Performance depends on GPU interconnect (NVLink, PCIe)
- NVLink provides significantly better performance than PCIe
- AllGather is commonly used in distributed training for collecting gradients or activations
