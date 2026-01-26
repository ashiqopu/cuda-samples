# ncclAllReduceTest

## Description

This sample demonstrates the NCCL AllReduce collective operation across multiple GPUs. AllReduce performs a reduction operation (sum, in this case) across all participating GPUs and distributes the result to all GPUs.

## Key Concepts

- NCCL Collective Operations
- Multi-GPU Communication
- AllReduce Pattern
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
make ncclAllReduceTest
```

## Usage

Basic usage:
```bash
./ncclAllReduceTest
```

With specific number of GPUs:
```bash
./ncclAllReduceTest --num_gpus=4
```

Custom configuration:
```bash
./ncclAllReduceTest --num_gpus=8 --size=128 --iterations=1000
```

## Command Line Options

- `--help`: Display help information
- `--num_gpus=N`: Number of GPUs to use (default: all available)
- `--size=N`: Buffer size in MB per GPU (default: 32)
- `--iterations=N`: Number of benchmark iterations (default: 100)
- `--warmup=N`: Number of warmup iterations (default: 10)

## How It Works

1. Initializes NCCL communicators for all participating GPUs
2. Allocates send and receive buffers on each GPU
3. Each GPU initializes its buffer with rank-specific values (rank+1)
4. Performs warmup iterations to stabilize GPU clocks
5. Runs benchmark iterations and measures performance
6. Verifies correctness by checking that all GPUs received the sum of all values
7. Reports performance metrics including bandwidth and latency

## Expected Output

```
NCCL AllReduce Test Starting...

Running AllReduce test with 4 GPUs
Buffer size per GPU: 32 MB
Iterations: 100 (warmup: 10)

Starting warmup...
Starting benchmark...

Performance Results:
-----------------------------------
Number of GPUs:         4
Buffer size per GPU:    32.00 MB
Average time:           2.1234 ms
Bus bandwidth:          60.23 GB/s
Algorithm bandwidth:    45.17 GB/s
-----------------------------------

Verifying results...
Rank 0: PASSED
Rank 1: PASSED
Rank 2: PASSED
Rank 3: PASSED

Test PASSED
```

## Performance Notes

- Bus bandwidth represents total data movement across all GPUs
- Algorithm bandwidth represents the effective per-GPU bandwidth
- Performance depends on GPU interconnect (NVLink, PCIe)
- NVLink provides significantly better performance than PCIe
