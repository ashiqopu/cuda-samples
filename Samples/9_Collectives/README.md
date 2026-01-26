# Collectives Samples

## Description

This directory contains samples demonstrating NCCL (NVIDIA Collective Communications Library) collective operations across multiple GPUs.

## Samples

### ncclAllReduceTest
Tests the AllReduce collective operation which performs reduction (e.g., sum) across all GPUs and distributes the result to all GPUs.

### ncclAllGatherTest
Tests the AllGather collective operation which gathers data from all GPUs and distributes the combined data to all GPUs.

## Building

These samples require NCCL to be installed on your system. To build:

```bash
cd /path/to/cuda-samples/build
cmake ..
make ncclAllReduceTest
make ncclAllGatherTest
```

Or build all collectives samples:
```bash
cd Samples/9_Collectives
make
```

## Running

### ncclAllReduceTest

Run with default settings (all available GPUs, 32 MB buffer):
```bash
./ncclAllReduceTest
```

Specify number of GPUs (TP size):
```bash
./ncclAllReduceTest --num_gpus=4
```

Customize buffer size and iterations:
```bash
./ncclAllReduceTest --num_gpus=8 --size=64 --iterations=1000 --warmup=20
```

Options:
- `--help`: Display help menu
- `--num_gpus=N`: Number of GPUs to use (TP size). Default: all available GPUs
- `--size=N`: Buffer size in MB per GPU. Default: 32
- `--iterations=N`: Number of benchmark iterations. Default: 100
- `--warmup=N`: Number of warmup iterations. Default: 10

### ncclAllGatherTest

Run with default settings:
```bash
./ncclAllGatherTest
```

Specify number of GPUs:
```bash
./ncclAllGatherTest --num_gpus=4
```

Customize buffer size and iterations:
```bash
./ncclAllGatherTest --num_gpus=8 --size=64 --iterations=1000 --warmup=20
```

Options:
- `--help`: Display help menu
- `--num_gpus=N`: Number of GPUs to use (TP size). Default: all available GPUs
- `--size=N`: Send buffer size in MB per GPU. Default: 32 (receive buffer will be N * num_gpus)
- `--iterations=N`: Number of benchmark iterations. Default: 100
- `--warmup=N`: Number of warmup iterations. Default: 10

## Requirements

- CUDA Toolkit 11.0 or later
- NCCL 2.0 or later
- Multiple CUDA-capable GPUs (minimum 2)
- GPUs should support peer-to-peer communication for optimal performance

## Performance Metrics

Both tests report:
- **Average time**: Time per iteration in milliseconds
- **Bus bandwidth**: Total data transferred divided by time
- **Algorithm bandwidth**: Effective bandwidth based on the collective algorithm

The tests also verify correctness of the results and report PASSED/FAILED for each GPU.

## Notes

- The tests use single-precision floating-point (float) data type
- AllReduce performs a SUM reduction operation
- Each GPU is initialized with rank-specific values for verification
- Results are verified after the benchmark to ensure correctness
