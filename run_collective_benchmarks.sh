#!/bin/bash

# Script to run AllGather and AllReduce benchmarks with various configurations

set -e

ALLGATHER="/cuda-samples/build/Samples/9_Collectives/ncclAllGatherTest/ncclAllGatherTest"
ALLREDUCE="/cuda-samples/build/Samples/9_Collectives/ncclAllReduceTest/ncclAllReduceTest"

# Number of GPUs: 2, 4, 8
GPUS=(2 4 8)
# Iterations
ITERATIONS=100

echo "======================================================"
echo "Running AllGather and AllReduce Collective Benchmarks"
echo "======================================================"
echo "GPU counts: ${GPUS[@]}"
echo "Iterations per test: $ITERATIONS"
echo "Message sizes: 1, 10, 100, 1000 MB (tested in iterative mode)"
echo "======================================================"
echo ""

# Run AllGather tests
echo "Starting AllGather benchmarks..."
for ngpu in "${GPUS[@]}"; do
    echo ""
    echo ">>> AllGather with $ngpu GPUs <<<"
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) $ALLGATHER \
        --num_gpus=$ngpu \
        --iterations=$ITERATIONS \
        --iterative
done

echo ""
echo "======================================================"
echo "Starting AllReduce benchmarks..."
for ngpu in "${GPUS[@]}"; do
    echo ""
    echo ">>> AllReduce with $ngpu GPUs <<<"
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) $ALLREDUCE \
        --num_gpus=$ngpu \
        --iterations=$ITERATIONS \
        --iterative
done

echo ""
echo "======================================================"
echo "All benchmarks completed!"
echo "Results written to benchmark_results/ directory"
echo "======================================================"

