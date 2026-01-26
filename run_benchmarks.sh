#!/bin/bash
# CUDA Samples Benchmark Runner
# This script runs selected benchmarks and saves results

set -e

# Create results directory
RESULTS_DIR="/cuda-samples/benchmark_results"
mkdir -p "$RESULTS_DIR"

# Get timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/benchmark_results_$TIMESTAMP.txt"

echo "======================================" | tee "$RESULTS_FILE"
echo "CUDA Samples Benchmark Results" | tee -a "$RESULTS_FILE"
echo "Timestamp: $(date)" | tee -a "$RESULTS_FILE"
echo "======================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Get GPU information
echo "=== GPU Information ===" | tee -a "$RESULTS_FILE"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

cd /cuda-samples/build

# Function to run a benchmark
run_benchmark() {
    local sample_path=$1
    local sample_name=$(basename "$sample_path")
    
    if [ -x "$sample_path" ]; then
        echo "=== Running: $sample_name ===" | tee -a "$RESULTS_FILE"
        "$sample_path" 2>&1 | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
        return 0
    else
        echo "Skipping $sample_name (not found or not executable)" | tee -a "$RESULTS_FILE"
        return 1
    fi
}

# # Performance-related samples
# echo "=== Performance Benchmarks ===" | tee -a "$RESULTS_FILE"

# # Matrix Multiplication
# run_benchmark "./Samples/0_Introduction/matrixMul/matrixMul"

# # Bandwidth Test
# run_benchmark "./Samples/1_Utilities/bandwidthTest/bandwidthTest"

# # Device Query
# run_benchmark "./Samples/1_Utilities/deviceQuery/deviceQuery"

# # Concurrent Kernels
# run_benchmark "./Samples/0_Introduction/concurrentKernels/concurrentKernels" || true

# # N-Body Simulation
# run_benchmark "./Samples/5_Domain_Specific/nbody/nbody" || true

# # Black-Scholes (Finance)
# run_benchmark "./Samples/4_CUDA_Libraries/BlackScholes/BlackScholes" || true

# # Fast Walsh Transform
# run_benchmark "./Samples/2_Concepts_and_Techniques/fastWalshTransform/fastWalshTransform" || true

# # Reduction
# run_benchmark "./Samples/0_Introduction/reduction/reduction" || true

# # Transpose
# run_benchmark "./Samples/6_Performance/transpose/transpose" || true

# P2P Benchmark
run_benchmark "./Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest" || true

echo "======================================" | tee -a "$RESULTS_FILE"
echo "Benchmark completed!" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "======================================" | tee -a "$RESULTS_FILE"

# Create a symlink to latest results
ln -sf "benchmark_results_$TIMESTAMP.txt" "$RESULTS_DIR/latest.txt"
