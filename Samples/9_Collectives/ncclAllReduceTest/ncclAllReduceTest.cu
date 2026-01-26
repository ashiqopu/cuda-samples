/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <helper_cuda.h>
#include <helper_timer.h>

const char *sSampleName = "NCCL AllReduce Test";

// Macro for checking NCCL errors
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// Macro for checking CUDA errors
#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed, CUDA error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void printHelp(void) {
    printf("Usage: ncclAllReduceTest [OPTION]...\n");
    printf("Tests NCCL AllReduce collective operation across multiple GPUs\n");
    printf("\n");
    printf("Options:\n");
    printf("--help               Display this help menu\n");
    printf("--num_gpus=N         Number of GPUs to use for the test (TP size)\n");
    printf("                     Default: use all available GPUs\n");
    printf("--size=N             Size of data buffer in MB per GPU\n");
    printf("                     Default: 32 MB\n");
    printf("--iterations=N       Number of iterations to run\n");
    printf("                     Default: 100\n");
    printf("--warmup=N           Number of warmup iterations\n");
    printf("                     Default: 10\n");
}

bool verifyAllReduce(float *h_result, float expected_sum, size_t count, int rank) {
    bool success = true;
    for (size_t i = 0; i < count; i++) {
        if (fabs(h_result[i] - expected_sum) > 1e-5) {
            if (success) {  // Only print first few errors
                printf("Rank %d: Verification failed at index %zu: expected %f, got %f\n", 
                       rank, i, expected_sum, h_result[i]);
            }
            success = false;
            if (i > 10) break;  // Don't print too many errors
        }
    }
    return success;
}

void runIterativeBenchmark(int num_gpus, const std::vector<size_t>& size_mbs, int iterations) {
    printf("\n=== Running Iterative AllReduce Benchmark ===\n");
    printf("GPUs: %d, Iterations: %d\n", num_gpus, iterations);
    
    // Create benchmark results directory
    char dirPath[512];
    snprintf(dirPath, sizeof(dirPath), "benchmark_results/allreduce_gpus_%d", num_gpus);
    mkdir("benchmark_results", 0755);
    mkdir(dirPath, 0755);
    
    // Setup NCCL communicators
    std::vector<cudaStream_t> streams(num_gpus);
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<int> devs(num_gpus);
    
    for (int i = 0; i < num_gpus; i++) {
        devs[i] = i;
    }
    
    NCCLCHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));
    
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Storage for consolidated results
    struct SizeStats {
        size_t size_mb;
        double time_min, time_max, time_median;
        double bw_min, bw_max, bw_median;
        double algbw_min, algbw_max, algbw_median;
    };
    std::vector<SizeStats> all_stats;
    
    // Test each message size
    for (size_t size_mb : size_mbs) {
        printf("\nTesting message size: %zu MB\n", size_mb);
        
        size_t count = (size_mb * 1024 * 1024) / sizeof(float);
        size_t bytes = count * sizeof(float);
        
        // Allocate buffers
        std::vector<float*> d_sendbuff(num_gpus);
        std::vector<float*> d_recvbuff(num_gpus);
        
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaMalloc(&d_sendbuff[i], bytes));
            CUDACHECK(cudaMalloc(&d_recvbuff[i], bytes));
            
            std::vector<float> h_buff(count, (float)(i + 1));
            CUDACHECK(cudaMemcpy(d_sendbuff[i], h_buff.data(), bytes, cudaMemcpyHostToDevice));
        }
        
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaDeviceSynchronize());
        }
        
        // Open result file for this message size
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/msg_size_%zuMB.txt", dirPath, size_mb);
        std::ofstream outfile(filename);
        outfile << "Iteration,Time_us,Bandwidth_GBps,AlgBW_GBps\n";
        
        std::vector<double> times_us;
        std::vector<double> bandwidths;
        std::vector<double> algbws;
        
        // Run iterations
        for (int iter = 0; iter < iterations; iter++) {
            StopWatchInterface *timer = NULL;
            sdkCreateTimer(&timer);
            sdkStartTimer(&timer);
            
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < num_gpus; i++) {
                NCCLCHECK(ncclAllReduce((const void*)d_sendbuff[i], (void*)d_recvbuff[i],
                                       count, ncclFloat, ncclSum, comms[i], streams[i]));
            }
            NCCLCHECK(ncclGroupEnd());
            
            for (int i = 0; i < num_gpus; i++) {
                CUDACHECK(cudaSetDevice(i));
                CUDACHECK(cudaStreamSynchronize(streams[i]));
            }
            
            sdkStopTimer(&timer);
            float time_ms = sdkGetTimerValue(&timer);
            double time_us = time_ms * 1000.0;
            sdkDeleteTimer(&timer);
            
            double bandwidth_gbps = (bytes * num_gpus / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
            double algbw_gbps = (bytes * 2.0 * (num_gpus - 1) / num_gpus / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
            
            times_us.push_back(time_us);
            bandwidths.push_back(bandwidth_gbps);
            algbws.push_back(algbw_gbps);
            
            outfile << iter << "," << time_us << "," << bandwidth_gbps << "," << algbw_gbps << "\n";
        }
        
        outfile.close();
        
        // Calculate statistics
        std::sort(times_us.begin(), times_us.end());
        std::sort(bandwidths.begin(), bandwidths.end());
        std::sort(algbws.begin(), algbws.end());
        
        double time_min = times_us[0];
        double time_max = times_us[times_us.size() - 1];
        double time_median = times_us.size() % 2 == 0 ?
            (times_us[times_us.size()/2 - 1] + times_us[times_us.size()/2]) / 2.0 :
            times_us[times_us.size()/2];
        
        double bw_min = bandwidths[0];
        double bw_max = bandwidths[bandwidths.size() - 1];
        double bw_median = bandwidths.size() % 2 == 0 ?
            (bandwidths[bandwidths.size()/2 - 1] + bandwidths[bandwidths.size()/2]) / 2.0 :
            bandwidths[bandwidths.size()/2];
        
        double algbw_min = algbws[0];
        double algbw_max = algbws[algbws.size() - 1];
        double algbw_median = algbws.size() % 2 == 0 ?
            (algbws[algbws.size()/2 - 1] + algbws[algbws.size()/2]) / 2.0 :
            algbws[algbws.size()/2];
        
        printf("  Time (us):        Min: %8.2f, Max: %8.2f, Median: %8.2f\n", time_min, time_max, time_median);
        printf("  Bandwidth (GB/s): Min: %8.2f, Max: %8.2f, Median: %8.2f\n", bw_min, bw_max, bw_median);
        printf("  AlgBW (GB/s):     Min: %8.2f, Max: %8.2f, Median: %8.2f\n", algbw_min, algbw_max, algbw_median);
        
        // Store statistics for consolidated table
        all_stats.push_back({size_mb, time_min, time_max, time_median, 
                            bw_min, bw_max, bw_median,
                            algbw_min, algbw_max, algbw_median});
        
        // Cleanup buffers for this message size
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaFree(d_sendbuff[i]));
            CUDACHECK(cudaFree(d_recvbuff[i]));
        }
    }
    
    // Cleanup NCCL
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
    
    // Write consolidated results table
    char consolidatedFile[512];
    snprintf(consolidatedFile, sizeof(consolidatedFile), "%s/consolidated_results.txt", dirPath);
    std::ofstream consFile(consolidatedFile);
    
    consFile << "AllReduce Collective Performance Results\n";
    consFile << "=========================================\n";
    consFile << "GPUs: " << num_gpus << "\n";
    consFile << "Iterations: " << iterations << "\n";
    consFile << "\nBuffer Configuration per Message Size:\n";
    for (const auto& stats : all_stats) {
        consFile << "  " << stats.size_mb << " MB: Send buffer per GPU = " << stats.size_mb 
                 << " MB, Recv buffer per GPU = " << stats.size_mb << " MB\n";
    }
    consFile << "\n";
    
    consFile << "Statistics by Data Size:\n";
    consFile << "Data Size | Metric         | Min      | Max      | Median  \n";
    consFile << "----------+----------------+----------+----------+---------\n";
    
    for (const auto& stats : all_stats) {
        consFile << std::setw(7) << stats.size_mb << "MB | Bandwidth GB/s | " 
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.bw_min << " | "
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.bw_max << " | "
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.bw_median << "\n";
        consFile << "          | AlgBW GB/s     | " 
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.algbw_min << " | "
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.algbw_max << " | "
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.algbw_median << "\n";
        consFile << "          | Time μs        | " 
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.time_min << " | "
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.time_max << " | "
                 << std::setw(8) << std::fixed << std::setprecision(2) << stats.time_median << "\n";
        consFile << "----------+----------------+----------+----------+---------\n";
    }
    
    consFile.close();
    
    printf("\nResults written to: %s/\n", dirPath);
    printf("Consolidated table: %s\n", consolidatedFile);
}

int main(int argc, char **argv) {
    printf("%s Starting...\n\n", sSampleName);

    // Parse command line arguments
    int num_gpus = -1;  // -1 means use all available
    size_t size_mb = 32;
    int iterations = 100;
    int warmup = 10;
    bool iterative = false;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--help", 6) == 0) {
            printHelp();
            return 0;
        } else if (strncmp(argv[i], "--iterative", 11) == 0) {
            iterative = true;
        } else if (strncmp(argv[i], "--num_gpus=", 11) == 0) {
            num_gpus = atoi(argv[i] + 11);
        } else if (strncmp(argv[i], "--size=", 7) == 0) {
            size_mb = atoi(argv[i] + 7);
        } else if (strncmp(argv[i], "--iterations=", 13) == 0) {
            iterations = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--warmup=", 9) == 0) {
            warmup = atoi(argv[i] + 9);
        }
    }

    // Get number of available GPUs
    int deviceCount = 0;
    CUDACHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        printf("No CUDA capable devices found!\n");
        return EXIT_FAILURE;
    }

    // Determine number of GPUs to use
    if (num_gpus == -1) {
        num_gpus = deviceCount;
    } else if (num_gpus > deviceCount) {
        printf("Requested %d GPUs but only %d are available. Using %d GPUs.\n", 
               num_gpus, deviceCount, deviceCount);
        num_gpus = deviceCount;
    }

    if (num_gpus < 2) {
        printf("AllReduce requires at least 2 GPUs. Found %d GPU(s).\n", num_gpus);
        return EXIT_FAILURE;
    }

    // If iterative mode, run the iterative benchmark
    if (iterative) {
        std::vector<size_t> sizes = {1, 10, 100, 1000};
        runIterativeBenchmark(num_gpus, sizes, iterations);
        return EXIT_SUCCESS;
    }

    printf("Running AllReduce test with %d GPUs\n", num_gpus);
    printf("Buffer size per GPU: %zu MB\n", size_mb);
    printf("Iterations: %d (warmup: %d)\n\n", iterations, warmup);

    // Calculate buffer size
    size_t count = (size_mb * 1024 * 1024) / sizeof(float);
    size_t bytes = count * sizeof(float);

    // Allocate buffers and setup NCCL
    std::vector<float*> d_sendbuff(num_gpus);
    std::vector<float*> d_recvbuff(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);
    std::vector<ncclComm_t> comms(num_gpus);
    std::vector<int> devs(num_gpus);

    // Initialize device list
    for (int i = 0; i < num_gpus; i++) {
        devs[i] = i;
    }

    // Initialize NCCL communicators
    NCCLCHECK(ncclCommInitAll(comms.data(), num_gpus, devs.data()));

    // Allocate buffers and create streams
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&d_sendbuff[i], bytes));
        CUDACHECK(cudaMalloc(&d_recvbuff[i], bytes));
        CUDACHECK(cudaStreamCreate(&streams[i]));

        // Initialize send buffer with rank-specific values
        std::vector<float> h_buff(count);
        for (size_t j = 0; j < count; j++) {
            h_buff[j] = (float)(i + 1);  // Rank 0 gets 1.0, rank 1 gets 2.0, etc.
        }
        CUDACHECK(cudaMemcpy(d_sendbuff[i], h_buff.data(), bytes, cudaMemcpyHostToDevice));
    }

    // Synchronize all devices
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaDeviceSynchronize());
    }

    printf("Starting warmup...\n");
    // Warmup iterations
    for (int iter = 0; iter < warmup; iter++) {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            NCCLCHECK(ncclAllReduce((const void*)d_sendbuff[i], (void*)d_recvbuff[i], 
                                   count, ncclFloat, ncclSum, comms[i], streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());
        
        // Wait for completion
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    printf("Starting benchmark...\n");
    // Benchmark iterations
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int iter = 0; iter < iterations; iter++) {
        NCCLCHECK(ncclGroupStart());
        for (int i = 0; i < num_gpus; i++) {
            NCCLCHECK(ncclAllReduce((const void*)d_sendbuff[i], (void*)d_recvbuff[i], 
                                   count, ncclFloat, ncclSum, comms[i], streams[i]));
        }
        NCCLCHECK(ncclGroupEnd());
        
        // Wait for completion
        for (int i = 0; i < num_gpus; i++) {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(streams[i]));
        }
    }

    sdkStopTimer(&timer);
    float elapsed_ms = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    // Calculate statistics
    float avg_time_ms = elapsed_ms / iterations;
    float bandwidth_gbps = (bytes * num_gpus / (1024.0 * 1024.0 * 1024.0)) / (avg_time_ms / 1000.0);
    float algbw_gbps = (bytes * 2.0 * (num_gpus - 1) / num_gpus / (1024.0 * 1024.0 * 1024.0)) / (avg_time_ms / 1000.0);

    printf("\nPerformance Results:\n");
    printf("-----------------------------------\n");
    printf("Number of GPUs:         %d\n", num_gpus);
    printf("Buffer size per GPU:    %.2f MB\n", (float)bytes / (1024.0 * 1024.0));
    printf("Average time:           %.4f ms\n", avg_time_ms);
    printf("Bus bandwidth:          %.2f GB/s\n", bandwidth_gbps);
    printf("Algorithm bandwidth:    %.2f GB/s\n", algbw_gbps);
    printf("-----------------------------------\n\n");

    // Verify results
    printf("Verifying results...\n");
    bool all_correct = true;
    float expected_sum = 0.0f;
    for (int i = 0; i < num_gpus; i++) {
        expected_sum += (float)(i + 1);
    }

    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        std::vector<float> h_result(count);
        CUDACHECK(cudaMemcpy(h_result.data(), d_recvbuff[i], bytes, cudaMemcpyDeviceToHost));
        
        bool correct = verifyAllReduce(h_result.data(), expected_sum, count, i);
        if (!correct) {
            all_correct = false;
        }
        printf("Rank %d: %s\n", i, correct ? "PASSED" : "FAILED");
    }

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_sendbuff[i]));
        CUDACHECK(cudaFree(d_recvbuff[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    printf("\nTest %s\n", all_correct ? "PASSED" : "FAILED");
    return all_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
