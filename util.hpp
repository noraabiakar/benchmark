#pragma once

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cassert>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <thread>

using timer = std::chrono::high_resolution_clock;
using usecs = std::chrono::microseconds;

inline double delta_t(const timer::time_point t0, const timer::time_point t1) {
  return (t1 - t0).count();
}

struct benchmark_parameters {
  size_t block_dim, grid_dim;
  size_t repetitions, epochs;
  size_t array_size, array_size_per_kernel;
  size_t slots, kernels_per_slot;
};

template<typename T>
inline T* offset_and_forward(T* ptr, size_t step, size_t idx) {
  return ptr + idx*step;
}

template<typename T>
inline T offset_and_forward(T val, size_t, size_t) {
  return val;
}

// read command line arguments
inline int read_arg(int argc, char** argv, int index, int default_value) {
    if(argc>index) {
        try {
            auto n = std::stoi(argv[index]);
            if(n<0) {
                return default_value;
            }
            return n;
        }
        catch (std::exception e) {
            std::cout << "error : invalid argument \'" << argv[index]
                      << "\', expected a positive integer." << std::endl;
            exit(1);
        }
    }
    return default_value;
}

static void check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

template <typename T>
T* malloc_device(size_t n) {
    void* p;
    auto status = cudaMalloc(&p, n*sizeof(T));
    check_status(status);
    return static_cast<T*>(p);
}

template <typename T>
void free_device(T* p) {
    cudaFree(p);
}

template <typename T>
T* malloc_host(size_t n) {
    return static_cast<T*>(malloc(n*sizeof(T)));
}

template <typename T>
void copy_to_device(T* from, T* to, size_t n) {
    auto status = cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyHostToDevice);
    check_status(status);
}

template <typename T>
void copy_to_host(T* from, T* to, size_t n) {
    auto status = cudaMemcpy(to, from, n*sizeof(T), cudaMemcpyDeviceToHost);
    check_status(status);
}

static void device_synch() {
    auto status = cudaDeviceSynchronize();
    check_status(status);
}

static void start_gpu_prof() {
    cudaProfilerStart();
}

static void stop_gpu_prof() {
    cudaProfilerStop();
}

static size_t num_blocks(size_t n, size_t threads) {
    return (n + threads - 1)/threads;
}
