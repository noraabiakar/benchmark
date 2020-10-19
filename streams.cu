#include <chrono> 
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "util_cuda.h"
#include "threading.hpp"

// read command line arguments
int read_arg(int argc, char** argv, int index, int default_value) {
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

namespace kernels {

__global__
void empty(unsigned n) {}

__global__
void axpy(double *y, const double* x, double alpha, unsigned n) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i<n) {
        y[i] += alpha*x[i];
    }
}

__device__
double f(double x) {
    return exp(cos(x))-2;
};

__device__
double fp(double x) {
    return -sin(x) * exp(cos(x));
};

__global__
void newton(double *x, unsigned n) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<n) {
        auto x0 = x[i];
        for(int iter=0; iter<7; ++iter) {
            x0 -= f(x0)/fp(x0);
        }
        x[i] = x0;
    }
}
} // namespace kernels

namespace validate {

void newton(double *x, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        auto x0 = x[i];
        for(int iter=0; iter<7; ++iter) {
            x0 -= (exp(cos(x0))-2)/(-sin(x0)*exp(cos(x0)));
        }
        x[i] = x0;
    }
}

void axpy(double *y, const double* x, double alpha, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        y[i] += alpha*x[i];
    }
}

void check_results(double *xd, double* xh, unsigned n) {
        double* xc = malloc_host<double>(n);
        std::copy(xh, xh+n, xc);

        newton(xc, n);

        copy_to_host<double>(xd, xh, n);

        for (unsigned i = 0; i < n; ++i) {
            if (std::abs(xc[i] - xh[i])>1e-9) {
                std::cout << "wrong at " << i << " " << xc[i] << " " << xh[i] << std::endl;
                exit(1);
            }
        }
        std::cout << "SUCCESS" << std::endl;
        std::free(xc);

}
} // namespace validate

long run(unsigned n_epochs,
         unsigned n_streams,
         unsigned n_kernels_per_stream,
         unsigned array_size,
         unsigned block_dim,
         double* x,
         double* y,
         bool multithreaded)
{
    // Total number of kernel launches over all the streams
    const unsigned n_kernels = n_kernels_per_stream * n_streams;

    // Size of the portion of the array that each kernel launch is processing
    const unsigned k_arr_size = array_size/n_kernels;

    // Rounded up division to determine the number of gpu thread blocks to spawn per kernel launch
    const unsigned grid_dim = (k_arr_size-1)/block_dim + 1;

    // Leftover size for the last array
    const unsigned k_arr_size_last = array_size - (k_arr_size * (n_kernels-1));

    // Size of the last grid of the last kernel launch
    const unsigned grid_dim_last = (k_arr_size_last-1)/block_dim + 1;

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) {
        create_stream(&streams[i]);
    }

    auto thread_runner = [&](unsigned stream_idx) {
        for (unsigned k = 0; k < n_kernels_per_stream; ++k) {
            unsigned kernel_idx   = k + stream_idx * n_kernels_per_stream;
            unsigned kernel_start = kernel_idx * k_arr_size;

            auto launch_arr_size =  (kernel_idx == (n_kernels-1)) ? k_arr_size_last : k_arr_size;
            auto launch_grid_dim =  (kernel_idx == (n_kernels-1)) ? grid_dim_last : grid_dim;

            kernels::newton<<<launch_grid_dim, block_dim, 0, streams[stream_idx]>>>(x+kernel_start, launch_arr_size);
        }
    };

    threading::task_system ts(n_streams);

    auto start = std::chrono::system_clock::now();
    if (multithreaded) {
        for (unsigned i = 0; i < n_epochs; ++i) {
            threading::parallel_for::apply(0, n_streams, &ts, [&](int i) {thread_runner(i);});
            device_synch();
        }
    }
    else {
        for (unsigned i = 0; i < n_epochs; ++i) {
            for (unsigned stream_idx = 0; stream_idx < n_streams; ++stream_idx) {
                thread_runner(stream_idx);
            }
            device_synch();
        }
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    for (int i = 0; i < n_streams; i++) {
        destroy_stream(streams[i]);
    }

    return elapsed.count();
}

int main(int argc, char** argv) {
    const unsigned n_epochs               = read_arg(argc, argv, 1, 10);
    const unsigned n_streams              = read_arg(argc, argv, 2, 1);
    const unsigned n_kernels_per_stream   = read_arg(argc, argv, 3, 4);
    const unsigned pow                    = read_arg(argc, argv, 4, 20);
    const unsigned block_dim              = read_arg(argc, argv, 5, 128);
    const unsigned multithreaded          = read_arg(argc, argv, 6, 0);

    const unsigned array_size = 2 << pow;

    // Run the newton kernel a bunch of times on a larger array to "warm up"
    {
        unsigned ni = 2<<24;
        unsigned grid_dim = (ni-1)/block_dim + 1;
        double* xhi = malloc_host<double>(ni);
        double* xdi = malloc_device<double>(ni);
        std::fill(xhi, xhi+ni, 2.3);
        copy_to_device<double>(xhi, xdi, ni);
        for (auto i=0; i<100; ++i) {
            kernels::newton<<<grid_dim, block_dim>>>(xdi, ni);
        }
        std::free(xhi);
        free_device(xdi);
    }

    double* xh = malloc_host<double>(array_size);
    double* yh = malloc_host<double>(array_size);

    double* xd = malloc_device<double>(array_size);
    double* yd = malloc_device<double>(array_size);

    std::fill(xh, xh+array_size, 2.0);
    std::fill(yh, yh+array_size, 1.0);

    copy_to_device<double>(xh, xd, array_size);
    copy_to_device<double>(yh, yd, array_size);

    device_synch();

    start_gpu_prof();
    auto time_us = run(n_epochs, n_streams, n_kernels_per_stream, array_size, block_dim, xd, yd, multithreaded);
    stop_gpu_prof();

    std::cout << n_epochs  << ", " 
              << n_streams << ", " 
              << n_kernels_per_stream << ", "
              << array_size << ", "
              << block_dim << ", "
              << multithreaded << ", "
              << n_epochs * (array_size * sizeof(double)) / (double)(time_us) << "\n";

    validate::check_results(xd, xh, array_size);

    std::free(xh);
    std::free(yh);
    free_device(xd);
    free_device(yd);

    return 0;
}
