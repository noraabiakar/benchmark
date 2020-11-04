#include "util.hpp"

#include "streams.hcu"
#include "graphs.hcu"
#include "simple.hcu"

int main(int argc, char** argv) {
    benchmark_parameters parameters;
    parameters.epochs           = read_arg(argc, argv, 1, 10);
    parameters.slots            = read_arg(argc, argv, 2, 1);
    parameters.kernels_per_slot = read_arg(argc, argv, 3, 4);
    parameters.array_size       = 1 << read_arg(argc, argv, 4, 20);
    parameters.block_dim        = read_arg(argc, argv, 5, 128);
    parameters.repetitions      = read_arg(argc, argv, 6, 10);

    auto total_kernels = parameters.slots*parameters.kernels_per_slot;
    assert(0 == parameters.array_size % total_kernels);
    parameters.array_size_per_kernel = parameters.array_size/total_kernels;
    parameters.grid_dim = num_blocks(parameters.array_size_per_kernel, parameters.block_dim);

    std::cout << "array_size          = " << parameters.array_size  << std::endl;
    std::cout << "array_size_per_task = " << parameters.array_size_per_kernel << std::endl;
    std::cout << "epochs              = " << parameters.epochs << std::endl;
    std::cout << "repetitions         = " << parameters.repetitions << std::endl;
    std::cout << "slots               = " << parameters.slots << std::endl;
    std::cout << "kernels_per_slot    = " << parameters.kernels_per_slot << std::endl;
    std::cout << "block_dim           = " << parameters.block_dim << std::endl;
    std::cout << "grid_dim            = " << parameters.grid_dim << std::endl;

    double* xh = malloc_host<double>(parameters.array_size);
    double* yh = malloc_host<double>(parameters.array_size);

    double* xd = malloc_device<double>(parameters.array_size);
    double* yd = malloc_device<double>(parameters.array_size);

    std::fill(xh, xh + parameters.array_size, 2.0);
    std::fill(yh, yh + parameters.array_size, 1.0);

    copy_to_device<double>(xh, xd, parameters.array_size);
    copy_to_device<double>(yh, yd, parameters.array_size);

    start_gpu_prof();
    auto result = CASE(parameters, newton, 5ul, xd, parameters.array_size_per_kernel);
    stop_gpu_prof();

    for (const auto t: result) {
      std::cout << t << ',';
    }
    std::cout << '\n';

    copy_to_host<double>(xd, xh, parameters.array_size);
    copy_to_host<double>(yd, yh, parameters.array_size);

    std::free(xh);
    std::free(yh);
    free_device(xd);
    free_device(yd);
}
