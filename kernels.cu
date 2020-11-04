__global__ void empty(size_t n) {}

__global__ void axpy(double *y, double* x, double alpha, size_t n) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<n) { y[i] += alpha*x[i]; }
}

__device__ double f(double x)  { return exp(cos(x)) - 2; }
__device__ double fp(double x) { return -sin(x) * exp(cos(x)); }

__global__ void newton(size_t n_iter, double *x, size_t n) {
    auto i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < n) {
        auto x0 = x[i];
        for(int iter = 0; iter < n_iter; ++iter) {
            x0 -= f(x0)/fp(x0);
        }
        x[i] = x0;
    }
}
