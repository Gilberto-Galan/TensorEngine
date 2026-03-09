#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <crt/device_functions.h>

// El Kernel sigue igual
__global__ void dot_product_kernel(const float* a, const float* b, float* partial_sums, int n) {
    extern __shared__ float cache[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (i < n) {
        temp += a[i] * b[i];
        i += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    for (int j = blockDim.x / 2; j > 0; j >>= 1) {
        if (cacheIndex < j) cache[cacheIndex] += cache[cacheIndex + j];
        __syncthreads();
    }
    if (cacheIndex == 0) partial_sums[blockIdx.x] = cache[0];
}

// ESTA ES LA FUNCIÓN PUENTE (Wrapper)
extern "C" void cuda_bridge_launch(int blocks, int threads, size_t shared, 
                                 const float* d_a, const float* d_b, 
                                 float* d_partial, int n) {
    // Aquí el compilador nvcc sí entiende los <<< >>>
    dot_product_kernel<<<blocks, threads, shared>>>(d_a, d_b, d_partial, n);
}