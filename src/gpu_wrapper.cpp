#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// --- DECLARACIÓN DEL PUENTE CUDA ---
// Esta función vive en matmul_kernel.cu y nos permite lanzar el kernel
// desde un archivo .cpp estándar.
extern "C" void cuda_bridge_launch(int blocks, int threads, size_t shared, 
                                  const float* d_a, const float* d_b, 
                                  float* d_partial, int n);

// --- WRAPPER PARA EL MOTOR DE TENSORES ---
// Usamos extern "C" para que el Linker lo encuentre como 'gpu_dot_product'
extern "C" float gpu_dot_product(const float* h_a, const float* h_b, int n) {
    float *d_a, *d_b, *d_partial;
    
    // Configuración para tu GTX 1650 (Arquitectura Turing)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 1. Reservar memoria en la VRAM de la GPU
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_partial, blocksPerGrid * sizeof(float));

    // 2. Transferencia de datos: Host (RAM) -> Device (VRAM)
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Lanzamiento del Kernel a través del puente
    // Usamos memoria compartida (shared memory) para optimizar la reducción parcial
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    cuda_bridge_launch(blocksPerGrid, threadsPerBlock, sharedMemSize, 
                       d_a, d_b, d_partial, n);

    // Esperar a que la GPU termine
    cudaDeviceSynchronize();

    // 4. Recuperar resultados parciales de la GPU
    std::vector<float> h_partial(blocksPerGrid);
    cudaMemcpy(h_partial.data(), d_partial, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Reducción final en CPU (Suma de los bloques parciales)
    float total = 0.0f;
    for(float val : h_partial) {
        total += val;
    }

    // 6. Limpieza de memoria en la GPU
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_partial);

    return total;
}