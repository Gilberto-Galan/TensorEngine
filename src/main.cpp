#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "../include/tensor.h"

extern "C" {
    float matmul_dot_avx(const float* a, const float* b, int len);
    float gpu_dot_product(const float* a, const float* b, int n);
}

float cpu_dot_product(const float* a, const float* b, int len) {
    float sum = 0.0f;
    for(int i = 0; i < len; ++i) sum += a[i] * b[i];
    return sum;
}

int main() {
    // SUBIMOS A 50 MILLONES para que la GTX 1650 realmente trabaje
    const int SIZE = 50 * 1024 * 1024; 
    
    Tensor A(1, SIZE);
    Tensor B(1, SIZE);
    A.randomize();
    B.randomize();

    std::cout << "--- Benchmarking Dot Product (" << SIZE << " elements) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // 1. C++
    auto s1 = std::chrono::high_resolution_clock::now();
    float res_cpp = cpu_dot_product(A.data.data(), B.data.data(), SIZE);
    auto e1 = std::chrono::high_resolution_clock::now();
    double ms_cpp = std::chrono::duration<double, std::milli>(e1 - s1).count();

    // 2. ASSEMBLY
    auto s2 = std::chrono::high_resolution_clock::now();
    float res_asm = matmul_dot_avx(A.data.data(), B.data.data(), SIZE);
    auto e2 = std::chrono::high_resolution_clock::now();
    double ms_asm = std::chrono::duration<double, std::milli>(e2 - s2).count();

    // 3. CUDA
    auto s3 = std::chrono::high_resolution_clock::now();
    float res_gpu = gpu_dot_product(A.data.data(), B.data.data(), SIZE);
    auto e3 = std::chrono::high_resolution_clock::now();
    double ms_gpu = std::chrono::duration<double, std::milli>(e3 - s3).count();

    // Aseguramos que el formato sea "Tiempo: X.XXXXms" sin espacios extraños
    std::cout << "Resultado C++: " << res_cpp << " | Tiempo: " << (ms_cpp < 0.0001 ? 0.0001 : ms_cpp) << "ms" << std::endl;
    std::cout << "Resultado ASM: " << res_asm << " | Tiempo: " << (ms_asm < 0.0001 ? 0.0001 : ms_asm) << "ms" << std::endl;
    std::cout << "Resultado CUDA: " << res_gpu << " | Tiempo: " << (ms_gpu < 0.0001 ? 0.0001 : ms_gpu) << "ms" << std::endl;
    
    return 0;
}