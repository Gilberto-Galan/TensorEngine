cmake_minimum_required(VERSION 3.18)

# Saltamos el bloqueo de versión de MSVC v18 para CUDA v12.8
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# Definimos el proyecto y los lenguajes (C++, Assembly y CUDA)
project(TensorEngine LANGUAGES CXX ASM_NASM CUDA)

# Especificamos la arquitectura Turing de tu GTX 1650
set(CMAKE_CUDA_ARCHITECTURES 75)

# Creamos el ejecutable
add_executable(TensorEngine 
    src/main.cpp 
    src/matmul_avx.asm
    src/matmul_kernel.cu
    src/gpu_wrapper.cpp
)

# --- ESTO ES LO QUE FALTABA PARA EL ERROR C1083 ---
# Le decimos a TODOS los archivos (especialmente a gpu_wrapper.cpp) 
# dónde encontrar cuda_runtime.h
target_include_directories(TensorEngine PRIVATE 
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include"
)

# Vinculamos la librería de tiempo de ejecución de CUDA
target_link_libraries(TensorEngine PRIVATE cudart)

# Optimizaciones adicionales para tu i3-10105 y la GPU
target_compile_options(TensorEngine PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:/O2 /arch:AVX2>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)