# 🚀 TensorEngine: High-Performance Tensor Operations

**TensorEngine** es un motor de cómputo de bajo nivel diseñado para realizar operaciones de álgebra lineal optimizadas para hardware específico. Este proyecto demuestra la aceleración de cálculos matemáticos pesados mediante el uso de tres paradigmas de programación distintos: **C++ Estándar**, **SIMD (Single Instruction, Multiple Data)** con Assembly AVX-2 y **GPGPU** con NVIDIA CUDA.



## 🛠️ Stack Tecnológico & Hardware

Para este desarrollo, se utilizó una arquitectura híbrida que exprime al máximo los recursos locales:

* **CPU:** Intel Core i3-10105 (4C/8T) - Optimizado con instrucciones **AVX-2**.
* **GPU:** NVIDIA GeForce GTX 1650 (4GB VRAM) - Arquitectura **Turing**.
* **Lenguajes:** C++17, x86-64 Assembly (NASM), CUDA C++ 12.8.
* **Build System:** CMake 3.18+ & NMake (MSVC v143).

## 🧠 Desafíos de Ingeniería (Problem Solving)

Este proyecto destaca por la resolución de conflictos críticos de entorno y arquitectura, demostrando habilidades en infraestructura de software:

1.  **Bypass de Compatibilidad:** Implementación de un hack técnico para utilizar **CUDA 12.8** con **MSVC v19.50 (Visual Studio 2025 Preview)** mediante el flag `-allow-unsupported-compiler`, superando los bloqueos de versión del toolkit de NVIDIA.
2.  **Orquestación Multi-Disco:** Configuración de rutas mediante `target_include_directories` en CMake para permitir una compilación exitosa en un entorno distribuido (Herramientas en unidad **D:** y CUDA Toolkit en **C:**).
3.  **Interoperabilidad Híbrida:** Gestión de enlazado mediante bloques `extern "C"` para evitar el *Name Mangling* de C++, permitiendo la comunicación binaria entre los objetos generados por NASM (ASM), NVCC (CUDA) y CL (MSVC).



## 📊 Benchmark: Producto Punto (50M+ elementos)

El sistema incluye un módulo de automatización en Python que mide y grafica el rendimiento real. Al escalar la carga de trabajo a 50 millones de elementos, se observan las capacidades de paralelismo:

| Implementación | Tecnología | Rendimiento | Ventaja |
| :--- | :--- | :--- | :--- |
| **C++ Naive** | Operaciones Secuenciales | Base | Referencia |
| **Assembly AVX-2** | Registros de 256-bits | Eficiente | ~2x - 4x Velocidad |
| **NVIDIA CUDA** | Cómputo Paralelo Masivo | **Extremo** | **10x+ Aceleración** |

> **Nota:** La GPU GTX 1650 utiliza un Kernel personalizado con reducción parcial para minimizar el cuello de botella del bus PCIe.



## 🚀 Instalación y Uso

### Requisitos previos
* NVIDIA CUDA Toolkit 12.x
* NASM (Netwide Assembler)
* Visual Studio Build Tools (MSVC)
* Visual Studio Community 2022/2025**: Es indispensable instalar el componente:
    * *Desktop development with C++* (Desarrollo para el escritorio con C++).
* Python 3.10+: Para la ejecución del benchmark y visualización

## 🛠️ Configuración y Construcción

### Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/TensorEngine
```

### Configurar y compilar con NMake
```cmd
mkdir build && cd build
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..
nmake
```

### Configurar el entorno de Python
```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno
# En Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# En Windows (CMD):
.\venv\Scripts\activate.bat

# Instalar dependencias de visualización
pip install matplotlib
```

## ⚙️ Configuración del Entorno (VS Code)

Para integrar **NVCC (CUDA)**, **NASM (Assembly)** y **MSVC (C++)** en un solo flujo de trabajo, se configuraron los siguientes archivos en la carpeta `.vscode`:

### 1. `c_cpp_properties.json` (IntelliSense)
Se configuró para que VS Code reconozca las librerías de CUDA y los encabezados de MSVC, eliminando los errores de "include no encontrado".
* **Rutas clave:** Se añadieron `${env:CUDA_PATH}/include` y las rutas del SDK de Windows para habilitar el autocompletado en archivos `.cu` y `.cpp`.

### 2. `tasks.json` (Automatización de Compilación)
Define la automatización del proceso de construcción sin salir del editor.
* **Flujo:** Se crearon tareas para ejecutar `cmake` y `nmake` de forma secuencial, vinculadas al entorno de **x64 Native Tools**.

### 3. `launch.json` (Depuración)
Configurado para permitir el **Debugging** del ejecutable `TensorEngine.exe`.
* **Debugger:** Utiliza `cppvsdbg` (Visual Studio Windows Debugger) para permitir el rastreo de variables y breakpoints directamente en el código fuente.

### 4. `settings.json` (Preferencias del Proyecto)
* **Asociación de archivos:** Se forzó a VS Code a tratar archivos `.cu` como C++ para mantener el resaltado de sintaxis.
* **CMake Tooling:** Configuración para que el plugin de CMake utilice específicamente el generador `NMake Makefiles`.

### Compilar el Motor (C++/CUDA/ASM)

> [!IMPORTANT]
> **Paso Crítico:** Este paso debe realizarse en la terminal **x64 Native Tools Command Prompt for VS**. No utilices PowerShell o CMD estándar para la compilación.

```bash
# Crear directorio de compilación
mkdir build && cd build

# Generar archivos de construcción con CMake
cmake -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ..

# Compilar el ejecutable final
nmake
```

## 📊 Ejecución del Benchmark

Una vez compilado el motor y activo tu entorno virtual de Python, ejecuta el script desde la raíz del proyecto para generar la comparativa de rendimiento:

```bash
python Scripts/benchmark.py
```

Al finalizar, el script abrirá una ventana con la gráfica comparativa y guardará una imagen en alta resolución llamada benchmark_results_final.png.


## 📂 Estructura del Proyecto

```text
TensorEngine/
├── .vscode/               # Configuraciones de IntelliSense y Debugging
│   ├── c_cpp_properties.json
│   ├── launch.json
│   ├── settings.json
│   └── tasks.json
├── build/                 # Binarios y archivos generados por CMake
│   ├── CMakeFiles/
│   ├── TensorEngine.exe   # Ejecutable final optimizado
│   ├── CMakeCache.txt
│   └── Makefile
├── include/               # Cabeceras (Headers)
│   └── tensor.h           # Definición de la clase Tensor y prototipos
├── Scripts/               # Herramientas de análisis
│   └── benchmark.py      # Automatización de pruebas y gráficas
├── src/                   # Código fuente (Híbrido)
│   ├── main.cpp           # Lógica del benchmark y C++ Naive
│   ├── gpu_wrapper.cpp    # Puente de comunicación Host-to-Device
│   ├── matmul_avx.asm     # Optimización SIMD en Assembly x86-64
│   └── matmul_kernel.cu   # Kernel de cómputo paralelo en CUDA
├── CMakeLists.txt         # Configuración del Build System (CMake)
├── .gitignore             # Exclusión de archivos binarios y temporales
└── readme.md              # Documentación técnica del proyecto