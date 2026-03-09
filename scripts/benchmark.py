import subprocess
import matplotlib.pyplot as plt
import re
import os

def safe_float(value_str):
    """Convierte strings a float manejando casos de 'inf' o errores de precisión."""
    try:
        f = float(value_str)
        # Si es infinito o demasiado pequeño, devolvemos un valor mínimo para no romper la gráfica
        if f == float('inf') or f <= 0:
            return 0.0001
        return f
    except ValueError:
        return 0.0001

def run_benchmark():
    # 1. Ruta al ejecutable
    executable = os.path.join("build", "TensorEngine.exe")
    
    if not os.path.exists(executable):
        print(f"❌ Error: No se encontró {executable}. Compila primero con nmake.")
        return

    try:
        print("🚀 Ejecutando motor de tensores (C++, ASM, CUDA)...")
        # Aumentamos el timeout por si el cálculo de 50M de elementos tarda
        result = subprocess.run([executable], capture_output=True, text=True, check=True, timeout=30)
        output = result.stdout
        print("\nSalida del motor:\n", output)
    except Exception as e:
        print(f"❌ Error al ejecutar el motor: {e}")
        return

    # 2. Extraer los tiempos (Regex mejorado)
    # Soporta: números, puntos, 'inf', y notación científica (e-05)
    times_found = re.findall(r"Tiempo:\s*([+-]?\d*\.\d+|inf|[+-]?\d+)ms", output)
    
    if len(times_found) < 3:
        print(f"❌ Error: Se encontraron {len(times_found)} tiempos, pero se necesitan 3.")
        print("Asegúrate de que el .exe imprima: 'Tiempo: X.XXXXms'")
        return

    # Convertir con seguridad
    cpp_time = safe_float(times_found[0])
    asm_time = safe_float(times_found[1])
    gpu_time = safe_float(times_found[2])

    # 3. Preparar datos para la gráfica
    labels = ['C++ Estándar', 'Assembly AVX-2', 'NVIDIA CUDA']
    values = [cpp_time, asm_time, gpu_time]
    colors = ['#e74c3c', '#3498db', '#2ecc71'] # Rojo, Azul (Intel), Verde (NVIDIA)

    # 4. Crear la gráfica profesional
    plt.style.use('ggplot') 
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Usamos escala logarítmica si la diferencia entre GPU y CPU es masiva
    # Esto evita que la barra de la GPU sea invisible si es 1000x más rápida
    if cpp_time / gpu_time > 100:
        ax.set_yscale('log')
        ax.set_ylabel('Tiempo de ejecución (ms) - Escala Logarítmica', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('Tiempo de ejecución (ms)', fontsize=12, fontweight='bold')

    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_title('Benchmark: Producto Punto (Matriz de Tensores)\nCPU Intel i3-10105 vs GPU GTX 1650', 
                 fontsize=15, pad=20, fontweight='bold')
    
    # 5. Calcular Speedup con seguridad
    speedup_text = ""
    if asm_time > 0.0001 and cpp_time > 0.0001:
        speedup_asm = cpp_time / asm_time
        speedup_text += f"ASM: {speedup_asm:.2f}x más rápido"
    
    if gpu_time > 0.0001 and cpp_time > 0.0001:
        speedup_gpu = cpp_time / gpu_time
        speedup_text += f"  |  CUDA: {speedup_gpu:.2f}x más rápido"
    
    if speedup_text:
        plt.figtext(0.5, 0.02, speedup_text, ha="center", fontsize=11, 
                    bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    # Etiquetas de datos sobre las barras
    for bar in bars:
        yval = bar.get_height()
        # Si el valor es muy pequeño, mostrar en formato científico
        label_text = f"{yval:.4f} ms" if yval > 0.001 else f"{yval:.2e} ms"
        ax.text(bar.get_x() + bar.get_width()/2, yval, 
                label_text, ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 6. Guardar y mostrar
    plt.tight_layout()
    output_img = 'benchmark_results_final.png'
    plt.savefig(output_img, dpi=300)
    print(f"\n✅ Gráfica generada con éxito: {output_img}")
    plt.show()

if __name__ == "__main__":
    run_benchmark()