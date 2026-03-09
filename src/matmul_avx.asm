section .text
global matmul_dot_avx

matmul_dot_avx:
    ; Convención de llamadas para WINDOWS x64:
    ; rcx = const float* A (Fila de la Matriz A)
    ; rdx = const float* B (Columna de la Matriz B)
    ; r8  = int len (Longitud)
    
    vzeroall                
    xor rax, rax            

.loop:
    ; Usamos rcx y rdx en lugar de rdi y rsi
    vmovups ymm1, [rcx + rax*4] 
    vmovups ymm2, [rdx + rax*4] 
    
    vfmadd231ps ymm0, ymm1, ymm2 
    
    add rax, 8              
    cmp rax, r8             ; Comparamos con r8 (len)
    jl .loop

    vextractf128 xmm1, ymm0, 1   
    vaddps xmm0, xmm0, xmm1      
    vhaddps xmm0, xmm0, xmm0     
    vhaddps xmm0, xmm0, xmm0
    
    ret