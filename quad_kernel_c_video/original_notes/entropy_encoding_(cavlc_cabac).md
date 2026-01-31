# Entropy Encoding (CAVLC/CABAC)

```markdown
# Entropy Encoding Kernel (Kernel C) - CAVLC/CABAC Implementation

## 1. Descripción Técnica Detallada

### 1.1 Visión General
Entropy Encoding es la etapa final en la codificación de video donde:
- **CAVLC** (Context-Adaptive Variable-Length Coding): Codificación basada en tablas VLC adaptativas
- **CABAC** (Context-Adaptive Binary Arithmetic Coding): Codificación aritmética binaria adaptativa

**Diferencias Clave**:
| Característica       | CAVLC              | CABAC               |
|----------------------|--------------------|---------------------|
| Complejidad          | Baja-Mediana       | Alta                |
| Eficiencia Compresión| ~15-20% menor      | Máxima             |
| Paralelización       | Fácil              | Limitada            |
| Uso típico           | Baseline Profile   | Main/High Profiles  |

### 1.2 Flujo de Datos en el Kernel
```
Raw Residuals → Quantization → Entropy Encoding (CAVLC/CABAC) → Bitstream Packaging
```

## 2. API/Interface en C Puro

### 2.1 Estructuras Principales
```c
typedef struct {
    int mode;               // CAVLC_MODE = 0, CABAC_MODE = 1
    int max_threads;        // Máximo de hilos permitidos
    bool low_latency_mode;  // Modo baja latencia
} EntropyConfig;

typedef struct {
    uint8_t* bitstream;     // Buffer de salida
    size_t bitstream_size;  // Tamaño actual
    size_t capacity;        // Capacidad total
} BitstreamBuffer;

// Contexto CABAC
typedef struct {
    uint32_t range;
    uint32_t low;
    uint8_t outstanding;
    const uint8_t* ctx_model;
} CABACState;
```

### 2.2 Funciones Principales
```c
// Inicialización
void entropy_init(EntropyConfig* config);

// Procesamiento principal
void entropy_encode_macroblock(
    MBData* macroblock, 
    BitstreamBuffer* bs, 
    CABACState* cabac_ctx
);

// Finalización
void entropy_flush(BitstreamBuffer* bs, CABACState* cabac_ctx);

// API Multi-hilo
int entropy_parallel_submit(TaskQueue* queue, FrameData* frame);
```

## 3. Algoritmos y Matemáticas

### 3.1 CAVLC: Codificación de Coeficientes
1. **Coeff Token**: 
   - `coeff_token = coeff_prefix + coeff_suffix`
   - Tablas VLC basadas en:
     - Número de coeficientes no cero (NC)
     - Trailing Ones (T1s) [1-3]

2. **Total Zeros**:
   - Codificación del total de ceros antes del último coeficiente
   - Tabla VLC dependiente del tamaño del bloque

### 3.2 CABAC: Modelo Probabilístico
**Ecuaciones Clave**:
```math
Interval = high - low
high = low + Interval * P(1)
low = low + Interval * (1 - P(0))
```

**Actualización de Probabilidad**:
```math
P_{new} = P_{old} + \alpha * (bit - P_{old})
```
Donde α = factor de adaptación (típicamente 0.05-0.1)

## 4. Implementación Paso a Paso

### 4.1 Pseudocódigo CABAC
```python
def cabac_encode_symbol(symbol, ctx_model):
    range = high - low
    p0 = probability_table[ctx_model]
    
    if symbol == 0:
        high = low + range * p0
    else:
        low = low + range * p0
        value = low
    
    # Normalización
    while high < 0x8000:
        write_bit((value >> 15) & 1)
        low <<= 1
        high <<= 1
        value <<= 1
```

### 4.2 Diagrama Flujo CABAC
```
      +---------------+
      | Binarización  | → Transformar a binario
      +-------+-------+
              |
      +-------v-------+
      | Modelo Contexto| → Seleccionar probabilidad
      +-------+-------+
              |
      +-------v-------+
      | Codificación  | → Actualizar rango
      | Aritmética    |
      +-------+-------+
              |
      +-------v-------+
      | Normalización | → Escribir bits
      +---------------+
```

## 5. Optimizaciones Hardware

### 5.1 Intel AVX-512
```c
// Procesamiento paralelo de 16 coeficientes
__m512i coeffs = _mm512_load_epi32(residuals);
__m512i signs = _mm512_srai_epi32(coeffs, 31);
__m512i abs_coeff = _mm512_abs_epi32(coeffs);
```

### 5.2 NVIDIA CUDA
```cuda
__global__ void cabac_kernel(uint8_t* bitstream, CABACState* states) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    CABACState local_state = states[tid];
    // Procesar macrobloque independiente
}
```

### 5.3 AMD ROCm
```hip
hipMallocManaged(&cabac_ctx, sizeof(CABACState)*num_blocks);
hipLaunchKernelGGL(cabac_encode, grids, blocks, 0, 0, ...);
```

## 6. Manejo de Memoria

### 6.1 Estrategias Clave
- **Memory Pools**: Reutilización de buffers de contexto
- **Zero-Copy**: Mapeo directo GPU-CPU para bitstreams
- **Prefetching**: Precarga de tablas VLC en caché L1

```c
// Alineación a 64 bytes para AVX
posix_memalign((void**)&cabac_table, 64, 1024*sizeof(CABACContext));
```

## 7. Benchmarks Esperados

### 7.1 Rendimiento 4K@120fps
| Algoritmo | CPU (Xeon 8380) | GPU (A100) | Latencia |
|-----------|-----------------|------------|----------|
| CAVLC     | 320 MB/s        | 2.1 GB/s   | 1.2 ms   |
| CABAC     | 180 MB/s        | 1.4 GB/s   | 2.8 ms   |

### 7.2 Consumo Recursos
| Recursos  | CAVLC  | CABAC  |
|-----------|--------|--------|
| CPU Usage | 15%    | 35%    |
| Memoria   | 80 MB  | 220 MB |
| BW PCIe   | 2 GB/s | 5 GB/s |

## 8. Casos de Uso

### 8.1 Live Streaming Ultra-Low Latency
```c
EntropyConfig config = {
    .mode = CAVLC_MODE,
    .max_threads = 32,
    .low_latency_mode = true
};
```

### 8.2 Video On Demand (Max Quality)
```c
EntropyConfig config = {
    .mode = CABAC_MODE,
    .max_threads = 16,
    .low_latency_mode = false
};
```

## 9. Integración con Otros Kernels

### 9.1 Flujo Completo Quad-Kernel
```
Kernel A (Capture) → Kernel B (DCT/Quant) → Kernel C (Entropy) → Kernel D (Network)
```

### 9.2 Interfaz de Transferencia
```c
void kernel_c_process(KernelBOutput* input, KernelDInput* output) {
    entropy_encode_macroblock(input->residuals, &output->bitstream);
}
```

## 10. Bottlenecks y Soluciones

### 10.1 Problemas Identificados
1. **Serialización CABAC**: Dependencias entre símbolos
2. **Branch Misses**: Decisiones probabilísticas
3. **Memory BW**: Acceso a tablas de contexto

### 10.2 Soluciones Implementadas
- **Parallel Slice Encoding**: División de frame en slices independientes
- **Branchless Code**:
  ```c
  int select = (bit >> 31) & 1;
  low = select ? low : new_low;
  high = select ? new_high : high;
  ```
- **Huffman Table Prefetch**:
  ```c
  __builtin_prefetch(&vlc_table[coeff_token]);
  ```

---

**Diagrama Final del Sistema**:
```
+----------------+     +----------------+     +----------------+     +----------------+
|  Kernel A      |     |  Kernel B      |     |  Kernel C      |     |  Kernel D      |
|  Video Capture |---->|  Transform/    |---->|  Entropy       |---->|  Network       |
|  4K@120fps     |     |  Quantization  |     |  Encoding      |     |  Streaming     |
+----------------+     +----------------+     +----------------+     +----------------+
```
```