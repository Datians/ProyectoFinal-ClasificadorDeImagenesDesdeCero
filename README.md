# ClasificadorDeImagenesDesdeCero (Proyecto Mascarilla)

> **Asignatura:** Fundamentos de Programación Concurrente y Distribuida  
> **Docente:** Prf. Alejandro Jaimes  
> **Fecha:** 26/06/2026  
> **Repositorio:** [https://github.com/Datians/ProyectoFinal-ClasificadorDeImagenesDesdeCero]()

---

## Equipo

| | Colaborador | GitHub |
|---|---|---|
| 👤 | Juan David Miranda Pelaez | [Gal4h4d](https://github.com/Gal4h4d) |
| 👤 | Cristhian David Parra Parada | [CristhianParada](https://github.com/CristhianParada) |
| 👤 | Andres Cuadrado | [Datians](https://github.com/Datians) |
| 👤 | Julio Martínez Triana | [Julsdev](https://github.com/JulssDev) |

## Proyecto y dataset

## Arquitectura de la red
Se implementó una red neuronal completamente conectada (Fully Connected Neural Network) utilizando CUDA para acelerar el entrenamiento y la inferencia sobre GPU.

La arquitectura está compuesta por tres capas principales:

```
Entrada (64x64 = 4096)
        │
        ▼
Capa Oculta (256 neuronas)
        │
      ReLU
        │
        ▼
Capa de Salida (10 neuronas)
        │
     Softmax
        │
        ▼
Predicción Final
```

La red recibe imágenes de **64 × 64 píxeles**, las cuales son transformadas en un vector de **4096 características** antes de ser procesadas.

Durante el entrenamiento se emplea:

- Función de activación ReLU en la capa oculta.
- Función Softmax en la capa de salida.
- Descenso por gradiente (Gradient Descent).
- Función de pérdida Cross-Entropy.
- Implementación paralela mediante CUDA.

---

## 2. Capas, tamaños, activaciones y número total de parámetros entrenables

### Configuración del conjunto de datos

El modelo fue entrenado utilizando un conjunto de **700 imágenes** distribuidas de la siguiente forma:

| Conjunto | Cantidad | Porcentaje |
|----------|---------:|-----------:|
| Entrenamiento | 490 | 70 % |
| Validación | 105 | 15 % |
| Prueba | 105 | 15 % |

Todas las imágenes poseen una resolución de:

```
64 × 64 píxeles
```

Por lo tanto, cada imagen produce un vector de entrada de:

```
64 × 64 = 4096 características
```

---

### Arquitectura del modelo

| Capa | Tamaño | Activación |
|------|--------|------------|
| Entrada | 4096 neuronas | — |
| Oculta | 256 neuronas | ReLU |
| Salida | 2 neuronas | Softmax |

---

### Parámetros entrenables

#### Capa Entrada → Oculta

Número de pesos:

```
4096 × 256 = 1 048 576
```

#### Capa Oculta → Salida

Número de pesos:

```
256 × 10 = 2 560
```

---

### Total de parámetros entrenables

| Conexión | Parámetros |
|----------|-----------:|
| Entrada → Oculta | 1 048 576 |
| Oculta → Salida | 2 560 |
| **Total** | **1 051 136** |

> **Nota:** En esta implementación únicamente se entrenan las matrices de pesos (`W1` y `W2`). No se utilizan vectores de sesgo (bias), por lo que estos no forman parte de los parámetros entrenables.

---

## Metricas finales
Las métricas fueron calculadas utilizando el conjunto de prueba compuesto por **105 imágenes**, el cual no participó durante el entrenamiento del modelo.

Las métricas de evaluación consideradas son:

- Exactitud (Accuracy)
- Precisión (Precision)
- Recall
- F1-Score
- Matriz de Confusión

---

### Exactitud (Accuracy) y Loss (Perdida)

![AccuracyGraphs](Evidencias/graficaRendimientoCUDA.png)

---

### Precisión (Precision)

![Accuracy](Evidencias/AccuracyCUDA.png)

---

### Recall

![Recall](Evidencias/recallCUDA.png)

---

### F1-Score y otras metricas

![F1Score](Evidencias/f1ScoreCUDA.png)

---

### Matriz de Confusión

![Confusion](Evidencias/matrizConfusionFinalCUDA.png)

---

## Resumen del Modelo

| Característica | Valor |
|---------------|-------|
| Tipo de red | Red neuronal completamente conectada (MLP) |
| Framework | CUDA C/C++ |
| Resolución de entrada | 64 × 64 |
| Tamaño de entrada | 4096 |
| Neuronas ocultas | 256 |
| Clases de salida | 10 |
| Activación oculta | ReLU |
| Activación salida | Softmax |
| Función de pérdida | Cross-Entropy |
| Tasa de aprendizaje | 0.01 |
| Épocas de entrenamiento | 500 |
| Dataset total | 700 imágenes |
| Entrenamiento | 490 imágenes |
| Validación | 105 imágenes |
| Prueba | 105 imágenes |
| Parámetros entrenables | 1 051 136 |

---

## Implementación

La implementación fue desarrollada en CUDA, aprovechando el procesamiento paralelo de la GPU para acelerar las operaciones de propagación hacia adelante (Forward Propagation), retropropagación (Backpropagation) y actualización de pesos.

Entre los kernels implementados se encuentran:

- Multiplicación de matrices.
- Función ReLU.
- Función Softmax.
- Cálculo de la pérdida Cross-Entropy.
- Retropropagación del error.
- Cálculo del gradiente.
- Actualización de pesos.

Esta estrategia permite reducir significativamente el tiempo de entrenamiento en comparación con una implementación secuencial ejecutada únicamente sobre CPU.

## Curvas de entrenamiento

## Rendimiento paralelo
### Tabla de tiempos: Secuencial vs. OpenMP (Etapa 1)

| Hilos | Tiempo (s) | Speedup | X veces más rápido |
| --- | --- | --- | --- |
| 1 | 16,861 | 1 | 1.00× |
| 2 | 7,43 | 2,26931 | 2.27× |
| 4 | 7,381 | 2,28438 | 2.28× |
| 6 | 7,237 | 2,32983 | 2.33× |
| 8 | 6,633 | 2,54199 | 2.54× |

### CPU vs GPU (Etapa 2) y Speedup



## Evidencias
### Tiempos Secuencial vs Paralelo y Speedup

### Tiempos
![Budget creado en Billing](Evidencias/Tiempos.png)

### Tabla Speedup 
![Budget creado en Billing](Evidencias/Speedup.png)

### Grafica
![Budget creado en Billing](Evidencias/Grafico_Speedup.png)

## Conclusiones

### Preguntas de reflexion (Etapa 1)

### 1. ¿Por qué este preprocesamiento es “vergonzosamente paralelo”? Den una analogía.

Este preprocesamiento es vergonzosamente paralelo porque cada imagen del dataset puede procesarse de forma completamente independiente. En el código, OpenMP distribuye las imágenes entre varios hilos mediante:

```C
#pragma omp parallel for
for (int i = 0; i < total_imagenes; i++) {
    process_image(rutas[i], dataset_features[i]);
}
```

Cada hilo ejecuta todo el pipeline (escala de grises, resize, filtro Gaussiano, Sobel, downsampling y normalización) sobre una imagen diferente, sin necesidad de intercambiar información con los demás hilos. Esto minimiza la sincronización y permite aprovechar varios núcleos del procesador.

Analogía:
Es como tener una pila de 8.000 fotografías para editar y repartirlas entre varios estudiantes. Cada estudiante recibe un conjunto de fotos y les aplica exactamente los mismos filtros. Ninguno necesita esperar a los demás ni compartir resultados durante el trabajo, por lo que todos pueden trabajar simultáneamente.


### 2. Si tienen 8 hilos pero el speedup se queda en 2.54×, ¿qué lo limita?


Aunque el procesamiento de imágenes es altamente paralelizable, los resultados muestran que al aumentar los hilos más allá de 2, la mejora es pequeña. Esto indica que el programa está limitado por factores distintos al número de hilos.

Según la Ley de Amdahl, la aceleración total está limitada por la fracción secuencial del programa. Además, en este caso existen otros cuellos de botella importantes:

Lectura de imágenes desde el disco.
Acceso simultáneo a archivos por varios hilos.
Gestión y sincronización de hilos OpenMP.
Limitaciones del ancho de banda de memoria.
Sobrecarga asociada a cargar cada imagen con stbi_load().

Los resultados muestran que pasar de 2 a 4 hilos apenas reduce el tiempo (7.43 s → 7.38 s), lo que sugiere que el sistema deja de estar limitado por la CPU y comienza a estar limitado principalmente por la entrada/salida (I/O) y el acceso a memoria.

Por ello, aunque se utilicen 8 hilos, el speedup obtenido es de aproximadamente 2.54×, muy inferior al speedup ideal de 8×.

### 3. Hagan un diagrama (Excalidraw) del flujo de una imagen desde foto cruda hasta vector de 4.096.

![Budget creado en Billing](Evidencias/Flujo.png)