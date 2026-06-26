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

## Metricas finales

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