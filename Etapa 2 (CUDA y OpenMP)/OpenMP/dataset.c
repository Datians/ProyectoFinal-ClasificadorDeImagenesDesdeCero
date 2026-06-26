#include <stdio.h>
#include <stdlib.h>


void load_dataset(const char* data_file, const char* labels_file, float** data, int** labels, int samples) {
    int pixels_per_sample = 4096; 

    *data = (float*)malloc(samples * pixels_per_sample * sizeof(float));
    *labels = (int*)malloc(samples * sizeof(int));

   
    if (*data == NULL || *labels == NULL) {
        printf("[ERROR] No se pudo asignar memoria RAM para el dataset.\n");
        exit(1);
    }

    // 2. Leer el archivo binario que contiene los datos de las imágenes (features)
    FILE* f_data = fopen(data_file, "rb");
    if (!f_data) {
        printf("[ERROR] No se pudo abrir el archivo de datos: %s\n", data_file);
        printf("Asegurate de que el archivo binario este en la misma carpeta que el ejecutable.\n");
        exit(1);
    }
    
    size_t data_read = fread(*data, sizeof(float), samples * pixels_per_sample, f_data);
    if (data_read != (size_t)(samples * pixels_per_sample)) {
        printf("[ADVERTENCIA] Se esperaban leer %d floats, pero se leyeron %zu desde %s\n", 
               samples * pixels_per_sample, data_read, data_file);
    }
    fclose(f_data);

    // 3. Leer el archivo binario que contiene las etiquetas (labels)
    FILE* f_labels = fopen(labels_file, "rb");
    if (!f_labels) {
        printf("[ERROR] No se pudo abrir el archivo de etiquetas: %s\n", labels_file);
        free(*data); 
        exit(1);
    }

    size_t labels_read = fread(*labels, sizeof(int), samples, f_labels);
    if (labels_read != (size_t)samples) {
        printf("[ADVERTENCIA] Se esperaban leer %d etiquetas, pero se leyeron %zu desde %s\n", 
               samples, labels_read, labels_file);
    }
    fclose(f_labels);

    printf("[DATASET] Archivos '%s' y '%s' cargados con exito \n", data_file, labels_file, samples);
}