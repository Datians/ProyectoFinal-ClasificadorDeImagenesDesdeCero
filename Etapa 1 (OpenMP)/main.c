#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <omp.h>
#include "preprocess.h"

#define MAX_IMAGES 8000
#define FEATURES_SIZE 4096

// Arrays globales para no desbordar la pila
char rutas[MAX_IMAGES][256];
int etiquetas[MAX_IMAGES];
float dataset_features[MAX_IMAGES][FEATURES_SIZE];
int total_imagenes = 0;

void cargar_rutas(const char* carpeta, int etiqueta) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(carpeta)) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (strstr(ent->d_name, ".jpg") || strstr(ent->d_name, ".png")) {
                sprintf(rutas[total_imagenes], "%s/%s", carpeta, ent->d_name);
                etiquetas[total_imagenes] = etiqueta;
                total_imagenes++;
            }
        }
        closedir(dir);
    }
}

void guardar_binario(const char* name_feat, const char* name_label, int* indices, int inicio, int fin) {
    FILE* f_feat = fopen(name_feat, "wb");
    FILE* f_label = fopen(name_label, "wb");
    for (int i = inicio; i < fin; i++) {
        int idx = indices[i];
        fwrite(dataset_features[idx], sizeof(float), FEATURES_SIZE, f_feat);
        fwrite(&etiquetas[idx], sizeof(int), 1, f_label);
    }
    fclose(f_feat);
    fclose(f_label);
}

// Función simple para mezclar arreglos (Shuffle)
void shuffle(int *array, int n) {
    for (int i = 0; i < n - 1; i++) {
        int j = i + rand() / (RAND_MAX / (n - i) + 1);
        int t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

int main() {
    printf("1. Buscando imagenes...\n");
    cargar_rutas("./dataset/with_mask", 0);
    cargar_rutas("./dataset/without_mask", 1);
    printf("Total imagenes: %d\n", total_imagenes);

    printf("2. Procesando imagenes con OpenMP...\n");
    double inicio_tiempo = omp_get_wtime();

    // La magia de OpenMP
    #pragma omp parallel for
    for (int i = 0; i < total_imagenes; i++) {
        process_image(rutas[i], dataset_features[i]);
    }

    double fin_tiempo = omp_get_wtime();
    printf("Tiempo paralelo: %f segundos\n", fin_tiempo - inicio_tiempo);

    printf("3. Dividiendo y guardando (.bin)...\n");
    int* indices = (int*)malloc(total_imagenes * sizeof(int));
    for(int i=0; i<total_imagenes; i++) indices[i] = i;
    shuffle(indices, total_imagenes);

    int train_end = (int)(total_imagenes * 0.70);
    int valid_end = train_end + (int)(total_imagenes * 0.15);

    guardar_binario("train.bin", "train_labels.bin", indices, 0, train_end);
    guardar_binario("valid.bin", "valid_labels.bin", indices, train_end, valid_end);
    guardar_binario("test.bin", "test_labels.bin", indices, valid_end, total_imagenes);

    free(indices);
    printf("¡Completado! Archivos listos para CUDA.\n");
    return 0;
}