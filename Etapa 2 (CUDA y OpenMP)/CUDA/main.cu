#include <stdio.h>
#include <stdlib.h>

#include "dataset.cuh"
#include "neural_network.cuh"

#define TRAIN_SAMPLES 350
#define VALID_SAMPLES 75
#define TEST_SAMPLES 75

int main(int argc, char* argv[])
{
    float* train_data;
    int* train_labels;

    float* valid_data;
    int* valid_labels;

    float* test_data;
    int* test_labels;

    load_dataset(
        "train.bin",
        "train_labels.bin",
        &train_data,
        &train_labels,
        TRAIN_SAMPLES);

    load_dataset(
        "valid.bin",
        "valid_labels.bin",
        &valid_data,
        &valid_labels,
        VALID_SAMPLES);

    load_dataset(
        "test.bin",
        "test_labels.bin",
        &test_data,
        &test_labels,
        TEST_SAMPLES);

    NeuralNetwork net;
    
    // Inicializamos punteros en NULL para controlar asignación interna
    net.d_w1 = NULL;
    net.d_w2 = NULL;

    // Si Streamlit nos pasa un argumento, hacemos la predicción individual sin entrenar
    if (argc > 1) {
        char* custom_image_path = argv[1];
        printf("\n[STREAMLIT] Modo Inferencia Activo.\n");
        
        // Intentamos cargar el modelo guardado previamente
        if (load_network(&net, "modelo_red.bin")) {
            printf("[STREAMLIT] Cargando imagen personalizada desde: %s\n", custom_image_path);
            
            FILE* f_custom = fopen(custom_image_path, "rb");
            if (f_custom) {
                float* custom_data = (float*)malloc(4096 * sizeof(float));
                fread(custom_data, sizeof(float), 4096, f_custom);
                fclose(f_custom);
                
                int fake_label = 0; 
                // Evaluamos la muestra individual para imprimir los resultados en la interfaz
                evaluate_network(&net, custom_data, &fake_label, 1);
                
                free(custom_data);
            } else {
                printf("[ERROR] No se pudo abrir la imagen de Streamlit.\n");
            }
        } else {
            printf("[ERROR] No se puede inferir. Primero debes entrenar el modelo ejecutando el programa sin argumentos.\n");
        }
    } else {
        // Ejecución normal por consola: Entrenamos desde cero y exportamos el modelo final
        initialize_network(&net);

        // Entrenamos el modelo por 500 épocas en la GPU
        train_network(&net, train_data, train_labels, TRAIN_SAMPLES, 500);

        // Guardamos el modelo recién entrenado para usos futuros
        save_network(&net, "modelo_red.bin");

        float valid_acc = evaluate_network(&net, valid_data, valid_labels, VALID_SAMPLES);
        float test_acc = evaluate_network(&net, test_data, test_labels, TEST_SAMPLES);
        printf("\nValidation Accuracy: %.2f%%\n", valid_acc);
        printf("Test Accuracy: %.2f%%\n", test_acc);
    }

    free_network(&net);

    free(train_data);
    free(train_labels);
    free(valid_data);
    free(valid_labels);
    free(test_data);
    free(test_labels);

    return 0;
}