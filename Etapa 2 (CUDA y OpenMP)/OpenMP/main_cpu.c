#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define INPUT_DIM 4096   
#define HIDDEN_DIM 256   
#define OUTPUT_DIM 10    
#define LEARNING_RATE 0.01f

#define TRAIN_SAMPLES 350
#define VALID_SAMPLES 75
#define TEST_SAMPLES 75

// Declaración de funciones externas (de dataset.c)
void load_dataset(const char* data_file, const char* labels_file, float** data, int** labels, int samples);

typedef struct {
    float* w1; 
    float* w2; 
} NeuralNetworkCPU;

// Prototipos de funciones para evitar errores de compilador
void initialize_network_cpu(NeuralNetworkCPU* net);
void train_network_cpu(NeuralNetworkCPU* net, float* train_data, int* train_labels, int samples, int epochs);
float evaluate_network_cpu(NeuralNetworkCPU* net, float* data, int* labels, int samples);
void free_network_cpu(NeuralNetworkCPU* net);
void save_network(NeuralNetworkCPU* net, const char* filename);
void save_predictions_csv(NeuralNetworkCPU* net, float* test_data, int* test_labels, int samples, const char* filename);

// ==========================================
// DEFINICIONES DE FUNCIONES
// ==========================================

void initialize_network_cpu(NeuralNetworkCPU* net) {
    int w1_size = INPUT_DIM * HIDDEN_DIM;
    int w2_size = HIDDEN_DIM * OUTPUT_DIM;

    net->w1 = (float*)malloc(w1_size * sizeof(float));
    net->w2 = (float*)malloc(w2_size * sizeof(float));

    for (int i = 0; i < w1_size; i++) {
        float val = (float)(i % 101) / 100.0f;
        net->w1[i] = (val - 0.5f) * 0.1f;
    }
    for (int i = 0; i < w2_size; i++) {
        float val = (float)(i % 101) / 100.0f;
        net->w2[i] = (val - 0.5f) * 0.1f;
    }
}

void train_network_cpu(NeuralNetworkCPU* net, float* train_data, int* train_labels, int samples, int epochs) {
    printf("Iniciando entrenamiento en CPU (OpenMP) por %d epocas...\n", epochs);

    float* hidden_layer = (float*)malloc(samples * HIDDEN_DIM * sizeof(float));
    float* output_layer = (float*)malloc(samples * OUTPUT_DIM * sizeof(float));
    float* grad_w1 = (float*)malloc(INPUT_DIM * HIDDEN_DIM * sizeof(float));
    float* grad_w2 = (float*)malloc(HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    float* hidden_error = (float*)malloc(samples * HIDDEN_DIM * sizeof(float));
    float* losses = (float*)malloc(samples * sizeof(float));

    for (int epoch = 0; epoch < epochs; epoch++) {
        // 1. FORWARD PASS
        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < samples; r++) {
            for (int c = 0; c < HIDDEN_DIM; c++) {
                float sum = 0.0f;
                for (int i = 0; i < INPUT_DIM; i++) {
                    sum += train_data[r * INPUT_DIM + i] * net->w1[i * HIDDEN_DIM + c];
                }
                hidden_layer[r * HIDDEN_DIM + c] = sum;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < samples * HIDDEN_DIM; i++) {
            hidden_layer[i] = hidden_layer[i] > 0.0f ? hidden_layer[i] : 0.0f;
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < samples; r++) {
            for (int c = 0; c < OUTPUT_DIM; c++) {
                float sum = 0.0f;
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    sum += hidden_layer[r * HIDDEN_DIM + i] * net->w2[i * OUTPUT_DIM + c];
                }
                output_layer[r * OUTPUT_DIM + c] = sum;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int r = 0; r < samples; r++) {
            int row_offset = r * OUTPUT_DIM;
            float max_val = output_layer[row_offset];
            for (int c = 1; c < OUTPUT_DIM; c++) {
                if (output_layer[row_offset + c] > max_val) max_val = output_layer[row_offset + c];
            }
            float sum = 0.0f;
            for (int c = 0; c < OUTPUT_DIM; c++) {
                sum += expf(output_layer[row_offset + c] - max_val);
            }
            for (int c = 0; c < OUTPUT_DIM; c++) {
                output_layer[row_offset + c] = expf(output_layer[row_offset + c] - max_val) / sum;
            }
        }

        // Cálculo paralelo de LOSS y ACCURACY
        float total_loss = 0.0f;
        int correct_count = 0;

        #pragma omp parallel for reduction(+:total_loss, correct_count) schedule(static)
        for (int i = 0; i < samples; i++) {
            int label = train_labels[i];
            float prob = output_layer[i * OUTPUT_DIM + label];
            if (prob < 1e-7f) prob = 1e-7f;
            
            losses[i] = -logf(prob);
            total_loss += losses[i];

            int row_offset = i * OUTPUT_DIM;
            float max_prob = output_layer[row_offset];
            int predicted_label = 0;
            for (int c = 1; c < OUTPUT_DIM; c++) {
                if (output_layer[row_offset + c] > max_prob) {
                    max_prob = output_layer[row_offset + c];
                    predicted_label = c;
                }
            }
            if (predicted_label == label) {
                correct_count++;
            }
        }

        // Imprimir métricas
        float avg_loss = total_loss / samples;
        float train_acc = ((float)correct_count / samples) * 100.0f;
        printf("  [CPU] Epoca %d/%d -> Loss: %.4f | Accuracy: %.2f%%\n", epoch + 1, epochs, avg_loss, train_acc);

        // 2. BACKWARD PASS
        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < samples; r++) {
            for (int c = 0; c < OUTPUT_DIM; c++) {
                float y_true = (train_labels[r] == c) ? 1.0f : 0.0f;
                output_layer[r * OUTPUT_DIM + c] -= y_true; 
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < HIDDEN_DIM; r++) {
            for (int c = 0; c < OUTPUT_DIM; c++) {
                float sum = 0.0f;
                for (int i = 0; i < samples; i++) {
                    sum += hidden_layer[i * HIDDEN_DIM + r] * output_layer[i * OUTPUT_DIM + c];
                }
                grad_w2[r * OUTPUT_DIM + c] = sum / samples;
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < samples; r++) {
            for (int c = 0; c < HIDDEN_DIM; c++) {
                float sum = 0.0f;
                for (int i = 0; i < OUTPUT_DIM; i++) {
                    sum += output_layer[r * OUTPUT_DIM + i] * net->w2[c * OUTPUT_DIM + i];
                }
                hidden_error[r * HIDDEN_DIM + c] = sum;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < samples * HIDDEN_DIM; i++) {
            if (hidden_layer[i] <= 0.0f) {
                hidden_error[i] = 0.0f;
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int r = 0; r < INPUT_DIM; r++) {
            for (int c = 0; c < HIDDEN_DIM; c++) {
                float sum = 0.0f;
                for (int i = 0; i < samples; i++) {
                    sum += train_data[i * INPUT_DIM + r] * hidden_error[i * HIDDEN_DIM + c];
                }
                grad_w1[r * HIDDEN_DIM + c] = sum / samples;
            }
        }

        // 3. ACTUALIZAR PESOS
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < INPUT_DIM * HIDDEN_DIM; i++) {
            net->w1[i] -= LEARNING_RATE * grad_w1[i];
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < HIDDEN_DIM * OUTPUT_DIM; i++) {
            net->w2[i] -= LEARNING_RATE * grad_w2[i];
        }
    }

    free(hidden_layer); free(output_layer);
    free(grad_w1); free(grad_w2);
    free(hidden_error); free(losses);
}

float evaluate_network_cpu(NeuralNetworkCPU* net, float* data, int* labels, int samples) {
    float* hidden_layer = (float*)malloc(samples * HIDDEN_DIM * sizeof(float));
    float* output_layer = (float*)malloc(samples * OUTPUT_DIM * sizeof(float));
    int correct_predictions = 0;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < samples; r++) {
        for (int c = 0; c < HIDDEN_DIM; c++) {
            float sum = 0.0f;
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += data[r * INPUT_DIM + i] * net->w1[i * HIDDEN_DIM + c];
            }
            hidden_layer[r * HIDDEN_DIM + c] = sum > 0.0f ? sum : 0.0f; 
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < samples; r++) {
        for (int c = 0; c < OUTPUT_DIM; c++) {
            float sum = 0.0f;
            for (int i = 0; i < HIDDEN_DIM; i++) {
                sum += hidden_layer[r * HIDDEN_DIM + i] * net->w2[i * OUTPUT_DIM + c];
            }
            output_layer[r * OUTPUT_DIM + c] = sum;
        }
    }

    for (int r = 0; r < samples; r++) {
        int row_offset = r * OUTPUT_DIM;
        float max_val = output_layer[row_offset];
        int predicted_label = 0;

        for (int c = 1; c < OUTPUT_DIM; c++) {
            if (output_layer[row_offset + c] > max_val) {
                max_val = output_layer[row_offset + c];
                predicted_label = c;
            }
        }
        if (predicted_label == labels[r]) {
            correct_predictions++;
        }
    }

    free(hidden_layer); free(output_layer);
    return ((float)correct_predictions / samples) * 100.0f;
}

// -----------------------------------------------------------
// NUEVA FUNCIÓN: Guarda las predicciones de prueba en un CSV
// -----------------------------------------------------------
void save_predictions_csv(NeuralNetworkCPU* net, float* test_data, int* test_labels, int samples, const char* filename) {
    float* hidden_layer = (float*)malloc(samples * HIDDEN_DIM * sizeof(float));
    float* output_layer = (float*)malloc(samples * OUTPUT_DIM * sizeof(float));

    // Forward pass para calcular resultados
    #pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < samples; r++) {
        for (int c = 0; c < HIDDEN_DIM; c++) {
            float sum = 0.0f;
            for (int i = 0; i < INPUT_DIM; i++) {
                sum += test_data[r * INPUT_DIM + i] * net->w1[i * HIDDEN_DIM + c];
            }
            hidden_layer[r * HIDDEN_DIM + c] = sum > 0.0f ? sum : 0.0f; 
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int r = 0; r < samples; r++) {
        for (int c = 0; c < OUTPUT_DIM; c++) {
            float sum = 0.0f;
            for (int i = 0; i < HIDDEN_DIM; i++) {
                sum += hidden_layer[r * HIDDEN_DIM + i] * net->w2[i * OUTPUT_DIM + c];
            }
            output_layer[r * OUTPUT_DIM + c] = sum;
        }
    }

    // Escribir a archivo (Secuencial, no OpenMP)
    FILE* f = fopen(filename, "w");
    if (f) {
        fprintf(f, "TrueLabel,PredictedLabel\n"); // Cabecera del CSV
        for (int r = 0; r < samples; r++) {
            int row_offset = r * OUTPUT_DIM;
            float max_val = output_layer[row_offset];
            int predicted_label = 0;

            for (int c = 1; c < OUTPUT_DIM; c++) {
                if (output_layer[row_offset + c] > max_val) {
                    max_val = output_layer[row_offset + c];
                    predicted_label = c;
                }
            }
            fprintf(f, "%d,%d\n", test_labels[r], predicted_label);
        }
        fclose(f);
        printf("[INFO] Predicciones guardadas en '%s'.\n", filename);
    } else {
        printf("[ERROR] No se pudo crear el archivo %s\n", filename);
    }

    free(hidden_layer); free(output_layer);
}

void save_network(NeuralNetworkCPU* net, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(net->w1, sizeof(float), INPUT_DIM * HIDDEN_DIM, f);
        fwrite(net->w2, sizeof(float), HIDDEN_DIM * OUTPUT_DIM, f);
        fclose(f);
        printf("[INFO] Modelo guardado en '%s' con exito.\n", filename);
    }
}

void free_network_cpu(NeuralNetworkCPU* net) {
    if (net->w1) free(net->w1);
    if (net->w2) free(net->w2);
}

// ==========================================
// FUNCIÓN PRINCIPAL
// ==========================================
int main() {
    float* train_data; int* train_labels;
    float* valid_data; int* valid_labels;
    float* test_data;  int* test_labels;

    printf("Cargando dataset...\n");
    load_dataset("train.bin", "train_labels.bin", &train_data, &train_labels, TRAIN_SAMPLES);
    load_dataset("valid.bin", "valid_labels.bin", &valid_data, &valid_labels, VALID_SAMPLES);
    load_dataset("test.bin", "test_labels.bin", &test_data, &test_labels, TEST_SAMPLES);

    NeuralNetworkCPU net;
    initialize_network_cpu(&net);

    double start_time = omp_get_wtime();
    train_network_cpu(&net, train_data, train_labels, TRAIN_SAMPLES, 500);
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;

    float train_final_acc = evaluate_network_cpu(&net, train_data, train_labels, TRAIN_SAMPLES);
    float valid_acc = evaluate_network_cpu(&net, valid_data, valid_labels, VALID_SAMPLES);
    float test_acc = evaluate_network_cpu(&net, test_data, test_labels, TEST_SAMPLES);

    printf("\n--- Resultados Finales ---\n");
    printf("Training Accuracy:   %.2f%%\n", train_final_acc);
    printf("Validation Accuracy: %.2f%%\n", valid_acc);
    printf("Test Accuracy:       %.2f%%\n", test_acc);
    printf("Tiempo Total:        %.4f segundos\n\n", total_time);

    // Guardar pesos y exportar CSV de predicciones
    save_network(&net, "modelo_entrenado.bin"); 
    save_predictions_csv(&net, test_data, test_labels, TEST_SAMPLES, "predicciones_test.csv");

    free_network_cpu(&net);
    free(train_data); free(train_labels);
    free(valid_data); free(valid_labels);
    free(test_data);  free(test_labels);

    return 0;
}