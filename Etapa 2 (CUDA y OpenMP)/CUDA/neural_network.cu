#include "neural_network.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Definiciones de la arquitectura de la red
#define INPUT_DIM 4096   // Tamaño de la entrada
#define HIDDEN_DIM 256   // Neuronas en la capa oculta
#define OUTPUT_DIM 10    // Clases de salida
#define LEARNING_RATE 0.01f

// ============================================================================
// 1. KERNELS DE CUDA (Funciones que corren en la GPU)
// ============================================================================

__global__ void compute_loss_kernel(const float* output, const int* labels, float* losses, int samples, int output_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < samples) {
        int label = labels[idx];
        float prob = output[idx * output_dim + label];
        if (prob < 1e-7f) prob = 1e-7f; 
        losses[idx] = -logf(prob);
    }
}

__global__ void init_weights_kernel(float* weights, int size, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = (float)(idx % 101) / 100.0f; 
        weights[idx] = (val - 0.5f) * 0.1f;
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_transA_kernel(const float* H, const float* ErrorOut, float* GradW2, int Hidden, int Output, int Samples) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < Hidden && col < Output) {
        float sum = 0.0f;
        for (int i = 0; i < Samples; i++) {
            sum += H[i * Hidden + row] * ErrorOut[i * Output + col];
        }
        GradW2[row * Output + col] = sum / Samples; 
    }
}

__global__ void matmul_transB_kernel(const float* ErrorOut, const float* W2, float* ErrorHidden, int Samples, int Hidden, int Output) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < Samples && col < Hidden) {
        float sum = 0.0f;
        for (int i = 0; i < Output; i++) {
            sum += ErrorOut[row * Output + i] * W2[col * Output + i];
        }
        ErrorHidden[row * Hidden + col] = sum;
    }
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
    }
}

__global__ void backprop_relu_kernel(float* d_hidden_error, const float* hidden_layer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (hidden_layer[idx] <= 0.0f) {
            d_hidden_error[idx] = 0.0f; 
        }
    }
}

__global__ void softmax_kernel(float* output, int samples, int output_dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < samples) {
        int row_offset = row * output_dim;
        float max_val = output[row_offset];
        for (int c = 1; c < output_dim; c++) {
            if (output[row_offset + c] > max_val) max_val = output[row_offset + c];
        }
        float sum = 0.0f;
        for (int c = 0; c < output_dim; c++) {
            sum += expf(output[row_offset + c] - max_val);
        }
        for (int c = 0; c < output_dim; c++) {
            output[row_offset + c] = expf(output[row_offset + c] - max_val) / sum;
        }
    }
}

__global__ void compute_output_error_kernel(float* output, const int* labels, int samples, int output_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < samples && col < output_dim) {
        int idx = row * output_dim + col;
        float y_true = (labels[row] == col) ? 1.0f : 0.0f;
        output[idx] = output[idx] - y_true; 
    }
}

__global__ void update_weights_kernel(float* weights, const float* gradient, int size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradient[idx];
    }
}

// ============================================================================
// 2. IMPLEMENTACIÓN DE LAS FUNCIONES DEL HOST (CPU)
// ============================================================================

void initialize_network(NeuralNetwork* net) {
    printf("Inicializando red neuronal en la GPU...\n");
    int w1_size = INPUT_DIM * HIDDEN_DIM;
    int w2_size = HIDDEN_DIM * OUTPUT_DIM;

    cudaMalloc((void**)&(net->d_w1), w1_size * sizeof(float));
    cudaMalloc((void**)&(net->d_w2), w2_size * sizeof(float));

    int blockSize = 256;
    init_weights_kernel<<<(w1_size + blockSize - 1) / blockSize, blockSize>>>(net->d_w1, w1_size, 1234ULL);
    init_weights_kernel<<<(w2_size + blockSize - 1) / blockSize, blockSize>>>(net->d_w2, w2_size, 5678ULL);
    cudaDeviceSynchronize();
}

void train_network(NeuralNetwork* net, float* train_data, int* train_labels, int samples, int epochs) {
    printf("Iniciando entrenamiento completo con exportación de métricas...\n");

    float* d_batch_data; int* d_batch_labels;
    cudaMalloc((void**)&d_batch_data, samples * INPUT_DIM * sizeof(float));
    cudaMalloc((void**)&d_batch_labels, samples * sizeof(int));
    cudaMemcpy(d_batch_data, train_data, samples * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_labels, train_labels, samples * sizeof(int), cudaMemcpyHostToDevice);

    float* d_hidden_layer; float* d_output_layer;
    cudaMalloc((void**)&d_hidden_layer, samples * HIDDEN_DIM * sizeof(float));
    cudaMalloc((void**)&d_output_layer, samples * OUTPUT_DIM * sizeof(float));

    float* d_grad_w1; float* d_grad_w2;
    cudaMalloc((void**)&d_grad_w1, INPUT_DIM * HIDDEN_DIM * sizeof(float));
    cudaMalloc((void**)&d_grad_w2, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    
    float* d_hidden_error;
    cudaMalloc((void**)&d_hidden_error, samples * HIDDEN_DIM * sizeof(float));

    float* h_output = (float*)malloc(samples * OUTPUT_DIM * sizeof(float));

    FILE* f_historial = fopen("historial_entrenamiento.txt", "w");
    if (f_historial == NULL) { printf("Error abriendo historial_entrenamiento.txt\n"); return; }
    fprintf(f_historial, "epoch,loss,accuracy\n");

    dim3 threadsPerBlock(16, 16);
    dim3 blocksHidden((HIDDEN_DIM + 15) / 16, (samples + 15) / 16);
    dim3 blocksOutput((OUTPUT_DIM + 15) / 16, (samples + 15) / 16);
    dim3 blocksW1Grad((HIDDEN_DIM + 15) / 16, (INPUT_DIM + 15) / 16);

    int blockSizeLinear = 256;
    int gridHiddenLinear = (samples * HIDDEN_DIM + blockSizeLinear - 1) / blockSizeLinear;
    int gridSoftmax = (samples + blockSizeLinear - 1) / blockSizeLinear;
    int gridW1Update = (INPUT_DIM * HIDDEN_DIM + blockSizeLinear - 1) / blockSizeLinear;
    int gridW2Update = (HIDDEN_DIM * OUTPUT_DIM + blockSizeLinear - 1) / blockSizeLinear;

    for (int epoch = 0; epoch < epochs; epoch++) {
        matmul_kernel<<<blocksHidden, threadsPerBlock>>>(d_batch_data, net->d_w1, d_hidden_layer, samples, HIDDEN_DIM, INPUT_DIM);
        relu_kernel<<<gridHiddenLinear, blockSizeLinear>>>(d_hidden_layer, samples * HIDDEN_DIM);
        matmul_kernel<<<blocksOutput, threadsPerBlock>>>(d_hidden_layer, net->d_w2, d_output_layer, samples, OUTPUT_DIM, HIDDEN_DIM);
        softmax_kernel<<<gridSoftmax, blockSizeLinear>>>(d_output_layer, samples, OUTPUT_DIM);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output_layer, samples * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        
        float total_loss = 0.0f;
        int correct_predictions = 0;

        for (int i = 0; i < samples; i++) {
            int target_label = train_labels[i];
            float prob = h_output[i * OUTPUT_DIM + target_label];
            if (prob < 1e-7f) prob = 1e-7f; 
            total_loss += -logf(prob);

            int highest_prob_class = 0;
            float max_prob = h_output[i * OUTPUT_DIM];
            for (int c = 1; c < OUTPUT_DIM; c++) {
                if (h_output[i * OUTPUT_DIM + c] > max_prob) {
                    max_prob = h_output[i * OUTPUT_DIM + c];
                    highest_prob_class = c;
                }
            }
            if (highest_prob_class == target_label) {
                correct_predictions++;
            }
        }
        
        float average_loss = total_loss / samples;
        float accuracy = ((float)correct_predictions / samples) * 100.0f;

        fprintf(f_historial, "%d,%.4f,%.2f\n", epoch + 1, average_loss, accuracy);

        if ((epoch + 1) % 50 == 0 || epoch == 0 || epoch == epochs - 1) {
            printf("  Época %d/%d -> Loss Promedio: %.4f | Accuracy: %.2f%%\n", epoch + 1, epochs, average_loss, accuracy);
        }

        compute_output_error_kernel<<<blocksOutput, threadsPerBlock>>>(d_output_layer, d_batch_labels, samples, OUTPUT_DIM);
        cudaDeviceSynchronize();

        matmul_transA_kernel<<<blocksOutput, threadsPerBlock>>>(d_hidden_layer, d_output_layer, d_grad_w2, HIDDEN_DIM, OUTPUT_DIM, samples);
        matmul_transB_kernel<<<blocksOutput, threadsPerBlock>>>(d_output_layer, net->d_w2, d_hidden_error, samples, HIDDEN_DIM, OUTPUT_DIM);
        
        backprop_relu_kernel<<<gridHiddenLinear, blockSizeLinear>>>(d_hidden_error, d_hidden_layer, samples * HIDDEN_DIM);
        cudaDeviceSynchronize();

        matmul_transA_kernel<<<blocksW1Grad, threadsPerBlock>>>(d_batch_data, d_hidden_error, d_grad_w1, INPUT_DIM, HIDDEN_DIM, samples);
        cudaDeviceSynchronize();

        update_weights_kernel<<<gridW1Update, blockSizeLinear>>>(net->d_w1, d_grad_w1, INPUT_DIM * HIDDEN_DIM, LEARNING_RATE);
        update_weights_kernel<<<gridW2Update, blockSizeLinear>>>(net->d_w2, d_grad_w2, HIDDEN_DIM * OUTPUT_DIM, LEARNING_RATE);
        cudaDeviceSynchronize();
    }
    fclose(f_historial);

    cudaFree(d_batch_data); cudaFree(d_batch_labels);
    cudaFree(d_hidden_layer); cudaFree(d_output_layer);
    cudaFree(d_grad_w1); cudaFree(d_grad_w2);
    cudaFree(d_hidden_error);
    free(h_output);
}

float evaluate_network(NeuralNetwork* net, float* data, int* labels, int samples) {
    float* d_data; float* d_hidden; float* d_output;
    cudaMalloc((void**)&d_data, samples * INPUT_DIM * sizeof(float));
    cudaMalloc((void**)&d_hidden, samples * HIDDEN_DIM * sizeof(float));
    cudaMalloc((void**)&d_output, samples * OUTPUT_DIM * sizeof(float));

    cudaMemcpy(d_data, data, samples * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksHidden((HIDDEN_DIM + 15) / 16, (samples + 15) / 16);
    dim3 blocksOutput((OUTPUT_DIM + 15) / 16, (samples + 15) / 16);

    matmul_kernel<<<blocksHidden, threadsPerBlock>>>(d_data, net->d_w1, d_hidden, samples, HIDDEN_DIM, INPUT_DIM);
    relu_kernel<<<(samples * HIDDEN_DIM + 255) / 256, 256>>>(d_hidden, samples * HIDDEN_DIM);
    matmul_kernel<<<blocksOutput, threadsPerBlock>>>(d_hidden, net->d_w2, d_output, samples, OUTPUT_DIM, HIDDEN_DIM);
    softmax_kernel<<<(samples + 255) / 256, 256>>>(d_output, samples, OUTPUT_DIM);

    float* h_output = (float*)malloc(samples * OUTPUT_DIM * sizeof(float));
    cudaMemcpy(h_output, d_output, samples * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    // --- MODIFICADO: Guarda en metricas_finales.txt el set que se esté evaluando (ej. Test) ---
    if (samples > 1) {
        FILE* f_metrics = fopen("metricas_finales.txt", "w");
        if (f_metrics != NULL) {
            fprintf(f_metrics, "real,predicho\n");
            for (int i = 0; i < samples; i++) {
                int highest_prob_class = 0;
                float max_prob = h_output[i * OUTPUT_DIM];
                for (int c = 1; c < OUTPUT_DIM; c++) {
                    if (h_output[i * OUTPUT_DIM + c] > max_prob) {
                        max_prob = h_output[i * OUTPUT_DIM + c];
                        highest_prob_class = c;
                    }
                }
                fprintf(f_metrics, "%d,%d\n", labels[i], highest_prob_class);
            }
            fclose(f_metrics);
            printf("Métricas de evaluación exportadas correctamente a 'metricas_finales.txt'.\n");
        }
    }

    int correct_predictions = 0;
    for (int i = 0; i < samples; i++) {
        int highest_prob_class = 0;
        float max_prob = h_output[i * OUTPUT_DIM];
        for (int c = 1; c < OUTPUT_DIM; c++) {
            if (h_output[i * OUTPUT_DIM + c] > max_prob) {
                max_prob = h_output[i * OUTPUT_DIM + c];
                highest_prob_class = c;
            }
        }
        if (highest_prob_class == labels[i]) {
            correct_predictions++;
        }
    }
    
    if (samples == 1) {
        int final_pred = 0;
        float max_p = h_output[0];
        for (int c = 1; c < OUTPUT_DIM; c++) {
            if (h_output[c] > max_p) {
                max_p = h_output[c];
                final_pred = c;
            }
        }
        printf("\nPRED_CLASS: %d\n", final_pred); 
    }

    cudaFree(d_data); cudaFree(d_hidden); cudaFree(d_output);
    free(h_output);

    return ((float)correct_predictions / samples) * 100.0f;
}

void free_network(NeuralNetwork* net) {
    printf("Liberando memoria de la GPU...\n");
    if (net->d_w1) cudaFree(net->d_w1);
    if (net->d_w2) cudaFree(net->d_w2);
}

// ============================================================================
// 3. PERSISTENCIA: GUARDAR Y CARGAR EL MODELO BINARIO
// ============================================================================

int save_network(NeuralNetwork* net, const char* filename) {
    printf("Guardando pesos del modelo en archivo binario: %s...\n", filename);
    int w1_size = INPUT_DIM * HIDDEN_DIM;
    int w2_size = HIDDEN_DIM * OUTPUT_DIM;

    float* h_w1 = (float*)malloc(w1_size * sizeof(float));
    float* h_w2 = (float*)malloc(w2_size * sizeof(float));

    if (!h_w1 || !h_w2) {
        printf("[ERROR] Fallo de memoria en Host para exportar pesos.\n");
        if (h_w1) free(h_w1); if (h_w2) free(h_w2);
        return 0;
    }

    cudaMemcpy(h_w1, net->d_w1, w1_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w2, net->d_w2, w2_size * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("[ERROR] No se pudo abrir %s para escritura.\n", filename);
        free(h_w1); free(h_w2);
        return 0;
    }

    fwrite(h_w1, sizeof(float), w1_size, file);
    fwrite(h_w2, sizeof(float), w2_size, file);
    fclose(file);

    free(h_w1); free(h_w2);
    printf("¡Pesos guardados correctamente!\n");
    return 1;
}

int load_network(NeuralNetwork* net, const char* filename) {
    printf("Cargando pesos del modelo desde: %s...\n", filename);
    int w1_size = INPUT_DIM * HIDDEN_DIM;
    int w2_size = HIDDEN_DIM * OUTPUT_DIM;

    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("[AVISO] No se encontró el archivo del modelo %s.\n", filename);
        return 0;
    }

    float* h_w1 = (float*)malloc(w1_size * sizeof(float));
    float* h_w2 = (float*)malloc(w2_size * sizeof(float));

    fread(h_w1, sizeof(float), w1_size, file);
    fread(h_w2, sizeof(float), w2_size, file);
    fclose(file);

    if (net->d_w1 == NULL) cudaMalloc((void**)&(net->d_w1), w1_size * sizeof(float));
    if (net->d_w2 == NULL) cudaMalloc((void**)&(net->d_w2), w2_size * sizeof(float));

    cudaMemcpy(net->d_w1, h_w1, w1_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net->d_w2, h_w2, w2_size * sizeof(float), cudaMemcpyHostToDevice);

    free(h_w1); free(h_w2);
    printf("¡Pesos cargados en la GPU con éxito!\n");
    return 1;
}