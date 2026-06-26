#ifndef NN_H
#define NN_H

struct NeuralNetwork
{
    float* d_w1;
    float* d_w2;
};

void initialize_network(NeuralNetwork* net);

void train_network(
    NeuralNetwork* net,
    float* train_data,
    int* train_labels,
    int samples,
    int epochs
);

float evaluate_network(
    NeuralNetwork* net,
    float* data,
    int* labels,
    int samples
);

// Funciones para guardar y cargar el modelo (Añadidas para solucionar el error)
int save_network(NeuralNetwork* net, const char* filename);
int load_network(NeuralNetwork* net, const char* filename);

void free_network(NeuralNetwork* net);

#endif