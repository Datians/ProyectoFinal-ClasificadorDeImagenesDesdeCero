#ifndef PREPROCESS_H
#define PREPROCESS_H

// Procesa una imagen completa y guarda 4096 floats en 'output_features'
int process_image(const char* filepath, float* output_features);

#endif