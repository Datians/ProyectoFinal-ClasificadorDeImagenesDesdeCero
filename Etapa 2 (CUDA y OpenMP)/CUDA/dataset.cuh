#ifndef DATASET_H
#define DATASET_H

bool load_dataset(
    const char* data_file,
    const char* label_file,
    float** data,
    int** labels,
    int samples
);

#endif