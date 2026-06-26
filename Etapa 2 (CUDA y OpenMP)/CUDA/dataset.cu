#include "dataset.cuh"
#include <stdio.h>
#include <stdlib.h>

bool load_dataset(
    const char* data_file,
    const char* label_file,
    float** data,
    int** labels,
    int samples)
{
    FILE* fd = fopen(data_file, "rb");
    FILE* fl = fopen(label_file, "rb");

    if(!fd || !fl)
        return false;

    *data = (float*)malloc(samples * 4096 * sizeof(float));
    *labels = (int*)malloc(samples * sizeof(int));

    fread(*data, sizeof(float), samples * 4096, fd);
    fread(*labels, sizeof(int), samples, fl);

    fclose(fd);
    fclose(fl);

    return true;
}