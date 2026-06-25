#include "preprocess.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define WORK_SIZE 256

// Función auxiliar para aplicar filtros 3x3
float aplicar_kernel(unsigned char* img, int x, int y, int w, int h, float kernel[3][3]) {
    float sum = 0.0;
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int px = x + kx;
            int py = y + ky;
            if (px < 0) px = 0;
            if (px >= w) px = w - 1;
            if (py < 0) py = 0;
            if (py >= h) py = h - 1;
            sum += img[py * w + px] * kernel[ky + 1][kx + 1];
        }
    }
    return sum;
}

// Reduce/agranda una imagen a otro tamaño promediando bloques de píxeles
void resize_box_average(unsigned char* src, int src_w, int src_h,
                         unsigned char* dst, int dst_w, int dst_h) {
    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            int x0 = x * src_w / dst_w, x1 = (x + 1) * src_w / dst_w;
            int y0 = y * src_h / dst_h, y1 = (y + 1) * src_h / dst_h;
            if (x1 <= x0) x1 = x0 + 1;
            if (y1 <= y0) y1 = y0 + 1;

            long sum = 0;
            int count = 0;
            for (int yy = y0; yy < y1 && yy < src_h; yy++) {
                for (int xx = x0; xx < x1 && xx < src_w; xx++) {
                    sum += src[yy * src_w + xx];
                    count++;
                }
            }
            dst[y * dst_w + x] = (unsigned char)(count > 0 ? sum / count : 0);
        }
    }
}

int process_image(const char* filepath, float* output_features) {
    int w, h, channels;
    unsigned char *img = stbi_load(filepath, &w, &h, &channels, 1);
    if (img == NULL) return 0; // Error al leer

    // 1. Reducir a un tamaño manejable ANTES de blur/sobel
    //    (clave para fotos de celular de alta resolución)
    unsigned char* work_img = (unsigned char*)malloc(WORK_SIZE * WORK_SIZE);
    if (!work_img) { stbi_image_free(img); return 0; }
    resize_box_average(img, w, h, work_img, WORK_SIZE, WORK_SIZE);
    stbi_image_free(img); // ya no necesitamos la imagen gigante original

    unsigned char* blur_img = (unsigned char*)malloc(WORK_SIZE * WORK_SIZE);
    unsigned char* sobel_img = (unsigned char*)malloc(WORK_SIZE * WORK_SIZE);
    if (!blur_img || !sobel_img) {
        free(work_img); free(blur_img); free(sobel_img);
        return 0;
    }

    float kernel_gauss[3][3] = {
        {1/16.0, 2/16.0, 1/16.0},
        {2/16.0, 4/16.0, 2/16.0},
        {1/16.0, 2/16.0, 1/16.0}
    };
    float kernel_sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float kernel_sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // 2. Filtro Gaussiano (ahora a 256x256, donde el kernel 3x3 sí importa)
    for (int y = 0; y < WORK_SIZE; y++) {
        for (int x = 0; x < WORK_SIZE; x++) {
            blur_img[y * WORK_SIZE + x] =
                (unsigned char)aplicar_kernel(work_img, x, y, WORK_SIZE, WORK_SIZE, kernel_gauss);
        }
    }

    // 3. Filtro Sobel (bordes)
    for (int y = 0; y < WORK_SIZE; y++) {
        for (int x = 0; x < WORK_SIZE; x++) {
            float gx = aplicar_kernel(blur_img, x, y, WORK_SIZE, WORK_SIZE, kernel_sobel_x);
            float gy = aplicar_kernel(blur_img, x, y, WORK_SIZE, WORK_SIZE, kernel_sobel_y);
            float mag = sqrt(gx * gx + gy * gy);
            if (mag > 255.0) mag = 255.0;
            sobel_img[y * WORK_SIZE + x] = (unsigned char)mag;
        }
    }

    // 4, 5 y 6. Reducir a 64x64 (con promedio), normalizar y aplanar
    unsigned char final_img[64 * 64];
    resize_box_average(sobel_img, WORK_SIZE, WORK_SIZE, final_img, 64, 64);
    for (int i = 0; i < 64 * 64; i++) {
        output_features[i] = (float)final_img[i] / 255.0f;
    }

    free(work_img);
    free(blur_img);
    free(sobel_img);

    return 1; // Éxito
}