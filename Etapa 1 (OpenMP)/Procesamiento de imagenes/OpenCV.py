"""
Script de preparación de dataset (PASO PREVIO a tu pipeline en C/OpenMP).

Qué hace:
1. Corrige la orientación EXIF de las fotos del celular (evita que salgan giradas).
2. Detecta la cara con un Haar Cascade de OpenCV.
3. Recorta la cara con un margen extra (para no perder mascarilla/cuello/orejas).
4. Guarda el recorte en una carpeta de salida, lista para que tu código en C la lea.

Requisitos:
    pip install opencv-python pillow

Uso:
    Ajusta las rutas en la sección CONFIGURACIÓN abajo y corre:
    python preparar_dataset.py
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps

# =========================
# CONFIGURACIÓN
# =========================
# Cada tupla es: (carpeta_fotos_crudas, carpeta_salida_recortada)
CARPETAS = [
    ("with_mask", "dataset/with_mask"),
    ("without_mask", "dataset/without_mask"),
]

MARGEN = 0.35  # cuánto márgen extra alrededor de la cara detectada (35%)
TAMANO_MINIMO_CARA = (80, 80)  # ignora detecciones diminutas (falsos positivos)

# =========================
# SCRIPT
# =========================

detector_cara = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Cascade específico para ojos CON lentes (el normal falla mucho con marcos de lentes)
detector_ojos = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

# Proporciones antropométricas aproximadas (ajusta si el recorte queda muy
# grande/pequeño al revisar los resultados):
#   ancho de cara  ~= 2.2 x distancia entre ojos
#   alto de cara   ~= 2.8 x distancia entre ojos
#   los ojos están ~40% hacia abajo desde la frente
FACTOR_ANCHO = 2.2
FACTOR_ALTO = 2.8
FACTOR_ARRIBA = 0.4


def corregir_orientacion(ruta_entrada):
    """Lee la imagen con Pillow y aplica la rotación EXIF real (no solo metadata)."""
    img_pil = Image.open(ruta_entrada)
    img_pil = ImageOps.exif_transpose(img_pil)
    img_pil = img_pil.convert("RGB")
    # Convertir de Pillow (RGB) a formato OpenCV (BGR, numpy array)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def _recortar_con_margen(img_bgr, x, y, w, h):
    mx = int(w * MARGEN)
    my = int(h * MARGEN)
    alto_img, ancho_img = img_bgr.shape[:2]
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(ancho_img, x + w + mx)
    y1 = min(alto_img, y + h + my)
    return img_bgr[y0:y1, x0:x1]


def _recortar_por_ojos(img_bgr, gris):
    """Plan B para fotos con mascarilla: ubica los ojos (con lentes) y estima
    el resto de la cara con proporciones antropométricas."""
    ojos = detector_ojos.detectMultiScale(
        gris, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25)
    )
    if len(ojos) < 2:
        return None

    # Tomar los 2 ojos más separados horizontalmente (evita falsos positivos pequeños)
    ojos_ordenados = sorted(ojos, key=lambda o: o[0])
    ox0, oy0, ow0, oh0 = ojos_ordenados[0]
    ox1, oy1, ow1, oh1 = ojos_ordenados[-1]

    cx0, cy0 = ox0 + ow0 / 2, oy0 + oh0 / 2
    cx1, cy1 = ox1 + ow1 / 2, oy1 + oh1 / 2

    distancia_ojos = abs(cx1 - cx0)
    if distancia_ojos < 10:  # los dos "ojos" detectados están casi en el mismo punto
        return None

    centro_x = (cx0 + cx1) / 2
    centro_y = (cy0 + cy1) / 2

    ancho_cara = distancia_ojos * FACTOR_ANCHO
    alto_cara = distancia_ojos * FACTOR_ALTO

    y_arriba = centro_y - alto_cara * FACTOR_ARRIBA
    x_izq = centro_x - ancho_cara / 2

    alto_img, ancho_img = img_bgr.shape[:2]
    x0 = max(0, int(x_izq))
    y0 = max(0, int(y_arriba))
    x1 = min(ancho_img, int(x_izq + ancho_cara))
    y1 = min(alto_img, int(y_arriba + alto_cara))

    return img_bgr[y0:y1, x0:x1]


def recortar_cara(img_bgr):
    """Intenta detectar cara completa; si falla (ej. por la mascarilla),
    usa la posición de los ojos para estimar el recorte. None si ninguna funciona."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    caras = detector_cara.detectMultiScale(
        gris, scaleFactor=1.1, minNeighbors=5, minSize=TAMANO_MINIMO_CARA
    )
    if len(caras) > 0:
        x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
        return _recortar_con_margen(img_bgr, x, y, w, h)

    return _recortar_por_ojos(img_bgr, gris)


def procesar_carpeta(carpeta_entrada, carpeta_salida):
    os.makedirs(carpeta_salida, exist_ok=True)

    extensiones = (".jpg", ".jpeg", ".png")
    archivos = [f for f in os.listdir(carpeta_entrada) if f.lower().endswith(extensiones)]

    exitosas = 0
    sin_cara = []

    for nombre in archivos:
        ruta_entrada = os.path.join(carpeta_entrada, nombre)

        try:
            img_bgr = corregir_orientacion(ruta_entrada)
        except Exception as e:
            print(f"  [ERROR] No se pudo abrir {nombre}: {e}")
            continue

        recorte = recortar_cara(img_bgr)

        if recorte is None:
            sin_cara.append(nombre)
            continue

        nombre_salida = os.path.splitext(nombre)[0] + ".jpg"
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)
        cv2.imwrite(ruta_salida, recorte, [cv2.IMWRITE_JPEG_QUALITY, 95])
        exitosas += 1

    print(f"  -> {exitosas}/{len(archivos)} procesadas correctamente.")
    if sin_cara:
        print(f"  -> {len(sin_cara)} SIN cara detectada (revisar manualmente): {sin_cara}")


def main():
    for carpeta_entrada, carpeta_salida in CARPETAS:
        print(f"Procesando: {carpeta_entrada} -> {carpeta_salida}")
        if not os.path.isdir(carpeta_entrada):
            print(f"  [AVISO] No existe la carpeta {carpeta_entrada}, se omite.")
            continue
        procesar_carpeta(carpeta_entrada, carpeta_salida)

    print("\n¡Listo! Las fotos recortadas están en 'dataset/', listas para tu pipeline en C.")


if __name__ == "__main__":
    main()