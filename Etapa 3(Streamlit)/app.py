import streamlit as st
import subprocess
import numpy as np
from PIL import Image
import os

# Configuración premium de la página
st.set_page_config(
    page_title="CUDA Face Mask Detector",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Parámetros del modelo
INPUT_DIM = 4096
HIDDEN_DIM = 256
MODEL_FILE = "modelo_red.bin"
TEMP_FILENAME = "temp_streamlit_sample.bin"

# Estilos CSS personalizados para pulir detalles visuales
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Diccionario de clases
CLASES = {
    0: "🔓 SIN MASCARILLA",
    1: "😷 CON MASCARILLA"
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def procesar_entrada(uploaded_file):
    """
    Mantiene el mismo preprocesamiento:
    imagen -> escala de grises -> 64x64 -> normalizar -> aplanar.
    Si el archivo es .bin, lo carga directamente como float32.
    """
    nombre_archivo = uploaded_file.name.lower()

    if nombre_archivo.endswith(("png", "jpg", "jpeg")):
        image_orig = Image.open(uploaded_file)
        image_gray = image_orig.convert("L")
        image_resized = image_gray.resize((64, 64))

        pixel_data = np.array(image_resized, dtype=np.float32) / 255.0
        flat_data = pixel_data.flatten()

        return flat_data, image_orig, image_resized, "Imagen"

    bytes_data = uploaded_file.read()
    flat_data = np.frombuffer(bytes_data, dtype=np.float32)

    return flat_data, None, None, "Archivo binario"


def ejecutar_cuda(flat_data):
    """
    Mantiene la lógica CUDA original:
    guarda temp_streamlit_sample.bin y ejecuta neural_net/neural_net.exe.
    """
    temp_filename = TEMP_FILENAME
    flat_data.tofile(temp_filename)

    try:
        ejecutable = "./neural_net.exe" if os.name == "nt" else "./neural_net"

        result = subprocess.run(
            [ejecutable, temp_filename],
            capture_output=True,
            text=True,
            check=True
        )

        return result.stdout

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def leer_salida_cuda(output_console):
    """
    Lee PRED_CLASS y PRED_CONF/PRED_PROB desde los logs de CUDA.
    """
    clase_num = None
    confianza = 0.0

    for line in output_console.split("\n"):
        if "PRED_CLASS:" in line:
            clase_num = int(line.split(":")[1].strip())
        if "PRED_CONF:" in line:
            confianza = float(line.split(":")[1].strip()) * 100.0
        if "PRED_PROB:" in line:
            confianza = float(line.split(":")[1].strip()) * 100.0

    return clase_num, confianza


@st.cache_data(show_spinner=False)
def cargar_pesos_cpu(model_file):
    """
    Carga modelo_red.bin y detecta automáticamente la cantidad de salidas.

    Formato esperado:
    W1 = 4096 x 256
    W2 = 256 x output_dim
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No se encontró {model_file} en la carpeta actual.")

    pesos = np.fromfile(model_file, dtype=np.float32)

    w1_size = INPUT_DIM * HIDDEN_DIM

    if pesos.size <= w1_size:
        raise ValueError("El archivo de pesos no tiene tamaño suficiente para W1 y W2.")

    remaining = pesos.size - w1_size

    if remaining % HIDDEN_DIM != 0:
        raise ValueError("El tamaño de modelo_red.bin no coincide con la arquitectura esperada.")

    output_dim = remaining // HIDDEN_DIM

    w1 = pesos[:w1_size].reshape(INPUT_DIM, HIDDEN_DIM)
    w2 = pesos[w1_size:].reshape(HIDDEN_DIM, output_dim)

    return w1, w2, output_dim, pesos.size


def softmax(logits):
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


def ejecutar_cpu(flat_data):
    """
    Hace inferencia en CPU con NumPy:
    entrada -> W1 -> ReLU -> W2 -> Softmax -> clase.
    """
    w1, w2, output_dim, total_pesos = cargar_pesos_cpu(MODEL_FILE)

    x = flat_data.astype(np.float32)

    hidden = np.matmul(x, w1)
    hidden = np.maximum(hidden, 0.0)

    logits = np.matmul(hidden, w2)
    probabilities = softmax(logits)

    clase_num = int(np.argmax(probabilities))
    confianza = float(probabilities[clase_num]) * 100.0

    return clase_num, confianza, probabilities, output_dim, total_pesos


def mostrar_panel_resultado(clase_num, confianza, metodo):
    """
    Renderiza el resultado para GPU o CPU.
    """
    if clase_num is not None:
        prediccion_detectada = CLASES.get(clase_num, f"Clase {clase_num}")

        st.subheader("🎯 Diagnóstico del Modelo")

        res_col1, res_col2 = st.columns([1.5, 2])

        with res_col1:
            if clase_num == 1:
                st.success(f"#### Veredicto {metodo}:  \n**{prediccion_detectada}**")
            else:
                st.error(f"#### Veredicto {metodo}:  \n**{prediccion_detectada}**")

        with res_col2:
            st.markdown(f"**Confianza de la Red:** `{confianza:.2f}%`")
            st.progress(min(max(confianza / 100.0, 0.0), 1.0))

        st.balloons()
    else:
        st.warning("⚠️ Inferencia completada, pero no se capturaron las etiquetas de salida en los logs.")


# ============================================================
# PANEL LATERAL TÉCNICO
# ============================================================

with st.sidebar:
    st.markdown("## 🛠️ Especificaciones de Hardware")
    st.info(
        "⚡ **Back-end GPU:** C++ / NVIDIA CUDA API\n\n"
        "🧠 **Back-end CPU:** NumPy\n\n"
        "⚙️ **Aceleración:** Cores de GPU dedicados cuando CUDA esté disponible"
    )
    st.markdown("---")
    st.markdown("### 📊 Arquitectura de la Red")
    st.markdown(
        "- **Entrada:** 4096 Neuronas (64x64 px)\n"
        "- **Capa Oculta:** 256 Neuronas (ReLU)\n"
        "- **Capa Salida:** Softmax"
    )

    st.markdown("---")
    st.markdown("### 📁 Archivos requeridos")

    if os.path.exists("./neural_net") or os.path.exists("./neural_net.exe"):
        st.success("Ejecutable neural_net encontrado.")
    else:
        st.warning("No se detectó neural_net en la carpeta actual.")

    if os.path.exists(MODEL_FILE):
        st.success("modelo_red.bin encontrado.")
    else:
        st.warning("No se detectó modelo_red.bin en la carpeta actual.")


# ============================================================
# CUERPO PRINCIPAL
# ============================================================

st.title("🧠 Detector de Mascarillas Inteligente")
st.caption("Clasificación binaria en tiempo real mediante GPU CUDA o inferencia alternativa en CPU.")
st.markdown("---")

tab_upload, tab_camera = st.tabs(["📤 Subir imagen", "📷 Tomar foto"])

uploaded_file = None
camera_file = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "📤 Arrastra o selecciona una imagen de prueba...",
        type=["bin", "png", "jpg", "jpeg"]
    )

with tab_camera:
    camera_file = st.camera_input("📷 Toma una foto con la cámara")

if camera_file is not None:
    uploaded_file = camera_file

if uploaded_file is not None:
    try:
        flat_data, image_orig, image_resized, tipo_entrada = procesar_entrada(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error al procesar la entrada: {e}")
        st.stop()

    # Crear dos columnas para visualización paralela
    col1, col2 = st.columns([1, 1])

    if image_resized is not None:
        with col1:
            st.markdown("##### 📸 Vista Original")
            st.image(image_orig, use_container_width=True)

        with col2:
            st.markdown("##### 📐 Entrada Redimensionada (64x64 Gray)")
            st.image(image_resized, width=180, caption="Entrada real a la Red")
    else:
        st.success(f"📦 Archivo binario cargado de forma directa. Elementos: {len(flat_data)}")

    st.markdown("---")

    met1, met2, met3 = st.columns(3)

    with met1:
        st.metric("Tipo de entrada", tipo_entrada)

    with met2:
        st.metric("Características", len(flat_data))

    with met3:
        st.metric("Rango", f"{flat_data.min():.3f} - {flat_data.max():.3f}")

    st.markdown("---")

    if len(flat_data) == 4096:
        col_gpu, col_cpu = st.columns(2)

        with col_gpu:
            boton_gpu = st.button("🚀 CLASIFICAR IMAGEN CON GPU CUDA", type="primary")

        with col_cpu:
            boton_cpu = st.button("🧠 CLASIFICAR IMAGEN CON CPU")

        if boton_gpu:
            with st.spinner("Ejecutando inferencia en memoria de video (VRAM)..."):
                try:
                    output_console = ejecutar_cuda(flat_data)
                    clase_num, confianza = leer_salida_cuda(output_console)

                    mostrar_panel_resultado(clase_num, confianza, "GPU CUDA")

                    with st.expander("🛠️ Ver métricas detalladas de los hilos de ejecución de la GPU"):
                        st.code(output_console)

                except Exception as e:
                    st.error(f"❌ Error crítico al comunicar con los hilos de ejecución de CUDA: {e}")

        if boton_cpu:
            with st.spinner("Ejecutando inferencia en CPU con NumPy..."):
                try:
                    clase_num, confianza, probabilities, output_dim, total_pesos = ejecutar_cpu(flat_data)

                    mostrar_panel_resultado(clase_num, confianza, "CPU")

                    with st.expander("🛠️ Ver detalles técnicos de la inferencia CPU"):
                        st.write(f"Total de pesos cargados: {total_pesos}")
                        st.write(f"Dimensión de salida detectada: {output_dim}")
                        st.write("Probabilidades por clase:")

                        for i, p in enumerate(probabilities):
                            st.write(f"Clase {i}: {p * 100:.4f}%")

                        if output_dim != 2:
                            st.warning(
                                "El modelo cargado no tiene 2 salidas. "
                                "Si el problema es binario, revisa que modelo_red.bin corresponda "
                                "a la arquitectura final del entrenamiento."
                            )

                except Exception as e:
                    st.error(f"❌ Error al ejecutar la inferencia CPU: {e}")

    else:
        st.error(
            f"⚠️ Vector inválido: La topología de la red requiere una matriz plana de 4096 píxeles "
            f"(Recibidos: {len(flat_data)})."
        )
else:
    st.info("Sube una imagen, carga un archivo .bin o toma una foto para comenzar.")
