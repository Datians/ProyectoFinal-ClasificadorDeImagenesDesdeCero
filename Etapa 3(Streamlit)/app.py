import streamlit as st
import subprocess
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="CUDA Mask Detector",
    page_icon="😷",
    layout="centered"
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, #151b2b 0%, #0e1117 45%, #07090f 100%);
            color: #f8fafc;
        }

        .hero-card {
            padding: 1.8rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(15,23,42,0.88));
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 16px 40px rgba(0,0,0,0.38);
            margin-bottom: 1.4rem;
        }

        .hero-title {
            font-size: 2.45rem;
            font-weight: 850;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1.03rem;
            line-height: 1.55;
        }

        .section-card {
            padding: 1.1rem;
            border-radius: 18px;
            background: rgba(15,23,42,0.68);
            border: 1px solid rgba(148,163,184,0.18);
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .ready-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(37,99,235,0.13);
            border: 1px solid rgba(96,165,250,0.35);
            color: #dbeafe;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .success-result {
            padding: 1.25rem;
            border-radius: 16px;
            background: rgba(22,163,74,0.22);
            border: 1px solid rgba(34,197,94,0.48);
            color: #4ade80;
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 0.8rem;
        }

        .error-result {
            padding: 1.25rem;
            border-radius: 16px;
            background: rgba(220,38,38,0.22);
            border: 1px solid rgba(248,113,113,0.48);
            color: #f87171;
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 0.8rem;
        }

        .warning-result {
            padding: 1.15rem;
            border-radius: 16px;
            background: rgba(234,179,8,0.18);
            border: 1px solid rgba(250,204,21,0.45);
            color: #fde047;
            font-size: 1.05rem;
            font-weight: 700;
            margin-top: 0.8rem;
        }

        div[data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 0.8rem;
            border: 1px dashed rgba(148,163,184,0.45);
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            padding: 0.75rem;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🧠 Detector de Mascarillas</div>
        <h3 style="margin-top: 0rem;">Acelerado por GPU CUDA</h3>
        <p class="hero-subtitle">
            Sube una imagen o toma una foto para clasificarla en tiempo real usando tu red neuronal en CUDA.
            La imagen será convertida a escala de grises, redimensionada a 64x64 y enviada al modelo.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Diccionario de clases (Ajusta el orden según tu dataset)
CLASES = {
    0: "🔓 SIN MASCARILLA",
    1: "😷 CON MASCARILLA"
}

with st.expander("ℹ️ Información del sistema"):
    col_info_1, col_info_2 = st.columns(2)

    with col_info_1:
        st.metric("Entrada esperada", "4096 características")
        st.metric("Tamaño procesado", "64x64")

    with col_info_2:
        if os.path.exists("./neural_net") or os.path.exists("./neural_net.exe"):
            st.success("Ejecutable neural_net encontrado.")
        else:
            st.warning("No se detectó neural_net en la carpeta actual.")

        if os.path.exists("modelo_red.bin"):
            st.success("modelo_red.bin encontrado.")
        else:
            st.warning("No se detectó modelo_red.bin en la carpeta actual.")

st.markdown("### 📸 Entrada de imagen")

tab_upload, tab_camera = st.tabs(["📁 Subir imagen", "📷 Tomar foto"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Elige una imagen o archivo binario",
        type=["bin", "png", "jpg", "jpeg"]
    )

with tab_camera:
    camera_file = st.camera_input("Toma una foto con la cámara")

if camera_file is not None:
    uploaded_file = camera_file

if uploaded_file is not None:
    if uploaded_file.name.endswith(('png', 'jpg', 'jpeg')):
        image = Image.open(uploaded_file).convert('L') # Escala de grises
        image_resized = image.resize((64, 64)) # 64x64 = 4096

        st.markdown("---")
        st.subheader("🧪 Datos procesados")

        col_img_1, col_img_2 = st.columns(2)

        with col_img_1:
            st.image(image, caption="Imagen original en escala de grises", width=300)

        with col_img_2:
            st.image(image_resized, caption="Imagen procesada para la Red (64x64)", width=220)

        pixel_data = np.array(image_resized, dtype=np.float32) / 255.0
        flat_data = pixel_data.flatten()
        tipo_entrada = "Imagen"
    else:
        bytes_data = uploaded_file.read()
        flat_data = np.frombuffer(bytes_data, dtype=np.float32)
        tipo_entrada = "Archivo binario"

        st.markdown("---")
        st.subheader("🧪 Datos procesados")
        st.success(f"Archivo binario cargado. Tamaño: {len(flat_data)} características.")

    col_data_1, col_data_2, col_data_3 = st.columns(3)

    with col_data_1:
        st.metric("Tipo de entrada", tipo_entrada)

    with col_data_2:
        st.metric("Características", len(flat_data))

    with col_data_3:
        st.metric("Rango", f"{flat_data.min():.3f} - {flat_data.max():.3f}")

    if len(flat_data) == 4096:
        st.markdown("---")


        if st.button("🚀 Clasificar con la GPU (CUDA)", type="primary", use_container_width=True):
            with st.spinner("Procesando en los Tensor Cores / CUDA Cores..."):
                temp_filename = "temp_streamlit_sample.bin"
                flat_data.tofile(temp_filename)

                try:
                    result = subprocess.run(
                        ["./neural_net", temp_filename],
                        capture_output=True,
                        text=True,
                        check=True
                    )

                    output_console = result.stdout

                    # Buscamos la línea mágica que imprimió CUDA
                    prediccion_detectada = None
                    for line in output_console.split("\n"):
                        if "PRED_CLASS:" in line:
                            clase_num = int(line.split(":")[1].strip())
                            prediccion_detectada = CLASES.get(clase_num, f"Clase desconocida ({clase_num})")

                    # --- DISEÑO DEL RESULTADO EN LA INTERFAZ ---
                    st.markdown("---")
                    st.subheader("🎯 Resultado de la Clasificación:")

                    if prediccion_detectada:
                        if "CON MASCARILLA" in prediccion_detectada:
                            st.markdown(
                                f"""
                                <div class="success-result">
                                    Predicción: {prediccion_detectada}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="error-result">
                                    Predicción: {prediccion_detectada}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            """
                            <div class="warning-result">
                                Se ejecutó bien, pero no se detectó la etiqueta PRED_CLASS en el código C++.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Colapsable para ver la consola de CUDA (Modo desarrollador)
                    with st.expander("🛠️ Ver logs técnicos de la GPU (Consola CUDA)"):
                        st.code(output_console)

                    st.balloons()

                except Exception as e:
                    st.error(f"Hubo un error al ejecutar el código CUDA: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
    else:
        st.error(f"Error: La red espera 4096 características, recibidas {len(flat_data)}.")
else:
    st.info("Sube una imagen, carga un archivo .bin o toma una foto para comenzar.")
