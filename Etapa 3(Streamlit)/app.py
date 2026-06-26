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

# --- PANEL LATERAL TÉCNICO ---
with st.sidebar:
    st.markdown("## 🛠️ Especificaciones de Hardware")
    st.info("⚡ **Back-end:** C++ / NVIDIA CUDA API\n\n⚙️ **Aceleración:** Cores de GPU dedicados")
    st.markdown("---")
    st.markdown("### 📊 Arquitectura de la Red")
    st.markdown("- **Entrada:** 4096 Neuronas (64x64 px)\n- **Capa Oculta:** 256 Neuronas (ReLU)\n- **Capa Salida:** Softmax de 2 Clases")

# --- CUERPO PRINCIPAL ---
st.title("🧠 Detector de Mascarillas Inteligente")
st.caption("Clasificación binaria en tiempo real mediante procesamiento masivo en paralelo con GPU.")
st.markdown("---")

CLASES = {
    0: "🔓 SIN MASCARILLA",
    1: "😷 CON MASCARILLA"
}

# Subida del archivo ocupando toda la fila superior
uploaded_file = st.file_uploader("📤 Arrastra o selecciona una imagen de prueba...", type=["bin", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Crear dos columnas para visualización paralela
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file.name.endswith(('png', 'jpg', 'jpeg')):
        image_orig = Image.open(uploaded_file)
        image_gray = image_orig.convert('L')
        image_resized = image_gray.resize((64, 64))
        
        with col1:
            st.markdown("##### 📸 Vista Original")
            st.image(image_orig, use_container_width=True)
            
        with col2:
            st.markdown("##### 📐 Entrada Redimensionada (64x64 Gray)")
            st.image(image_resized, width=180, caption="Entrada real a la Red CUDA")
            
        pixel_data = np.array(image_resized, dtype=np.float32) / 255.0
        flat_data = pixel_data.flatten()
    else:
        bytes_data = uploaded_file.read()
        flat_data = np.frombuffer(bytes_data, dtype=np.float32)
        st.success(f"📦 Archivo binario cargado de forma directa. Elementos: {len(flat_data)}")

    st.markdown("---")

    if len(flat_data) == 4096:
        # Botón destacado para disparar la ejecución
        if st.button("🚀 CLASIFICAR IMAGEN CON LA GPU"):
            with st.spinner("Ejecutando inferencia en memoria de video (VRAM)..."):
                temp_filename = "temp_streamlit_sample.bin"
                flat_data.tofile(temp_filename)
                
                try:
                    # Ejecutar ejecutable de Windows sumando el .exe
                    ejecutable = "./neural_net.exe" if os.name == 'nt' else "./neural_net"
                    result = subprocess.run(
                        [ejecutable, temp_filename], 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    
                    output_console = result.stdout
                    
                    # Parsing avanzado de la consola de CUDA
                    clase_num = None
                    confianza = 0.0
                    
                    for line in output_console.split("\n"):
                        if "PRED_CLASS:" in line:
                            clase_num = int(line.split(":")[1].strip())
                        if "PRED_CONF:" in line:
                            confianza = float(line.split(":")[1].strip()) * 100.0
                    
                    # --- RENDERIZADO DEL PANEL DE RESULTADOS ---
                    if clase_num is not None:
                        prediccion_detectada = CLASES.get(clase_num, f"Clase {clase_num}")
                        
                        st.subheader("🎯 Diagnóstico del Modelo")
                        
                        # Layout dinámico en base al veredicto
                        res_col1, res_col2 = st.columns([1.5, 2])
                        
                        with res_col1:
                            if clase_num == 1:
                                st.success(f"#### Veredicto:  \n**{prediccion_detectada}**")
                            else:
                                st.error(f"#### Veredicto:  \n**{prediccion_detectada}**")
                        
                        with res_col2:
                            st.markdown(f"**Confianza de la Red:** `{confianza:.2f}%`")
                            # Barra de progreso visual usando colores temáticos
                            st.progress(min(max(confianza / 100.0, 0.0), 1.0))
                            
                        st.balloons()
                    else:
                        st.warning("⚠️ Inferencia completada, pero no se capturaron las etiquetas de salida en los logs.")
                    
                    # Logs técnicos colapsados elegantemente
                    with st.expander("🛠️ Ver métricas detalladas de los hilos de ejecución de la GPU"):
                        st.code(output_console)
                        
                except Exception as e:
                    st.error(f"❌ Error crítico al comunicar con los hilos de ejecución de CUDA: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
    else:
        st.error(f"⚠️ Vector inválido: La topología de la red requiere una matriz plana de 4096 píxeles (Recibidos: {len(flat_data)}).")